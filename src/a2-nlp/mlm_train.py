"""
mlm_train.py -- the masked-language-model builders and the pretraining loop.

The counterpart of models.py + train_loop.py for the group's study. Two things differ from the
causal side and both are deliberate:

  the model is a Hugging Face one.  Their downstream fine-tuning loads checkpoints with
    AutoModelForSequenceClassification and AutoModelForTokenClassification, so the pretrained
    artifact has to be a save_pretrained directory, not our .pt. We build the config ourselves
    and let transformers build the model, which keeps the fine-tuning half of their notebook
    working untouched.

  the schedule is counted in STEPS, not epochs.  Their POC found the study compute-bound, so
    steps are the axis that moves the result and "epochs over a 2M-token rung" is a misleading
    unit -- the 2M rung makes ~25 passes and the 32M rung ~1.5 at the same step count. The loop
    below takes a step budget and reports how many passes that turned out to be.

The JSONL it writes is the same schema the causal runs write, treating each logging interval as
an "epoch", so dashboard.py displays these runs with no changes at all.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

RUNS = os.path.join(os.path.dirname(__file__), 'runs')

# Their two size presets, unchanged, so numbers stay comparable with the POC. 'poc' is the fast
# one the notebook uses; 'afriberta' is the size the real study would run at.
SIZE_PRESETS = {
    'poc': dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                intermediate_size=2048),
    'afriberta': dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                      intermediate_size=3072),
}


def compact(n: int) -> str:
    """A short, collision-free rendering of a count: 327k, 2M, 32M, 1500.

    Rounding to the nearest million was the obvious choice and the wrong one -- every cell under
    a million rendered as '0M', so two different runs shared a tag and the second silently
    overwrote the first's checkpoint and history. The same rounding produced a chart legend
    reading '0M tokens' for two different series, which is why this is public: the plots label
    their axes with it too, and one formatter means the plot and the run name cannot disagree.
    """
    for unit, size in (('M', 1_000_000), ('k', 1_000)):
        if n >= size:
            return f'{n / size:.1f}'.rstrip('0').rstrip('.') + unit
    return str(n)


def cell_tag(corpus: str, tokens: int, steps: int, seed: int = 0, preset: str = 'poc') -> str:
    """The canonical name for one grid cell.

    Defined once and imported by the notebook API, the single-cell runner, and the fleet, so all
    three address the same run by the same name. When these were three separate copies they
    could drift, and a drifted tag means the fleet writes a record the notebook cannot find.
    """
    stem = f'{corpus}_{compact(tokens)}_{compact(steps)}'
    if preset != 'poc':
        stem += f'_{preset}'
    return f'{stem}_s{seed}'


def build_model(vocab_size: int, tokenizer, preset: str = 'poc', max_len: int = 128):
    """A fresh RoBERTa-style masked LM at one of the size presets.

    max_position_embeddings is max_len + 2 because RoBERTa offsets positions past its padding
    id; giving it exactly max_len produces an index error only on the longest sequences, which
    is the kind of bug that survives a smoke test and dies overnight.
    """
    from transformers import AutoModelForMaskedLM, RobertaConfig

    cfg = RobertaConfig(vocab_size=vocab_size, max_position_embeddings=max_len + 2,
                        type_vocab_size=1, pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id, **SIZE_PRESETS[preset])
    return AutoModelForMaskedLM.from_config(cfg)


def n_params(model) -> tuple[int, int]:
    """(total, non-embedding) parameter counts -- the pair the POC prints."""
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    return total, total - emb


@torch.no_grad()
def evaluate(model, val_batches) -> float:
    """Mean masked-token loss over the fixed validation batches."""
    model.eval()
    total, n = 0.0, 0
    for xm, y in val_batches:
        total += model(input_ids=xm, labels=y).loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def save_random_init(vocab_size, tokenizer, preset, max_len, out_dir) -> str:
    """Write an untrained model of the same architecture -- the control that decides everything.

    Whatever the pretrained checkpoints score ABOVE this one is what pretraining actually
    bought. Without it, a from-scratch model beating a multilingual baseline could just as
    easily be the language-specific tokenizer plus a few hundred labelled examples.
    """
    model = build_model(vocab_size, tokenizer, preset, max_len)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def pretrain(ds, tokenizer, tag: str, steps: int, preset: str = 'poc', batch: int = 128,
             lr: float = 5e-4, mlm_prob: float = 0.15, seed: int = 0, clip: float = 1.0,
             log_every: int | None = None, out_dir: str | None = None,
             val_batches=None, amp_dtype=None) -> dict:
    """Pretrain one grid cell and return its record.

    ds is an mlm_data.MlmTokens already sliced to this cell's token budget, so "how much unique
    data" is decided by the caller and "how much compute" by `steps` -- the two axes the POC
    established have to be separated.

    Returns the dict the results notebook reads: where the checkpoint went, how long it took,
    the final validation loss, and the throughput the cost estimator needs.
    """
    device = ds.device
    torch.manual_seed(seed)

    model = build_model(ds.vocab_size, tokenizer, preset, ds.seq_len).to(device)
    total_p, nonemb_p = n_params(model)

    # bf16 on this hardware rather than the POC's fp16: same speed, no GradScaler to babysit,
    # and no loss-scale underflow to diagnose at 3am. Falls back to fp16 where bf16 is absent.
    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.98), eps=1e-6)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06)

    if val_batches is None:
        val_batches = ds.fixed_val_batches(mlm_prob=mlm_prob)
    log_every = log_every or max(1, steps // 10)
    out_dir = out_dir or os.path.join(RUNS, tag)

    tokens_per_step = batch * ds.seq_len
    passes = steps * tokens_per_step / ds.n
    random_loss = math.log(ds.vocab_size)
    print(f'[{tag}] {total_p/1e6:.1f}M params ({nonemb_p/1e6:.1f}M non-emb) | '
          f'{ds.n/1e6:.0f}M tokens | {steps:,} steps x {tokens_per_step:,} tok = '
          f'{passes:.1f} passes | random-loss {random_loss:.2f}', flush=True)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    jsonl = os.path.join(RUNS, f'{tag}.jsonl')
    os.makedirs(RUNS, exist_ok=True)
    if os.path.exists(jsonl):
        os.remove(jsonl)

    history, ema, best = [], None, float('inf')
    t0 = time.time()
    t_window = time.time()
    model.train()

    for step, (xm, y) in enumerate(ds.masked_batches(batch, steps, mlm_prob, gen), start=1):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=True):
            loss = model(input_ids=xm, labels=y).loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        sch.step()

        # One .item() per step is a host sync, which the causal loop was careful to avoid. Here
        # the step is far heavier than a small LM's, so the sync is a rounding error, and the
        # EMA it feeds is what makes a noisy MLM loss readable while a run is in flight.
        li = loss.item()
        ema = li if ema is None else 0.99 * ema + 0.01 * li

        if step % log_every == 0 or step == steps:
            dt = time.time() - t_window
            tok_s = log_every * tokens_per_step / max(dt, 1e-9)
            vl = evaluate(model, val_batches)
            is_best = vl < best
            best = min(best, vl)
            row = {'epoch': step // log_every, 'step': step, 'lr': sch.get_last_lr()[0],
                   'elapsed': time.time() - t0, 'is_best': is_best,
                   'train': {'loss': ema, 'ppl': math.exp(min(20.0, ema)),
                             'sec': dt, 'tok_s': tok_s},
                   'val': {'loss': vl, 'ppl': math.exp(min(20.0, vl))}}
            history.append(row)
            with open(jsonl, 'a') as f:
                f.write(json.dumps(row) + '\n')
            print(f'[{tag}] step {step:>7,}/{steps:,}{" *" if is_best else "  "}| '
                  f'train(EMA) {ema:5.3f} | val {vl:5.3f} | '
                  f'{tok_s/1e3:6.0f}k tok/s | {time.time()-t0:6.0f}s', flush=True)
            t_window = time.time()

    secs = time.time() - t0
    val_loss = evaluate(model, val_batches)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Which vocabulary this run scored against. Two runs are only comparable if these match --
    # a loss of 2.9 over one 16,000-token vocabulary is not the same measurement as 2.9 over a
    # different one, and without this recorded the difference is invisible.
    try:
        import mlm_data as _D
        vocab_fp = _D.tokenizer_fingerprint(tokenizer)
    except Exception:
        vocab_fp = None

    record = {'tag': tag, 'path': out_dir, 'preset': preset, 'params': total_p,
              'vocab_fingerprint': vocab_fp,
              'nonemb_params': nonemb_p, 'n_tokens': int(ds.n), 'steps': steps, 'batch': batch,
              'seq_len': ds.seq_len, 'lr': lr, 'seed': seed, 'passes': passes,
              'store_dtype': str(ds.store_dtype).replace('torch.', ''),
              'store_gb': ds.gb(), 'val_loss': val_loss, 'best_val_loss': best,
              'val_ppl': math.exp(min(20.0, val_loss)), 'random_loss': random_loss,
              'seconds': secs, 'tokens_per_s': steps * tokens_per_step / secs,
              'history': history}
    with open(os.path.join(RUNS, f'{tag}_result.json'), 'w') as f:
        json.dump(record, f, indent=2)

    print(f'[{tag}] done in {secs/60:.1f} min | val {val_loss:.3f} | '
          f'{record["tokens_per_s"]/1e3:.0f}k tok/s -> {out_dir}', flush=True)

    del model, opt, scaler
    torch.cuda.empty_cache()
    return record


def measure_throughput(ds, tokenizer, preset: str = 'poc', batch: int = 128,
                       steps: int = 20, warmup: int = 5) -> float:
    """Tokens per second for one preset, measured rather than guessed.

    The cost estimator needs a throughput number, and the POC's Gate 7 got one by extrapolating
    from a completed grid. This measures it in a few seconds instead, so a budget can be checked
    BEFORE committing a night -- which is the point of having an estimator at all.
    """
    device = ds.device
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(ds.vocab_size, tokenizer, preset, ds.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    model.train()

    def run(n):
        for xm, y in ds.masked_batches(batch, n, 0.15, gen):
            with torch.autocast('cuda', dtype=amp_dtype, enabled=True):
                loss = model(input_ids=xm, labels=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    run(warmup)
    torch.cuda.synchronize(device)
    t0 = time.time()
    run(steps)
    torch.cuda.synchronize(device)
    tok_s = steps * batch * ds.seq_len / (time.time() - t0)

    del model, opt
    torch.cuda.empty_cache()
    return tok_s


def estimate_hours(tok_s: float, cells: list[tuple[int, int]], batch: int, seq_len: int,
                   preset_from: str = 'poc', preset_to: str | None = None) -> dict:
    """Predict wall-clock for a grid, optionally rescaled to a larger model.

    The rescaling is the POC's own rule: cost per token grows with hidden^2 and with depth. It
    is an approximation -- it ignores attention's sequence-length term and every fixed overhead
    -- so it is honest for "does this fit a weekend", not for "is this 3.2 or 3.4 hours".
    """
    scale = 1.0
    if preset_to and preset_to != preset_from:
        a, b = SIZE_PRESETS[preset_from], SIZE_PRESETS[preset_to]
        scale = ((b['hidden_size'] / a['hidden_size']) ** 2
                 * (b['num_hidden_layers'] / a['num_hidden_layers']))
    effective = tok_s / scale

    rows = []
    for n_tokens, steps in cells:
        seen = steps * batch * seq_len
        rows.append({'n_tokens': n_tokens, 'steps': steps, 'tokens_seen': seen,
                     'hours': seen / effective / 3600,
                     'passes': seen / n_tokens})
    return {'tok_s_measured': tok_s, 'scale': scale, 'tok_s_effective': effective,
            'cells': rows, 'total_hours': sum(r['hours'] for r in rows)}
