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

# Peak learning rate per preset. The poc value is well established by every run in the study;
# the afriberta value is the best of what has been measured and is NOT a guarantee -- read on
# before trusting it at that scale.
#
# What is solid: at afriberta scale (86M), 5e-4 and 1e-3 collapse the model outright. Validation
# loss goes flat at ~6.76 from the first logged point and never moves. 3e-4 and 1e-4 descend
# normally over 4,000 steps. That is why the first afriberta ladder came out worse at every rung
# than the much smaller poc model.
#
# At 3e-4 over 4,000 steps it is reliable: three seeds landed at 5.614, 5.609 and 5.610.
#
# TWO conditions have to hold together, which is what made this confusing to diagnose. The rate
# must be low enough -- 5e-4 collapsed the 64M cell despite that run having the longest warmup
# of any -- AND the warmup must be long enough in absolute steps, since a 1,500-step run at the
# safe 3e-4 collapsed on 90 warmup steps. MIN_WARMUP_STEPS below fixes the second condition so
# short exploratory runs stop failing for a reason that has nothing to do with the experiment.
#
# The stall detector further down catches whatever still slips through, within a couple of
# minutes rather than at the end. The 64M afriberta cell ran 69 minutes and produced a model
# that had learned nothing.
PRESET_LR = {'poc': 5e-4, 'afriberta': 3e-4}

# Floor on warmup, in absolute steps -- see the schedule construction in pretrain().
MIN_WARMUP_STEPS = 250


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
    easily be the language-specific tokenizer plus a few hundred labeled examples.
    """
    model = build_model(vocab_size, tokenizer, preset, max_len)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def _state_path(out_dir: str) -> str:
    """Where the resumable training state lives, beside the HF checkpoint."""
    return os.path.join(out_dir, 'train_state.pt')


def save_train_state(out_dir, model, opt, sch, scaler, micro, history, ema, best, stall_warned):
    """Everything needed to continue this run, written atomically.

    Temp file then rename, so a crash during the write can destroy the .tmp but never the good
    state underneath -- the discipline train_loop.py already uses. A multi-day run WILL be
    interrupted; the only question is whether that costs minutes or days.
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp = _state_path(out_dir) + '.tmp'
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sch.state_dict(), 'scaler': scaler.state_dict(),
                'micro': micro, 'history': history, 'ema': ema, 'best': best,
                'stall_warned': stall_warned}, tmp)
    os.replace(tmp, _state_path(out_dir))


def load_train_state(out_dir, model, opt, sch, scaler, device):
    """Restore a previous run's state. None if there is nothing to resume from."""
    p = _state_path(out_dir)
    if not os.path.exists(p):
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['optimizer'])
    sch.load_state_dict(ck['scheduler'])
    if scaler.is_enabled() and ck.get('scaler'):
        scaler.load_state_dict(ck['scaler'])
    return ck


def pretrain(ds, tokenizer, tag: str, steps: int, preset: str = 'poc', batch: int = 128,
             lr: float | None = None, mlm_prob: float = 0.15, seed: int = 0, clip: float = 1.0,
             log_every: int | None = None, out_dir: str | None = None,
             val_batches=None, amp_dtype=None, accum: int = 1,
             resume: bool = True, ckpt_every: int | None = None,
             warmup: float | None = None) -> dict:
    """Pretrain one grid cell and return its record.

    ds is an mlm_data.MlmTokens already sliced to this cell's token budget, so "how much unique
    data" is decided by the caller and "how much compute" by `steps` -- the two axes the POC
    established have to be separated.

    Returns the dict the results notebook reads: where the checkpoint went, how long it took,
    the final validation loss, and the throughput the cost estimator needs.
    """
    device = ds.device
    torch.manual_seed(seed)

    # None means "whatever trains this preset" -- see PRESET_LR. An explicit value is honoured,
    # including one known to collapse, because a sweep has to be able to ask for a bad rate.
    if lr is None:
        lr = PRESET_LR.get(preset, 5e-4)

    model = build_model(ds.vocab_size, tokenizer, preset, ds.seq_len).to(device)
    total_p, nonemb_p = n_params(model)

    # bf16 on this hardware rather than the POC's fp16: same speed, no GradScaler to babysit,
    # and no loss-scale underflow to diagnose at 3am. Falls back to fp16 where bf16 is absent.
    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.98), eps=1e-6)
    # Warm up for a fraction of the run, but never fewer than MIN_WARMUP_STEPS. A flat 6% is
    # fine for a long run and far too short for a brief one: a 1,500-step run got 90 warmup
    # steps and collapsed at a learning rate that trains reliably (3/3 seeds) over 4,000 steps
    # with 240. Short runs are exactly what people use to try things out, so the schedule has to
    # survive them.
    #
    # `warmup` overrides that rule outright, because the rule is now itself under test: at 86M
    # roughly a third of seeds never leave the unigram plateau within their budget, and a warmup
    # too short for that width is one of the two hypotheses that would explain it. A sweep has to
    # be able to ask for a warmup the heuristic would not choose.
    pct_start = warmup if warmup is not None else \
        min(0.25, max(0.06, MIN_WARMUP_STEPS / max(steps, 1)))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=pct_start)

    if val_batches is None:
        val_batches = ds.fixed_val_batches(mlm_prob=mlm_prob)
    # Ten points over a whole run is enough to plot and useless to watch: on a two-day run that
    # is one update every five hours, and a dashboard polling every three seconds shows a frozen
    # screen. Aim for ~60 points, capped so a very long run still updates every few minutes.
    log_every = log_every or max(1, min(steps // 60 or 1, 500))
    out_dir = out_dir or os.path.join(RUNS, tag)

    tokens_per_step = batch * accum * ds.seq_len
    passes = steps * tokens_per_step / ds.n
    random_loss = math.log(ds.vocab_size)
    print(f'[{tag}] {total_p/1e6:.1f}M params ({nonemb_p/1e6:.1f}M non-emb) | '
          f'{ds.n/1e6:.0f}M tokens | {steps:,} steps x {tokens_per_step:,} tok = '
          f'{passes:.1f} passes | random-loss {random_loss:.2f}', flush=True)
    # dashboard.py counts progress in "epochs" and reads the total off this line. Our epochs are
    # logging intervals, so tell it how many there will be -- otherwise it falls back to a
    # hardcoded 30 and every MLM run displays as 10/30 forever.
    n_intervals = (steps + log_every - 1) // log_every
    print(f'[{tag}] {n_intervals} epochs, batch {batch} x {ds.seq_len} tok '
          f'(logging every {log_every:,} steps)', flush=True)
    # Total work, for the dashboard's finish-time estimate. It cannot infer this: its causal
    # heuristic is epochs x corpus-size, which for an MLM run means logging intervals times the
    # WHOLE corpus rather than the subset actually being trained on -- off by any factor at all.
    # Stating the real figure lets it divide by observed throughput and be right.
    print(f'[{tag}] total work {steps * tokens_per_step:,} tokens', flush=True)

    # The same facts as JSON. Dashboards used to recover these by regexing the console log,
    # which broke for any run whose log was redirected elsewhere -- no total, no progress bar,
    # no finish estimate. Structured data belongs in a file, not in printed prose.
    meta = {'tag': tag, 'corpus': getattr(ds, 'name', None), 'preset': preset, 'steps': steps, 'batch': batch, 'accum': accum,
            'seq_len': ds.seq_len, 'tokens_per_step': tokens_per_step,
            'total_work': steps * tokens_per_step, 'n_tokens': int(ds.n), 'lr': lr, 'warmup': pct_start, 'clip': clip,
            'seed': seed, 'vocab_size': ds.vocab_size, 'random_loss': random_loss,
            'log_every': log_every, 'started': time.time(),
            'params': total_p, 'nonemb_params': nonemb_p}
    with open(os.path.join(RUNS, f'{tag}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    os.makedirs(RUNS, exist_ok=True)

    history, ema, best = [], None, float('inf')
    stall_warned = False
    diverged_for, diverged_at = 0, None
    start_micro = 0

    # Resume BEFORE touching the JSONL, because whether this is a fresh run decides whether the
    # old history is thrown away -- reading start_micro before setting it raised UnboundLocalError
    # on the first real restart.
    #
    # The RNG stream is not restored, so resumed windows differ from what an uninterrupted run
    # would have drawn. That is acceptable -- the windows are random anyway -- and far better
    # than losing a day of training to a reboot.
    if resume:
        prev = load_train_state(out_dir, model, opt, sch, scaler, device)
        if prev:
            start_micro = prev['micro']
            history, ema = prev['history'], prev['ema']
            best, stall_warned = prev['best'], prev['stall_warned']
            done = start_micro // accum
            print(f'[{tag}] resuming at step {done:,}/{steps:,} '
                  f'({done / max(steps, 1):.0%} done, best val {best:.3f})', flush=True)

    jsonl = os.path.join(RUNS, f'{tag}.jsonl')
    if os.path.exists(jsonl) and not start_micro:
        os.remove(jsonl)          # a fresh run never appends to an old run's history

    ckpt_every = ckpt_every or max(1, steps // 20)
    t0 = time.time()
    t_window = time.time()
    last_log_step = 0
    model.train()

    # accum micro-batches make one optimizer step, so the EFFECTIVE batch is batch * accum while
    # memory only ever holds `batch` sequences. That is the only way to reach the batch sizes a
    # model this size was designed for -- RoBERTa-base trained near two million tokens per step
    # and we hold sixteen thousand -- without a card that can hold them all at once.
    micro_total = steps * accum
    if start_micro >= micro_total:
        # Nothing left to do. Report the finished run rather than crashing on a `step` that the
        # loop never got to define.
        print(f'[{tag}] already complete ({steps:,} steps) -- '
              f'pass resume=False to train it again', flush=True)
        done_path = os.path.join(RUNS, f'{tag}_result.json')
        if os.path.exists(done_path):
            with open(done_path, encoding='utf-8') as f:
                return json.load(f)

    for micro, (xm, y) in enumerate(
            ds.masked_batches(batch, micro_total - start_micro, mlm_prob, gen),
            start=start_micro + 1):
        with torch.autocast('cuda', dtype=amp_dtype, enabled=True):
            loss = model(input_ids=xm, labels=y).loss
            # Scale so the accumulated gradient is the MEAN over micro-batches, not the sum;
            # without this the effective learning rate silently multiplies by accum.
            loss_for_backward = loss / accum

        scaler.scale(loss_for_backward).backward()

        if micro % accum:
            continue                      # keep accumulating; no optimizer step yet
        step = micro // accum

        if clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sch.step()

        # One .item() per step is a host sync, which the causal loop was careful to avoid. Here
        # the step is far heavier than a small LM's, so the sync is a rounding error, and the
        # EMA it feeds is what makes a noisy MLM loss readable while a run is in flight.
        li = loss.item()
        ema = li if ema is None else 0.99 * ema + 0.01 * li

        if step % log_every == 0 or step == steps:
            dt = time.time() - t_window
            # Count the steps this window ACTUALLY covered. The final window is whatever is left
            # over when steps is not a multiple of log_every -- 9 steps out of 1,171 on one run --
            # and assuming a full window there reported 24.4M tok/s for a model doing 189k.
            window_steps = step - last_log_step
            tok_s = window_steps * tokens_per_step / max(dt, 1e-9)
            last_log_step = step
            vl = evaluate(model, val_batches)
            is_best = vl < best
            best = min(best, vl)
            # Number the intervals 1..n by counting them, not by dividing. step // log_every
            # gives the FINAL (partial) interval the same number as the one before it, so a
            # finished run reported 10 of 11 intervals and the dashboard drew it as STOPPED.
            # `epoch` is a LOGGING-INTERVAL COUNTER, not an epoch. It is named that only
            # because the causal study's dashboard reads that key, and inheriting the name has
            # confused every reader who met it -- there is no sense in which this run is on its
            # fourth epoch. `passes` is the real thing: how many times the model has now seen the
            # corpus. The two diverge wildly and on purpose, because the compute axis is steps:
            # 62,500 steps is one pass over a 1B-token rung and 256 passes over a 4M-token one.
            row = {'epoch': len(history) + 1, 'step': step, 'lr': sch.get_last_lr()[0],
                   'passes': step * tokens_per_step / max(ds.n, 1),
                   'total_passes': steps * tokens_per_step / max(ds.n, 1),
                   'elapsed': time.time() - t0, 'is_best': is_best,
                   'train': {'loss': ema, 'ppl': math.exp(min(20.0, ema)),
                             'sec': dt, 'tok_s': tok_s},
                   'val': {'loss': vl, 'ppl': math.exp(min(20.0, vl))}}
            # Has this run learned anything yet? A collapsed MLM sits flat just below the
            # uniform-prediction loss and stays there; it will not recover, and every further
            # step is wasted. Warn loudly at the halfway mark rather than aborting, because a
            # slow start is not the same as a dead one and only the operator can tell.
            # A DIFFERENT failure, and the one the stall detector below cannot see: a run that
            # learns normally and then loses it. eng_1b_1024M_afriberta_s1 reached 6.182 at step
            # 20,000 and ended at 7.469 -- English's unigram entropy is 7.491, so it fell all the
            # way back to predicting word frequencies. The stall check compares against the FIRST
            # point, so a run that descended and then diverged still shows plenty of movement and
            # passes silently. It cost an hour of card time each of the four times it happened.
            #
            # Threshold on the best loss ever seen, not on the previous point, because the loss
            # is noisy step to step and a single bad batch is not a divergence. Two consecutive
            # evaluations 0.5 above the best is not noise at this scale: within a healthy run the
            # interval-to-interval spread is under 0.1.
            if best < random_loss - 0.5 and vl > best + 0.5:
                diverged_for += 1
                if diverged_for >= 2 and diverged_at is None:
                    diverged_at = step
                    print(f'[{tag}] WARNING: val loss has risen to {vl:.3f} from a best of '
                          f'{best:.3f} at step {step:,} and stayed there. This run is diverging, '
                          f'not converging slowly -- it learned and is losing it. The checkpoint '
                          f'on disk is the best one, not the current one. A lower peak learning '
                          f'rate is the first thing to try.', flush=True)
            else:
                diverged_for = 0

            if history and step >= steps // 2 and not stall_warned:
                moved = history[0]['val']['loss'] - vl
                if moved < 0.15:
                    stall_warned = True
                    print(f'[{tag}] WARNING: val loss has moved only {moved:.3f} in '
                          f'{step:,} steps (random-loss is {random_loss:.2f}). This run looks '
                          f'collapsed rather than slow. At this model size try a lower peak '
                          f'learning rate -- see PRESET_LR in mlm_train.py.', flush=True)

            history.append(row)
            with open(jsonl, 'a') as f:
                f.write(json.dumps(row) + '\n')

            if step % ckpt_every == 0 or step == steps:
                save_train_state(out_dir, model, opt, sch, scaler, micro,
                                 history, ema, best, stall_warned)
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
              'seq_len': ds.seq_len, 'lr': lr, 'warmup': pct_start,
              'clip': clip, 'seed': seed,
              'passes': passes,
              'store_dtype': str(ds.store_dtype).replace('torch.', ''),
              'store_gb': ds.gb(), 'val_loss': val_loss, 'best_val_loss': best,
              'val_ppl': math.exp(min(20.0, val_loss)), 'random_loss': random_loss,
              'seconds': secs, 'tokens_per_s': steps * tokens_per_step / secs,
              'lr_used': lr, 'stalled': stall_warned,
              'diverged': diverged_at is not None, 'diverged_at': diverged_at,
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
