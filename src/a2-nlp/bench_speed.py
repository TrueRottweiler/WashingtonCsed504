"""What would make a run faster without changing what it measures.

Learning rate is not on this list on purpose. A different LR is a different experiment, not a
faster one -- and 5e-4 is already known to collapse the 86M preset (report 03 6d). The levers
here change how the same arithmetic is executed: the attention kernel, and whether the graph is
compiled. Both must leave the loss alone to be usable, so this checks that too.

    bash src/a2-nlp/py.sh bench_speed.py --preset afriberta
"""
from __future__ import annotations

import argparse
import time

import torch

import mlm_data as D
import mlm_train as M

STEPS = 40
WARMUP = 10


def build(corpus, preset, seq_len, attn):
    """A fresh model, optionally asking transformers for a specific attention kernel."""
    from transformers import AutoModelForMaskedLM, RobertaConfig
    stats = D.T.load_stats(corpus)
    tok = D.load_tokenizer(corpus)
    cfg = RobertaConfig(vocab_size=stats['vocab_size'], max_position_embeddings=seq_len + 2,
                        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
                        eos_token_id=tok.eos_token_id, **M.SIZE_PRESETS[preset])
    kw = {'attn_implementation': attn} if attn else {}
    return AutoModelForMaskedLM.from_config(cfg, **kw)


def timed(model, ds, batch, seq_len, label, compile_it=False):
    dev = torch.device('cuda:0')
    model = model.to(dev)
    if compile_it:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    g = torch.Generator(device=dev).manual_seed(0)

    losses = []
    for i in range(STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            t0 = time.time()
        x = ds.windows(batch, generator=g)
        inp, lab = ds.mask(x, generator=g)
        with torch.autocast('cuda', dtype=amp, enabled=True):
            out = model(input_ids=inp, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
    torch.cuda.synchronize()
    el = time.time() - t0

    tps = (STEPS - WARMUP) * batch * seq_len / el
    # The mean over the timed window, not the last step: a single step is noisy and what matters
    # is whether the whole trajectory moved, which is what would betray a numerics change.
    mean_loss = sum(losses[WARMUP:]) / (STEPS - WARMUP)
    print(f'  {label:34s} {tps/1e3:7.0f}k tok/s   mean loss {mean_loss:.4f}')
    del model, opt
    torch.cuda.empty_cache()
    return tps, mean_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='eng_1b')
    p.add_argument('--preset', default='afriberta', choices=list(M.SIZE_PRESETS))
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--seq-len', type=int, default=128)
    args = p.parse_args()

    dev = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ds = D.MlmTokens(dev, args.corpus, 'train', seq_len=args.seq_len, subset=8_000_000)

    print(f'\n{args.preset} preset, batch {args.batch} x seq {args.seq_len}, '
          f'{STEPS - WARMUP} timed steps\n')
    base = None
    for label, attn, comp in (('as shipped (default attention)', None, False),
                              ('attn_implementation=sdpa', 'sdpa', False),
                              ('sdpa + torch.compile', 'sdpa', True)):
        try:
            model = build(args.corpus, args.preset, args.seq_len, attn)
            tps, loss = timed(model, ds, args.batch, args.seq_len, label, comp)
        except Exception as e:                       # noqa: BLE001 -- report and keep going
            print(f'  {label:34s} unavailable: {repr(e)[:90]}')
            continue
        if base is None:
            base = (tps, loss)
        else:
            print(f'  {"":34s} {tps/base[0]:6.2f}x speed, '
                  f'loss delta {loss - base[1]:+.4f}')


if __name__ == '__main__':
    main()
