"""
diagnose.py -- what this machine is, and where its bottleneck is.

The factory's numbers are not portable. A batch size that saturates a 96 GB workstation card
starves a Colab T4, and a model that is too small to keep one GPU busy may be exactly right for
another. So rather than shipping our measurements as advice, this measures YOURS and tells you
what they mean.

It answers four questions, in order:

    what am I running on?      GPUs, memory, bf16, and whether this is Colab
    what batch should I use?   swept on this hardware, not taken from a table
    what is my bottleneck?     a verdict derived from the sweep, with the reasoning shown
    what will my study cost?   projected from the throughput just measured

Nothing here trains anything you keep -- the sweep runs a few dozen throwaway steps.

Usage:
    python diagnose.py                          # hardware + bottleneck, no corpus needed
    python diagnose.py --corpus yor             # also sizes your corpus and your grid
    python diagnose.py --corpus yor --preset afriberta
    python diagnose.py --corpus yor --json diag.json     # for the notebook to plot
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

BATCHES = [16, 32, 64, 128, 256, 512]


# -- environment ---------------------------------------------------------------------------

def describe_environment() -> dict:
    """Everything about this machine that changes what the right settings are."""
    import torch

    env = {
        'platform': platform.system(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'colab': 'google.colab' in sys.modules or os.path.exists('/content'),
        'cpu_count': os.cpu_count(),
        'gpus': [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        env['gpus'].append({
            'index': i, 'name': p.name,
            'memory_gb': p.total_memory / 1e9,
            'capability': f'{p.major}.{p.minor}',
            'multiprocessors': p.multi_processor_count,
        })
    env['n_gpu'] = len(env['gpus'])
    env['bf16'] = bool(env['n_gpu']) and torch.cuda.is_bf16_supported()
    return env


def print_environment(env: dict) -> None:
    where = 'Google Colab' if env['colab'] else env['platform']
    print(f'\n  running on   {where} | python {env["python"]} | torch {env["torch"]}'
          f' | CUDA {env["cuda"]}')
    if not env['gpus']:
        print('  GPUs         none found -- pretraining needs one')
        return
    for g in env['gpus']:
        print(f'  GPU {g["index"]}        {g["name"]} | {g["memory_gb"]:.0f} GB | '
              f'compute {g["capability"]} | {g["multiprocessors"]} SMs')
    print(f'  precision    {"bf16 (no loss scaler needed)" if env["bf16"] else "fp16 + GradScaler"}')
    if env['n_gpu'] > 1:
        print(f'  parallelism  {env["n_gpu"]} cards -> mlm_fleet.py runs {env["n_gpu"]} cells at once')
    else:
        print('  parallelism  one card -- the fleet still works, it just runs cells in sequence')


# -- the sweep -----------------------------------------------------------------------------

def sweep_batches(corpus: str, preset: str, seq_len: int, gpu: int,
                  steps: int = 12, warmup: int = 4, batches=BATCHES) -> list[dict]:
    """Time a handful of real training steps at each batch size, stopping at the memory wall.

    Uses the corpus if one is prepared and synthetic tokens otherwise, because the question
    "what batch fits on this card" does not depend on which language the tokens came from --
    and a new user should be able to run this before preparing anything.
    """
    import torch

    device = torch.device(f'cuda:{gpu}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if corpus:
        import mlm_api as factory
        ds = factory.stream(corpus, tokens=4_000_000, seq_len=seq_len, gpu=gpu)
        tok = factory.load_tokenizer(corpus)
        vocab = ds.vocab_size
        source = f'corpus {corpus!r}'
    else:
        ds, tok, vocab = None, _throwaway_tokenizer(), 16_000
        source = 'synthetic tokens (no corpus prepared)'

    print(f'\n  sweeping batch sizes on {source}, preset {preset!r}, seq_len {seq_len} ...')
    rows = []
    for b in batches:
        try:
            rows.append(_time_batch(b, ds, tok, vocab, preset, seq_len, device, amp,
                                    steps, warmup))
            r = rows[-1]
            print(f'    batch {b:>4}  {r["tok_s"]/1e3:>7.0f}k tok/s   '
                  f'{r["peak_gb"]:>6.1f} GB peak   {r["tflops"]:>5.0f} TFLOP/s')
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'    batch {b:>4}  out of memory -- this is your ceiling')
            break
    return rows


def _throwaway_tokenizer():
    """A minimal tokenizer stand-in, so the sweep runs before any corpus is prepared."""
    class _T:
        pad_token_id, bos_token_id, eos_token_id, mask_token_id = 1, 0, 2, 4
    return _T()


def _time_batch(batch, ds, tok, vocab, preset, seq_len, device, amp, steps, warmup) -> dict:
    import torch
    import mlm_train as M

    torch.cuda.empty_cache()
    model = M.build_model(vocab, tok, preset, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    def batches(n):
        if ds is not None:
            yield from ds.masked_batches(batch, n, 0.15, gen)
        else:
            for _ in range(n):
                x = torch.randint(5, vocab, (batch, seq_len), device=device, generator=gen)
                y = x.clone()
                y[torch.rand(x.shape, device=device, generator=gen) > 0.15] = -100
                yield x, y

    def run(n):
        for xm, y in batches(n):
            with torch.autocast('cuda', dtype=amp):
                loss = model(input_ids=xm, labels=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    run(warmup)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    run(steps)
    torch.cuda.synchronize(device)
    dt = time.time() - t0

    nonemb = sum(p.numel() for p in model.parameters()) - model.get_input_embeddings().weight.numel()
    head = model.config.hidden_size * vocab
    tok_s = steps * batch * seq_len / dt
    peak = torch.cuda.max_memory_allocated(device) / 1e9

    del model, opt
    torch.cuda.empty_cache()
    return {'batch': batch, 'tok_s': tok_s, 'peak_gb': peak,
            'tflops': tok_s * 6 * (nonemb + head) / 1e12,
            'ms_per_step': dt / steps * 1000}


# -- the verdict ---------------------------------------------------------------------------

def diagnose(rows: list[dict], env: dict, total_gb: float) -> dict:
    """Turn the sweep into a recommendation and a named bottleneck, showing the reasoning."""
    if not rows:
        return {'verdict': 'no measurements', 'best_batch': None}

    fastest = max(rows, key=lambda r: r['tok_s'])

    # Recommend the KNEE, not the peak: the smallest batch within a few percent of the best.
    # Picking the outright maximum trades real costs for noise -- a larger batch uses more
    # memory and, at a fixed budget of tokens, takes proportionally fewer optimizer steps, which
    # changes the optimization rather than just the speed. On this workstation the peak sits at
    # 256 and the knee at 128: 2% slower, half the memory, twice the steps.
    KNEE = 0.97
    best = min((r for r in rows if r['tok_s'] >= KNEE * fastest['tok_s']),
               key=lambda r: r['batch'])

    peak_mem_frac = best['peak_gb'] / total_gb if total_gb else 0
    hit_oom = len(rows) < len(BATCHES)
    at_top = fastest['batch'] == rows[-1]['batch']

    # Did throughput actually stop improving, or did we just run out of batch sizes to try?
    gain_at_best = None
    if len(rows) > 1:
        prev = [r for r in rows if r['batch'] < best['batch']]
        if prev:
            gain_at_best = best['tok_s'] / prev[-1]['tok_s']

    findings = []
    if hit_oom:
        bottleneck = 'memory'
        findings.append(f'the sweep hit an out-of-memory error, so batch size is capped by the '
                        f'{total_gb:.0f} GB on this card, not by throughput')
    elif at_top and (gain_at_best or 1) > 1.05:
        bottleneck = 'batch size (untested headroom)'
        findings.append('throughput was still climbing at the largest batch tried -- there may '
                        'be more available above it')
    elif peak_mem_frac < 0.25:
        bottleneck = 'model size (launch-bound)'
        findings.append(f'throughput plateaus while using only {peak_mem_frac:.0%} of memory, '
                        f'so the card is not short of room -- the model is too small to keep it '
                        f'busy, and the time goes into starting and finishing small kernels')
    else:
        bottleneck = 'balanced'
        findings.append('throughput plateaus with memory meaningfully used -- this is a '
                        'reasonable operating point')

    if env['n_gpu'] > 1:
        findings.append(f'{env["n_gpu"]} cards are available and one run uses one card, so the '
                        f'fleet is worth up to {env["n_gpu"]}x on a multi-cell grid')
    else:
        findings.append('only one card, so there is no fleet speed-up here -- cells run in '
                        'sequence and wall-clock equals the sum of them')

    if best['batch'] != fastest['batch']:
        findings.append(f'batch {fastest["batch"]} was marginally faster '
                        f'({fastest["tok_s"]/1e3:.0f}k vs {best["tok_s"]/1e3:.0f}k tok/s) but '
                        f'costs {fastest["peak_gb"]/best["peak_gb"]:.1f}x the memory and halves '
                        f'the optimizer steps at a fixed token budget, so the smaller one is '
                        f'recommended')

    return {'best_batch': best['batch'], 'best_tok_s': best['tok_s'],
            'peak_gb': best['peak_gb'], 'peak_fraction': peak_mem_frac,
            'fastest_batch': fastest['batch'], 'fastest_tok_s': fastest['tok_s'],
            'bottleneck': bottleneck, 'findings': findings,
            'hit_oom': hit_oom}


def print_verdict(d: dict, env: dict) -> None:
    print(f'\n  {"-" * 70}')
    print(f'  RECOMMENDED BATCH  {d["best_batch"]}   '
          f'({d["best_tok_s"]/1e3:.0f}k tok/s, {d["peak_gb"]:.1f} GB peak)')
    print(f'  BOTTLENECK         {d["bottleneck"]}')
    print(f'  {"-" * 70}')
    for f in d['findings']:
        print(f'    - {f}')
    print(f'\n  use it with:  --batch {d["best_batch"]}'
          + ('' if env['n_gpu'] < 2 else f'   (and mlm_fleet.py will use all {env["n_gpu"]} cards)'))


# -- cost ----------------------------------------------------------------------------------

def project(tok_s: float, preset: str, scale_to: str | None, rungs, passes: int) -> float:
    """What the study costs on THIS machine, at the throughput just measured.

    Returns the GPU-hours for one language. Only throughput and model size enter the arithmetic:
    the batch shape is already baked into tok_s, and the corpus only decides the rungs.
    """
    import mlm_train as M

    scale = 1.0
    if scale_to and scale_to != preset:
        a, b = M.SIZE_PRESETS[preset], M.SIZE_PRESETS[scale_to]
        scale = ((b['hidden_size'] / a['hidden_size']) ** 2
                 * (b['num_hidden_layers'] / a['num_hidden_layers']))
    eff = tok_s / scale

    print(f'\n  cost on this machine, {passes} passes per rung'
          + (f', scaled to {scale_to!r} ({scale:.1f}x cost/token)' if scale > 1 else ''))
    print(f'\n  {"rung":>14}{"tokens seen":>16}{"GPU-hours":>12}')
    total = 0.0
    for n in rungs:
        h = n * passes / eff / 3600
        total += h
        print(f'  {n:>14,}{n * passes:>16,}{h:>12.2f}')
    print(f'  {"per language":>14}{"":>16}{total:>12.2f}')
    return total


# -- corpus ---------------------------------------------------------------------------------

def size_corpus(corpus: str, rungs) -> None:
    """Does the planned ladder actually fit in the text this language has?"""
    import mlm_api as factory

    info = factory.corpus_info(corpus)
    n = info['n_tokens']['train']
    print(f'\n  corpus {corpus!r}: {n:,} train tokens, vocab {info["vocab_size"]:,}'
          + (f', {info["chars_per_token"]:.2f} chars/token' if 'chars_per_token' in info else ''))
    print(f'  resident store: {info["store_gb_int16"]:.3f} GB as int16')
    print(f'\n  {"rung":>14}{"share of corpus":>18}')
    for r in rungs:
        share = r / n
        flag = '' if share <= 1 else '   DOES NOT FIT'
        print(f'  {r:>14,}{share:>17.0%}{flag}')
    if max(rungs) > n:
        print('\n  at least one rung is larger than the corpus. Lower it, or add sources.')
    elif max(rungs) / n > 0.8:
        print('\n  the top rung uses most of the available text -- there is little headroom '
              'to grow the data axis.')


def main():
    p = argparse.ArgumentParser(description='Measure this machine and name its bottleneck.')
    p.add_argument('--corpus', default=None, help='prepared corpus to size (optional)')
    p.add_argument('--preset', default='poc')
    p.add_argument('--scale-to', default=None, help='project cost at a larger preset')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--steps', type=int, default=12, help='timed steps per batch size')
    p.add_argument('--rungs', type=int, nargs='+',
                   default=[4_000_000, 16_000_000, 64_000_000])
    p.add_argument('--passes', type=int, default=12)
    p.add_argument('--json', default=None)
    args = p.parse_args()

    print('=' * 74)
    print('  FACTORY DIAGNOSTICS')
    print('=' * 74)

    env = describe_environment()
    print_environment(env)
    if not env['gpus']:
        raise SystemExit('\n  no GPU -- nothing further to measure.')

    if args.corpus:
        size_corpus(args.corpus, args.rungs)

    rows = sweep_batches(args.corpus, args.preset, args.seq_len, args.gpu, steps=args.steps)
    d = diagnose(rows, env, env['gpus'][args.gpu]['memory_gb'])
    print_verdict(d, env)

    total = project(d['best_tok_s'], args.preset, args.scale_to, args.rungs, args.passes)
    if env['n_gpu'] > 1:
        print(f'  {"on " + str(env["n_gpu"]) + " cards":>14}{"":>16}'
              f'{total / env["n_gpu"]:>12.2f}  wall-clock hours')

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'environment': env, 'sweep': rows, 'diagnosis': d,
                       'corpus': args.corpus, 'preset': args.preset,
                       'seq_len': args.seq_len}, f, indent=2)
        print(f'\n  wrote {args.json}')
    print()


if __name__ == '__main__':
    main()
