"""What one of our training runs costs on whatever machine you happen to have.

The poster tells a student that a run is 1.024 billion tokens and ninety minutes. Ninety minutes
on *what* is the question they will actually ask, because almost nobody reading it owns two
Blackwell cards. This measures the same two models on any machine -- a MacBook, a Colab session,
a laptop with a mobile RTX -- and converts the answer into the only unit that matters to someone
deciding whether to try: how long the run would take, and whether it fits in memory at all.

Deliberately self-contained. It needs no corpus, no tokenizer and no repository data: the token
stream is random integers, which is worthless for learning and identical for timing, because a
transformer's cost per step does not depend on which token ids arrive. So this file can be pasted
straight into a fresh Colab cell.

    bash src/a2-nlp/py.sh bench_portable.py                 # both sizes, this machine
    python bench_portable.py --steps 30 --preset poc        # anywhere else

The one thing it will not tell you is throughput under contention. Numbers taken while another
job holds the same card are lower, and the script prints a warning when it can detect that.
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import torch

# The two model shapes this project trains, copied rather than imported so the file stands alone.
PRESETS = {
    'poc':       dict(hidden_size=512, num_hidden_layers=8,
                      num_attention_heads=8, intermediate_size=2048),
    'afriberta': dict(hidden_size=768, num_hidden_layers=12,
                      num_attention_heads=12, intermediate_size=3072),
}
VOCAB, SEQ, BATCH = 16_000, 128, 128
FULL_RUN_STEPS = 62_500          # what the study actually runs, for the extrapolation


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), torch.cuda.get_device_name(0)
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps'), f'Apple {platform.machine()} (MPS)'
    return torch.device('cpu'), platform.processor() or platform.machine()


def amp_dtype(device):
    """bf16 where the hardware really has it, fp16 on older CUDA, nothing elsewhere.

    MPS and CPU are left in fp32 on purpose: autocast on those paths is either unsupported or
    slower, and a benchmark that silently changes precision between machines is comparing two
    different computations.
    """
    if device.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build(preset, device):
    from transformers import AutoModelForMaskedLM, RobertaConfig
    cfg = RobertaConfig(vocab_size=VOCAB, max_position_embeddings=SEQ + 2,
                        type_vocab_size=1, **PRESETS[preset])
    return AutoModelForMaskedLM.from_config(cfg).to(device)


def bench(preset, device, dev_name, steps, warmup):
    model = build(preset, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - model.get_input_embeddings().weight.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dt = amp_dtype(device)

    # One fixed batch, reused. Generating fresh ids each step would time the RNG as well as the
    # model, and the model is the thing under test.
    ids = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    labels = ids.clone()

    def one_step():
        opt.zero_grad(set_to_none=True)
        if dt is not None:
            with torch.autocast('cuda', dtype=dt):
                loss = model(input_ids=ids, labels=labels).loss
        else:
            loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()

    for _ in range(warmup):                      # warmup is not optional: the first steps pay
        one_step()                               # for kernel autotuning and allocator growth
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    dt_s = time.perf_counter() - t0

    tok_s = steps * BATCH * SEQ / dt_s
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == 'cuda' else None
    return {'preset': preset, 'device': dev_name, 'dtype': str(dt or torch.float32),
            'params_m': round(n_params / 1e6, 1), 'backbone_m': round(n_backbone / 1e6, 1),
            'tokens_per_s': round(tok_s), 'peak_gb': round(peak, 2) if peak else None,
            'full_run_hours': round(FULL_RUN_STEPS * BATCH * SEQ / tok_s / 3600, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--preset', default=None, help='poc or afriberta; both if omitted')
    ap.add_argument('--out', default=None, help='append the rows to this JSON file')
    a = ap.parse_args()

    device, dev_name = pick_device()
    print(f'device: {dev_name}   torch {torch.__version__}   dtype {amp_dtype(device) or "fp32"}')
    if device.type == 'cuda':
        # mem_get_info is the DEVICE's free/total, across every process. memory_reserved() only
        # sees this process, which is useless for the question being asked -- the first version
        # of this check reported a clean card while another job was using half of it, and the
        # measurement came out at half speed with no warning attached.
        free, total = torch.cuda.mem_get_info()
        used_gb = (total - free) / 1024 ** 3
        print(f'memory: {total/1024**3:.0f} GB total, {used_gb:.1f} GB already in use')
        if used_gb > 1.0:
            print('WARNING: another process is on this card. These numbers will read LOW -- on '
                  'our own box a contended measurement came out at half the sustained rate.')
    print(f'batch {BATCH} x seq {SEQ} = {BATCH*SEQ:,} tokens per step, '
          f'{a.steps} timed steps after {a.warmup} warmup\n')

    rows = []
    for preset in ([a.preset] if a.preset else ['poc', 'afriberta']):
        try:
            r = bench(preset, device, dev_name, a.steps, a.warmup)
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as e:
            # Not a failure of the benchmark. "It does not fit" is one of the answers a student
            # needs, so it is recorded as a result rather than raised.
            print(f'{preset:>10}: DOES NOT FIT -- {type(e).__name__}')
            rows.append({'preset': preset, 'device': dev_name, 'error': 'out of memory'})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"{preset:>10}: {r['params_m']:>5}M params ({r['backbone_m']}M backbone)  "
              f"{r['tokens_per_s']:>8,} tok/s  "
              f"peak {r['peak_gb'] if r['peak_gb'] else '--'} GB  "
              f"=> one 62,500-step run takes {r['full_run_hours']:.2f} h")

    if a.out:
        try:
            old = json.load(open(a.out, encoding='utf-8'))
        except (OSError, ValueError):
            old = []
        json.dump(old + rows, open(a.out, 'w', encoding='utf-8'), indent=2)
        print(f'\nappended {len(rows)} rows to {a.out}')
    else:
        print('\n' + json.dumps(rows))


if __name__ == '__main__':
    main()
