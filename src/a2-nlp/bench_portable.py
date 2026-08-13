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
# What the whole A2 project consumed on the workstation, so any machine can be told what the
# same term of work would have cost it. This is the number that decides whether a student can
# attempt a study like this at all, and it is more useful than tokens per second.
PROJECT_GPU_HOURS = 83.3
REF_TOK_S = {'poc': 381_817, 'afriberta': 184_329}   # our sustained medians, 96 and 55 runs


def pick_device():
    # A TPU runtime has torch but no CUDA, so without this check it would silently fall through
    # to CPU and report a number that describes neither the TPU nor a sensible CPU baseline.
    try:
        import torch_xla                                        # noqa: F401
        raise SystemExit('This is a TPU runtime. The benchmark needs CUDA, MPS or CPU -- '
                         'torch_xla would need a different training loop, and a TPU row is not '
                         'comparable to the others anyway. Pick a GPU or CPU runtime.')
    except ImportError:
        pass
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

    THE CHECK IS THE COMPUTE CAPABILITY, NOT is_bf16_supported(). This function used to ask
    `torch.cuda.is_bf16_supported()`, whose signature is `(including_emulation: bool = True)` --
    so on a Turing card it falls through to `_check_bf16_tensor_supported()`, finds that a
    bfloat16 tensor can be created, and returns True. bf16 then runs in software.

    A Colab T4 measured **11,566 tok/s that way, against 381,817 on the workstation -- 33x**, and
    reported the whole project as 114 days. The real gap is nearer 8x. Nothing errored and nothing
    warned; the benchmark simply produced a number that was wrong by 4x in the direction that
    would have killed the board's central claim, which is that you do not need the workstation.

    bf16 tensor cores arrive with Ampere (sm_80). Asking the hardware directly is both the honest
    question and one fewer library behaviour to track.
    """
    if device.type != 'cuda':
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def build(preset, device):
    from transformers import AutoModelForMaskedLM, RobertaConfig
    cfg = RobertaConfig(vocab_size=VOCAB, max_position_embeddings=SEQ + 2,
                        type_vocab_size=1, **PRESETS[preset])
    return AutoModelForMaskedLM.from_config(cfg).to(device)


def bench(preset, device, dev_name, steps, warmup, batch=BATCH):
    model = build(preset, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - model.get_input_embeddings().weight.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dt = amp_dtype(device)

    # One fixed batch, reused. Generating fresh ids each step would time the RNG as well as the
    # model, and the model is the thing under test.
    ids = torch.randint(0, VOCAB, (batch, SEQ), device=device)
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

    tok_s = steps * batch * SEQ / dt_s
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == 'cuda' else None
    # Compute capability and SM count identify the GENERATION, which is what actually predicts
    # whether bf16 exists and how the card will behave. A student reading "T4" has no way to know
    # it is a 2018 part; reading "7.5" against our "12.0" makes the gap obvious.
    cc = sms = None
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        cc, sms = f'{props.major}.{props.minor}', props.multi_processor_count
    return {'preset': preset, 'device': dev_name, 'compute_capability': cc, 'sms': sms,
            'dtype': str(dt or torch.float32),
            'params_m': round(n_params / 1e6, 1), 'backbone_m': round(n_backbone / 1e6, 1),
            'tokens_per_s': round(tok_s), 'peak_gb': round(peak, 2) if peak else None,
            'batch': batch,
            # The projection is over the TOKEN budget, not the step count, which is why a smaller
            # batch does not invalidate it -- 62,500 steps at batch 128 is 1.024B tokens of
            # updates, and the same budget at batch 32 is 250,000 steps of the same experiment.
            # That is Panel 1's argument doing real work: because the compute axis is tokens
            # rather than steps, a card that cannot hold batch 128 is slower here, not excluded.
            'full_run_hours': round(FULL_RUN_STEPS * BATCH * SEQ / tok_s / 3600, 2)}


def bench_with_fallback(preset, device, dev_name, steps, warmup, floor=8):
    """Measure at batch 128, and on an out-of-memory halve the batch and try again.

    "DOES NOT FIT" is the wrong answer to the question this benchmark exists to ask. A Colab T4
    reported exactly that for the 86M model, and it is misleading: the model fits in 15 GB
    comfortably, what does not fit is 128 sequences of activations alongside it. A student reading
    "does not fit" concludes they cannot do the study. They can -- at a smaller batch.

    That is only true because of how this project defines compute. The budget is 1.024B TOKENS of
    updates, so batch 32 for 250,000 steps is the same experiment as batch 128 for 62,500, and the
    projected hours stay comparable. On a project that counted steps instead, halving the batch
    would halve the work and the row would be a lie.

    Reports the batch it succeeded at, because a throughput number without its batch is not
    reproducible -- and small batches lose throughput to launch overhead, which the figure should
    show rather than hide.
    """
    batch = BATCH
    while True:
        try:
            return bench(preset, device, dev_name, steps, warmup, batch=batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= floor:
                raise
            batch //= 2
            print(f'  {preset}: out of memory at batch {batch * 2}, retrying at {batch}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--preset', default=None, help='poc or afriberta; both if omitted')
    ap.add_argument('--out', default=None, help='append the rows to this JSON file')
    ap.add_argument('--note', default='',
                    help='free text recorded with the result, e.g. "plugged in". On a laptop '
                         'this matters: our own a1-cv notes measured a 17%% swing from boost '
                         'behaviour, so a battery reading is not comparable to a mains one.')
    a = ap.parse_args()

    device, dev_name = pick_device()
    # On CPU the defaults would run for tens of minutes and nobody would wait. Scale down and say
    # so, rather than appearing to hang -- a benchmark people abandon produces no data at all.
    if device.type == 'cpu':
        if a.steps > 6:
            a.steps, a.warmup = 4, 1
            print('CPU detected: dropping to 4 timed steps. Even so, expect minutes, and expect '
                  'the 98M model to be very slow.')
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

    if a.note:
        print(f'note: {a.note}')
    elif device.type != 'cuda' or 'Laptop' in dev_name or 'Mobile' in dev_name:
        print('NOTE: no --note given. If this is a laptop, say whether it was on mains -- a '
              'battery reading is not comparable.')
    rows = []
    for preset in ([a.preset] if a.preset else ['poc', 'afriberta']):
        try:
            r = bench_with_fallback(preset, device, dev_name, a.steps, a.warmup)
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as e:
            # Not a failure of the benchmark. "It does not fit" is one of the answers a student
            # needs, so it is recorded as a result rather than raised.
            print(f'{preset:>10}: DOES NOT FIT -- {type(e).__name__}')
            rows.append({'preset': preset, 'device': dev_name, 'error': 'out of memory',
                         'note': a.note})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        # How this machine compares to the box the project ran on, and what the whole term of
        # work would have cost here. A ratio is easier to reason about than a raw rate.
        ratio = REF_TOK_S[preset] / r['tokens_per_s']
        r['vs_workstation'] = round(ratio, 2)
        r['project_hours_here'] = round(PROJECT_GPU_HOURS * ratio, 1)
        r['note'] = a.note
        rows.append(r)
        print(f"{preset:>10}: {r['params_m']:>5}M params ({r['backbone_m']}M backbone)  "
              f"{r['tokens_per_s']:>8,} tok/s  "
              f"peak {r['peak_gb'] if r['peak_gb'] else '--'} GB")
        print(f"{'':>10}  one 62,500-step run: {r['full_run_hours']:.2f} h"
              f"   |  {ratio:.1f}x the workstation"
              f"   |  the whole 83-GPU-hour project: "
              f"{r['project_hours_here']:.0f} h ({r['project_hours_here']/24:.1f} days)")

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
