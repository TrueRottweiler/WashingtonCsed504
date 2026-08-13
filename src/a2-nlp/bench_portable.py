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

A preset that does not fit is a configuration problem, not a verdict. The 98M model wants
~10 GB at the study's batch and the most common student card has 8, so on an out-of-memory --
a real OOM, or the silent spill into system RAM that Windows calls working, which times the
PCIe bus instead of the GPU -- the script retries the same 16,384-token step in smaller
pieces: gradient accumulation first (mlm_train.pretrain has the same accum= knob), activation
checkpointing after. Every configuration still does 128 x 128 tokens of updates per optimizer
step, so the row stays the same experiment in the project's own unit, and it records how it
had to fit. --cpu ignores the GPU and measures the machine without it: the baseline that says
what the GPU is actually buying you.
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
#
# A LITERAL ON PURPOSE, AND THEREFORE PINNED. This file has to stand alone -- it is pasted into a
# fresh Colab cell with no repository behind it, so it cannot compute this from runs/. That makes
# it exactly the shape of constant this project keeps writing panels about: measured once, correct
# then, and quietly deciding an answer somewhere else later.
#
# It said 83.3 until 12 August, by which point the project had reached 148.0, so a Colab T4 was
# told the whole term would cost it 492 hours when the answer was 874 -- 20 days against 36. The
# number was not wrong when it was written. It was wrong by the time somebody used it, which is
# the whole failure mode.
#
# test_board_numbers.py now asserts these against the live records, so the next drift fails a test
# rather than reaching a student.
PROJECT_GPU_HOURS = 148.0        # recomputed 12 Aug 2026 from mlm_api.results() + ft_api.results()
REF_TOK_S = {'poc': 381_817, 'afriberta': 184_329}   # sustained medians; 127 and 70 runs today


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
    reported the whole project as 114 days. Re-run with this fix the same card gives **64,644
    tok/s, 5.9x**, and 36 days. Nothing errored and nothing warned; the benchmark simply produced
    a number wrong by 5.6x, in the direction that would have killed the board's central claim --
    which is that you do not need the workstation.

    The memory reading was wrong too, and for the same reason. The 86M model reported
    OutOfMemoryError on 15 GB under emulated bf16; in fp16 it fits at batch 128 with 9.85 GB peak.
    So "DOES NOT FIT" was not a property of the card at all.

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


class MemorySpill(RuntimeError):
    """The driver oversubscribed VRAM into system RAM instead of raising OOM (Windows does
    this). The step 'works' while timing the PCIe bus -- our first laptop measurement of the
    98M model came out at 5,075 tok/s against an honest 32,267 -- so it is treated exactly
    like an OOM: not a measurement, try a smaller footprint."""


def bench(preset, device, dev_name, steps, warmup, micro=None, ckpt=False):
    micro = micro or BATCH
    if BATCH % micro:
        raise SystemExit(f'--micro-batch must divide {BATCH}')
    accum = BATCH // micro

    free_start = None
    if device.type == 'cuda':
        torch.cuda.empty_cache()                 # release anything a failed attempt cached
        torch.cuda.reset_peak_memory_stats()
        free_start, _ = torch.cuda.mem_get_info()

    model = build(preset, device)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - model.get_input_embeddings().weight.numel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dt = amp_dtype(device)

    # One fixed micro-batch, reused. Generating fresh ids each step would time the RNG as well
    # as the model, and the model is the thing under test. `accum` micro-batches make one
    # optimizer step, the same way mlm_train.pretrain(accum=) does it, so a step is always
    # BATCH x SEQ tokens of updates however little memory it is squeezed through.
    ids = torch.randint(0, VOCAB, (micro, SEQ), device=device)
    labels = ids.clone()

    def one_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            if dt is not None:
                with torch.autocast('cuda', dtype=dt):
                    loss = model(input_ids=ids, labels=labels).loss / accum
            else:
                loss = model(input_ids=ids, labels=labels).loss / accum
            loss.backward()
        opt.step()

    one_step()                                   # the first step allocates everything at once:
    if device.type == 'cuda':                    # params, grads, optimizer state, activations
        torch.cuda.synchronize()
        # Allocations beyond what was free at the start cannot have come out of VRAM. The 0.95
        # leaves margin for other processes' usage drifting between the two readings; a false
        # positive only costs falling back to a config that certainly fits.
        if torch.cuda.max_memory_allocated() > free_start * 0.95:
            raise MemorySpill(f'peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB '
                              f'against {free_start / 1024**3:.1f} GB free')
    for _ in range(warmup - 1):                  # warmup is not optional: the first steps pay
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
    # Compute capability and SM count identify the GENERATION, which is what actually predicts
    # whether bf16 exists and how the card will behave. A student reading "T4" has no way to know
    # it is a 2018 part; reading "7.5" against our "12.0" makes the gap obvious.
    cc = sms = None
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        cc, sms = f'{props.major}.{props.minor}', props.multi_processor_count
    r = {'preset': preset, 'device': dev_name, 'compute_capability': cc, 'sms': sms,
         'dtype': str(dt or torch.float32),
         'params_m': round(n_params / 1e6, 1), 'backbone_m': round(n_backbone / 1e6, 1),
         'tokens_per_s': round(tok_s), 'peak_gb': round(peak, 2) if peak else None,
         # `batch` is the EFFECTIVE batch and it is always 128: accumulation folds how the step
         # is computed, not what the step is. The projection is over the token budget, and a
         # 16,384-token step at micro-batch 32 is the same experiment to four decimal places.
         'batch': BATCH,
         'full_run_hours': round(FULL_RUN_STEPS * BATCH * SEQ / tok_s / 3600, 2)}
    # Only when the run had to be squeezed: a row without these keys is the plain configuration,
    # and old rows stay comparable with new ones.
    if accum > 1:
        r['micro_batch'], r['grad_accum'] = micro, accum
    if ckpt:
        r['checkpointing'] = True
    return r


# What to try when a preset does not fit, cheapest compromise first: accumulation costs a few
# percent, checkpointing recomputes the forward pass and costs about a third. The order encodes
# the advice we would give in person.
FALLBACKS = [dict(micro=64), dict(micro=32), dict(micro=32, ckpt=True),
             dict(micro=16, ckpt=True), dict(micro=8, ckpt=True)]


def fit_label(micro, ckpt):
    bits = []
    if micro and micro != BATCH:
        bits.append(f'micro-batch {micro} x {BATCH // micro}')
    if ckpt:
        bits.append('checkpointing')
    return ' + '.join(bits) or f'full batch {BATCH}'


def bench_with_fallback(preset, device, dev_name, steps, warmup, micro=None, ckpt=False):
    """Measure as asked; when memory says no, fold the batch rather than shrink it.

    "DOES NOT FIT" is the wrong answer to the question this benchmark exists to ask. A Colab T4
    reported exactly that for the 86M model, and it was misleading: the model fits in 15 GB
    comfortably, what does not fit is 128 sequences of activations alongside it. Our own 8 GB
    laptop then failed worse -- Windows spilled the overflow into system RAM and reported a
    "working" 5,075 tok/s that was really the PCIe bus. A student reading either conclusion
    decides they cannot do the study. They can.

    An earlier version answered by halving the true batch, leaning on the token budget to keep
    the row comparable. Gradient accumulation is the stricter fix: run the same 128-sequence
    step as `accum` micro-batches with the loss averaged across them -- identical update math,
    identical schedule, the batch never stops being 128. That matters twice over. Panel 1's
    "steps and tokens are interchangeable" argument only holds while the batch is frozen, and
    the 86M preset's learning rate is fragile (PRESET_LR in mlm_train.py) in ways nobody has
    re-tuned at batch 32. mlm_train.pretrain() takes the same accum= knob, so the configuration
    this measures is one the factory can actually run, unchanged.

    Measured on the 8 GB card: full batch spills at a peak of 9.1 GB, micro-batch 64 x 2 fits
    at 5.98 GB and 32,267 tok/s, and 32 x 4 measures within 2% of that -- accumulation depth is
    nearly free. Checkpointing is the last resort, not the first. The row records what it took,
    because a throughput number without its configuration is not reproducible.
    """
    asked = dict(micro=micro or BATCH, ckpt=ckpt)
    attempts = [asked] + [c for c in FALLBACKS
                          if c.get('micro', BATCH) < asked['micro']
                          or (c.get('micro', BATCH) <= asked['micro']
                              and c.get('ckpt') and not asked['ckpt'])]
    oom = (torch.cuda.OutOfMemoryError, MemorySpill) if device.type == 'cuda' \
        else (RuntimeError,)
    for cfg in attempts:
        try:
            return bench(preset, device, dev_name, steps, warmup,
                         micro=cfg.get('micro'), ckpt=cfg.get('ckpt', False))
        except oom as e:
            # str() rather than the exception itself: e.__traceback__ pins bench()'s frame,
            # and with it the failed model, which would shrink every later attempt's memory.
            why = str(e) if isinstance(e, MemorySpill) else type(e).__name__
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if cfg is attempts[-1]:
                raise
            print(f'{preset:>10}: no room for {fit_label(cfg.get("micro"), cfg.get("ckpt", False))} '
                  f'({why}) -- retrying smaller')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--preset', default=None, help='poc or afriberta; both if omitted')
    ap.add_argument('--out', default=None, help='append the rows to this JSON file')
    ap.add_argument('--micro-batch', type=int, default=None,
                    help='sequences per forward pass; must divide 128. Gradient accumulation '
                         'keeps every optimizer step at 128 x 128 tokens, so rates stay '
                         'comparable however small the card.')
    ap.add_argument('--checkpointing', action='store_true',
                    help='recompute activations in the backward pass: ~1/3 slower, much smaller')
    ap.add_argument('--cpu', action='store_true',
                    help='ignore any GPU and measure the CPU: the baseline that answers '
                         'whether your laptop GPU is worth using')
    ap.add_argument('--no-fallback', action='store_true',
                    help='record "does not fit" instead of retrying smaller configurations')
    ap.add_argument('--note', default='',
                    help='free text recorded with the result, e.g. "plugged in". On a laptop '
                         'this matters: our own a1-cv notes measured a 17%% swing from boost '
                         'behaviour, so a battery reading is not comparable to a mains one.')
    a = ap.parse_args()

    if a.cpu:
        device, dev_name = torch.device('cpu'), platform.processor() or platform.machine()
    else:
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
            print('NOTE: other processes hold part of this card. If one of them is COMPUTING, '
                  'these numbers will read low -- a contended measurement on our own box came '
                  'out at half the sustained rate. A desktop merely holding memory (Windows '
                  'uses ~1 GB for the display) costs headroom rather than speed, and the '
                  'fallback below accounts for it.')
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
            if a.no_fallback:
                r = bench(preset, device, dev_name, a.steps, a.warmup,
                          micro=a.micro_batch, ckpt=a.checkpointing)
            else:
                r = bench_with_fallback(preset, device, dev_name, a.steps, a.warmup,
                                        micro=a.micro_batch, ckpt=a.checkpointing)
        except ((torch.cuda.OutOfMemoryError, MemorySpill)
                if device.type == 'cuda' else RuntimeError) as e:
            # Not a failure of the benchmark. "It does not fit" is one of the answers a student
            # needs, so it is recorded as a result rather than raised.
            print(f'{preset:>10}: DOES NOT FIT -- {type(e).__name__}')
            rows.append({'preset': preset, 'device': dev_name, 'error': 'out of memory',
                         'note': a.note})
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
        # How this machine compares to the box the project ran on, and what the whole term of
        # work would have cost here. A ratio is easier to reason about than a raw rate.
        ratio = REF_TOK_S[preset] / r['tokens_per_s']
        r['vs_workstation'] = round(ratio, 2)
        r['project_hours_here'] = round(PROJECT_GPU_HOURS * ratio, 1)
        r['note'] = a.note
        rows.append(r)
        how = ''
        if 'micro_batch' in r or 'checkpointing' in r:
            how = f"  via {fit_label(r.get('micro_batch'), r.get('checkpointing', False))}"
        print(f"{preset:>10}: {r['params_m']:>5}M params ({r['backbone_m']}M backbone)  "
              f"{r['tokens_per_s']:>8,} tok/s  "
              f"peak {r['peak_gb'] if r['peak_gb'] else '--'} GB{how}")
        print(f"{'':>10}  one 62,500-step run: {r['full_run_hours']:.2f} h"
              f"   |  {ratio:.1f}x the workstation"
              f"   |  the whole {PROJECT_GPU_HOURS:.0f}-GPU-hour project: "
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
