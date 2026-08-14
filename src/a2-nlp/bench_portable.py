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
a real OOM, or one of the two silent spills that no operating system will raise for you (Windows
pages VRAM over PCIe and calls it working; a Mac takes the overflow out of the same pool the
rest of the machine is using) -- the script retries the same 16,384-token step in smaller
pieces: gradient accumulation first (mlm_train.pretrain has the same accum= knob), activation
checkpointing after. Every configuration still does 128 x 128 tokens of updates per optimizer
step, so the row stays the same experiment in the project's own unit, and it records how it
had to fit. --cpu ignores the GPU and measures the machine without it: the baseline that says
what the GPU is actually buying you.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
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

# WHAT THE RATIO DIVIDES BY, AND WHY IT IS NOW THE WORKSTATION'S OWN BENCHMARK.
#
# It was the median over 127 and 70 real training runs until 13 August, on the reasoning that a
# real median beats a synthetic sitting. It does, as a description of what this box delivered --
# but not as the denominator of a ratio, because every numerator is another machine running THIS
# SCRIPT. Dividing a benchmark by a real-run median compares two different measurements and
# flatters whichever side is measured the friendlier way.
#
# So the reference is now this script, on one idle card, timing the same realistic loop every
# other row times: 442,510 and 186,534. Comparable by construction.
#
# It is also validated, which no earlier version of this constant was. The 98M preset's real
# runs, which average 93 minutes, have a median of 183,697 and a p90 of 187,594 -- the benchmark
# lands at 1.006 of that p90 and 1.015 of the median. It predicts the real thing on the one
# machine where the real thing can be checked.
#
# The 33.8M preset agrees at the top and not in the middle: p90 427,932 (0.97 of this), median
# 381,817 (0.86). That is not the benchmark being wrong, it is 9-minute runs spanning 1.73x from
# p10 to p90 while 93-minute runs span 1.11x. Short runs are at the mercy of the machine's other
# work; the ratio below is therefore a CEILING, and REALISTIC_FRACTION says so.
#
# test_board_numbers.py asserts both against the live records, so the next drift fails a test.
REF_TOK_S = {'poc': 442_510, 'afriberta': 186_534}    # this script, idle card 0, 13 Aug 2026

# What fraction of that ceiling the project's own runs actually sustained, at the median. The
# honest thing to hand a student alongside a ratio: the benchmark says what your machine can do
# when it is yours alone, and this says what ours delivered in practice over 190 real runs.
REALISTIC_FRACTION = {'poc': 0.86, 'afriberta': 0.98}      # median real run / this benchmark


TICK = 20.0                      # seconds between progress lines while a preset is timing


def say(msg):
    """Print immediately. Colab buffers stdout, and a progress line that arrives at the end of
    the run is worse than none -- it looks like the script sat silent and then lied about it."""
    print(msg, flush=True)


# The data path, copied from mlm_data.MlmTokens rather than imported, so this file still pastes
# into a bare Colab cell. What it holds is random rather than Yoruba, which is worthless for
# learning and identical for timing -- a gather and a comparison do not care what the ids mean.
POOL_TOKENS = 64_000_000         # real runs held 16M (poc median) to 1,024M; 64M sits between
POOL_TOKENS_SMALL = 16_000_000   # unified memory has no headroom to spare -- see make_pool()
MLM_PROB, MASK_ID, N_SPECIAL = 0.15, 4, 5


def make_pool(device):
    """The token stream a real run trains from: resident on the device, not fed from a loader.

    int16 where the device is a GPU, because that is what mlm_data's `store_dtype='auto'` picks
    for a 16k vocabulary and the gather bandwidth is half what int32 costs. MPS and CPU take
    int32, which is the same data and the safer index path on backends that are patchier about
    narrow integer types; the whole data path is under a millisecond either way.

    SMALLER ON MPS, and #89 is the reason. A Mac has no free VRAM to spend -- the 98M preset
    already lands at 16.2 GB against Metal's 17.8 GB recommended working set, and past that line
    the machine does not fail, it degrades silently and hands back 286 tok/s. 256 MB of int32
    pool is not worth a quarter of the remaining headroom, and 16M tokens is the median corpus
    the `poc` runs actually used, so it is the more faithful number anyway.
    """
    if device.type == 'cuda':
        return torch.randint(0, VOCAB, (POOL_TOKENS,), device=device, dtype=torch.int16)
    return torch.randint(0, VOCAB, (POOL_TOKENS_SMALL,), device=device, dtype=torch.int32)


def masked_batch(pool, micro, device):
    """One training batch: random windows out of the stream, then BERT 80/10/10 corruption.

    Line for line what MlmTokens.windows() and .mask() do, including the two boolean-mask
    scatters, because those are the part with a cost. Of the positions selected at mlm_prob,
    80% become <mask>, 10% become a random real token and 10% are left alone; labels are -100
    everywhere the model is not asked to predict.
    """
    # Bounded by the pool that was actually built, not by POOL_TOKENS. Those differ on MPS, and
    # reading the constant here would have indexed a 64M-token window into a 16M-token tensor --
    # on the one backend nobody here can test against.
    starts = torch.randint(0, pool.numel() - SEQ - 1, (micro,), device=device)
    x = pool[starts.view(-1, 1) + torch.arange(SEQ, device=device)].long()
    labels = x.clone()
    sel = torch.rand(x.shape, device=device) < MLM_PROB
    labels[~sel] = -100
    r = torch.rand(x.shape, device=device)
    out = x.clone()
    out[sel & (r < 0.8)] = MASK_ID
    rnd = torch.randint(N_SPECIAL, VOCAB, x.shape, device=device)
    swap = sel & (r >= 0.8) & (r < 0.9)
    out[swap] = rnd[swap]
    return out, labels


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


def other_compute_processes(device):
    """Who else is on this card. Returns a list of pids, or None if it could not be determined.

    THIS REPLACES A MEMORY THRESHOLD THAT CRIED WOLF ON EVERY RUN. The old check asked
    torch.cuda.mem_get_info(), called `total - free` "already in use", and warned above 1 GB.
    But mem_get_info can only run after CUDA is initialised, and the context itself is ~1.6 GB
    of the device under Windows -- so on 13 August an idle card with 0 MiB in nvidia-smi still
    printed `1.6 GB already in use` and fired the warning, with torch.cuda.memory_reserved()
    sitting at exactly 0.00 GB. The warning was measuring us.

    That is worse than no warning. This file tells the reader in three places never to read past
    that line, and the line appears on every run including the clean ones -- which is precisely
    how three genuinely contended workstation readings got waved through as "Windows display
    memory". A signal that is always on carries no information.

    So ask the driver who is actually attached instead. Two details make it work where the
    obvious version does not:

      - Match on the device UUID, not the index. CUDA_VISIBLE_DEVICES renumbers what torch sees
        while nvidia-smi keeps reporting physical indices, so `device 0` means two different
        cards to the two tools during exactly the runs this check exists for.
      - Do not ask for per-process memory. nvidia-smi reports used_gpu_memory as [N/A] under
        Windows WDDM, so a memory-based version of this check silently degrades to nothing on
        the platform the project's own workstation runs.

    Graphics-only processes do not appear in --query-compute-apps, which is the behaviour we
    want: a desktop drawing windows costs headroom, not throughput, and it is the compute
    clients (an ollama server, someone else's training run) that make a number wrong.
    """
    if device.type != 'cuda':
        return []
    try:
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                              '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except Exception:
        return None                     # no nvidia-smi, or a torch too old to expose the uuid
    mine, others = os.getpid(), []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2 and uuid in parts[0] and parts[1].isdigit() and int(parts[1]) != mine:
            others.append(int(parts[1]))
    return others


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
    tok/s, 5.9x**, and 36 days. (Both of those are the old 40-step burst. Held for three minutes
    the same card gives 54,144 -- 7.1x and 50 days -- which is the number the figure now quotes;
    the T4 is the only tier where the two methods disagree by more than 2%.) Nothing errored and
    nothing warned; the benchmark simply produced
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
    """The driver oversubscribed device memory into system RAM instead of raising OOM. The step
    'works' while timing the wrong hardware -- our first laptop measurement of the 98M model came
    out at 5,075 tok/s against an honest 32,267 -- so it is treated exactly like an OOM: not a
    measurement, try a smaller footprint.

    TWO OPERATING SYSTEMS DO THIS, FOR DIFFERENT REASONS, AND NEITHER RAISES ANYTHING.
    On Windows the driver pages VRAM over PCIe. On a Mac there is no separate VRAM to overflow --
    the GPU and the CPU share one pool, so going past what Metal recommends does not fail, it
    just starts swapping against the rest of the machine. The first MacBook reading of the 98M
    model was 286 tok/s: four steps in 229 seconds, the rate decaying 1,029 -> 701 -> 515 as the
    machine dug itself further in. Held under the recommended working set it is 6,000+ -- a
    factor of twenty, in the direction that says a Mac cannot do this project."""


def sync(device):
    """Wait for the device to actually finish, on whichever backend this is.

    CUDA and MPS both queue work asynchronously, so a clock read without this records when the
    work was SUBMITTED. The timing loop always got this right for CUDA and, after one fix, for
    the marks on MPS -- but the final elapsed and the memory readings around warmup were still
    CUDA-only, which is exactly how a Mac row ends up with numbers nobody checked."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def empty_cache(device):
    """Hand cached blocks back before the next attempt, so a failed configuration does not
    shrink the memory available to the smaller one that follows it."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def mem_budget(device):
    """How much memory this run can occupy before the machine starts paging, or None if the
    backend cannot say.

    The two backends answer slightly different questions and both answers are the right one for
    their platform. CUDA reports what is FREE on the card right now, across every process, which
    is what a discrete card can still hand out. MPS reports Metal's RECOMMENDED WORKING SET --
    on a 24 GB M4 Pro that is 17.8 GB -- because unified memory has no free/total to speak of:
    the GPU can always have more, it just takes it from the operating system and everything else
    running. Past that line the machine does not fail, it degrades, which is the harder failure
    to notice and the reason this function exists."""
    if device.type == 'cuda':
        return torch.cuda.mem_get_info()[0]
    if device.type == 'mps':
        return torch.mps.recommended_max_memory()
    return None


def mem_held(device):
    """High-water bytes this process has taken from the device, or None.

    CUDA has a true peak counter that can be reset. MPS has no peak statistic in torch 2.5, so
    this reports what the driver is currently holding -- which serves as a high-water mark
    because the caching allocator does not give blocks back until empty_cache()."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated()
    if device.type == 'mps':
        return torch.mps.driver_allocated_memory()
    return None


def check_spill(device, budget, where):
    """Raise if this configuration is working outside the memory the device can comfortably give.

    The 0.95 leaves margin for other processes drifting between two readings; a false positive
    only costs falling back to a configuration that certainly fits.

    CHECKED TWICE, after the first step and again after warmup. On CUDA the first step really
    does allocate everything at once -- params, grads, optimizer state, activations -- so one
    check was enough. MPS grows into its allocation: the 98M model at full batch reads 12.6 GB
    after step one and 15.6 GB by step two. A single early check is a check that passes."""
    held = mem_held(device)
    if budget and held and held > budget * 0.95:
        raise MemorySpill(f'{held / 1024**3:.1f} GB held {where} against a '
                          f'{budget / 1024**3:.1f} GB budget')


def bench(preset, device, dev_name, steps, warmup, micro=None, ckpt=False,
          seconds=180.0, bare_seconds=60.0):
    micro = micro or BATCH
    if BATCH % micro:
        raise SystemExit(f'--micro-batch must divide {BATCH}')
    accum = BATCH // micro

    empty_cache(device)                          # release anything a failed attempt cached
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    budget = mem_budget(device)

    say(f'{preset:>10}: building the model ...')
    model = build(preset, device)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = n_params - model.get_input_embeddings().weight.numel()
    dt = amp_dtype(device)

    # THE STEP THIS TIMES IS THE ONE pretrain() RUNS, NOT A STRIPPED-DOWN COUSIN.
    #
    # It was the cousin until 13 August, and the difference was 11% on the 33.8M model. The old
    # loop reused one fixed micro-batch and called plain AdamW, reasoning that "the model is the
    # thing under test" -- but the thing under test is how long a RUN takes, and a run also builds
    # every batch, clips gradients and reads the loss back to the host. Measured on an idle card,
    # one ingredient at a time:
    #
    #                                        33.8M       98M
    #     the old bench_portable step       475,473   190,492
    #     + mlm_train's AdamW settings        -1.9%     -0.1%
    #     + clip_grad_norm_(1.0)              -2.2%     -0.7%
    #     + loss.item() every step            -2.6%     -0.8%
    #     + a freshly masked batch            -2.1%     -0.3%
    #     ------------------------------------------------------
    #     predicted                         434,874   186,797
    #     pretrain() itself, same card      427,290         -    (within 1.8%)
    #
    # Every ingredient is a roughly FIXED cost per step -- kernel launches, a host round trip, a
    # norm reduction -- so it is 2% of a 34 ms step and 0.5% of an 86 ms one. That is why the
    # error was never a constant anyone could divide out: it scales with how cheap the step is,
    # so it also shrinks on slower hardware. The old benchmark was LEAST accurate on the fastest
    # machine here, which is the one every other row gets divided by.
    #
    # loss.item() is the sharpest of them. mlm_train.py:340 calls it "a rounding error" because
    # "the step is far heavier than a small LM's" -- true at 98M, where it costs 0.8%, and the
    # largest single ingredient at 33.8M, where it costs 2.6%. The comment was right about the
    # model it was written beside.
    #
    # The bare step is still measured, in the same sitting, because the gap between the two is a
    # property of the MACHINE and not of ours. A card whose step is slow hides fixed costs that a
    # fast one cannot, so every row carries its own gap rather than inheriting this one.
    # ONE optimizer, shared by both loops, built the way mlm_train.pretrain() builds it -- the
    # absence of fused= included, because that is what the factory actually runs. Sharing it is
    # deliberate on two counts. Two AdamW states for the 98M model is 1.6 GB, which is real money
    # on the 8 GB card this has to fit and would inflate the peak_gb we report as "what a run
    # needs". And it makes the bare/realistic gap mean one clean thing: the cost of building
    # batches, clipping, and reading the loss back. Not the cost of AdamW's keyword arguments.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                            betas=(0.9, 0.98), eps=1e-6)
    pool = make_pool(device)
    fixed_ids = torch.randint(0, VOCAB, (micro, SEQ), device=device)
    fixed = (fixed_ids, fixed_ids.clone())

    # `accum` micro-batches make one optimizer step, the same way mlm_train.pretrain(accum=)
    # does, so a step is always BATCH x SEQ tokens of updates however little memory it is
    # squeezed through.
    def make_step(bare):
        def one_step():
            opt.zero_grad(set_to_none=True)
            loss = None
            for _ in range(accum):
                x, y = fixed if bare else masked_batch(pool, micro, device)
                if dt is not None:
                    with torch.autocast('cuda', dtype=dt):
                        loss = model(input_ids=x, labels=y).loss / accum
                else:
                    loss = model(input_ids=x, labels=y).loss / accum
                loss.backward()
            if not bare:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if not bare:
                loss.item()                  # the host sync a real run pays for its EMA readout

        return one_step

    one_step, bare_step = make_step(bare=False), make_step(bare=True)

    one_step()                                   # on CUDA the first step allocates everything at
    sync(device)                                 # once: params, grads, optimizer state, activations
    check_spill(device, budget, 'after the first step')
    say(f'{preset:>10}: {warmup} warmup steps ...')
    for _ in range(warmup - 1):                  # warmup is not optional: the first steps pay
        one_step()                               # for kernel autotuning and allocator growth
    sync(device)
    # Again, now that the allocator has stopped growing. This is the check that catches a Mac.
    check_spill(device, budget, 'after warmup')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # TIME A DURATION, NOT A STEP COUNT, AND REPORT THE END RATHER THAN THE AVERAGE.
    #
    # Jeffrey asked how a two-minute test can be representative of a laptop, and the honest
    # answer is that it was not representative of anything. At 40 steps the timed section is
    # 1.3 seconds of work on the workstation and 20 on the laptop -- a burst, taken while every
    # card is still cold, and the number it returns is a boost clock rather than a rate anything
    # can hold.
    #
    # The proof is on the one machine where both readings exist, and pulling it apart took three
    # tries. The old 40-step burst read 490,338 tok/s on the workstation against a 381,817 median
    # over 127 real runs -- a gap of 1.28x this comment used to blame entirely on the burst, and
    # which a later attempt blamed mostly on the method. Both were wrong, and wrong for the same
    # reason: the stages were measured in sequence on a card that sheds ~5% over its first
    # minutes, so whichever ran later was charged for the drift. Interleaved, on the 33.8M model:
    #
    #     490,338   40-step burst, cold card
    #     454,544   the bare step, held and interleaved       (-7.3%  the burst)
    #     442,510   the realistic loop, idle card             (-2.6%  the method, fixed above)
    #     427,932   p90 of 120 real runs                      (-3.3%  eval and checkpoint writes)
    #     381,817   the MEDIAN of those runs                  (-10.8% the conditions)
    #
    # Only the first two were the benchmark's to fix, and both now are. What is left is not bias,
    # it is DISPERSION: real runs span 1.73x from p10 to p90 on this preset, because a 9-minute
    # run is at the mercy of whatever else the box does during those 9 minutes. The 98M preset,
    # whose runs average 93 minutes, spans only 1.11x and its p90 sits 1.006 of this benchmark.
    #
    # So the honest reading of any row here is A CEILING, and a well-behaved run reaches it. How
    # far below it you land is a property of your machine's other work, not of this code, and the
    # only way to know it for a machine is to measure that machine more than once.
    #
    # So: run for a fixed number of SECONDS, and report the last third separately from the first.
    # The ratio between them is the throttle, measured rather than assumed, and on a laptop it is
    # a finding rather than an artefact.
    #
    # It also prints while it works. Three minutes a preset is six minutes of silence for the two,
    # and on a Colab cell silence is indistinguishable from a hang -- Jeffrey watched one for
    # several minutes not knowing whether it had died. Every 20 seconds it says where it is and
    # what it is currently getting, so a slow machine looks slow rather than broken, and the
    # running figure lets you see the throttle happening rather than waiting for the verdict.
    def time_phase(step_fn, run_for, label, alt_fn=None, alt_total=0.0, block=20.0):
        """Time step_fn for run_for seconds. If alt_fn is given, INTERLEAVE the two.

        Interleaved because measuring them in sequence does not compare them. The first version
        of this ran the realistic loop for three minutes and then the bare step for one, and
        reported the bare step as 1.01x -- against 1.09x from a staged experiment earlier the
        same hour. Neither number was the gap. This card sheds about 5% over its first couple of
        minutes, so whichever loop went second was measured at a lower clock, and the drift was
        sitting inside the comparison with its own sign. The staged experiment had the same bug
        pointing the other way: it added ingredients cheapest-first, so every ingredient was
        credited with the thermal decay that happened while it was being measured.

        Alternating ten-second blocks cancels any monotonic drift to first order -- both loops
        see the same clock history, near enough -- and it costs nothing but bookkeeping. The
        progress line and the throttle still describe the realistic loop, which is the headline.
        """
        # sync(device) before every clock read, or a mark records when the work was SUBMITTED
        # rather than when it finished. That helper is the Mac's, from #89, and it is why this
        # loop is correct on MPS as well as CUDA -- the device where throttle matters most.
        def run_block(fn, budget, count):
            """Run fn until budget seconds of ITS OWN time are spent. Returns (steps, seconds)."""
            b0, k = time.perf_counter(), 0
            while time.perf_counter() - b0 < budget:
                fn()
                k += 1
                if (count + k) % 10 == 0:
                    sync(device)
                    marks.append((count + k, own + time.perf_counter() - b0))
            sync(device)
            return k, time.perf_counter() - b0

        # The alternate gets alt_total seconds spread over the same number of blocks, so it is
        # sampled across the whole sitting rather than bolted on at one end -- but it costs a
        # third of the wall clock rather than doubling it. Three minutes of realistic loop and
        # one of bare step is four minutes a preset, which is a number people will actually sit
        # through; the point of the bare figure is the ratio, and a ratio converges quickly.
        n_blocks = max(1, int(run_for / block + 0.999)) if alt_fn is not None else 1
        alt_block = alt_total / n_blocks
        say(f'{preset:>10}: timing {label} for {run_for:.0f}s ...')
        marks, n, own, next_report = [], 0, 0.0, TICK
        alt_n, alt_own = 0, 0.0
        def alt_turn():
            """One block of the alternate loop, which never writes the throttle marks."""
            nonlocal alt_n, alt_own, marks
            saved, marks = marks, []
            k, took = run_block(alt_fn, alt_block, alt_n)
            alt_n, alt_own, marks = alt_n + k, alt_own + took, saved

        # WHICH LOOP LEADS THE CYCLE ALTERNATES, AND ON A DECAYING CARD THAT IS THE WHOLE BALLGAME.
        #
        # Interleaving was supposed to cancel drift, and it does cancel drift BETWEEN cycles. What
        # the first version did not cancel was drift WITHIN one: the alternate always ran second,
        # so on a card whose clocks fall it was always sampled a few seconds later and a little
        # cooler. The T4 sheds 20% across three minutes -- throttle 1.19 to 1.22 over three
        # sittings -- and the bias showed up unmistakably as `bare_over_real` of 0.99: the
        # stripped-down step, which does strictly less work, reading SLOWER than the loop that
        # builds and masks a batch on top of it. An impossible number is a gift; a plausible one
        # would have gone on the board.
        #
        # Leading with the alternate on odd cycles puts both loops at the same average point in
        # the thermal history. Everything measured before this fix on a machine with throttle
        # near 1.0 is unaffected -- the workstation and the Mac -- and the T4's own rows read
        # about 1% optimistic on the 33.8M preset until they are taken again.
        cycle = 0
        while own < run_for:
            if alt_fn is not None and cycle % 2 and own > 0:
                alt_turn()
            k, took_k = run_block(step_fn, min(block if alt_fn else run_for, run_for - own), n)
            n, own = n + k, own + took_k
            if own >= next_report:
                say(f'{"":>10}  {own:5.0f}s / {run_for:.0f}s   {n:6,} steps'
                    f'   {n * BATCH * SEQ / own:9,.0f} tok/s so far')
                next_report += TICK
            if alt_fn is not None and not cycle % 2 and own < run_for:
                alt_turn()
            cycle += 1

        # First third against last third, from the marks, so the throttle needs no second run.
        # `own` time throughout, so interleaving does not leak into it.
        early = late = None
        if len(marks) >= 3:
            a, b = marks[len(marks) // 3 - 1], marks[-1]
            mid = marks[(2 * len(marks)) // 3 - 1]
            early = a[0] * BATCH * SEQ / a[1]
            late = (b[0] - mid[0]) * BATCH * SEQ / (b[1] - mid[1])
        alt = (dict(tok_s=alt_n * BATCH * SEQ / alt_own, steps=alt_n, took=alt_own)
               if alt_fn is not None and alt_own > 0 else None)
        return dict(tok_s=n * BATCH * SEQ / own, steps=n, took=own,
                    early=early, late=late, alt=alt)

    if bare_seconds > 0:
        for _ in range(max(2, warmup // 2)):     # the bare loop wants its own warmup
            bare_step()
        sync(device)
    real = time_phase(one_step, seconds, 'a real training step, against the bare step',
                      alt_fn=bare_step if bare_seconds > 0 else None,
                      alt_total=bare_seconds, block=max(5.0, seconds / 9))
    bare = real['alt']
    dt_s, steps, early, late, tok_s = (real['took'], real['steps'], real['early'],
                                       real['late'], real['tok_s'])
    # Reported on MPS too now, via #89's mem_held(). A row whose peak is blank is a row nobody
    # can sanity-check, and the Mac was the machine that most needed checking: 20.1 GB against a
    # 17.8 GB budget was sitting there in the driver the whole time it reported 286 tok/s.
    held = mem_held(device)
    peak = held / 1024 ** 3 if held else None
    # Compute capability and SM count identify the GENERATION, which is what actually predicts
    # whether bf16 exists and how the card will behave. A student reading "T4" has no way to know
    # it is a 2018 part; reading "7.5" against our "12.0" makes the gap obvious.
    cc = sms = None
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        cc, sms = f'{props.major}.{props.minor}', props.multi_processor_count
    r = {'preset': preset, 'device': dev_name, 'compute_capability': cc, 'sms': sms,
         # WHICH STEP THIS TIMED. Every row written before 13 August 2026 is 'bare-step' and
         # reads high -- on this workstation by 11% at 33.8M and 2% at 98M. The two are not
         # comparable and nothing should ever median them together, which is why the field is
         # written explicitly rather than inferred from a date or a missing key.
         'method': 'realistic-loop',
         # What was ASKED for, beside what it took. The overrun guard in test_board_numbers
         # compares the two, and it used to compare against a hardcoded 180 -- which is right
         # until somebody legitimately runs --seconds 600 to check whether three minutes reaches
         # the settled rate. A test that reads the window off the row cannot be broken by using
         # the flag the script advertises.
         'asked_seconds': round(seconds, 1),
         'timed_seconds': round(dt_s, 1), 'timed_steps': steps,
         # The rate at the start against the rate at the end. 1.0 means the card held its clocks;
         # anything well above means the projection above is a boost number and the machine
         # cannot sustain it for a real run.
         'tok_s_first_third': round(early) if early else None,
         'tok_s_last_third': round(late) if late else None,
         'throttle': round(early / late, 2) if early and late else None,
         'dtype': str(dt or torch.float32),
         'params_m': round(n_params / 1e6, 1), 'backbone_m': round(n_backbone / 1e6, 1),
         'tokens_per_s': round(tok_s), 'peak_gb': round(peak, 2) if peak else None,
         # The same model on the same card in the same sitting, timing the old bare step. The
         # ratio is how much of this machine's step is fixed overhead, and it is a property of
         # the machine: a slow card hides per-step costs a fast one cannot. Measured here rather
         # than assumed, because assuming the workstation's 1.11x applied everywhere is exactly
         # the mistake this pair of numbers exists to prevent.
         'tok_s_bare': round(bare['tok_s']) if bare else None,
         'bare_seconds': round(bare['took'], 1) if bare else None,
         'bare_over_real': round(bare['tok_s'] / tok_s, 3) if bare else None,
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


def bench_with_fallback(preset, device, dev_name, steps, warmup, seconds=180.0, micro=None,
                        ckpt=False, bare_seconds=60.0):
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
    nearly free. (Those three are 40-step bursts, compared against each other. Held for three
    minutes the same 64 x 2 configuration gives 31,836, and 23,850 with the mains unplugged.)
    Checkpointing is the last resort, not the first. The row records what it took,
    because a throughput number without its configuration is not reproducible.
    """
    asked = dict(micro=micro or BATCH, ckpt=ckpt)
    attempts = [asked] + [c for c in FALLBACKS
                          if c.get('micro', BATCH) < asked['micro']
                          or (c.get('micro', BATCH) <= asked['micro']
                              and c.get('ckpt') and not asked['ckpt'])]
    # MemorySpill is a RuntimeError, so it is already inside the non-CUDA tuple -- named anyway,
    # because the whole point of this ladder on a Mac is that nothing else will ever be raised.
    oom = (torch.cuda.OutOfMemoryError, MemorySpill) if device.type == 'cuda' \
        else (MemorySpill, RuntimeError)
    for cfg in attempts:
        try:
            return bench(preset, device, dev_name, steps, warmup, seconds=seconds,
                         bare_seconds=bare_seconds,
                         micro=cfg.get('micro'), ckpt=cfg.get('ckpt', False))
        except oom as e:
            # str() rather than the exception itself: e.__traceback__ pins bench()'s frame,
            # and with it the failed model, which would shrink every later attempt's memory.
            why = str(e) if isinstance(e, MemorySpill) else type(e).__name__
            empty_cache(device)
            if cfg is attempts[-1]:
                raise
            print(f'{preset:>10}: no room for {fit_label(cfg.get("micro"), cfg.get("ckpt", False))} '
                  f'({why}) -- retrying smaller')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=180.0,
                    help='how long to hold each preset under load. The default of 3 '
                         'minutes is long enough for a laptop to reach its steady '
                         'clocks; shorter readings are boost numbers and the row '
                         'says so via `throttle`.')
    ap.add_argument('--bare-seconds', type=float, default=60.0,
                    help='how long to hold the OLD step-only loop, in the same sitting, for '
                         'comparison. The ratio between the two is how much of this machine\'s '
                         'step is fixed overhead, which is a property of the machine rather '
                         'than something our workstation can tell you. 0 skips it.')
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
        # The loop times a DURATION now, so there is no step count to shrink -- what a CPU needs
        # is a shorter duration and one warmup step, or the 98M preset alone takes the best part
        # of ten minutes. Said out loud because a benchmark people abandon produces no data.
        a.warmup = 1
        if a.seconds > 60:
            a.seconds = 60.0
            a.bare_seconds = min(a.bare_seconds, 20.0)
            print(f'CPU detected: timing for {a.seconds:.0f}s per preset rather than 180. Expect '
                  f'minutes even so, and expect the 98M model to be very slow.', flush=True)
    print(f'device: {dev_name}   torch {torch.__version__}   dtype {amp_dtype(device) or "fp32"}')
    if device.type == 'cuda':
        # mem_get_info is the DEVICE's free/total across every process, which is the right
        # question for "will this fit" -- it is kept for the headroom line and the fallback
        # ladder. It is no longer asked "is the card busy", because it cannot separate another
        # job's allocation from our own CUDA context. See other_compute_processes().
        free, total = torch.cuda.mem_get_info()
        used_gb = (total - free) / 1024 ** 3
        others = other_compute_processes(device)
        if others is None:
            print(f'memory: {total/1024**3:.0f} GB total, {used_gb:.1f} GB held (some of it this '
                  f'process). Could not ask nvidia-smi who else is on this card -- check by '
                  f'hand before trusting a slow number.')
        elif others:
            print(f'memory: {total/1024**3:.0f} GB total, {used_gb:.1f} GB held')
            print(f'NOTE: {len(others)} other process(es) are computing on this card '
                  f'(pid {", ".join(str(p) for p in others)}). THESE NUMBERS WILL READ LOW -- a '
                  f'contended measurement on our own box came out at half the sustained rate, '
                  f'and was then used to build two wrong explanations. Stop them and re-run.')
        else:
            print(f'memory: {total/1024**3:.0f} GB total, {used_gb:.1f} GB held by our own CUDA '
                  f'context; no other process is computing on this card.')
    elif device.type == 'mps':
        # Unified memory has no free/total worth printing -- the GPU can always take more, from
        # the rest of the machine. What matters is Metal's recommended working set, because
        # crossing it is the failure this backend actually has, and it is silent.
        print(f'memory: {torch.mps.recommended_max_memory() / 1024**3:.1f} GB recommended '
              f'working set, shared with the OS (unified memory, no separate VRAM)')
    # Print the timing mode the loop actually uses. This line used to say "40 timed steps" from
    # the vestigial --steps flag while the loop timed a duration -- and a transcript that lies
    # about its own method is how burst numbers sneak back onto the figure.
    print(f'batch {BATCH} x seq {SEQ} = {BATCH*SEQ:,} tokens per step, '
          f'timing {a.seconds:.0f}s per preset after {a.warmup} warmup steps\n')

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
                          seconds=a.seconds, bare_seconds=a.bare_seconds,
                          micro=a.micro_batch, ckpt=a.checkpointing)
            else:
                r = bench_with_fallback(preset, device, dev_name, a.steps, a.warmup,
                                        seconds=a.seconds, bare_seconds=a.bare_seconds,
                                        micro=a.micro_batch, ckpt=a.checkpointing)
        except ((torch.cuda.OutOfMemoryError, MemorySpill)
                if device.type == 'cuda' else RuntimeError) as e:
            # Not a failure of the benchmark. "It does not fit" is one of the answers a student
            # needs, so it is recorded as a result rather than raised.
            print(f'{preset:>10}: DOES NOT FIT -- {type(e).__name__}')
            rows.append({'preset': preset, 'device': dev_name, 'error': 'out of memory',
                         'note': a.note})
            empty_cache(device)
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
        if r.get('tok_s_bare'):
            print(f"{'':>10}  the bare step alone was {r['tok_s_bare']:,} tok/s -- "
                  f"{r['bare_over_real']:.2f}x. That difference is what building batches, "
                  f"clipping and\n{'':>10}  reading the loss back cost this machine, and it is "
                  f"the part a step-only benchmark leaves out.")
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
