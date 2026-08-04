# Making the factory earn its keep: a throughput investigation

*A2-NLP · August 2026 · a running record of what we measured, what we changed, and what it cost*

This is the engineering log behind the Part 2 claim that the factory makes the group's study
cheaper. It exists because "the GPU usage was miserable" is an observation, not a diagnosis, and
the difference between the two is measurement. Several of the obvious explanations turned out to
be wrong, and those are recorded here too — a log that only lists the successful guesses is not
a log, it is marketing.

Hardware throughout: one workstation, 2 × RTX PRO 6000 Blackwell Max-Q (96 GB each), PyTorch
2.11 + CUDA 12.8 on Windows, bf16 autocast.

---

## 1. The starting point

The group's proof-of-concept notebook ran its 2×2 pretraining grid in-notebook, one cell after
another, at batch 64 × seq-len 128. Measured on the prepared Yoruba corpus:

| cell | data | steps | val MLM loss | wall-clock |
|---|---|---|---|---|
| `yor_2M_6k_s0` | 2M | 6,000 | 5.711 | 2.6 min |
| `yor_2M_24k_s0` | 2M | 24,000 | 4.775 | 10.1 min |
| `yor_32M_6k_s0` | 32M | 6,000 | 5.635 | 2.5 min |
| `yor_32M_24k_s0` | 32M | 24,000 | 2.958 | 10.0 min |
| | | | **total** | **25.2 min** |

Throughput held steady around 318–333k tok/s across all four.

## 2. What we measured, including the wrong guesses

Batch-size and sync sweep, `poc` preset, seq-len 128, on one card:

| batch | per-step `.item()` | tok/s | ms/step | peak GB |
|---|---|---|---|---|
| 64 | yes (what we shipped) | 356k | 23.0 | 3.3 |
| 64 | no | 362k | 22.6 | 3.3 |
| 128 | no | 482k | 34.0 | 6.1 |
| 256 | no | 487k | 67.3 | 11.7 |
| 512 | no | 464k | 141.4 | 22.8 |
| 1024 | no | 457k | 286.6 | 45.1 |
| 2048 | no | 387k | 677.3 | 89.7 |

**Wrong guess #1 — the per-step host sync.** `train_loop.py` carries a hard-won lesson from
Part 1: every `.item()` is a GPU-to-CPU sync, and at 32×32 that habit alone cost about 7%. The
MLM loop calls `.item()` once per step to drive the EMA of the training loss. Removing it is
worth **2%** (356k → 362k). The Part 1 lesson does not transfer, because an MLM step at this
size is ~23 ms against a CNN step's fraction of a millisecond — the sync is genuinely in the
noise here. Kept the `.item()`; a readable live loss is worth 2%.

**Wrong guess #2 — memory headroom.** Peak allocation at the shipped configuration is **3.3 GB
of 96 GB**. Nothing about this workload is memory-bound, and the instinct to reach for gradient
accumulation or a memory-saving trick would have been wasted effort.

**Wrong guess #3 — bigger is better.** Throughput *peaks* at batch 128–256 and then falls: batch
2048 is slower than batch 128 in tokens per second while using 89.7 GB. There is no reward for
filling the card.

**What was actually true.** The shipped batch of 64 leaves **1.33×** on the table, and the second
card was idle for the entire run.

## 3. Why utilization looks bad, and why that is mostly not fixable

Converting throughput to arithmetic rate (6·N FLOPs per token for forward+backward, counting
non-embedding parameters plus the output head):

| preset | params | batch | tok/s | TFLOP/s | peak GB |
|---|---|---|---|---|---|
| poc | 33.8M | 64 | 364k | 74 | 3.3 |
| poc | 33.8M | 128 | 484k | 98 | 6.1 |
| poc | 33.8M | 256 | 489k | 99 | 11.7 |
| afriberta | 86M | 64 | 171k | 101 | 6.2 |
| afriberta | 86M | 128 | 209k | **123** | 10.6 |
| afriberta | 86M | 256 | 201k | 118 | 19.8 |

A 33.8M-parameter model at sequence length 128 is a small workload. It is launch- and
overhead-bound rather than compute-bound, and no amount of batching fixes that — the kernels
themselves are short. The useful observation for the group is the last two rows: utilization
rises by about a quarter at AfriBERTa scale, so **the real study will use the hardware better
than the POC does**. The poor number is a property of the proof-of-concept's model size, not of
the factory.

Two things were ruled out rather than tried:

- `torch.compile` — the Windows PyTorch wheels ship without Triton, and `triton-windows` is
  unreliable on Blackwell. Not available here.
- Attention backend — already `sdpa`; nothing to win.

## 4. The change, and the trap in it

**Batch 64 → 128, and the compute axis re-expressed in tokens of updates.**

The second half of that sentence is the part that matters, and it is easy to get wrong. The
group's grid defines its compute axis in *optimizer steps*. Steps are not comparable across batch
sizes: 6,000 steps at batch 64 and 6,000 steps at batch 128 differ by a factor of two in work
performed. Raising the batch while holding steps fixed would not have been a faster run of the
same experiment — it would have been a **different, larger experiment that scores better**, and
reporting it as a speed-up would have been wrong.

So `mlm_fleet.QUEUES` now specifies **tokens of updates** (`steps × batch × seq_len`), which is
batch-invariant, and derives steps from it:

```
steps = update_tokens / (batch * seq_len)
```

The POC's 6,000 and 24,000 steps at batch 64 are 49,152,000 and 196,608,000 tokens of updates.
At batch 128 those same budgets buy 3,000 and 12,000 steps. Verified identical:

```
budget  49,152,000 -> b64:  49,152,000 | b128:  49,152,000
budget 196,608,000 -> b64: 196,608,000 | b128: 196,608,000
```

Everything else — learning rate 5e-4, OneCycle schedule, weight decay, masking probability, seed
— was held constant, so the A/B below has exactly one independent variable: the batch shape, and
the step count that follows from it.

**Deliberately not changed: the learning rate.** The linear-scaling rule would suggest 1e-3 at
double the batch. We left it at 5e-4 to keep the comparison clean; 128 is still a small batch,
and 1e-3 on a 33.8M RoBERTa risks instability that would have confounded the result. If the
batch-128 run comes out measurably worse, LR scaling is the first thing to try.

## 5. Results of the A/B

Same four cells, same token budgets, same recipe. Left column is the POC's batch 64 run on one
card; right column is batch 128 through `mlm_fleet` on both cards.

| cell | b64 val | b128 val | Δ | b64 min | b128 min | b64 tok/s | b128 tok/s |
|---|---|---|---|---|---|---|---|
| 2M × 49M upd | 5.711 | 5.701 | −0.010 | 2.6 | 1.9 | 318k | 442k |
| 2M × 197M upd | 4.775 | **3.494** | **−1.281** | 10.1 | 7.7 | 323k | 423k |
| 32M × 49M upd | 5.635 | 5.626 | −0.009 | 2.5 | 1.9 | 328k | 428k |
| 32M × 197M upd | 2.958 | 2.886 | −0.072 | 10.0 | 7.6 | 328k | 429k |

**Throughput: 1.31×**, against 1.33× predicted from the sweep. The prediction held.

**GPU-time: 25.2 → 19.1 GPU-minutes (1.32× less).** This is the honest efficiency number — it is
what the batch change alone bought, independent of how many cards are running.

**Wall-clock: 25.2 min → 12.2 min (2.07×).** The extra factor is the second card, which the
notebook never used. Observed utilization during the fleet run was **91% and 93%** on the two
cards at 300 W each, against a single card and an idle one before.

**A scheduling bug found by watching the run.** Card 1 finished its single long cell at 7.9 min
and then sat at 1–3% for four minutes while card 0 worked through the tail. The queue was in
natural reading order, so both short cells landed on the same card as a long one. a1-cv's fleet
had already learned this and orders its overnight queue longest-first; the MLM queues did not.
Fixed — all three `QUEUES` are now longest-budget-first, which balances the poc grid at ~246M
tokens of updates per card and should give **~9.9 min, or 2.55×**. That figure is projected from
the measured per-cell times, not yet observed.

### The one result that is not a speed-up, and should not be reported as one

The `2M × 197M` cell improved by **1.281** in validation loss. That is far too large to be a
batch-size effect and it should not be written up as "batch 128 trains better."

The likely explanation is visible in the training curves. Every cell in this grid has a sharp
phase transition where validation loss falls off a cliff — the model stops predicting unigram
frequencies and starts using context. In the batch-64 run, the `2M × 197M` cell hit that
transition very late (loss 5.564 at step 16,800, then 5.107, 4.827, 4.775 in the last three
logging intervals) and was still falling when the budget ran out. It was cut off mid-transition.
The batch-128 run got through it. The cell where this matters most is exactly the one with the
most repetition — 98 passes over a 2M-token corpus.

So the difference is most plausibly **when the phase transition fired**, which is stochastic,
not a property of the batch size. This is a single seed per configuration. **Do not quote the
1.281 without seeds** — `mlm_fleet.py --corpus yor --queue seeds` exists for exactly this, and
until it is run, the defensible claim is: *throughput improved 1.31×, and validation loss was
unchanged in three of four cells.*

## 5a. Changes made to the factory

| change | file | why |
|---|---|---|
| compute axis in **tokens of updates**, not steps | `mlm_fleet.py` | steps are not comparable across batch sizes; this makes the batch a free throughput knob |
| default batch **64 → 128** | `mlm_train`, `mlm_run`, `mlm_api` | 1.33× measured, saturates past 256 |
| queues ordered **longest budget first** | `mlm_fleet.py` | a short job started last idles a card; a long one sets the finish time |
| `pretrain(..., reuse=True)` | `mlm_api.py` | re-running the notebook to iterate on the *fine-tuning* gates should not spend 20 minutes reproducing checkpoints that are already on disk |

`reuse` is the one that changes day-to-day work most: the notebook can now be re-executed end to
end in the time the downstream gates take, because the grid is read from `runs/` instead of
retrained.

## 6. Where the remaining wall-clock goes

The fine-tuning gates (SIB-200, MasakhaNER) are where GPU utilization looks worst: batch 16 over
701 labelled examples, every sequence padded to 128. They are also only about two minutes of the
run, so optimizing them would buy time nobody notices. Deliberately left alone — the factory
does not take over fine-tuning, and this is why.

## 6a. The missing control, recovered

Re-running the notebook with the Colab path fixed produced the MasakhaNER random-init row that
had been failing. It changes the reading of the NER gate:

| model | SIB-200 macro-F1 | MasakhaNER entity-F1 |
|---|---|---|
| random-init (no pretraining) | 0.100 | **0.346** |
| from-scratch 32M×24k | 0.527 | 0.698 |
| XLM-R | 0.127 | 0.843 |
| mmBERT | 0.537 | 0.848 |

Pretraining buys **+0.352 entity-F1** over the untrained control — on the task specifically
chosen because it is harder to solve from surface lexical cues. So the from-scratch model's 0.698
is real learning, not tokenizer luck, even though it does not reach the multilingual baselines.
Both readings survive: from-scratch pretraining works, *and* it has further to go on NER than
SIB-200 alone would suggest.

The re-run also reproduced every other number to within fine-tuning seed noise (mmBERT on Yoruba
moved 0.535 → 0.537; everything else identical), which is a useful incidental check that the
`reuse` path returns the same checkpoints rather than quietly retraining something different.
All four cells reported `reusing the completed run`, so the 25 minutes of pretraining were
skipped entirely and the re-run spent its time only on the gates being iterated.

## 6b. The faster run also gave a cleaner scientific reading

An unplanned benefit. With all eight runs on disk, the results notebook had to stop plotting
*steps* on its compute axis — 6,000 steps at batch 64 and 6,000 at batch 128 are different
budgets, so a mixed grid rendered as four compute levels instead of two. Switching that axis to
tokens of updates (and reading the verdict off one batch size rather than across both) changed
what the grid says:

Read at batch 128, val MLM loss:

| | 49.2M upd | 196.6M upd | compute effect |
|---|---|---|---|
| **2M tokens** | 5.701 | 3.494 | −2.207 |
| **32M tokens** | 5.626 | 2.886 | −2.740 |
| **data effect** | −0.075 | −0.608 | |

**Compute-bound**, and not marginally: compute moves validation loss by 2.2–2.7 while data moves
it by 0.08–0.61. Data does begin to matter, but only once the compute budget is large enough to
get through the phase transition — at the small budget it is worth 0.075, which is nothing.

This is a firmer version of the same conclusion the batch-64 grid pointed at, and it is firmer
precisely because the `2M × 196.6M` cell is no longer truncated mid-transition. The earlier
reading ("both axes bind and they interact") over-weighted the data axis because one cell had
been cut off. The corrected reading is: **compute binds; data is a second-order effect that only
appears once compute is sufficient.**

For the group's study design this is the actionable finding. Spending the budget on more unique
Yoruba text buys little; spending it on more updates buys a lot — which is fortunate, because
§8 shows unique Yoruba text is the resource they are about to run out of.

## 6c. How much of that is noise? (seeds, finally)

Every claim above was one run per cell until now. The headline cell was repeated at three seeds:

| seed | val loss | wall-clock |
|---|---|---|
| 0 | 2.8793 | 7.7 min |
| 1 | 2.9776 | 7.8 min |
| 2 | 2.9874 | 7.6 min |
| | **mean 2.948, sd 0.049, range 0.108** | |

Holding each grid effect against that spread:

| effect | size | × sd | verdict |
|---|---|---|---|
| compute, at 32M tokens | 2.747 | 56× | real |
| compute, at 2M tokens | 2.208 | 45× | real |
| data, at 197M updates | 0.614 | 13× | real |
| **data, at 49M updates** | **0.075** | **1.5×** | **inside the noise** |

This sharpens §6b rather than overturning it. The compute effects are so far outside the seed
spread that no plausible amount of noise touches them. The data effect at *high* compute is real
at 13× the spread. But the data effect at *low* compute — 0.075, which §6b called "nothing" —
is now measurably nothing: it is smaller than the gap between two runs of the same configuration.

So the precise claim is: **more unique text has no measurable effect until there is enough
compute to use it.** That is a stronger statement than "data matters less", and it is the one to
put in the report.

Two caveats worth keeping honest. The spread was measured on one cell and is assumed to
characterise the others; that is reasonable but not verified. And a re-run of seed 0 landed at
2.8793 against the 2.886 recorded earlier — a difference of 0.007 at *identical* seed, from
cuDNN autotune choosing different kernels. Single-seed figures quoted elsewhere in these reports
are seed 0 and carry that much wobble on their own.

## 6d. The recipe does not survive the model size the real study uses

The most consequential result here, and it only appeared because we ran the study ladder at both
scales instead of projecting the larger one from the smaller.

| rung | POC 33.8M | AfriBERTa 86M | bigger is |
|---|---|---|---|
| 4M | 5.670 | 6.789 | +1.118 worse |
| 16M | 3.008 | 5.509 | +2.501 worse |
| 64M | **2.612** | 6.733 | +4.121 worse |

A larger model, given the same budget of tokens-of-updates, came out **worse at every rung** —
and the 64M cell finished worse than the 16M cell despite four times the steps. That is not slow
learning. The 64M run's validation loss was flat at 6.73 for its entire 69 minutes: it never left
the plateau where a model predicts unigram frequencies and stops.

**Why this matters more than any speed number in this document.** The group's real study is
planned at AfriBERTa scale. Run with the recipe as it stood, every cell would have produced a
model that learned nothing, and the natural conclusion — "from-scratch pretraining does not work
for Yoruba" — would have been an artifact of the optimizer settings.

### Two conditions, which is why it looked random

Diagnosing this took two rounds because *neither cause alone explains the pattern*.

**The learning rate.** A sweep at 86M (16M tokens, 4,000 steps):

| lr | val loss | trajectory |
|---|---|---|
| 1e-4 | 5.677 | 6.09 → 5.68, descending |
| **3e-4** | **5.610** | 6.06 → 5.61, descending |
| 5e-4 *(the old default)* | 6.759 | 6.79 → 6.76, **flat from step one** |
| 1e-3 | 6.758 | 6.79 → 6.76, **flat from step one** |

That explains the erratic ladder: at 5e-4 the 4M and 64M cells collapsed and the 16M cell
happened to survive.

**The warmup length.** Setting the per-preset rate to 3e-4 looked like the fix — and the first
verification run at 3e-4 *collapsed anyway*. It ran 1,500 steps, and `pct_start=0.06` gave it 90
warmup steps, where the 4,000-step run that trained had 240. A percentage warmup is fine for a
long run and far too short for a brief one, which is exactly the kind people use to try things.

With both conditions met the run is reliable — three seeds at 3e-4 over 4,000 steps landed at
**5.614, 5.609, 5.610**. Re-running the config that had collapsed, with only the warmup floor
added, took it from **6.763 to 5.783**.

### What changed in the factory

- `PRESET_LR` — peak learning rate per preset, measured. 5e-4 for poc, 3e-4 for afriberta.
- `MIN_WARMUP_STEPS = 250` — a floor in absolute steps, so short runs stop failing for a reason
  that has nothing to do with the experiment.
- **A stall detector.** At the halfway mark, if validation loss has moved less than 0.15 from the
  first logged point, the run prints a loud warning and records `stalled: true`. It cannot
  prevent a collapse, but it turns a 69-minute silent failure into a two-minute one. It fired
  correctly on the collapsed verification run.

The original AfriBERTa rows are kept in `runs/` under a `collapsed-lr5e4_` prefix as the evidence
for this finding. They are not results.

## 6e. The ladder, re-run — and the answer is not the one we expected

With the corrected recipe the AfriBERTa runs no longer collapse (`stalled: false` on all three,
the 64M cell moving from 6.733 to 5.315). **The smaller model still wins at every rung.**

| rung | POC 33.8M | AfriBERTa old (collapsed) | AfriBERTa fixed |
|---|---|---|---|
| 4M | **5.670** | 6.789 | 5.706 |
| 16M | **3.008** | 5.509 | 5.494 |
| 64M | **2.612** | 6.733 | 5.315 |

The training curves on the 64M rung say why. Both models start at exactly the same place:

```
POC 33.8M    5.65  5.55  5.47  5.42  3.46  2.95  2.75  2.67  2.62  2.61  2.61
AfriBERTa    5.65  5.56  5.49  5.46  5.43  5.38  5.35  5.33  5.32  5.32  5.32
```

The smaller model **breaks through** at around 40% of training — the cliff from 5.42 to 3.46 is
the point where it stops predicting unigram frequencies and starts using context. The larger
model never does. It grinds down the plateau and converges there: the last three points move by
0.003, so this is not a run that would arrive with a little more patience. It has finished, at a
far worse place.

### What this means for the study design

At this compute budget, **scaling from 33.8M to 86M is not merely unhelpful, it is harmful.** The
larger model costs 2.5× the compute per token *and* fails to reach the regime where a language
model becomes useful. Both halves of that are bad; together they are decisive. The budget is
better spent on more updates to the smaller model.

That inverts the plan in the proposal, which sizes the real study at AfriBERTa scale on the
assumption that bigger is better once the pipeline works.

### What we have not established

Only that the larger model loses *at this budget and this recipe*. We tried two learning rates at
that scale (5e-4 collapses, 3e-4 converges to the plateau) and did not search schedules, warmup
lengths beyond the floor, or batch sizes. It is entirely possible that a longer run, or a
different schedule, gets the 86M model through the transition — large models are known to need
disproportionately more compute before they overtake small ones.

### Run: does more compute rescue it? No.

The 16M rung at AfriBERTa scale with **three times** the step budget — 35,156 steps, 54 minutes:

| configuration | val loss |
|---|---|
| AfriBERTa 86M, 1× budget | 5.494 |
| **AfriBERTa 86M, 3× budget** | **5.385** |
| POC 33.8M, 1× budget | **3.008** |

```
5.69  5.61  5.53  5.51  5.47  5.44  5.42  5.40  5.39  5.39  5.39
```

Tripling the compute bought **0.109**, and the run converged on the plateau again — the last
three intervals are identical to two decimal places. The larger model does not break through
with more compute at this scale; it settles.

For contrast, the smaller model reached 3.008 on the same rung with **a third** of that budget.

So the recommendation stands without the qualifier: **at this budget and batch shape, do not
scale the model up.** More compute does not rescue it, and the compute spent trying is compute
not spent on the smaller model that works.

### What is still unsearched, and the most promising lead

We have now tested four learning rates (1e-4, 3e-4, 5e-4, 1e-3) and two compute budgets (1× and
3×) at 86M. We have *not* varied the batch shape, and that is where suspicion should fall next.

Our batch is 128 sequences × 128 tokens = **16,384 tokens per step**. RoBERTa-base — which the
afriberta preset closely resembles — was trained at 8,000 sequences, on the order of two million
tokens per step. We are running a model of that size at roughly a hundredth of the batch it was
designed for, and large models trained with small batches are known to struggle to escape
exactly this kind of plateau: the gradient is too noisy for the model to organise.

That is a testable hypothesis, not a conclusion. The experiment is the 16M rung at batch 512 or
1024 with the step count reduced to hold tokens-of-updates fixed. `diagnose.py` shows batch 512
costs about 22.8 GB at poc scale, so it fits comfortably. Not run.

All AfriBERTa figures here are one seed.

## 7. Bugs this investigation surfaced

- **Colab path in the NER gate.** The v3→v4 notebook transform patched the hardcoded
  `/content/random_init` in the SIB-200 fine-tuning cell but not the identical reference in the
  MasakhaNER cell, so the NER random-init control died on
  `OSError: Repo id must be in the form ...`. The transform now rewrites every occurrence in
  live code (leaving the commented "before" blocks alone) and fails loudly if it finds fewer
  than two.
- **`--force` silently did nothing.** A new `sample_docs` parameter was inserted ahead of `force`
  in `prepare_corpus`, and the CLI passed its arguments positionally — so `--force` was being
  bound to `sample_docs` (`False` → keep zero documents) and `force` kept its default. The
  language-ID gate got an empty document sample and re-preparation quietly no-opped. Neither
  raised. Fixed by switching the CLI to keyword arguments and making an empty sample an error.
- **Tag collision at sub-million counts.** `tokens // 1_000_000` rendered every cell below a
  million tokens as `0M`, so two different runs shared a tag and the second overwrote the first's
  checkpoint. The same rounding put "0M tokens" on two different chart legend entries.

## 8. What this means for the group's budget

At batch 128 the estimator now measures **376k tok/s** median at the `poc` preset (up from
338–344k at batch 64). Projected to AfriBERTa scale (the hidden²×depth rule, 3.4× cost per
token), 12 passes per rung:

| rung | tokens seen | GPU-h | 2-card h |
|---|---|---|---|
| 4M | 48M | 0.12 | 0.06 |
| 16M | 192M | 0.48 | 0.24 |
| 64M | 768M | 1.92 | 0.96 |
| **per language** | | **2.52** | **1.26** |

Three languages × 1 seed: **7.6 GPU-h**, roughly **3.8 h wall-clock on two cards**, against a
budget of about 20 GPU-h — down from 8.2 GPU-h before the change. It fits with room for the seed
repeats the study needs, because the downstream differences between grid cells are currently
inside the seed spread.

One hard constraint found along the way: FineWeb-2's Yoruba shard is **259,864,169 characters in
total** — it exhausts in seven seconds of streaming — which is 69.1M tokens at 3.73 chars/token.
The 64M-token rung therefore consumes **93% of all the Yoruba text that exists in that source**,
and the 128M rung contemplated in the POC is not reachable from it at all.
