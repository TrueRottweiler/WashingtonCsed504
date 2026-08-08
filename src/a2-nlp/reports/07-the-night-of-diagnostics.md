# Two results, and a third that was nearly wrong

*A2-NLP · August 2026 · an overnight queue, and what it took to trust it*

Three questions went into the cards overnight: whether the 86M model's instability can be fixed,
whether the tokenizer-fit gradient survives being measured on more than four languages, and
whether from-scratch quality tracks XLM-R's coverage the way the tokenizer penalty does.

All three came back. One of them came back wrong the first time.

---

## 1. Tighter gradient clipping fixes the 86M model

Three configurations at the same cell — 64M tokens, 1.024B tokens of updates, three seeds each.
Only one thing changes from the baseline in each.

| | peak lr | clip | mean | sd | final losses |
|---|---|---|---|---|---|
| baseline | 3e-4 | 1.0 | 3.824 | 1.363 | 2.818, 3.278, 5.376 |
| lower rate | **1.5e-4** | 1.0 | 5.624 | 0.049 | 5.585, 5.608, 5.679 |
| tighter clip | 3e-4 | **0.5** | **2.829** | **0.520** | 2.493, 2.567, 3.428 |

**Clipping at 0.5 is the fix.** It moves the mean by a full nat, cuts the seed spread by more than
half, and its worst seed (3.428) beats the baseline's median. For scale, the 33.8M model reaches
2.217 at this cell — so the 86M model goes from losing by 1.6 to losing by 0.6, on a single
changed hyperparameter.

**A lower peak rate is not the fix, and its failure is instructive.** Every seed at 1.5e-4 landed
at 5.6 with a seed spread of 0.049 — beautifully reproducible and completely useless. It prevents
divergence by preventing learning: at half the rate the model never leaves the unigram plateau
within the budget. A configuration can be perfectly stable and worthless, and a stability metric
alone would have scored it best of the three.

Together with the warmup result from the same sweep — 0 of 3 diverged at warmup 0.06, 2 of 2 at
0.15 — the picture is that the 86M model is not undertrained or badly sized. **It is
badly clipped.** That is a materially different message for anyone about to run an AfriBERTa-scale
study, and it is a one-line change.

### The whole ladder, re-run at clip 0.5

One cell is not a recommendation, so the 86M ladder was re-run: four data rungs, three seeds
each, everything else identical. All at 1.024B tokens of updates.

| data | 86M, clip 1.0 | 86M, clip 0.5 | 33.8M | 86M − 33.8M |
|---|---|---|---|---|
| 4M | 6.720 *(n=1)* | 6.738 ±0.018 | 3.284 *(n=1)* | +3.455 |
| 16M | 4.207 ±1.290 | 4.347 ±1.620 | 2.544 ±0.179 | +1.803 |
| 64M | 3.824 ±1.363 | **2.829 ±0.520** | 2.282 ±0.115 | +0.547 |
| 256M | 2.755 ±0.123 | **2.481 ±0.026** | 2.387 ±0.154 | **+0.094** |
| 1024M | 4.365 ±2.699 | **2.544 ±0.071** | 2.193 *(n=1)* | +0.351 |

**Clipping helps only where there is enough data.** At 4M and 16M it changes nothing — the two
columns are within each other's spread, and both are terrible. From 64M upward it is decisive.

**The reproducibility change is larger than the loss change.** At the top rung the seed spread
falls from **2.699 to 0.071** — a factor of thirty-eight. At 256M it falls to 0.026, which is
*smaller than the 33.8M model's own spread of 0.154*. The 86M model was never unusable; it was
unpredictable, and clipping is what makes it a measurement rather than a coin flip.

**At 256M the two models are indistinguishable.** 2.481 ±0.026 against 2.387 ±0.154 — a gap of
0.094 against a spread that covers it. This is the closest the bigger model has come, and it is
the first time the comparison has been made with both sides properly seeded.

**There is still no crossover.** The 86M model does not beat the 33.8M model at any rung. The gap
narrows monotonically to 256M and then widens slightly at 1024M, which is within the spread and
should not be read as a trend.

Where the failures live has also moved. Runs ending on the plateau, by rung:

```
clip 1.0    4M 1/1   16M 0/3   64M 0/3   256M 0/3   1024M 1/3
clip 0.5    4M 3/3   16M 1/3   64M 0/3   256M 0/3   1024M 0/3
```

Under clip 1.0 the failures were scattered, including one at the largest rung. Under clip 0.5
they are confined to the two smallest rungs, where the model has too little data to break through
within the budget whatever the gradient norm. That is a different and more tractable failure than
the one this report opened with.

### What this does to earlier claims

[Report 05](05-when-data-stops-mattering.md) §4 says the 86M column should be read as anecdote
because its seed spread is 1.369. That stands for the runs it describes. It also says the failures
are seeds that "never break through the plateau", which was already corrected once — the failure
is mid-training divergence — and can now be corrected again with a cause: the gradient norm.

The 86M ladder is worth re-running at clip 0.5 before anything is concluded from it. That has not
been done.

---

## 2. The tokenizer gradient holds across seventeen corpora

The study's contrast rested on two languages either side. This is the same measurement — a
language's own 16k BPE against XLM-R's 250k vocabulary — across everything now prepared.

| covered by XLM-R | | not covered | |
|---|---|---|---|
| Mandarin | 0.95 | Wolof | 1.31 |
| Indonesian | 1.01 | Luganda | 1.50 |
| English | 1.04 | Chichewa | 1.57 |
| French | 1.04 | Shona | 1.59 |
| Afrikaans | 1.06 | Kinyarwanda | 1.59 |
| Swahili | 1.17 | **Yoruba** | **1.76** |
| Somali | 1.21 | Igbo | 1.82 |
| Hausa | 1.27 | | |
| Amharic | 1.33 | | |
| Xhosa | 1.42 | | |
| **mean 1.150** | | **mean 1.593** | |

Restricting both sides to African languages — which removes region and script as explanations —
the gap narrows but survives: **1.244 against 1.593.**

One exception, stated rather than smoothed: **Wolof at 1.31 sits inside the covered range**, below
Xhosa's 1.42. Six of seven uncovered languages are above every covered one; Wolof is not.

Yoruba's 1.76 is second-highest of seventeen. The group's original 1.65× was not a fluke of one
language, and it is near the top of a gradient rather than an isolated number.

---

## 3. The nearly-wrong result

The first version of §2 reported **perfect separation** — every uncovered language above every
covered one, Mann-Whitney U of 70 out of 70, p ≈ 5×10⁻⁵. It was measured on the first 400
documents of each corpus, which is what the tooling had always used.

The boundary was Xhosa at 1.3871 against Wolof at 1.3934: a gap of **0.0063**, or half a percent.
That is small enough to be worth checking, so it was:

| sample | Xhosa | Wolof |
|---|---|---|
| 100 docs | 1.2465 | 1.3931 |
| 200 docs | 1.3267 | 1.4112 |
| 400 docs | 1.3871 | 1.3934 |
| 800 docs | **1.4603** | **1.3584** |

The ordering reverses. "Perfect separation" was an artifact of where the sample happened to stop,
and the significance test built on top of it was measuring the sample size rather than the
languages.

The measurement now uses every committed sample document, and §2 reports a gradient with one
exception instead of a clean split. The weaker claim is the true one.

This is the third time in this project that a result has been produced by a fixed arbitrary
constant rather than by the thing under study — after the 0.049 seed spread applied to
experiments it did not describe, and the `epoch` field that counted logging intervals. It is worth
naming as a pattern rather than three separate mistakes.

---

## 4. From-scratch quality does not track coverage

The same twelve languages, pretrained identically: 33.8M model, 50M tokens, 196.6M tokens of
updates. Measured as context gained — the corpus's unigram entropy minus the final loss — because
raw loss across different vocabularies compares nothing.

Restricted to the ten languages that had the full 50M tokens:

| | n | mean | range |
|---|---|---|---|
| covered by XLM-R | 6 | 4.618 | 4.061 – 5.624 |
| not covered | 4 | 4.808 | 4.101 – 5.172 |

**The ranges overlap almost entirely.** The highest is Xhosa, which XLM-R covers; the lowest is
Hausa, which XLM-R also covers. Whatever makes a language easy or hard to learn from scratch, it
is not whether a multilingual model was trained on it.

Set beside §2, that is the study's argument in two lines: **the tokenizer penalty separates by
coverage and the from-scratch learnability does not.** The disadvantage a multilingual model
carries on an under-represented language is in its vocabulary, not in the language being harder.

This is the strongest version of the group's thesis the project has produced, and it does not
depend on any XLM-R fine-tuning run — which matters, because
[report 06](06-when-a-number-is-not-a-result.md) withdrew those.

---

## What this does not settle

Every language in §4 is one seed. The seed spread for a 33.8M model at a comparable budget is
~0.1–0.15, and several of the gaps in §4 are smaller than that; only the overall overlap is being
claimed, not any ordering within it.

Wolof (4.8M tokens) and Luganda (20.1M) never had the full 50M, so their models made many more
passes over much less text. Both are excluded from §4's comparison and both are included in §2,
where corpus size does not enter.

§1 tests one cell. Whether clip 0.5 helps at every rung, or whether it merely moves the point at
which divergence starts, needs the ladder re-run.

Nothing here is downstream. These are pretraining measurements, and the group's headline is a
fine-tuning score.
