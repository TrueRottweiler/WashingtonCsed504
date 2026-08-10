# Reports

Written investigations behind the A2-NLP factory. Each one is self-contained and every number in
them was measured on the CSED 504 workstation (2 × RTX PRO 6000 Blackwell) unless labeled as a
projection.

| | report | what it answers |
|---|---|---|
| 01 | [What we're building](01-what-were-building.md) | What is the group's research question, what does the factory provide, and where are the data and hardware limits? Start here. |
| 02 | [What the model actually learned](02-what-the-model-learned.md) | We trained a model — what did it *do*? Fill-in-the-blank predictions from the real checkpoint, what a loss of 2.886 means, and what the corpus does and doesn't contain. |
| 03 | [The throughput investigation](03-efficiency.md) | Why GPU utilization was poor, what we measured, what we changed, and which of our guesses were wrong. The engineering log. |
| 04 | [The language gradient](04-the-language-gradient.md) | Five languages, prepared identically. Is Yoruba hard, or under-served? What a multilingual vocabulary actually costs, and where the tooling had Latin-script assumptions in it. |
| 05 | [When more data stops helping](05-when-data-stops-mattering.md) | The English ladder — 256× of data at fixed compute — plus the Yoruba rungs that check whether its threshold transfers. Where the data axis saturates and whether the bigger model ever catches up. |
| 06 | [When a number is not a result](06-when-a-number-is-not-a-result.md) | The downstream runs. A tokenizer comparison run on the wrong Unicode normalization, a step budget inherited from an old notebook that decided the answer, an untrained control quoted at the wrong budget — and what survives of the study's downstream claims. |
| 07 | [Two results, and a third that was nearly wrong](07-the-night-of-diagnostics.md) | Tighter clipping fixes the 86M instability; the tokenizer gradient holds across seventeen corpora; from-scratch quality does not track XLM-R coverage. And how a fixed sample size nearly produced a fourth result that was not there. |
| 08 | [What the tokenizer actually costs](08-what-the-tokenizer-costs.md) | The swap experiment: does a badly-fitting vocabulary cost anything? At matched compute, 0.144 bits/char. At matched *steps*, nothing — and why that reading was wrong. Plus the downstream rows, where a 33.8M from-scratch model comes out ahead of mmBERT on topic classification. |
| **09** | **[The plain-language version](09-the-poster.md)** | **Start here if you are not on this project.** The whole study explained for someone who has taken one ML course: the problem, what we built, what it cost in hardware and electricity, what we found, and the times a setting nobody questioned decided a result, and what the factory could work out about its own training runs from nothing but their stored curves. Fourteen sections, one per poster panel. |
| **10** | **[The next night](10-the-next-night.md)** | Three questions about the factory rather than about Yoruba, all answerable with checkpoints we already have: does validation loss predict downstream usefulness (19 models, never checked), does a tuned learning rate transfer to a new language (our "one function call" claim, untested), and how many of ten new languages need manual intervention. ~19 GPU-hours, one night. |
| **11** | **[Selecting on the dev split](11-selecting-on-the-dev-split.md)** | SIB-200 ships a 99-item validation split and this project had never used it — every learning rate it quoted was picked on the 204 test items it then reported. Nine rates, five arms, selected on dev. The from-scratch margin over mmBERT grows to 0.106 and clears the floor; XLM-R drops *below* an untrained model of its own architecture; and the practice turns out to have inflated exactly one row — the one whose conclusion rested on it. **Supersedes the SIB-200 tables in 06 and 08.** |

## Where the symmetric learning-rate sweeps live

The work that removed the sweep asymmetry from report 08 — Patrick raised it, and he was right —
is in the repository but not in a commit named after it, so this is the pointer.

- **The narrative and the corrected tables:** [report 08 §2b, "Three passes to a fair
  comparison"](08-what-the-tokenizer-costs.md). Three passes, each unfair in a different
  direction, and the finding that **three of the five sweeps in this project peaked at their own
  boundary** — a best-of-sweep number means nothing if the sweep does not contain the best.
- **The scripts:** `sweep_fromscratch.py` and `sweep_ner_baselines.py`.
- **The records:** 24 `runs/ft_*` files covering both tasks at every rate.

It was opened as PR #43 and closed unmerged, because PR #44 was branched from a tree that already
contained all of it and swept it in. Nothing was lost; only the commit title is misleading. Left
here because "grep the log for the sweep commit" would otherwise come up empty.

## The short version

- **Yoruba is scarce.** All of it on FineWeb-2 is 69.1M tokens — less than one English benchmark
  dataset. The group's top rung uses 93% of everything available.
- **The small from-scratch model works.** 33.8M parameters trained on 64M tokens of Yoruba. With
  every arm's rate chosen on the 99-item validation split it is **ahead of** mmBERT on topic
  classification (0.688 vs 0.582) and loses entity recognition by 0.026 (0.837 vs 0.863). Say
  *ahead*, not *beats*: the margin of 0.106 does now clear the project's 0.06 floor, but the
  intervals still overlap by 0.004 ([0.631, 0.734] against [0.518, 0.635]). NER has not had the
  same treatment — those two rates were picked on the items they are scored on.
  [Report 11](11-selecting-on-the-dev-split.md). Earlier versions of this line quoted 0.527/0.537,
  0.698/0.848 and 0.666/0.595, from before the step budget, the Unicode normalization, the
  learning rates and the selection split were fixed in turn.
- **The study is compute-bound, not data-bound.** More training moves validation loss by 2.2–2.7
  against a measured seed spread of 0.049 — 45–56× the noise. More *text* does nothing measurable
  until the training budget is large enough to use it (0.075 at low compute, 1.5× the spread).
  Convenient, since text is what they've run out of — and [report 05](05-when-data-stops-mattering.md)
  shows the text they lack would have bought almost nothing anyway.
- **The factory made the same work 2.07× faster** — 1.33× from batch size, the rest from using the
  second card at all — and the full study fits in 7.6 GPU-hours against a ~20 hour budget.
- **Yoruba is under-served, not hard.** At matched data and compute, a from-scratch Yoruba model
  gains 4.114 nats of context against English's 3.906 — measured after subtracting each corpus's
  unigram entropy, since raw loss across two vocabularies compares nothing. Nothing about the
  language resists modeling.
- **A multilingual vocabulary is not inherently worse.** XLM-R costs English 1.04×, Indonesian
  1.00×, and Mandarin 0.95× — better than a dedicated 16k BPE. It costs Yoruba **1.65×**. The
  penalty appears only where the language is under-represented, which is the group's thesis with
  a control arm attached.
- **Past 64M tokens, more English text buys nothing measurable.** Sixteen times more data moves
  validation loss by −0.080 against a measured seed spread of 0.185. Where saturation *begins* is
  not resolvable at three seeds. In Yoruba the effect arrives earlier still — its 16M → 64M gain
  is +0.053, 0.4× its own spread — so inside the 69.1M that exists, more Yoruba text is already
  buying nothing. The study cannot rest on "there is too little Yoruba text"; it has to rest on
  tokenizer fit.

## The one that changes the study plan

> **Largely superseded — read [report 07](07-the-night-of-diagnostics.md) §1 first.** The runs
> below are one seed per cell, at gradient clipping 1.0. At 86M one seed is a coin flip rather
> than a measurement, and clipping at 0.5 changes the answer: at 256M the two model sizes become
> indistinguishable (2.481 ±0.026 against 2.387 ±0.154) where this section has the larger model
> losing by 2.7. What survives is that the 86M model never wins. What does not survive is the
> explanation below — it was not "finished, not merely slow", it was badly clipped.

**Scaling the model up made it worse, at every rung.** The proposal sizes the real study at
AfriBERTa scale (86M) on the assumption that bigger is better once the pipeline works. Run at
matched budgets against the 33.8M model, it lost every time — 2.612 vs 5.315 on the 64M rung.

The curves say why. Both start at 5.65. The smaller model *breaks through* the unigram plateau
at ~40% of training and converges at 2.61; the larger one grinds along the plateau and converges
at 5.32, its last three checkpoints moving 0.003. It is finished, not merely slow. So the larger
model costs 2.5× the compute per token **and** never reaches the regime where a language model
becomes useful.

Getting to that answer took fixing a separate bug first: at 86M the old recipe *collapsed*
outright (loss flat at 6.73 for 69 minutes), because the peak learning rate was too high for
that size and the percentage-based warmup gave short runs far too few steps. Both are fixed and
verified across three seeds. Had the study run on the old settings, every cell would have
produced a dead model and "from-scratch pretraining doesn't work for Yoruba" would have looked
like the answer. Detail in [03-efficiency.md §6d–6e](03-efficiency.md).

**More compute does not rescue it.** The 16M rung re-run at 3× the step budget reached 5.385
against 5.494 at 1× — 0.109 for triple the compute — and converged on the plateau again. The
smaller model reaches 3.008 on that rung with a third of the budget.

**Batch shape was the promising lead, and it is not the answer either.** Quadrupling the batch
to 65,536 tokens per step moved the 16M rung from 5.494 to 5.342 — 0.15 against a gap of 2.3 —
and converged on the plateau just the same. Larger batches help in the expected direction and
by nowhere near enough.

## The seed spread is not one number

It is a property of the cell — the model size, the corpus and the compute budget — and this
project spent most of its life judging every comparison against a single borrowed value.

| | measured seed spread |
|---|---|
| 33.8M, Yoruba, 196.6M updates | 0.049 *(the figure everything was judged against)* |
| 33.8M, English, 1.024B updates | **0.149** |
| 33.8M, Yoruba, 1.024B updates | **0.103** *(one repeated cell)* |
| 86M, English, 1.024B updates | **1.369** *(and see below — this is the wrong summary)* |

So the old threshold understated the noise by 2–3× on the 33.8M ladders and by **27×** on the 86M
column. [Report 05](05-when-data-stops-mattering.md) §2 restates the ladder against measured
spreads; the results notebook now computes it per cell instead of holding a constant.

And at 86M the spread is the wrong summary altogether. Thirteen runs on the English corpus land
as **nine below 3.8 and four above 5.3, with nothing in between** — two populations, not a
distribution. Some seeds break through the unigram plateau within the budget and some do not, at
a failure rate of roughly 31%. So the practical rule for anyone continuing this work is that
**at 86M, one seed is not a measurement.**

## What the tokenizer costs, measured at last

Everything before [report 08](08-what-the-tokenizer-costs.md) measured that the penalty *exists*.
The swap measures what it *costs*: same Yoruba text, same architecture, only the vocabulary
differs.

| | steps | min/seed | bits/char |
|---|---|---|---|
| our 16k BPE | 12k | 8 | 1.135 ±0.035 |
| XLM-R's 250k | 12k | **41** | 1.056 ±0.142 |
| our 16k BPE | 62.5k | 40 | **0.912 ±0.042** |

**At matched compute our vocabulary wins by 0.144 bits/char, 1.6× the seed spread.** At matched
*steps* the two look identical — but the 250k arm burned 5.1× the compute to get there, because
its output projection is fifteen times wider. Same three runs, opposite conclusions, and only one
of the two axes is the question anyone has.

## Downstream: the small model wins the semantic task

At 1,056 steps, the budget where models actually train:

Every model at its own best learning rate. SIB-200 is selected on the 99-item validation split and
scored on test at five seeds ([report 11](11-selecting-on-the-dev-split.md)); MasakhaNER is still
best-on-test at three seeds.

| SIB-200 (topic) | | MasakhaNER (entities) | |
|---|---|---|---|
| **from-scratch, ours** | **0.688** | mmBERT | 0.863 |
| mmBERT | 0.582 | XLM-R | 0.851 |
| our arch, untrained | 0.429 | **from-scratch, ours** | **0.837** |
| XLM-R arch, untrained | 0.382 | our arch, untrained | 0.414 |
| XLM-R | 0.358 | | |

A 33.8M model pretrained on 64M tokens of Yoruba is **ahead of mmBERT on topic classification by
0.106** — clearing the project's 0.06 floor, though the intervals still overlap by 0.004 — and
loses entity recognition by 0.026, not the 0.145 report 06 recorded, most of which was the wrong
Unicode normalization and an unswept learning rate.

What the controls settle. **XLM-R lands below a randomly initialised model of its own
architecture**: −0.024 across five seeds, or +0.052 if the one collapsed seed is discarded, inside
the 0.06 floor either way. For Yoruba its pretraining contributes nothing measurable. (Report 08
first put this at +0.039 *above* the control; that compared its best of five against a control run
once. Symmetric selection reverses the sign.)

> **Under revision.** The explanation that used to sit here — that the NER floor is 49% of
> achievable against SIB-200's 64%, and that this is why the two tasks disagree — was retracted on
> 9 August. The percentages depend on which denominator is used, and the better-supported account
> is that scores on NER barely move across from-scratch models (a band of 0.044) while SIB-200
> scores vary three times as much (0.143). Note also that the NER control is still a single cell
> at 3e-5 while its baselines are best-of-a-sweep, so any NER floor figure inherits exactly the
> asymmetry report 11 removed from SIB-200.

## The strongest form of the thesis

**The tokenizer penalty separates by XLM-R coverage. From-scratch learnability does not.**

Across seventeen corpora, XLM-R's vocabulary costs covered languages 1.150x on average and
uncovered ones 1.593x — 1.244 against 1.593 if both sides are restricted to African languages.
But pretrain a model from scratch on each and the context it gains is 4.618 against 4.808, with
ranges that overlap almost entirely.

So the disadvantage a multilingual model carries on an under-represented language lives in its
vocabulary, not in the language being harder to learn. That is the group's argument, on seventeen
languages instead of four, and it does not depend on any XLM-R fine-tuning run — which matters,
since [report 06](06-when-a-number-is-not-a-result.md) withdrew those.

One exception, not smoothed: Wolof at 1.31 sits inside the covered range.

## The 86M model was unpredictable, not incapable

Clipping gradients at **0.5** instead of 1.0, across the whole ladder at three seeds a rung:

| data | clip 1.0 | clip 0.5 | 33.8M |
|---|---|---|---|
| 64M | 3.824 ±1.363 | **2.829 ±0.520** | 2.282 ±0.115 |
| 256M | 2.755 ±0.123 | **2.481 ±0.026** | 2.387 ±0.154 |
| 1024M | 4.365 ±2.699 | **2.544 ±0.071** | 2.193 |

The reproducibility change is the bigger one: at the top rung the seed spread falls from 2.699 to
**0.071**, a factor of thirty-eight. At 256M the two model sizes become indistinguishable — 2.481
±0.026 against 2.387 ±0.154. There is still no crossover, and clipping changes nothing at 4M and
16M, where the model has too little data to break through whatever the gradient norm.

A *lower* learning rate does the opposite of helping: every seed lands at 5.6 with a spread of
0.049 — perfectly reproducible and completely useless. See
[report 07](07-the-night-of-diagnostics.md) §1.

## Things not to quote yet

- **XLM-R's 0.127 on topic classification is a non-result, and the configuration question is now
  settled.** [Report 06](06-when-a-number-is-not-a-result.md) originally called it a degenerate
  fine-tune; the sweep since shows it was **undertrained** — 2 of 25 seeds clear uniform random at
  352 steps, 18 of 25 at 1056. But at 1056 steps XLM-R reaches 0.408 against a random-init control
  of **0.403**, intervals overlapping, so there is still no working XLM-R baseline. The headline
  contrast stays withdrawn, and not for want of a better learning rate.
- **The AfriBERTa ladder rows in `runs/` are collapsed runs**, kept as evidence for the finding
  above. They are not results and the ladder needs re-running at the corrected settings.
- **Downstream comparisons need a control measured at the same step budget.** The random-init
  floor is **0.107 at 352 steps and 0.403 at 1056** — it is a measurement, not a constant, and
  carrying it across budgets is what made XLM-R's 0.408 look like a recovered baseline. On
  SIB-200 the usable range between that floor and the best mmBERT cell is 0.171, which at the
  0.06 resolution of 204 test items is under three distinguishable levels — so *which* pretraining
  grid cell fine-tunes best is still not answerable. **MasakhaNER has no control at all yet**, so
  0.841 against 0.851 should not be read as a gap over anything.

## Reproducing

All three reports are generated from artifacts in `runs/` and `data/`, which the factory writes.
See the parent [README](../README.md) for how to run the pipeline, and `explain_model.py`,
`store_bench.py`, and `results_factory_mlm.ipynb` for the tools that produced the figures.
