# When a number is not a result

*A2-NLP · August 2026 · the first downstream runs, and what it took before any of them could be
trusted*

Ninety-two pretraining runs exist in `runs/`. Until this week, zero downstream runs did — and the
group's claim is a downstream score. Every finding in
[report 04](04-the-language-gradient.md) and [report 05](05-when-data-stops-mattering.md) lives in
loss-space, on a project that has twice caught validation loss failing to predict fine-tuned
quality.

This note is the record of closing that gap, and of how much had to be corrected on the way. The
headline is not the new numbers. It is that **four of the numbers the study quoted downstream were
not measurements of what they appeared to measure** — and that each one was found by a check
firing, not by anyone noticing something looked wrong.

Two of them were bugs in the ordinary sense: a baseline that never trained, and a tokenizer
comparison run on text in the wrong Unicode normalization form, which reversed its own conclusion.
The other two are more uncomfortable, because nothing was broken. A fixed step budget and an
untrained control both looked like neutral scaffolding, and both turned out to be deciding the
answer.

---

## 1. The harness moved into a module

The fine-tuning half lived inline in `POC_v4_factory.ipynb` — `macro_f1`, `pooled_ci`,
`finetune_once`, the SIB-200 and MasakhaNER loaders. The open questions are all *comparisons*, and
each wants its own small notebook. Copied into three notebooks, that becomes three harnesses
inside a week, and a seed count changed in one silently makes its numbers incomparable with the
others.

It is now `ft_api.py`, alongside `mlm_api.py` and mirroring its shape. One behavioral change was
made deliberately, and it matters more than it looks:

**The epoch loop became a fixed step budget.** The notebook trained `for _ in range(FT_EPOCHS)`.
On 701 SIB-200 examples at batch 16 that is 352 updates; on 6,876 MasakhaNER sentences it is
3,440. So the existing SIB-200-versus-NER comparison varied the update budget by 10× at the same
time as the task and the label count — and the planned experiment that subsamples MasakhaNER to
701 sentences would, under that harness, have cut its budget by 10× as well, then reported the
budget cut as a label-count effect.

The defaults — `FT_STEPS = 352` and `NER_STEPS = 2150` — equal what the old loop spent on the full
splits, so existing numbers reproduce.

**That decision is correct and it is also how the biggest error in this report survived.** Fixing
the budget was right; inheriting its *value* from an 8-epoch loop written for a different purpose
was never examined, and §5 shows that 352 decided the study's central downstream conclusion. The
harness change made the constant reproducible without making it interrogated. Reproducibility and
correctness are not the same property.

Every cell now writes one record per `(model, task, lang, n_train, lr, steps, max_length,
normalize, seed-set)` to `runs/ft_*.json`. The last three are in the key because each of them
moves the number, and a key missing one does not collide so much as silently return the wrong
run's result.

---

## 2. XLM-R never learned SIB-200 — at the budget it was given

The study quotes XLM-R at **0.127** macro-F1 on Yoruba topic classification against mmBERT's
0.537. [Report 05](05-when-data-stops-mattering.md) already flagged it as *"almost certainly a
fine-tuning failure"* on the grounds that the same model scores 0.843 on Yoruba entity
recognition, and a model with no usable Yoruba cannot do that. Re-run through the extracted
harness, it is worse than a failure.

At `lr 2e-5, 352 steps` — the configuration the study quoted — five seeds give
**0.057, 0.057, 0.105, 0.241, 0.089**, mean 0.110.

Now the reference points. SIB-200's Yoruba test split is 204 items across 7 classes, distributed
19 / 17 / 22 / 30 / 51 / 25 / 40:

| behavior | macro-F1 |
|---|---|
| collapse to a single class | 0.022 – **0.057** |
| collapse to two classes | 0.038 – 0.099 |
| uniform random guessing | **0.133 – 0.135** |
| perfectly balanced chance | 0.143 |

**0.057 is exactly the score for predicting the majority class on every item** — precision
51/204, recall 1.0, F1 0.400, averaged across seven classes. Two of the five seeds hit it to three
decimal places.

Four of five seeds score below uniform random guessing, and the cell mean sits below the untrained
control's 0.107. A fully pretrained multilingual encoder finishing below an untrained one is as
clear as this gets: **0.127 was never a measurement of XLM-R's Yoruba coverage.**

This is not a property of the harness. mmBERT, through the identical code path, lands at
0.502–0.558 across five seeds and reproduces the notebook's 0.537 to within 0.016.

### The confidence interval does not catch this

The project's stated practice is to lean on pooled bootstrap CIs. On this failure mode they are
worse than useless:

| cell | 95% CI | width |
|---|---|---|
| XLM-R, SIB-200 — collapsed | [0.094, 0.125] | **0.031** |
| mmBERT, SIB-200 — learned | [0.455, 0.573] | **0.118** |

The collapsed run's interval is nearly **4× narrower** than the working one's. A model that
answers "class 4" for every item is consistently wrong, so resampling test items finds almost no
variation, and the interval reads as precision. Nothing inside [0.094, 0.125] signals that the
number is an artifact.

Seed spread does not catch it either. Only the comparison against chance does, which is why
`ft_api` now derives `chance = 1/k`, records `chance` and `degenerate` beside the score, and
prints the per-seed values, so a collapsed run is read as an absent number rather than a small
one.

### What this section got wrong

The first version of this report diagnosed the collapse as *"the textbook degenerate fine-tune,
where a freshly initialised classification head never receives usable gradient"*, and called the
failure model-specific. **That explanation is wrong**, and §5 replaces it. The observation — that
0.127 is not a result — survives intact; the mechanism proposed for it did not.

---

## 3. MasakhaNER ships decomposed, and that reversed the tokenizer result

[Report 04](04-the-language-gradient.md)'s central measurement is that a multilingual vocabulary
is not inherently worse — it costs English 1.04×, Indonesian 1.00×, Mandarin 0.95× — but costs
Yoruba **1.65×**. That was measured on FineWeb-2, the pretraining corpus. The obvious next step is
to check it on the datasets the study actually evaluates on.

Done naively, it says the opposite.

**MasakhaNER 2.0 is distributed in decomposed form.** 17% of its characters are combining marks —
Yoruba tone is stored as separate codepoints rather than precomposed characters. The five
committed 16k BPEs are byte-level with **no normalizer at all**, and were trained on FineWeb-2,
which is precomposed. So on the raw files `yor-bpe16k` sees `náà` as five codepoints and cuts it
into four tokens instead of one.

XLM-R does not care. Its SentencePiece carries a `Precompiled` charsmap that folds the difference
away. mmBERT's `Replace` normalizer does not.

Tokens per word, and fraction of items exceeding a 128-token window, on MasakhaNER train+test:

| tokenizer | normalizer | raw t/w | NFC t/w | raw >128 | NFC >128 |
|---|---|---|---|---|---|
| `yor-bpe16k` | none | 3.15 | **1.67** | 18.9% | **0.2%** |
| `xlm-roberta-base` | Precompiled | 2.83 | 2.91 | 11.8% | 13.3% |
| `mmBERT-base` | Replace | 3.25 | 2.68 | 20.8% | 6.3% |

Read the raw column and the study's own tokenizer is *worse* than the multilingual one, on the one
dataset where the thesis says it should look best. Read the NFC column and it is better by 1.74×.
Same data, same tokenizers, opposite conclusion — and the difference is a preprocessing decision
nobody had made explicitly.

SIB-200 is 1.2% decomposed, so it was never badly affected, which is precisely why the problem
went unnoticed: the dataset the group looked at most is the one that hides it.

**The fix belongs in the data, not the tokenizer.** Adding a normalizer to `yor-bpe16k` would move
its fingerprint off `15abd33de5af` and make every Yoruba pretraining run incomparable. `ft_api`
normalizes to NFC on the way in, records the setting on every result, and puts it in the record
key so an NFC run and a raw run cannot land on the same file.

**The thesis comes out stronger.** With the encoding corrected, XLM-R costs:

| corpus | XLM-R costs |
|---|---|
| SIB-200 Yoruba | 1.60× |
| MasakhaNER Yoruba | 1.74× |
| FineWeb-2 Yoruba (report 04) | 1.65× |

The tokenizer gradient holds on the *evaluation* data, not only on the pretraining corpus. It has
since been extended to seventeen languages in
[report 07](07-the-night-of-diagnostics.md), where it separates by XLM-R coverage.

**This section is unaffected by everything that follows.** It is CPU arithmetic over committed
tokenizers, and no revision to the fine-tuning results touches it.

---

## 4. Truncation is not the mechanism

The tokenizer finding has been framed as *"65% of the context window spent on fragments"*. That is
a claim about **truncation**, and it had never been checked. SIB-200 items are FLORES sentences
and MasakhaNER items are single sentences; at `max_length=128` they may simply fit.

Fraction of items over each window, on NFC text:

| dataset | tokenizer | 128 | 192 | 256 |
|---|---|---|---|---|
| SIB-200 | `yor-bpe16k` | 0.1% | 0.0% | 0.0% |
| SIB-200 | `xlm-roberta-base` | 4.0% | 0.1% | 0.0% |
| MasakhaNER | `yor-bpe16k` | 0.2% | 0.0% | 0.0% |
| MasakhaNER | `xlm-roberta-base` | 13.3% | 0.5% | 0.0% |
| MasakhaNER | `mmBERT-base` | 6.3% | 0.2% | 0.0% |

Almost nothing is truncated for a fitted vocabulary, and at 256 nothing is truncated for anyone.
**The context window is not the mechanism**, and the sentence has to go. The penalty, if it acts,
acts through representation quality — more fragments meaning each token carries less — which is a
different claim needing different evidence.

### Measured, not just inferred

The encoding fix supplied the controlled version of this by accident. mmBERT's MasakhaNER
truncation fell from **20.8% to 6.3%** under NFC — a 14.5-point change, same model, same
2,150-step budget — and its entity F1 moved from 0.848 to **0.851**, against a seed sd of 0.007.
XLM-R, whose truncation moved the other way (11.8% → 13.3%), went 0.843 to **0.841**.

A 14.5-point swing in truncation is worth 0.003. The reason is in how the metric is built: gold
labels are truncated identically to predictions, so truncated words leave the evaluation set
rather than counting as errors. Truncation shrinks what is scored; it does not penalise.

That cuts both ways. It confirms the window is not the mechanism — and it also means re-running
the NER table at `max_length=256` buys accuracy of about 0.003. Worth doing for tidiness, not for
correctness.

The same fact stated as capacity survives, and is what the poster should say instead:

> In a 128-token window, a Yoruba-fitted vocabulary holds **77 Yoruba words**. XLM-R holds **44**.
> mmBERT holds 48.

---

## 5. The step budget was the cause

§2 established that XLM-R's number was not a result. It did not establish why, and the mechanism
it proposed was wrong.

The sweep that settled it: five learning rates extending *downward* from 2e-5, two step budgets,
five seeds each — 50 fine-tuning runs, reported as **the fraction of seeds clearing the
uniform-random bar** rather than as a mean, because a mean over collapsed seeds averages a broken
optimizer with a working one and reports the result as a property of the model.

| lr | 352 steps | 1056 steps |
|---|---|---|
| 5e-6 | 0/5 — 0.099 | **5/5** — 0.295 |
| 1e-5 | 0/5 — 0.089 | **5/5** — **0.408** |
| 2e-5 | 1/5 — 0.110 | 4/5 — 0.394 |
| 3e-5 | 1/5 — 0.139 | 4/5 — 0.358 |
| 5e-5 | 0/5 — 0.057 | 0/5 — 0.057 |
| | **2 of 25 seeds** | **18 of 25 seeds** |

**The budget decides it and the learning rate mostly does not.** At 1056 steps — 24 epochs against
the original 8 — four of five learning rates converge. XLM-R was never missing usable Yoruba and
the classification head was never failing to receive gradient. It was undertrained, at a step
count nobody had chosen for it.

`FT_STEPS = 352` came from `POC_v4_factory.ipynb`'s 8-epoch loop, preserved so the extraction
would be behavior-preserving (§1). It was never selected for this question, and it produced the
study's central downstream conclusion. **This is the sixth or seventh instance of the project's
recurring failure mode — a result produced by a fixed arbitrary constant rather than by the thing
under study — and the most consequential one so far.**

Two details worth keeping. **lr 5e-5 is dead at both budgets**: all ten seeds land on 0.057, the
exact majority-class score, a deterministic collapse rather than a noisy one. And the
fraction-reporting earned its place at `3e-5 / 352`, whose mean of 0.139 sits above the bar while
only one seed of five cleared it — the mean alone would have called that cell converged. The same
pattern appears at 1056: `2e-5` and `3e-5` have seed sds of 0.140 and 0.161 because each carries
one collapsed seed among four working ones.

---

## 6. But the fix does not produce a baseline

XLM-R reaching 0.408 looks like the recovered baseline the downstream table needed. It is not, and
the reason is the second piece of neutral-looking scaffolding.

**The untrained control was also a 352-step number.** Random init scored 0.107 there — at chance,
as intended, and quoted throughout as *the* floor. Measured at the budget where XLM-R actually
trains, the floor moves:

| 1056 steps | best fully-converged cell | mean | 95% CI | over control |
|---|---|---|---|---|
| mmBERT | lr 5e-5 | **0.574** | [0.501, 0.627] | **+0.170**, disjoint |
| mmBERT | lr 2e-5 | 0.556 | [0.490, 0.599] | +0.153, disjoint |
| XLM-R | lr 1e-5 | 0.408 | [0.351, 0.455] | **+0.004, intervals overlap** |
| random init | lr 5e-5 | 0.403 | [0.351, 0.446] | — |

**XLM-R does not separate from a randomly initialised encoder.** Same lower CI bound to three
decimals, means 0.004 apart, two orders of magnitude below the 0.06 floor that 204 test items
resolve. mmBERT clears the same control by 0.170 with disjoint intervals.

So the sweep's "XLM-R converges at 1056 steps" was the *classifier head* learning from 701 labels,
which a random encoder does equally well. The verdict does not depend on which mmBERT cell is
chosen: 2e-5 instead of 5e-5 still gives +0.153.

A control quoted at one budget is not a control at another. **0.107 was a fact about 352 steps,
not about untrained encoders**, and it was carried across a 3× budget change without anyone
noticing it had to move too.

### Three caveats that belong with any use of this

- **The control confounds pretraining with tokenizer fit.** `yor-random-init` is the 33.8M Yoruba
  architecture on `yor-bpe16k` — a quarter of XLM-R's size *and* a far better Yoruba vocabulary.
  So the honest statement is that a randomly initialised model with a fitted tokenizer matches a
  pretrained multilingual model with a poor one. That points the same way as the group's thesis
  but does not demonstrate it. Isolating the two needs a randomly initialised **XLM-R** — same
  architecture, same vocabulary, no pretrained weights — which the factory does not currently
  build.
- **A random encoder reaching 0.403 bounds the benchmark.** SIB-200 at this budget is
  substantially solvable from its 701 training examples alone, which limits how much any encoder
  comparison on it can show.
- **Each model is represented by its best fully-converged cell, chosen on the test set.** SIB-200
  ships a 99-item validation split; selection belongs there before any single number is quoted.
  The full grid is reported above precisely so the selection is visible.

---

## 7. The canonical tables

Every cell is five seeds unless marked, NFC text, fixed step budget, no early stopping, with a 95%
interval bootstrapped over test items and pooled across seeds. **The proposal and the poster
should quote these rather than any number printed in a notebook.**

**SIB-200 Yoruba topic classification** — 701 train / 99 dev / 204 test, 7 classes, batch 16.
Uniform-random guessing is ~0.133 and 1/k is 0.143.

*At 352 steps (8 epochs) — the original budget, retained for continuity:*

| model | lr | macro-F1 | sd | 95% CI | |
|---|---|---|---|---|---|
| mmBERT base | 2e-5 | **0.521** | 0.020 | [0.455, 0.573] | |
| XLM-R base | 2e-5 | 0.110 | 0.068 | [0.094, 0.125] | 1/5 seeds trained — **not a result** |
| random init (control) | 5e-5 | 0.107 | 0.011 | [0.086, 0.128] | 3 seeds; at chance |
| from-scratch 33.8M | 5e-5 | *not measured* | | | needs a checkpoint off the workstation |

*At 1056 steps (24 epochs) — where every model trains:*

> **Superseded 9 August by [report 11](11-selecting-on-the-dev-split.md).** Every cell below was
> selected on the same 204 test items it is reported on, which this report itself flags two
> sections down. Report 11 re-selects all five arms on SIB-200's 99-item validation split; the
> table there is the one to quote. The finding this table was written to support — that XLM-R
> does not clear the untrained control — survives and gets stronger.

| model | lr | macro-F1 | sd | 95% CI | |
|---|---|---|---|---|---|
| mmBERT base | 5e-5 | **0.574** | 0.035 | [0.501, 0.627] | |
| mmBERT base | 2e-5 | 0.556 | 0.016 | [0.490, 0.599] | lower variance; the safer quote |
| XLM-R base | 1e-5 | 0.408 | 0.037 | [0.351, 0.455] | **does not clear the control** |
| random init (control) | 5e-5 | 0.403 | 0.047 | [0.351, 0.446] | |
| from-scratch 33.8M | 5e-5 | *not measured* | | | |

**MasakhaNER 2.0 Yoruba entity recognition** — 6,876 / 983 / 1,964, 2,150 steps, batch 16, lr
3e-5, three seeds:

| model | entity F1 | sd | 95% CI | |
|---|---|---|---|---|
| mmBERT base | **0.851** | 0.007 | [0.837, 0.865] | |
| XLM-R base | 0.841 | 0.008 | [0.827, 0.856] | |
| from-scratch 33.8M | *not re-measured* | | | the one row the encoding fix can have moved |

The two multilingual baselines are **indistinguishable on NER** — 0.841 against 0.851, intervals
overlapping across most of their width — so that task does not separate them. No control has been
run on NER at all, which after §6 is a gap rather than a detail.

---

## 8. What this does to the study's claims

| claim | status |
|---|---|
| XLM-R scores 0.127 on Yoruba topic classification | **Withdraw.** At 352 steps one seed in five trains; the cell mean of 0.110 sits below the untrained control. |
| XLM-R collapsed because of a degenerate fine-tune | **Withdraw.** It was undertrained. At 1056 steps 18 of 25 seeds converge. |
| mmBERT beats XLM-R by 0.41 on topic classification | **Replace.** At a matched 1056-step budget, mmBERT leads by **0.166** with disjoint intervals. The margin is real; the size was an artifact of comparing a trained model against an untrained one. |
| XLM-R is a usable Yoruba baseline on SIB-200 | **Withdraw.** At the only budget where it trains it is 0.004 above a randomly initialised encoder, with overlapping intervals. |
| The untrained control sits at chance (0.107) | **Withdraw as a general fact.** That is a 352-step number. At 1056 steps the control reaches 0.403. |
| The from-scratch model beats XLM-R on topic classification | **Withdraw.** It beat a baseline that never ran, and the from-scratch row has still not been measured under NFC. |
| A multilingual vocabulary costs Yoruba ~1.65× | **Hold, strengthened.** 1.60× and 1.74× on the two evaluation sets; extended to seventeen languages in report 07. |
| 65% of the context window is spent on fragments | **Withdraw.** Nothing is truncated, and a 14.5-point change in truncation is worth 0.003 F1; use the 77-vs-44-words framing. |
| MLM loss does not predict downstream quality | **Hold, one leg weaker.** The surviving support is the 2M-token cell where loss improved 5.72 → 4.57 while F1 fell 0.489 → 0.451. |
| XLM-R and mmBERT both score ~0.84 on Yoruba NER | **Hold.** 0.841 and 0.851 under NFC, overlapping CIs. |
| The from-scratch model loses NER by 0.145 | **Withdraw — it loses by 0.053.** Measured under NFC in [report 08](08-what-the-tokenizer-costs.md) §2: 0.7877 against 0.8410 and 0.8507. The prediction on this row held exactly — both baselines moved by ≤0.003 and the from-scratch row, the only one whose tokenizer fertility changed, moved by 0.092. |

---

## 9. What is not settled

- **Whether XLM-R's pretraining contributes anything to Yoruba topic classification.** §6 says it
  does not clear an untrained encoder, but that control differs in size and vocabulary as well as
  in pretraining. A randomly initialised XLM-R — same architecture, same vocabulary — is the one
  measurement that separates the three, and it is cheap: one architecture change and five seeds.
  **This is the highest-value open experiment in this report.**
- **No control on MasakhaNER.** The NER table compares two baselines against each other and
  against nothing. After §6, an untrained row at 2,150 steps is the obvious first thing to add
  before that table is quoted.
- **The from-scratch rows, on both tasks.** Still unmeasured off the workstation, for want of a
  checkpoint. The NER row is the one the encoding fix can plausibly have moved.
- **Whether the tokenizer penalty causes anything.** Nothing in the project establishes this. The
  swap experiment — same architecture, same corpus, same budget, one model on `yor-bpe16k` and one
  on XLM-R's vocabulary — is the only design that isolates it, and §6 raises rather than lowers
  the stakes on running it, because the downstream comparison that was supposed to carry the
  argument no longer does.

---

## 10. Reproducing

The scores come from `ft_api.evaluate`, which writes `runs/ft_*.json`; `ft_api.results()` and
`ft_api.table()` read them back. The pretraining count in the opening line is
`len(mlm_api.results())` — the `*_result.json` files less the Part 1 causal runs it filters out.
**Recompute it rather than quoting it.** It moves whenever a ladder is merged, and an earlier
version of this report was wrong by 18 because the figure had been carried forward by hand. The two notebooks are
[`exp_xlmr_lr_sweep.ipynb`](../exp_xlmr_lr_sweep.ipynb) and
[`exp_budget_matched_baselines.ipynb`](../exp_budget_matched_baselines.ipynb), both committed with
their outputs. The collapse reference points in §2 are computed directly from the SIB-200 test
labels, and both notebooks recompute the uniform-random bar from the test set rather than
hardcoding it — after §5, a typed-in constant is not something this project should still be doing.

The tokenizer measurements in §3 and §4 are deterministic CPU arithmetic over the committed
`tokenizers/yor-bpe16k` and the two hub tokenizers, and need no GPU.

Fine-tuning ran on a Colab runtime with an **NVIDIA A100-SXM4-80GB**, not the workstation's RTX
PRO 6000 Blackwell. Wall-clock figures therefore do not transfer from reports 03–05, which are all
Blackwell measurements. The card is recorded in the `gpu` field of every record, and it is worth
checking rather than assuming — Colab does not always allocate the same hardware.

Seconds per seed:

| cell | 352 steps | 1056 steps | 2150 steps |
|---|---|---|---|
| SIB-200, mmBERT base | 41 | 122 | |
| SIB-200, XLM-R base | 32 | 95 | |
| SIB-200, random init (33.8M) | 10 | 30 | |
| MasakhaNER, mmBERT base | | | 254 |
| MasakhaNER, XLM-R base | | | 196 |

Roughly 2.3 GPU-hours in total: 27 minutes for the original table, 53 for the sweep, 57 for the
budget-matched baselines. Nothing in §2–§8 depends on the hardware — the tokenizer measurements
are CPU arithmetic and every score is a comparison within a fixed budget — but any *projection*
from these timings needs the A100 figure, not a Blackwell one.
