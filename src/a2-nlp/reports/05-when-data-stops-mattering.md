# When more data stops helping

*A2-NLP · August 2026 · the English ladder, and what it says about Yoruba*

The Yoruba ladder found that data barely moved validation loss while compute moved it forty-fold.
That reading was too generous to itself. Every rung there received roughly the same tokens of
updates, so the models never trained far enough to exhaust what even the smallest rung could
teach them — the data axis was not measured, it was suppressed.

It also could not be measured in Yoruba. All of FineWeb-2 Yoruba is 69.1M tokens, so the 64M rung
already consumes 93% of the language and there is no higher rung to climb to.

This is the same experiment in a language with headroom: five data rungs spanning **256×**, from
4M to 1.024B tokens, with **compute held fixed at 1.024B tokens of updates** at every rung. Fixed
is the point. Scaling compute alongside data would confound the two axes and leave us unable to
say which one moved the loss.

Both model sizes at every rung, because the 86M preset losing at every Yoruba rung reads as
undertraining rather than a verdict on capacity.

---

## 1. The ladder

| unique tokens | 33.8M model | 86M model | gap |
|---|---|---|---|
| 4M | 3.284 | 6.720 | +3.436 |
| 16M | 2.361 | 3.200 | +0.839 |
| 64M | 2.217 | 3.278 | +1.061 |
| 256M | 2.210 | 2.896 | +0.686 |
| 1024M | **2.193** | **2.575** | +0.382 |

Corpus: 1,096,781,996 tokens of FineWeb-Edu, tokenized with the committed `eng-bpe16k`
(fingerprint `7820319faa75`), so these are directly comparable with the `multi_eng` run in
[report 04](04-the-language-gradient.md).

**These are the original single-seed runs.** §2 replaces the 33.8M column's middle rungs with
three-seed means; the 4M and 1024M cells are still one draw each. The 86M column is one draw
throughout, and §4 shows its seed spread is **1.327** — read it as anecdote, not measurement.

---

## 2. The data axis, with the noise measured

**Everything below is three seeds per cell at 16M, 64M and 256M.** Earlier versions of this
section were one seed per cell, judged against a spread of 0.049 borrowed from a different corpus
at a fifth the compute. That borrowing was the mistake; the numbers it produced are kept in the
table so the difference is visible.

Pooled within-cell seed spread, 33.8M model, English, 1.024B tokens of updates: **0.149.**

| step | 3-seed gain | 1-seed gain (earlier) | ÷ spread | reading |
|---|---|---|---|---|
| 4M → 16M | **+0.739** | +0.923 | 4.96× | real |
| 16M → 64M | +0.263 | +0.144 | 1.76× | **cannot resolve** |
| 64M → 256M | −0.105 | +0.007 | 0.70× | noise |
| 256M → 1024M | +0.194 | +0.017 | 1.30× | cannot resolve |

Three bands, not two. A step below the spread is noise; above twice it, real; between them this
many seeds cannot say. Collapsing that middle band into either neighbour is how a single lucky
draw became a reported finding the first time.

**The headline survives.** From 64M to 1024M — a sixteen-fold increase in unique text — the total
change is **+0.089**, which is 0.6× the seed spread. Past 64M, more English text buys nothing this
study can measure.

**Where saturation begins does not.** The 16M → 64M step is +0.263 at 1.76× the spread: larger
than the single-seed version suggested, and still not resolvable. Saturation sets in somewhere at
or before 64M and this experiment cannot localise it more tightly. Two earlier drafts said
"around 64M" and then "between 64M and 256M"; both were reading single draws as measurements, and
neither was entitled to a number.

Note also that 64M → 256M is *negative* and 256M → 1024M positive. Neither is resolvable, and the
non-monotonicity is what a flat curve looks like through noise of this size.

---

## 3. The Yoruba check

All of FineWeb-2 Yoruba is 69.1M tokens, so the ladder's upper rungs cannot exist in it. What can
be compared is the shape over the rungs both languages share. Yoruba's 64M cell is three seeds
(mean 2.4305, sd 0.103); its 4M and 16M cells are single runs.

| step | Yoruba | English | apart by |
|---|---|---|---|
| 4M → 16M | +1.644 | +0.739 | 0.905 |
| 16M → 64M | **+0.053** | **+0.263** | 0.209 |

Pooled spread across both languages: **0.122.**

**The "curves track closely" claim from the single-seed draft is withdrawn.** It rested on the two
16M → 64M gains differing by 0.025. With the 64M cells seeded they differ by **0.209 — 1.7× the
spread.** They are not measurably the same shape.

They differ in the direction that helps the conclusion, which is worth stating plainly rather
than quietly banking: **Yoruba's gain from 16M to 64M is +0.053, which is 0.4× its own spread —
nothing.** English at the same step is still ambiguously gaining. So on the rungs we can measure,
Yoruba stops responding to more text *earlier* than English does, not later.

**What this supports:** the group's case cannot rest on *"there is too little Yoruba text."* By
64M — inside the 69.1M that exists — more Yoruba text is buying nothing measurable. The argument
has to rest on the tokenizer-fit result in [report 04](04-the-language-gradient.md) §4, where
Yoruba pays XLM-R a 1.65× penalty no other language in the set pays.

**What this does not support:** any claim that the two languages saturate at the same point, or
that English's behaviour predicts Yoruba's. The measurement says Yoruba saturates earlier, on a
16M rung that is still a single seed.

**What cannot be fixed:** Yoruba's 4M and 16M cells deserve seeding, and that is 80 minutes of GPU
time. The rung above 64M does not exist and no amount of compute will create it.

---

## 4. The bigger model still has not crossed over

*Read this section against the seed measurement at the end of it — the single-seed numbers below
turn out to sit inside a spread far larger than the differences they describe.*

The 86M model loses at every rung. It has lost at every rung of every ladder run so far.

But it is the only one still improving. Its last two steps gained 0.382 and 0.321 nats while the
smaller model gained 0.007 and 0.017, and the gap between them closes monotonically once past the
smallest rungs: **0.839 → 0.686 → 0.382**.

That is consistent with the undertraining reading — the extra capacity is real but needs more
compute and more data than it has been given — and inconsistent with "86M is simply the wrong
architecture here". Extrapolating the gap naively puts a crossover somewhere past 4B tokens of
data at a proportionally larger compute budget, which is the two-day run and is not yet
justified by anything measured.

**It has not crossed over. Nothing here licenses saying that it will.** What can be said is that
the negative AfriBERTa result in the group's study is explained by budget rather than by
architecture, which is a materially different claim from "the bigger model does not work".

### The seed check, and what it cost this section

The paragraphs above were written from one seed per cell. The 86M model scored 3.278 at 64M
tokens against 3.200 at 16M — worse with four times the data, the only inversion in either
column — and that was flagged as needing a repeat. Repeating it changed more than the inversion.

Three seeds per cell, 86M model, same 1.024B tokens of updates:

| unique tokens | s0 | s1 | s2 | mean | sd |
|---|---|---|---|---|---|
| 16M | 3.200 | 3.760 | 5.661 | 4.207 | **1.290** |
| 64M | 3.278 | 2.818 | 5.376 | 3.824 | **1.363** |

**The inversion is gone** — 64M now comes out 0.383 *better* than 16M, the expected direction. It
was noise.

**The noise is the finding.** The seed spread within a single 86M cell is **1.327 nats**. The
figure this project has been using as its significance threshold is 0.049, measured on the 33.8M
model, and it is **27× too small for this preset.** Everything in this section that compares one
86M rung to another was reading differences of 0.3–0.4 against a spread four times larger.

What survives:

- **The 86M model loses badly to 33.8M.** 4.207 against 2.361 at 16M, 3.824 against 2.217 at 64M.
  That gap is larger than the spread and is not in doubt.
- **It is undertrained rather than broken.** All six seeds were still descending at 62,500 steps —
  none had flattened onto the plateau, so this is not the learning-rate collapse of
  [report 03](03-efficiency.md) §6d. The seeds differ in *when* they break through the unigram
  plateau, and at this budget some have barely started.

What does not survive: **"the gap closes monotonically, 0.839 → 0.686 → 0.382"** and **"it is the
only one still improving"** as quantitative claims. Both were computed from single seeds of a
distribution with sd 1.3. The direction may well be right; the numbers cannot carry the weight
this section put on them, and the 256M and 1024M rungs have not been re-seeded at all.

Treat the whole 86M column of §1 as one draw each, not as measurements.

---

## 5. What this cost

Ten cells, both cards, no failures. 86M cells ran ~93 min each at 184k tokens/sec; 33.8M cells
~40–52 min at 408k. Total wall clock about six hours.

Corpus preparation was 20 minutes for 1.1B tokens: FineWeb-Edu streams at 49.5M chars/sec on this
machine and the BPE encodes at 9.1M, so encoding dominates and the download is nearly free. Both
rates were measured before committing to the run rather than estimated after it.

---

## What this does not settle

The saturation threshold is one model size and one compute budget. It has now been measured in
two languages rather than assumed from one, and they agree to within half the seed spread — but
the step that shows the curve going flat (64M → 256M) exists only in English, and no amount of
work will make it exist in Yoruba.

**The seed spread is a property of the cell, not a constant.** Measured at 1.024B updates:
**0.149** for 33.8M English, **0.103** for 33.8M Yoruba, **1.327** for the 86M preset. The 0.049
this project quoted everywhere came from 33.8M Yoruba at 196.6M updates — a different corpus at a
fifth the compute — and using it as a universal threshold understated the noise by 2–3× on the
33.8M ladders and by 27× on the 86M column. The results notebook now measures it per cell rather
than holding a number.

Still unseeded: the 4M and 1024M English cells, the 4M and 16M Yoruba cells, and the 256M and
1024M cells of the entire 86M column.

No fine-tuning here — this is pretraining loss only, and the group's headline comparison is a
downstream task score. A model that wins on validation loss has not thereby been shown to win on
SIB-200 or MasakhaNER.
