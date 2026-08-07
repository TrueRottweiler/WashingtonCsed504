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
throughout, and §4 shows that column is bimodal rather than noisy — read it as anecdote, not
measurement.

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

## 4. The bigger model, and why its column cannot be read as a curve

Three seeds at every rung except 4M. The single-seed version of this section, and the two claims
it made, are gone — what replaced them is stranger and more useful.

| data | 86M mean | 86M sd | 33.8M mean | gap | 86M, 1 seed (earlier) |
|---|---|---|---|---|---|
| 4M | 6.7196 | *(1 seed)* | 3.2836 | 3.44 | 6.720 |
| 16M | 4.2071 | 1.290 | 2.5444 | 1.66 | 3.200 |
| 64M | 3.8241 | 1.364 | 2.2819 | 1.54 | 3.278 |
| 256M | **2.7547** | **0.123** | 2.3869 | **0.37** | 2.896 |
| 1024M | **4.3649** | **2.699** | 2.1930 | **2.17** | 2.575 |

**The variance is not a constant, and it is not monotonic.** At 256M the 86M model is as
reproducible as the small one — sd 0.123 — and posts its best result, within 0.37 of a model less
than half its size. At 1024M the spread is **2.699**, which is larger than the entire span of the
33.8M column across a 256× range of data. That cell is a coin flip between a model that learned
and one that barely left the plateau.

**Withdrawn.** The earlier draft said the gap "closes monotonically, 0.839 → 0.686 → 0.382" and
that the 86M model was "the only one still improving". On seeded means the gap goes
**1.54 → 0.37 → 2.17**, and the 2.575 reported at 1024M was a lucky draw from a distribution whose
mean is 4.365. Neither claim survives; both were single draws read as a trend.

**What can be said.** The 86M model is not incapable — 2.7547 ± 0.123 at 256M is a real result
from a stable cell. It is *unreliable*, and unreliable in a way that depends on the rung. Whatever
governs whether a seed breaks through the unigram plateau within 62,500 steps is not simply "more
data helps"; at 1024M it evidently gets worse.

**What this means for the group's AfriBERTa result.** It is still explained by budget rather than
by architecture — but the mechanism is not the one this report proposed. It is not that the model
is uniformly undertrained and improving steadily with scale. It is that at this step budget the
model *sometimes* trains and sometimes does not, and a single run per cell cannot tell you which
happened. A study that reports one seed per cell at 86M is reporting coin flips.

That is the most practical finding in this report for anyone continuing the work: **at 86M, one
seed is not a measurement.**

### Correction in progress: two failure modes, not one

This section describes the 86M failures as seeds that "never break through the unigram plateau."
That is true of some of them and **false of others**, and collapsing the two cost the report a
claim it should not have made.

`eng_1b_1024M_afriberta_s1` reached **6.182 at step 20,000** and finished at **7.469**. English's
unigram entropy is 7.491, so it did not fail to leave the plateau — it left, learned for twenty
thousand steps, and fell all the way back. That single run is what gives the 1024M cell its
standard deviation of 2.699.

A schedule sweep run to test the warmup hypothesis found the same thing deliberately, and gave a
clean result on the way:

| warmup | diverged | 
|---|---|
| 0.06 (the default) | **0 of 3** |
| 0.15 | **2 of 2** |

Both 0.15 seeds peaked — 6.209 and 6.190 — and then fell back to ~7.48. Longer warmup holds the
peak learning rate higher for longer, and at 86M that is enough to turn a run that was learning
into one that is not. The remaining configuration at 0.25 was dropped rather than run: it would
have held the rate higher still, and the direction was no longer in question.

So the mechanism is **mid-training divergence, not failure to start**, and the recommendation
that follows is the opposite of the hypothesis: at this width, *less* warmup and a *lower* peak
rate. Sections above will be restated once the lower-rate and tighter-clipping runs finish.

### What would explain it

Not tested, and stated as hypotheses so nobody mistakes them for results:

- The warmup floor (`MIN_WARMUP_STEPS = 250`, so `pct_start` bottoms out at 0.06) may be too short
  for this width at long budgets, leaving the model's early trajectory seed-dependent.
- The plateau breakout looks like a **threshold crossing rather than a gradual effect**, in which
  case the standard deviations above are the wrong summary entirely. All thirteen 86M runs on
  this corpus, sorted:

  ```
  2.57  2.67  2.70  2.82  2.90  3.05  3.20  3.28  3.76   |   5.38  5.66  6.72  7.47
  ```

  Nine below 3.8, four above 5.3, and **nothing between 3.8 and 5.3**. That is not a spread, it
  is two populations: runs that broke through the unigram plateau and runs that did not. A mean
  and an sd describe neither, and the honest parameter is the failure rate — 4 of 13, about 31%,
  on this evidence.

Either would be worth a learning-rate or warmup sweep before anyone runs an 86M study for real.
The second also means the sd column above should be read as "how often did this cell fail",
not as a precision.

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
**0.149** for 33.8M English (three repeated cells), **0.103** for 33.8M Yoruba (one), and
**1.369** for the 86M preset — where §4 shows a standard deviation is the wrong summary. The 0.049
this project quoted everywhere came from 33.8M Yoruba at 196.6M updates — a different corpus at a
fifth the compute — and using it as a universal threshold understated the noise by 2–3× on the
33.8M ladders and by 27× on the 86M column. The results notebook now measures it per cell rather
than holding a number.

Still unseeded: the 4M and 1024M English cells, the 4M and 16M Yoruba cells, and the 256M and
1024M cells of the entire 86M column.

No fine-tuning here — this is pretraining loss only, and the group's headline comparison is a
downstream task score. A model that wins on validation loss has not thereby been shown to win on
SIB-200 or MasakhaNER.
