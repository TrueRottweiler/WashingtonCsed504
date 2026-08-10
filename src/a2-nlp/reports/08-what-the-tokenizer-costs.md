# What the tokenizer actually costs

*A2-NLP · August 2026 · the swap experiment, the downstream rows, and a comparison that gave two opposite answers*

Every earlier report measures that the tokenizer penalty **exists**. Report 04 found Yoruba pays
1.65× under XLM-R's vocabulary; report 07 found that holds across seventeen corpora and separates
languages XLM-R covers from languages it does not. None of them measures what the penalty
**costs**.

This one does, twice — once in modeling quality and once downstream — and the first attempt got
the answer backwards for a reason worth more than the answer.

---

## 1. The same experiment, two opposite conclusions

Same Yoruba text, same architecture, three seeds each side. The only difference is which
vocabulary turned characters into tokens.

| arm | vocab | chars/token | steps | min/seed | bits/char |
|---|---|---|---|---|---|
| our BPE | 16,000 | 3.734 | 12,000 | 8 | 1.135 ±0.035 |
| XLM-R's | 250,002 | 2.133 | 12,000 | **41** | 1.056 ±0.142 |
| our BPE | 16,000 | 3.734 | 62,500 | 40 | **0.912 ±0.042** |

**Read as matched steps** — the first two rows — the vocabularies are indistinguishable. 1.135
against 1.056, a difference of 0.079 against a pooled spread of 0.089. On that reading the
penalty costs nothing and the study's central argument is in trouble.

**Read as matched compute** — rows two and three, both about forty minutes a seed — our
vocabulary wins by **0.144 bits per character**, 1.6× the spread.

Both numbers come from the same three runs. The difference is entirely which axis was held
fixed, and the first pass held the wrong one.

### Why matched steps is the wrong axis here

A 250k output projection is not a detail. At hidden size 512 the language-model head is 128M
parameters against 8.2M, and its matmul dominates the forward pass — the body is roughly 25M
FLOPs per token and the head roughly 128M. Measured, the 250k arm runs at 80k tokens/sec against
408k: **5.1× the cost per step.**

So "12,000 steps each" gave one arm five times the compute of the other. That is not a controlled
comparison, and it is the comparison the first pass ran.

Matched compute is also the question anyone actually has. Nobody chooses a vocabulary and then
buys whatever hardware it needs; they have a budget and want the best model it will buy.

### This is the fourth constant to decide a result here

The pattern is now established well enough to name. Each of these was a value chosen sensibly for
one purpose that then silently determined an answer somewhere else:

| the constant | chosen for | what it decided |
|---|---|---|
| seed spread 0.049 | one grid, measured honestly | significance everywhere, understated 2–27× |
| `epoch` in the JSONL | dashboard compatibility | that nobody could tell steps from passes |
| 400 documents | a default nobody revisited | a "perfect separation" that reverses at 800 |
| `FT_STEPS = 352` | reproducing an extraction | the project's central downstream conclusion |
| **matched steps** | **the obvious way to hold compute fixed** | **that the tokenizer penalty was zero** |

Five, then. The common shape is that none was a bug — each was defensible where it was written,
and each stopped being defensible somewhere else without anyone noticing it had moved.

### What the swap does not settle

The 250k model carries 120M more parameters. Matching compute does not match capacity, and a
reader is entitled to ask whether our vocabulary wins because it fits Yoruba or because the small
head leaves more compute for the body. Those are not separable in this design, and the honest
answer is that for a fixed budget it does not matter which — but for a mechanism, it does.

One of the three XLM-R seeds (0.892 bits/char) sits well below the other two (1.129, 1.147). With
three seeds that is a wide arm rather than an outlier to discard, and it is why the matched-steps
difference is inside the spread.

---

## 2. Downstream: the from-scratch model wins the semantic task

All at 1,056 steps, the budget at which models actually train — see
[report 06](06-when-a-number-is-not-a-result.md) for why 352 was not that budget.

**SIB-200, Yoruba topic classification**

> **Superseded 9 August.** Every row below picked its learning rate on the same 204 test items it
> is scored on. [Report 11](11-selecting-on-the-dev-split.md) re-selects all five arms on the
> 99-item validation split and replaces this table. It is kept here because the two claims that
> follow it were drawn from it, and both are corrected in place below — one of them reverses
> sign. The NER table further down is not affected.

| | macro-F1 | 95% CI | best lr |
|---|---|---|---|
| **from-scratch, ours (33.8M)** | **0.6659** | [0.603, 0.711] | 3e-5 |
| mmBERT | 0.5949 | [0.520, 0.652] | 7e-5 |
| XLM-R | 0.4077 | [0.351, 0.455] | 1e-5 |
| our architecture, untrained | 0.4034 | [0.351, 0.446] | — |
| XLM-R's architecture, untrained | 0.3692 | [0.318, 0.417] | — |

**A 33.8M model pretrained on 64M tokens of Yoruba is ahead of mmBERT by 0.071**, which was
pretrained on 3 trillion tokens across 1,800 languages. The intervals overlap, so this is not a
decisive win — but it is ahead, and it is the result the study set out to look for. Write it as
*ahead*, not *beats*, wherever it is quoted.

**Update, 9 August.** Under dev-split selection the margin is **+0.106** — 0.688 against 0.582 —
and clears the project's 0.06 floor for the first time. The intervals still overlap, by 0.004, so
*ahead* rather than *beats* survives the correction intact.

Every model here is its own best of a learning-rate sweep. That symmetry took three passes to
reach and is the subject of §2b.

One caveat §2b does not close: every arm chose its rate on the same 204 test items it is then
reported on, which inflates all of them. Patrick is re-selecting on SIB-200's 99-item validation
split — shipped with the dataset and never used by this project until now — and scoring only the
winner on test. **They landed on 9 August — [report 11](11-selecting-on-the-dev-split.md). Those
are the numbers to quote.**

The bottom three rows are the more interesting part. **XLM-R's pretraining is worth +0.039 over
the same architecture with random weights**, with intervals that overlap almost completely. For
Yoruba, XLM-R's pretraining contributes nothing measurable. That is a sharper statement than
"XLM-R is weak on Yoruba": whatever it learned from 100 languages does not reach this one.

**Corrected 9 August — the sign reverses and the conclusion hardens.** The +0.039 set XLM-R's
best of five against a control run at a single rate, so it was an upper bound, as §2b says of
itself. Selecting both arms the same way on the dev split gives **−0.024**: XLM-R lands *below* a
randomly initialised model of its own architecture. It also survives the obvious objection. One
of XLM-R's five seeds collapses to 0.057; discard that seed outright and the margin is +0.052,
still inside the 0.06 floor. Either way "contributes nothing measurable" holds — and it no longer
depends on reading an interval overlap, which is the weaker form of the argument.

**MasakhaNER 2.0, Yoruba entity recognition** (2,150 steps, NFC)

| | entity F1 | 95% CI | best lr |
|---|---|---|---|
| mmBERT | 0.8628 | [0.850, 0.877] | 7e-5 |
| XLM-R | 0.8513 | [0.836, 0.866] | 7e-5 |
| **from-scratch, ours** | **0.8373** | [0.821, 0.852] | 1e-4 |
| our architecture, untrained | 0.4140 | [0.390, 0.435] | — |

Report 06 recorded the from-scratch NER gap as **0.145** and predicted this row was the only one
the Unicode fix could move, because the from-scratch tokenizer was the only one whose fertility
changed (by 47%). Under NFC it became 0.053, exactly where predicted. Sweeping every model's
learning rate takes it to **0.014 against XLM-R** — overlapping intervals, though the baselines
are still ahead.

So most of what looked like a deficit for from-scratch pretraining on entity recognition was
measurement: the wrong Unicode normalization, then an unswept learning rate. What remains is
0.014.

### 2b. Three passes to a fair comparison

The sweep that produced those numbers had to be run three times, and each pass was unfair in a
different direction:

| | SIB-200 (ours − mmBERT) | NER (ours − XLM-R) |
|---|---|---|
| everything at defaults | +0.058 | −0.145 → −0.053 after NFC |
| **only our model swept** | +0.092 | −0.004 |
| every model swept | **+0.071** | **−0.014** |

The middle row is the trap. Our downstream numbers were at a default while both baselines had
been swept on SIB-200 — an asymmetry against us — so sweeping ours looked like simple fairness.
It was, on SIB-200. On NER *nobody* had swept anything, so the same action created an asymmetry
in our favor, and for a few hours the project believed it had drawn level with XLM-R.

Two further edge cases fell out of checking. Our NER peak sat at 5e-5, the top of the range, so
the range was extended and the true peak is 1e-4. mmBERT's SIB-200 peak sat at 5e-5, the top of
Patrick's range, so that was extended too and its true peak is 7e-5 — which is why the SIB
margin is 0.071 and not the 0.092 an unextended comparison would have given.

**A best-of-sweep number is only meaningful if the sweep contains the best.** Three of the five
sweeps in this project peaked at their own boundary.

**A fourth pass followed on 9 August**, and the +0.071 in the table above is superseded by
**+0.106** — see [report 11](11-selecting-on-the-dev-split.md). Every pass here still selected on
the 204 test items it reported; the fourth selects on the 99-item validation split instead. The
boundary problem recurred there too, on the untrained control, and the count is now four of six.

### The two tasks disagree, and the control explains why

Our model wins topic classification by 0.059 and loses entity recognition by 0.053. The floors
say why: **the untrained control scores 0.414 on NER and 0.403 on SIB-200 — but 0.414 out of
0.851 is 49% of the achievable score, against 0.403 out of 0.632, or 64%.**

> **Retracted 9 August, and the retraction reinforced on 10 August.** Both percentages moved and
> the gap between them mostly closed: the SIB-200 control is 0.429 once it gets its own swept
> rate ([report 11](11-selecting-on-the-dev-split.md)), and the denominators depend on which model
> counts as the ceiling. The better-supported account of the same divergence is that NER scores
> barely move across from-scratch models — a band of **0.044** — while SIB-200 scores vary three
> times as much, at **0.143**. The paragraphs below are kept because the *direction* survives; the
> two percentages should not be quoted.
>
> **Quote the raw bands, not the normalised ones.** The NER control was a single cell at 3e-5.
> Swept over twelve rates it peaks at **0.6261**, so the 0.414 below is not the floor — it is the
> floor's value at a rate one-tenth of the best one. That moves any figure with headroom in its
> denominator: the normalised band on NER goes from 9.8% to **18.6%**, so "a spread of 0.044,
> under 10% of its headroom" is wrong.
>
> And the denominator is unstable in a second way. Patrick puts SIB-200's normalised band at 55.0%
> and this recomputation puts it at 51.7% — neither is a mistake, we simply took different
> ceilings, his the dev-selected headline of 0.6881 and mine the best of the sixteen models in the
> correlation set at 0.7047. A statistic whose value depends on which model you decide is the
> ceiling, and which also moved by half when the floor was swept, is not one to print at 90 point.
>
> The raw ratio has neither problem: **0.044 against 0.143 over the same sixteen models is 3.24×,
> and no choice of floor or ceiling can move it.** Both bands are over the identical sixteen
> models at val_loss < 3.1 — NER [0.7537, 0.7977], SIB-200 [0.5621, 0.7047] — which Patrick
> checked independently and this agrees with to four decimals.
>
> Second time the headroom framing has been bitten by the same denominator: once when the
> floor-share reading was retracted on the 9th, again when the floor was swept on the 10th. That
> is enough. Patrick's call to drop it, and he is right.

Read the other way round: NER hands 0.414 to a model that knows no Yoruba at all. Capitalisation
and name shape carry it, and those transfer from any language. What the multilingual models add
on top is largely more of the same surface knowledge, which is why they are ahead there and not
on the task that needs semantics.

This is the SIB-200/NER divergence the project spent a fortnight treating as an anomaly. It was
not an anomaly and it was not a bug. It is what the tasks measure.

---

## 3. What this does to the study's claims

| claim | status |
|---|---|
| A language-specific vocabulary is worth building for an under-served language | **Hold, now measured.** 0.144 bits/char at matched compute, 1.6× the seed spread. Previously only the fertility ratio was measured. |
| The tokenizer penalty is 1.65× for Yoruba | **Hold.** Reproduced three independent ways: fertility on the pretraining corpus, on two evaluation sets, and as a 1.75× token-count ratio in the swap corpus. |
| From-scratch pretraining beats multilingual transfer for Yoruba | **Hold on topic classification**, and strengthened under dev-split selection: 0.688 vs 0.582, a margin of 0.106 that clears the 0.06 floor, though the intervals still overlap by 0.004 ([report 11](11-selecting-on-the-dev-split.md)). **Does not hold on entity recognition** (0.837 vs 0.863), though the gap is 0.026 rather than the 0.145 first recorded. The task decides. |
| XLM-R is a usable Yoruba baseline | **Withdrawn** in report 06, and now explained: under symmetric dev-split selection its pretraining is worth **−0.024** against the same architecture untrained — below it, not above ([report 11](11-selecting-on-the-dev-split.md)). The +0.039 first recorded here compared its best of five against a control run once. |
| The from-scratch model loses NER by 0.145 | **Withdrawn.** 0.014 against XLM-R once the Unicode normalization is right and every model gets its own best learning rate. |
| MLM loss does not predict downstream quality | **Weakened further.** Our model has the best SIB-200 score and the third-best NER score; loss ordering predicts one and not the other. |

---

## What this does not settle

Every downstream row is now three seeds at each model's own best learning rate, and three of the
five sweeps had to be extended because they peaked at their own boundary. The two untrained
controls are still single-rate: they are floors rather than competitors, but a swept floor could
be higher than the one quoted, and neither has been swept.

The swap is one language and one model size, at one compute budget. It says the penalty costs
something on Yoruba at 33.8M parameters and forty minutes. It does not say the cost scales, or
that it holds where the vocabulary fits better.

Nothing here is a downstream *tokenizer* comparison. Both downstream tables compare pretrained
models; neither isolates the vocabulary the way §1 does for language modeling. That experiment —
the same architecture and budget fine-tuned under two vocabularies — is still unrun, and it is the
one that would connect the two halves of this report.
