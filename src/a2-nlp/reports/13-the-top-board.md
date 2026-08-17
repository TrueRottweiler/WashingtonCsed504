# When is it worth training your own model? Yoruba, from scratch against transfer

*A2-NLP · CSED 504 · the top board*

**What goes on the wall is the nine cells and three strip blocks below.** The fourteen *Panel*
sections after them are the long form — the argument behind each cell, kept because a cell that
makes somebody want the whole case needs somewhere to send them. The bottom board has the same
pair: [report 12](12-the-bottom-board.md) is what to print, [report 09](09-the-bottom-report.md)
is the long form.

Every number here was recomputed from `src/a2-nlp/runs/` on **11 August 2026**, through
`ft_api.results()` and `mlm_api.results()`, not transcribed from a report or an email. Where a
figure is not regenerable from the records, it says so in the panel.
`test_board_numbers.py` pins each one to the record it came from.

---

## Format — the same board as the bottom, and the same measure

24 × 36 in portrait, three columns at x = 1.50 / 8.75 / 16.13 in, each **6.35 in** wide; body type
**18 pt**. Three rows of 6.70 in with 0.25 in gutters, then a 4.45 in strip across the foot. The
full derivation is in [report 12](12-the-bottom-board.md#format--measured-from-the-template-not-assumed)
and is not repeated here — **both boards print at the same size and must not drift apart.**

**The word budget is a hard limit, not a target.** A cell with a figure has about 1.9 in of column
left, which is **~55 words at 18 pt**; a cell without a figure gets about **110**; a strip block
about **100**. Every big number has to fit **about 18 characters per line**, which is why each is
written below as two lines separated by ` / `.

**Nine cells, seven figures.** Cells 1 and 4 carry no figure and run long as type. That is the
whole of the layout decision — unlike the bottom board, no cell here is carrying a second block.

---

## Title block

> ## When is it worth training your own model?
> *Yoruba, from scratch against multilingual transfer*
>
> **Transfer** inherits a hundred languages and a vocabulary built mostly for other people.
> **From scratch** fits the language and starts from nothing. On the task that needs meaning,
> from scratch wins — and the reason is the **vocabulary**, not the data.

Author line, in the header band: *Patrick Kwok · Jeffrey Stall · Leon · A2-NLP · CSED 504 · the
lower board is the machinery this experiment ran on.*

Both boards carry all three names; the role split is named once, on the bottom board's strip.

**Goals**, set as one line in the header band under the author line, at 20 pt. These are the
proposal's six, in its own order, and they are the one rubric item neither board carried until
17 August:

> **What we set out to do.** Establish baselines against a reproduced published floor · measure
> from-scratch quality against pretraining budget on two tasks · **separate labelled-data quantity
> from task type**, which the proposal called the decisive experiment · locate where from-scratch
> overtakes transfer · test whether the pattern generalises across languages · ask whether MLM
> loss predicts downstream quality. **All six are closed.** Goals 1–5 are the cells below; goal 6
> is answered **no**, in report 10.

---

## The nine cells

Read left to right, top to bottom. The top row rules out the obvious explanation and puts the real
one in its place; the middle row is the result and the floors that make it honest; the bottom row
is the causal evidence and what it actually buys you.

### 1 · Why train your own model at all?

**Big number:** `69.1M` / `tokens, all of it` · **Figure:** *none — set as type*

> For a language with little text the received answer is transfer: fine-tune a multilingual
> encoder someone else pretrained — XLM-R across 100 languages, mmBERT across 1,800 and about
> 3 trillion tokens. You inherit everything they learned, **including a vocabulary built mostly
> for other languages**. The alternative is to pretrain your own small encoder on whatever
> in-language text exists, with a vocabulary fitted to it, and get a model that fits, built from
> far less. **Yoruba is the test case**: about 47 million speakers, and all of FineWeb-2 Yoruba is
> 69.1M tokens. The received answer is transfer. The interesting question is when it stops being
> right.

*105 words · no figure · report 13 Panel 1*

### 2 · Is Yoruba starved of text?

**Big number:** `−0.080` / `for 16× the text` · **Figure:** `05-data-saturation.svg`

> Scarcity is the natural story, and it does not survive measurement. On an English ladder at
> fixed compute, sixteen times the text — 64M to 1,024M tokens — moves validation loss
> **−0.080**: 0.43× the seed spread, and the wrong way. All of FineWeb-2 Yoruba is 69.1M, so
> **at these budgets Yoruba has enough text.**

*54 words · `runs/*_result.json`, the `eng_1b_*` ladder*

### 3 · Then where does the disadvantage live?

**Big number:** `1.15× vs 1.59×` / `covered vs not` · **Figure:** `02-tokenizer-gradient.svg`

> Seventeen languages. XLM-R's 250k vocabulary costs **1.150** tokens per word on languages it
> covers and **1.593** on those it does not — 1.244 against 1.593 with both sides African, ruling
> out script and region. Learnability does not separate at all: 4.618 against 4.808, overlapping.
> **The disadvantage lives in the vocabulary, not the language.**

*53 words · `runs/gradient_table.json` · report 07 §4*

### 4 · What does a vocabulary that does not fit cost?

**Big number:** `77 words` / `against 44` · **Figure:** *none — set as type*

> A vocabulary that does not fit cuts words into more pieces. Yoruba costs **1.76 tokens for every
> one** a fitted vocabulary uses — second-highest of seventeen, not a quirk of one language.
> Concretely: **a 128-token context window holds 77 Yoruba words under our `yor-bpe16k` and 44
> under XLM-R's.** The same effect arrives from a different direction: encoding the identical
> 260M characters gives 69,096,452 tokens under our vocabulary against 121,339,416 under XLM-R's
> — a ratio of **1.756**, reproducing the 1.76 measured on word counts. The first time two of our
> measurements agreed by accident rather than by construction.

*97 words · no figure · `runs/swap_*_result.json`, `n_tokens`*

### 5 · Does a small from-scratch model actually win?

**Big number:** `0.688` / `against 0.582` · **Figure:** `01-headline.svg`

> SIB-200 topic classification: 204 test items, five seeds, every arm swept over nine learning
> rates and ranked on the **dev** split rather than the items it is scored on. **A 33.8M model
> trained on 64M Yoruba tokens reaches 0.688 macro-F1 against mmBERT's 0.582** — mmBERT saw about
> 3 trillion tokens across 1,800 languages. Say **ahead**, not *beats*: the intervals still
> overlap by 0.004.

*63 words · `runs/ft_sib200_*.json` · report 11*

### 6 · What does a model that knows no Yoruba score?

**Big number:** `0.626` / `knowing no Yoruba` · **Figure:** `12-floors.svg`

> Draw the floor, or the chart lies. On MasakhaNER, 0.863 / 0.851 / 0.837 looks like three strong
> models — until an untrained encoder scores **0.626**. Most of every bar is capitalisation and
> name shape, which transfer from any language. On topic the floor bites harder: **XLM-R scores
> 0.358, below the 0.382 of an untrained model of its own architecture.**

*60 words · `runs/ner_control_sweep.json` · `runs/ft_sib200_*.json`*

### 7 · Is that the task, or just the labels?

**Big number:** `43%` / `of the way there` · **Figure:** `18-label-quantity.svg`

> We win topic by 0.106 and lose entities by 0.026. Across the identical sixteen from-scratch
> models, entity scores span **0.044** against topic's **0.143** — entity recognition barely
> notices which model it is given. So we cut NER to 701 labels at a fixed step budget.
> **Threefold does nothing; tenfold moves the models 1.5× further apart, and still less than half
> way.**

*61 words · `runs/label_quantity.json` · `runs/downstream_correlation.json`*

### 8 · Does the vocabulary *cause* it?

**Big number:** `+0.144` / `vocabulary alone` · **Figure:** `03-matched-steps-vs-compute.svg`

> Same architecture, same Yoruba text, same compute — one model on our 16k vocabulary, one on
> XLM-R's 250k. Four seeds a side, dev-swept. **Every seed of ours beats every seed of theirs, on
> both tasks**: +0.144 on topic, +0.061 on entities. The only design here that shows the penalty
> *causing* a difference — and only because compute was held fixed, not steps.

*62 words · `runs/swap_downstream.json`*

### 9 · So what does a bad vocabulary actually cost?

**Big number:** `F = 15.1` / `spread, not mean` · **Figure:** `17-tokenizer-lottery.svg`

> We reported for two days that a bad vocabulary costs 0.144 bits per character. Six
> **pre-registered** seeds a side say otherwise: the gap falls to 0.059 (*p* = 0.374) and the arms
> interleave. **The spreads separate instead** — 0.145 against 0.037, *F* = 15.1. A vocabulary
> that does not fit **decides how much of a gamble the run is.**

*59 words · `runs/tokenizer_seeds.json`*

---

## The strip — three blocks across the foot

### A number that is not a result

**Big number:** `5 constants` / `one on purpose`

> The recurring failure mode: *a constant chosen for one context silently deciding a result in
> another.* A seed spread of 0.049, applied to experiments it did not describe. Fertility measured
> on the first 400 documents — a separation that reverses at 800. An untrained floor printed for a
> fortnight at 0.4140, one cell of a sweep. And **`FT_STEPS = 352`, inherited to preserve
> behaviour, which decided this study's central downstream conclusion** — the best of them,
> because keeping it was correct.

*81 words · report 13 Panel 12*

### What we do not claim

**Big number:** `3 limits` / `stated, not hidden`

> **Nobody on this team reads Yoruba**, so every quality judgement here is a benchmark number
> rather than a judgement about whether the output is good Yoruba. We did not compare masked
> against next-token modelling — encoder and decoder cannot be scored comparably on these tasks.
> We did not use translation-based augmentation, because we could not audit whether a translation
> preserved meaning. And the causal evidence in cell 8 is **one language**, four seeds a side.

*75 words · report 13 Panel 13 · `runs/claims_audit.json`*

### What we would do next

**Big number:** `4–5` / `languages next`

> **The gap is causal breadth.** Cell 8 shows the vocabulary causing a downstream difference in
> one language; cell 3 shows the penalty tracking coverage across seventeen, but never through to
> a task. **The experiment that closes it is cell 8 repeated across the coverage gradient** — the
> swap in four or five languages spanning 1.15 to 1.59, with the untrained floor measured in each.
> If the downstream gap tracks the fertility ratio, the argument becomes quantitative.

*76 words · report 13 Panel 14*

---

## Does the board answer the assignment?

Checked against the rubric rather than assumed. **The pair of boards is what gets marked**, so
this column says plainly where an item lives even when that is downstairs — duplicating a
requirement across both boards spends wall space twice, and the brief warns that space is tight.

| required | where it is |
|---|---|
| Team member names | header band, both boards · role split on the bottom strip |
| Problem / motivation | title block + **cell 1** |
| Goals | **header band**, all six, added 17 Aug · full audit in report 13 §Goals |
| Tool stack | **bottom board cells 2–5** — that board is the tool stack |
| Pre-existing vs from scratch | **cells 1 and 5** |
| How performance was explored | **cells 5–9**: dev-split selection, five seeds, floors, exact tests |
| Dataset EDA | **cell 2** (corpus size and saturation), **cells 3–4** (tokenizer fit across 17 languages) |
| Results / summary statistics | **cells 5–9**, every one a measured number with its spread |
| Discussion / limitations | **cells 6–9** and **strip 2** |
| Ethical impact | **strip 2** here and on the bottom board — the one item deliberately said twice |
| Next steps | **strip 3** |
| Citations + AI statement | **bottom board strip 3**, with all 19 references in report 09 |

**Two of these were genuinely missing before this pass, and neither was visible from the panels.**
Goals were on no board at all — the proposal's six had been closed one at a time in the state file
and never collected anywhere a reader could see them. And Dataset EDA was being claimed by the
bottom board's coverage table as *"the top board's panel 2"* while the top board had no settable
panel 2 to point at, because it had no settable panels. **A coverage table that cites a document
which does not yet say the thing is the same defect as a caption that cites uncommitted records.**

---

## The long form — the fourteen panels

*Everything below is the argument behind the cells above. It is not set on the board.*

---

## Panel 1 — The question

For a language with little text and few labels, you have two options.

**Transfer.** Take a multilingual encoder someone else pretrained — XLM-R, 100 languages;
mmBERT, 1,800 languages and about 3 trillion tokens — and fine-tune it on your task. You inherit
everything those models learned, including a vocabulary built mostly for other languages.

**From scratch.** Pretrain your own small encoder on whatever in-language text exists, with a
vocabulary fitted to that language, then fine-tune it. You get a model that fits, built from far
less.

The received answer is transfer, and the interesting question is when it stops being right.
**Yoruba is the test case**: ~47 million speakers, and all of FineWeb-2 Yoruba is 69.1M tokens.

**Our finding is that from scratch wins on the task that needs meaning, and that the reason is
the vocabulary rather than the data.** The rest of the board is how that was established and
where it stops.

---

## Panel 2 — The obvious explanation is wrong: Yoruba is not data-starved

*Figure 05 — data saturation*

The natural story is scarcity: too little Yoruba text, so a from-scratch model cannot be good.
**That story does not survive measurement.** An English ladder at fixed compute, 33.8M-parameter
model, three seeds per rung:

| training tokens | val loss | sd (n=3) |
|---|---|---|
| 4M | 3.621 | 0.333 |
| 16M | 2.544 | 0.179 |
| **64M** | **2.282** | 0.115 |
| 256M | 2.387 | 0.154 |
| 1024M | 2.362 | 0.146 |

**From 64M to 1024M — sixteen times the text — the loss moves −0.080, which is 0.43× the seed
spread, and points the wrong way.** Past roughly 64M tokens, more text buys nothing this study can
measure. Yoruba arrives there earlier still: its 16M → 64M gain is +0.053 against its own measured
spread of 0.103, i.e. noise — though that rung is a single seed against a three-seed spread, so it
supports the English result rather than standing alone.

All of FineWeb-2 Yoruba is 69.1M tokens. **So at the budgets anyone here trains at, Yoruba has
enough text**, and "there is too little Yoruba" is not available as the explanation for anything.

Read three bands, not two: below the spread is noise, above twice it is real, and in between this
many seeds cannot say. The 16M → 64M step (1.4× the spread) sits in that middle band, which is why
this panel says saturation begins "at or before 64M" and declines to be more precise.

---

## Panel 3 — The thesis

*Figure 02 — the tokenizer gradient*

> **The tokenizer penalty separates by XLM-R coverage. From-scratch learnability does not.**
> The disadvantage a multilingual model carries on an under-represented language lives in its
> **vocabulary**, not in the language being harder to learn.

Seventeen languages. **The tokenizer penalty** is what XLM-R's 250k vocabulary costs against each
language's own 16k BPE, measured as tokens per word:

| | languages | mean |
|---|---|---|
| **covered by XLM-R** | cmn 0.95, ind 1.01, eng 1.04, fra 1.04, afr 1.06, swh 1.17, som 1.21, hau 1.27, amh 1.33, xho 1.42 | **1.150** |
| **not covered** | wol 1.31, lug 1.50, nya 1.57, sna 1.59, kin 1.59, **yor 1.76**, ibo 1.82 | **1.593** |

Restricted to African languages on both sides — which rules out "it is really about script or
region" — **1.244 against 1.593**.

**Learnability**, over the same languages pretrained identically, measured as *context gained* =
corpus unigram entropy − final loss, because raw loss across different vocabularies compares
nothing:

| | n | mean | range |
|---|---|---|---|
| covered by XLM-R | 6 | 4.618 | 4.061 – 5.624 |
| not covered | 4 | 4.808 | 4.101 – 5.172 |

**The ranges overlap almost entirely.** The best is Xhosa, which XLM-R covers; the worst is Hausa,
which XLM-R also covers.

Two honesty notes that travel with this panel. **Wolof at 1.31 sits inside the covered range**, so
this is a gradient with one exception rather than a clean split — six of seven uncovered languages
are above every covered one. And an earlier version of this result claimed *perfect* separation at
p ≈ 5×10⁻⁵; that was an artifact of measuring fertility on the first 400 documents, and it reverses
at 800. Panel 12 is about that class of mistake.

*Regenerable here from `runs/gradient_table.json`. One trap: the file has 18 rows for 17 languages
— `eng` and `eng_1b` are the same language at two corpus sizes — and averaging the raw rows gives
1.140 over "11 covered languages" instead of 1.150 over 10. The learnability table's entropies live
in report 07 §4 rather than in a JSON.*

---

## Panel 4 — What the penalty actually is

A vocabulary that does not fit cuts words into more pieces. Yoruba costs **1.76 tokens for every
one** a fitted vocabulary uses — second-highest of the seventeen, so the original measurement was
not a quirk of one language.

What that means concretely, and this is the sentence for the board:

> **A 128-token context window holds 77 Yoruba words under our `yor-bpe16k`, and 44 under
> XLM-R's.**

The same effect appears from a completely different direction. Encoding the identical 260M
characters of Yoruba gives **69,096,452 tokens** under our vocabulary and **121,339,416** under
XLM-R's — a ratio of **1.756**, independently reproducing the 1.76 fertility penalty measured on
word counts. **This is the first time two of the project's measurements agreed by accident rather
than by construction.**

---

## Panel 5 — The headline

*Figure 01 — the headline*

**SIB-200 topic classification**, 701 train / 99 dev / 204 test, 7 classes, 1056 steps. Every arm
swept over the same nine learning rates, ranked on the 99-item **dev** split, and only the winner
scored on the 204 test items, at five seeds:

| model | lr | macro-F1 | sd | 95% CI | per-seed |
|---|---|---|---|---|---|
| **from-scratch 33.8M** | 3e-5 | **0.688** | 0.024 | [0.631, 0.734] | .667 .695 .655 .712 .712 |
| mmBERT base | 7e-5 | 0.582 | 0.023 | [0.518, 0.635] | .590 .586 .538 .594 .604 |
| random init, our arch | 1e-4 | 0.429 | 0.036 | [0.379, 0.468] | .408 .372 .468 .430 .466 |
| random init, XLM-R arch | 3e-5 | 0.382 | 0.017 | [0.330, 0.425] | .396 .377 .394 .395 .350 |
| XLM-R base | 3e-5 | 0.358 | **0.161** | [0.311, 0.398] | .543 .386 .409 .396 **.057** |

**A 33.8M model trained on 64M Yoruba tokens is ahead of mmBERT — which saw about 3 trillion
tokens across 1,800 languages — by 0.106.**

Two qualifications belong on the board beside it, not in a footnote.

**Say "ahead", not "beats".** The margin clears the project's own 0.06 floor, but the bootstrap
intervals still overlap by 0.004. The two statistics answer different questions and both are
honest: the bootstrap resamples the 204 **test items** with predictions pooled across seeds, so it
carries item-sampling uncertainty and is blind to seeds; a Welch t-test over the five seeds gives
**t = 6.41, p < 1e-4** and is blind to the test set. The accurate sentence is that **the difference
is far larger than seed noise while the test set is too small to place it precisely.**

**XLM-R's row is a mixture, not a mean.** Four of five seeds trained and one collapsed to 0.057,
below the 0.143 chance line. Quote it as *4 of 5 seeds trained*; the honest summaries are 0.358
with the failure and 0.434 without.

---

## Panel 6 — The sharpest negative result: XLM-R does not clear its own floor

Run a randomly initialised model of **XLM-R's own architecture** — same size, same vocabulary, no
pretraining — and fine-tune it identically. Dev-selected, both arms:

| | macro-F1 |
|---|---|
| XLM-R base, pretrained on 100 languages | 0.358 |
| the same architecture, untrained | **0.382** |

**XLM-R scores below an untrained model of its own architecture.** Discard its one collapsed seed
and it is +0.052 — still inside the 0.06 floor. Either way:

> **Whatever XLM-R learned from 100 languages does not reach this one.**

And a second sentence comes free: *a randomly initialised encoder is a steadier predictor of Yoruba
topic than XLM-R is* — sd 0.017 against 0.161.

This result also has a history worth one line, because it is the study's cleanest example of a
control being a measurement rather than a constant. At an earlier 352-step budget XLM-R scored
0.110 against a floor of 0.107 and looked collapsed; at 1056 steps it trains, but so does the
floor, and the gap does not appear. **A control quoted at one budget is not a control at another.**

---

## Panel 7 — Draw the floor, or the chart lies

*Figure 12 — floors*

**MasakhaNER entity F1**, 6,876 / 983 / 1,964 sentences, 2,150 steps:

| model | entity F1 | lr |
|---|---|---|
| mmBERT base | **0.8628** | 7e-5 |
| XLM-R base | 0.8513 | 7e-5 |
| from-scratch 33.8M | 0.8373 | 1e-4 |
| **untrained control** | **0.6261** | 3e-4 — best of twelve rates |

A chart of 0.863 / 0.851 / 0.837 looks like three strong models. **The same chart with 0.626 drawn
in is a much more honest picture:** a model that knows no Yoruba at all scores 0.626 where the best
model scores 0.863, so most of the height of every bar is capitalisation and name shape — surface
features that transfer from any language. What the three pretrained models add on top is largely
more of the same, which is why they lead here and trail on the task that needs meaning.

*Stated as a difference, not a share. Panel 8 explains why a percentage-of-ceiling on this chart is
the least stable number on the board.*

The floor is also this project's best example of an unswept constant. It was **0.4140** for a
fortnight — and that turned out to be the **3e-5 cell of a sweep whose best is 3e-4**, a rate one
tenth of the right one, quoted only because nobody had swept it. Fixing it halved the from-scratch
model's apparent lead over the floor, from 0.423 to 0.211.

Selection is on test for the control, because MasakhaNER as loaded ships no dev split. That makes
the floor an **upper** bound, which is the conservative direction for every gap measured against it.

---

## Panel 8 — The two tasks disagree, and the disagreement is the result

Our model wins topic classification by 0.106 and loses entity recognition by 0.026. That is not an
inconsistency to explain away; it is the finding.

Across the **identical sixteen** from-scratch models measured on both tasks:

| | band (range) | between-model sd | span |
|---|---|---|---|
| MasakhaNER entity F1 | **0.0441** | 0.0129 | 0.7537 – 0.7977 |
| SIB-200 topic | **0.1426** | 0.0457 | 0.5621 – 0.7047 |

**Entity recognition barely notices which from-scratch model it is given. Topic classification
varies three and a quarter times as much.** Together with the floor in panel 7, the account is:
NER leans on surface form, which transfers without knowing any Yoruba; topic classification needs
semantics, which does not.

**Lead with the raw bands and do not normalise them.** A "share of headroom" version of this panel
has been bitten twice in two days — once when the floor moved (9.8% → 18.6%) and once because the
answer depends on which model you call the ceiling (55.0% or 51.7%, both defensible). The raw ratio
is 3.24× under any floor and any ceiling. **A statistic whose denominator is the least stable number
on the board has no business being set in 90-point type.**

---

## Panel 9 — The decisive experiment: neither explanation, and it said so in advance

*Figure 18 — the labelled-data axis*

The two tasks differ in **kind** and in **label count** — 6,876 against 701 — and both differences
predict what panel 8 shows. The original study design called separating them *"the decisive
experiment"*: subsample the larger task's training set to match the smaller one, holding the step
budget fixed so a subsample does not silently become a compute cut.

The instrument is the **band**, re-measured at fewer labels. The decision rule was written before
the data existed:

| outcome | reading |
|---|---|
| the band widens toward topic's | label **quantity** |
| the band stays near its full-data value | task **type** |

**All four rows below are over the identical sixteen models**, which is the only way the comparison
is legitimate — a range grows with the number of models in it, so a band measured over one set and
compared against a constant from another set is measuring arithmetic:

| labels | band (range) | between-model sd | vs full split |
|---|---|---|---|
| 6,876 (full split) | 0.0441 | 0.0129 | — |
| **2,000** | 0.0559 | 0.0127 | ×1.27 range, **×0.99 sd** |
| **701** | 0.0691 | 0.0195 | ×1.57 range, **×1.51 sd** |
| *SIB-200 topic, for scale* | *0.1426* | *0.0457* | |

**The rule returns BETWEEN THE TWO, and the honest panel says so.** At SIB-200's label count NER
does discriminate more between models — and gets **43% of the way** there, on a statistic that does
not grow with the set size. Task type is not the whole story and label quantity does not account
for it either.

**The sharper finding is the dose-response, and it is only visible because we asked for 2,000 as
well as 701.** At 2,000 labels the between-model sd is 0.0127 against the full split's 0.0129 — no
widening at all. If spread grew steadily as labels shrank, 2,000 would sit between 6,876 and 701.
It does not; it sits on top of the full split.

> **Cutting the labels threefold does nothing. Cutting them tenfold moves the models 1.5× further
> apart, and still less than half way to topic classification.**

Context arms, not part of the band: mmBERT 0.6938 at 701 and 0.7856 at 2,000; the untrained floor
0.3519 and 0.4650.

*A seventeenth from-scratch model has a 701 cell but no 2,000 cell, so the study's own report shows
the 701 band over seventeen: the same range, 0.0691, and a between-model sd of 0.0190 against the
matched sixteen's 0.0195. The table above uses the sixteen that carry all three levels, because the
dose-response compares the levels with each other and that only means anything on a matched set.*

*That is also why this panel reads 43% where the study's console output reads 42% — 0.0195 / 0.0457
against 0.0190 / 0.0457, each correct over its own set. Do not reconcile them by changing one: the
figure and this table are the matched sixteen throughout, and a percentage carried between two set
sizes is the mistake panel 12 is about.*

---

## Panel 10 — The causal evidence: swap the vocabulary and nothing else

*Figure 03 — matched steps against matched compute*

Panels 3 and 4 show the tokenizer penalty exists and tracks coverage. **Only this experiment shows
it causes anything.** Same architecture, same Yoruba text, same compute — one model on our
`yor-bpe16k`, one on XLM-R's 250k vocabulary. Four pretraining seeds a side, both arms swept over
nine learning rates on the dev split:

| task | XLM-R's vocabulary | our vocabulary | gap | exact p |
|---|---|---|---|---|
| SIB-200 topic | .4597 .4794 .4826 .4891 → **0.478** | .5943 .6278 .6303 .6339 → **0.622** | **+0.144** | 0.029 |
| MasakhaNER | .6818 .6888 .7506 .7667 → **0.722** | .7758 .7820 .7855 .7870 → **0.783** | **+0.061** | 0.029 |

**Every seed of our vocabulary beats every seed of theirs, on both tasks.** Divided by the pooled
seed spread — **how far apart** the two vocabularies land — the gaps are **9.1× on topic** and
**2.0× on entities**, so topic is decisive where entities barely clears.

Panel 11 divides one arm's spread by the other's, which is a different question about the same
eight runs. Its table sets the two side by side, because calling both "the spread ratio" is what
made this read as hedging.

Three things must travel with this table.

**0.029 is the floor of the test, not a measurement of separation.** At three seeds a side an exact
permutation test cannot return anything below 0.10 however cleanly the arms separate; a fourth seed
takes the floor to 0.029, which is exactly where both tasks sit. The spread ratios are what measure
the separation.

**The gaps shrank when the fourth seed arrived** — 0.157 → 0.144 on topic, 0.074 → 0.061 on
entities — because our fourth seed is the weakest of the four. Ordinary regression, and the reason
three seeds flatter a result.

**These are 12k-step swap models, not the headline model.** Our 0.622 here is not the 0.688 of
panel 5. Do not mix the two tables.

The design also carries the project's best methodological lesson, and it is worth its own line on
the board: **hold compute fixed, not steps.** A 250k output projection is 128M parameters against
8.2M, so the large-vocabulary arm runs at 80k tokens/s against 408k — **5.1× the cost per step**.
"12,000 steps each" silently hands it five times the compute, and the experiment comes out saying
the tokenizer penalty costs nothing: fully seeded, internally consistent, and backwards.

---

## Panel 11 — What a bad vocabulary actually costs is *predictability*

*Figure 17 — the tokenizer lottery*

This panel replaces a claim we withdrew, and the replacement is better.

We reported for two days that a badly-fitting vocabulary costs **0.144 bits per character** at
matched compute. Six pre-registered seeds a side — the sample size fixed in advance from a power
calculation, and written into the script before the runs started — say otherwise:

```
250k vocabulary   0.832  0.854  0.892  1.079  1.129  1.147     mean 0.989   sd 0.145
16k  vocabulary   0.871  0.910  0.929  0.937  0.955  0.979     mean 0.930   sd 0.037
```

**The means do not separate.** The gap fell to 0.059 — Welch *p* = 0.374, exact *p* = 0.335 — and
the arms *interleave*: three of the six large-vocabulary runs land below the small-vocabulary
median. There is no direction left to report.

**The spreads separate decisively.** 0.145 against 0.037 — **F = 15.1, p = 0.0098**.

> **A vocabulary that does not fit does not reliably cost you bits per character. It decides how
> much of a gamble the run is.**

### The two questions, and why they point opposite ways

Panel 10 and this panel measure different things on the same eight runs, and the temptation is to
reconcile them. They do not need reconciling. A difference in means and a difference in variances
are independent, and the fact that they disagree on topic is the most interesting thing here:

| | topic | entities |
|---|---|---|
| **how far apart** — gap ÷ pooled seed spread | **9.1×**, exact *p* = 0.029 | **2.0×**, exact *p* = 0.029 |
| **how consistent** — one arm's spread ÷ the other's | **0.7×**, *F*-test *p* = 0.553 | **8.6×**, *F*-test *p* = 0.005 |

> **The vocabulary decides how good the topic model is, and how reliable the entity model is.**

Two notes that have to travel with the table. (The repeated 0.029 in the top row is the exact test's
floor at four seeds a side, not a coincidence — panel 10 says why.)

**The two rows deliberately use different tests, and using one for both is a trap we walked into
while checking this.** Permuting *raw scores* between two arms whose means differ inflates the
spread of every reshuffled group, so the observed grouping looks unusually tight and a location
difference comes back as a variance finding. On topic it returns *p* = 0.057 against the
*F*-test's 0.553 — a significant variance difference conjured entirely out of the 0.144 gap, on
the one task where the whole point is that consistency does *not* differ.

The fix is not to abandon permutation testing; it is to remove the means first. A two-sided exact
permutation test on each arm's *residuals* — every score minus its own arm's mean, with
|log(sd ratio)| as the statistic — is valid, distribution-free, and lands where the *F*-test does:
**0.571 on topic and 0.029 on entities, against the *F*-test's 0.553 and 0.005.** The two agree
because subtracting each arm's mean is the property the bottom row needs, and the *F*-test has it
built in. The board quotes the *F*-test because the rest of the project already does.

**Entities' 0.029 is the floor, and it is the same 0.029 as the top row's** — `2/C(8,4)`, four
seeds a side, both rows, for the same reason. Neither could have gone lower whatever the data
showed. That the two rows floor identically and still disagree on topic is the cleanest evidence
that they are measuring different things.

This is worth stating precisely rather than as "permutation tests are wrong for variance", because
the true version is the more useful one and it is panel 12's shape exactly: a tool that is correct
one question over, used one question too far. Two drafts of this paragraph got it wrong in two
different ways — the first said the permutation test was wrong for dispersion at all, and the
second used `|ratio − 1|`, which is not symmetric under swapping the arms and so was quietly
one-sided; it returned 1/70 on entities, a value no two-sided test at four a side can produce,
because every split is enumerated alongside its complement and both always count. **The paragraph
about naming your test went two drafts without naming its own.** It is
`poster_figures.residual_permutation()` now, so it regenerates and `test_board_numbers.py` pins
it.

**The same mean-to-variance shift appears upstream, on pretraining loss.** In bits per character
the gap is 0.059 and does not clear (exact *p* = 0.335, Welch *p* = 0.374) while the spreads do
(*F* = 15.1, *p* = 0.0098) — the numbers at the top of this panel. So the pattern holds on
pretraining loss and on entity recognition, and **topic classification is the one place it
reverses**: there the vocabularies separate cleanly and their variability does not differ at all.
A pattern with an exception is worth more than a pattern.

**Note this reads in bits per character, not nats per token.** Per-token loss cannot compare two
vocabularies at all, and the two factors pull opposite ways: a vocabulary that fits badly makes
more tokens per character, which raises bits/char even where per-token loss looks better.

---

## Panel 12 — A number that is not a result

**The most distinctive thing this project has, and it is a methods panel rather than a finding.**
The recurring failure mode, in Jeffrey's wording, is:

> *a constant chosen for one context silently deciding a result in another.*

Every substantive correction this fortnight came from a check firing, not from insight. The
clearest instances:

| the constant | what it decided |
|---|---|
| a seed spread of **0.049** | measured on one cell and applied to experiments it did not describe; the real values are 0.103–0.149, and 1.369 at 86M |
| **fertility on the first 400 documents** | gave perfect coverage separation at p ≈ 5×10⁻⁵; it reverses at 800 |
| **`FT_STEPS = 352`** | inherited from a frozen notebook's 8-epoch loop *to preserve behaviour*, and it decided the study's central downstream conclusion |
| **matched steps** in the swap experiment | 5.1× more compute to one arm; a fully seeded, internally consistent, backwards result |
| an **untrained floor at 3e-5** | printed for a fortnight as 0.4140; it is one cell of a sweep whose best is 3e-4, and fixing it halved a gap on this board |

**`FT_STEPS = 352` is the best of them, because keeping it was correct.** A bug is something you fix
and forget; a constant held for a good reason that turns out to be load-bearing is a harder lesson.
Reproducibility and correctness are not the same property.

**Two additions from 12 August, both a different shape from the table above.** Each is an input
nobody wrote down as an input at all.

*Prose about code goes stale like any other prose.* `fig_tokenizer_lottery`'s caption is computed
— that was fixed once already — but its **docstring** still asserted "4.0× wider on topic, 7.7× on
entities": three-seed values, the topic one never significant at *p* = 0.115 and reversed outright
at four seeds. The caption learned and the paragraph above it did not. We had been treating
"computed rather than hardcoded" as a property of the *output*, and a docstring is the thing the
next person reads before deciding whether to trust the function.

*"Generated from the records so it cannot drift" assumes a fixed renderer, and never said so.*
Figure 18 was rendered on a cloud runtime rather than the workstation — the first figure in this
project not drawn on one machine. `save()` already pins the SVG hash salt and strips the date so an
unchanged figure re-renders byte-identically; the matplotlib version was the one input left to
whichever machine you happened to be on. The runtime ships 3.10.0 and the other sixteen figures are
3.11.0, so the first commit of it would have regenerated differently on the workstation at every
staleness pass, forever — "all 16 figures byte-identical" quietly becoming 17 of 18, with nothing
wrong and nothing to find. Pinning `matplotlib==3.11.0` and re-rendering fixed it, and all
seventeen SVGs now report one renderer.

**And it goes one level below where we found it.** With matplotlib pinned, the SVGs are byte-
identical across both machines and **the PNGs still are not** — 406,856 bytes against 372,398 for
the same figure. The images are *pixel*-identical, all 9.2 million of them, to a channel delta of
zero; what differs is the zlib stream compressing them, which belongs to the Python build rather
than to matplotlib. So the version pin fixed the format that carries a version stamp and left the
one that does not, and the check that would eventually have caught it — `git status` after a
regeneration — is the same check the pin was protecting. **An undeclared input usually has a
second one underneath it, and pinning the visible one is what hides the rest.**

**End the panel on the catches, not the misses** — predicting the pattern is a better ending than
surviving it:

- Two guards written before the data existed **fired on their first run**: a grid-edge check caught
  an arm selecting at the top of its learning-rate range, and a chance check caught a reported mean
  that was a mixture of four trained seeds and one collapsed.
- `claims_audit.py` — a gate that states the null for each comparative claim and computes what
  would refute it — **rejected the 0.144 bits/char penalty at three seeds while four places in the
  repository asserted it, and the six pre-registered seeds sided with the gate.** That is the
  pattern caught by a tool built for it, before the poster carried it.
- And one from the day this board was written. The label-quantity study (panel 9) reads its model
  set out of the records rather than hardcoding it, precisely so the set cannot drift. It drifted
  anyway: a downstream sweep elsewhere gave four tokenizer-swap checkpoints their missing rows, the
  set went from 16 models to 21, the band went 0.069 → 0.182 and the printed verdict flipped from
  BETWEEN THE TWO to LABEL QUANTITY — with no change to the study at all. **A set computed from a
  shared directory is a query against other people's work.** Computing a constant instead of
  writing it down is necessary and not sufficient.

---

## Panel 13 — What we did not do, and why

Three scoping decisions, stated rather than hidden.

**We did not compare masked-language modelling against next-token prediction.** Encoder and decoder
models cannot be scored comparably on these two tasks, and a decoder trained on 69.1M Yoruba tokens
would be weak for well-understood reasons. The comparison would have measured the architecture
mismatch, not the question.

**We did not use translation-based data augmentation.** None of us reads Yoruba, so we could not
audit whether a translated example preserved meaning. An unauditable augmentation on a low-resource
language is a way to generate confident numbers about nothing.

**We did not fine-tune the 86M models downstream.** SIB-200 at this budget spans 0.429 (untrained)
to 0.688, which at the 0.06 floor is under three distinguishable levels, and the 86M and 33.8M
models sit 0.094 apart in loss — inside the seed spread. Two models that match in loss, on a
benchmark with under three resolvable levels, will not separate.

The last one has a corollary worth stating: **at AfriBERTa scale the larger model never beat the
smaller one at any data rung.** The 86M model was not too big for the data; it was unstable at the
default gradient norm. Clipping at 0.5 moves it a full nat and cuts its seed spread by up to 38×,
and at 256M tokens the two sizes then become indistinguishable.

---

## Panel 14 — What we would do next

**The gap this study did not close is causal breadth.** Panel 10 is the only design in the project
that shows the tokenizer penalty *causing* a downstream difference, and it does so in one language
with four seeds a side. Panel 3 shows the penalty tracks coverage across seventeen languages, but
only as a property of the vocabulary, never carried through to a task.

**The experiment that closes it is panel 10 repeated across the coverage gradient** — the swap, in
four or five languages spanning the 1.15-to-1.59 range, with the untrained floor measured in each.
If the downstream gap tracks the fertility ratio, the argument becomes quantitative rather than
directional.

Two smaller ones. **Panel 9's dose-response wants a rung between 701 and 2,000**, because the
effect appears somewhere in there and three points cannot locate it. And **panel 11's variance
result wants a mechanism**: we can show a 250k vocabulary makes a run a gamble, and we cannot yet
say whether that is the output projection's optimisation dynamics or the sparsity of updates to
rarely-used embeddings.

---

## Sources

Every table above regenerates from the repository. `mlm_api.results()` returns **197** pretraining
runs and `ft_api.results()` returns **161** reportable downstream records, plus 117 dev-split cells
that `results()` hides by default because a cell selected on the items it is scored on is not a
reportable number.

| panel | file |
|---|---|
| 2 | `runs/*_result.json`, the `eng_1b_*` ladder |
| 3 | `runs/gradient_table.json`; report 07 §4 for context gained |
| 4 | `runs/swap_*_result.json`, `n_tokens` |
| 5, 6 | `runs/ft_sib200_*.json` |
| 7, 8 | `runs/ft_masakhaner_*.json`, `runs/ner_control_sweep.json`, `runs/downstream_correlation.json` |
| 9 | `runs/label_quantity.json`; `study_label_quantity.py --report` |
| 10 | `runs/swap_downstream.json` |
| 11 | `runs/tokenizer_seeds.json` |

Reports behind the panels: **05** (saturation), **07** (the gradient and learnability), **11** (the
dev-split selection and the SIB-200 table), **08** (the swap experiment), **06** (the step budget
and the control that were both deciding the answer).
