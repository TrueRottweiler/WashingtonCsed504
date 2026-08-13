# Selecting on the dev split

*A2-NLP · August 2026 · the SIB-200 numbers, chosen on data they are not scored on — and the one
row where it mattered*

SIB-200 ships three splits: 701 training sentences, 99 for validation, 204 for test. Until 9
August this project had used two of them. Every learning rate it ever quoted was picked by running
a grid on the **test** set and reporting the best cell — on the same 204 items.

`exp_budget_matched_baselines.ipynb` flagged this in its own output back on 7 August:

> *That is a choice made on the test set — SIB-200 ships a 99-item validation split, so select
> there instead before quoting a single number in a writeup.*

Nothing acted on it. This report is what happened when something did.

---

## 1. Two asymmetries, one of them already closed

[Report 08](08-what-the-tokenizer-costs.md) §2b describes the first: mmBERT and XLM-R were swept
over learning rates and their best cell quoted, while the from-scratch model and both untrained
controls were run at a single default. That biases in **opposite directions** — against us on
ours-vs-mmBERT, in XLM-R's favour on XLM-R-vs-its-own-control — and the weekend sweeps closed most
of it by giving every headline arm its own grid.

The second was untouched, and it is the larger one. Selecting the maximum of a grid and reporting
it on the items you selected on inflates every swept arm by an amount nobody here had measured.
Both controls were also still at one rate each, so XLM-R's reported **+0.039** over its own
untrained architecture remained an upper bound: sweeping a control can only raise it.

---

## 2. Method

Nine learning rates — 5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4 — across all five arms
at 1,056 steps. Three seeds per cell on the 99 validation items, which only has to *rank* the
rates. The winner of each arm is then run at five seeds on the 204 test items, and that is the
only number reported.

Two mechanics make this safe to keep alongside the older records. Dev-scored cells carry `_onval`
in their record tag, so they cannot collide with a test cell of the same configuration. And
`ft_api.results()` excludes them by default — a cell selected on the items it is scored on is not
a reportable number, and mixed into the canonical table it would be indistinguishable from one.

45 dev cells and 5 test cells, on a Colab A100. The seven-rate dev pass took 135.5 minutes; §5
explains the last two rates.

---

## 3. The table

**SIB-200 Yoruba topic classification, 1,056 steps, dev-selected, five seeds on test.**
Chance is 1/7 = 0.143.

| | macro-F1 | sd | 95% CI | lr | per-seed |
|---|---|---|---|---|---|
| **from-scratch, ours (33.8M)** | **0.688** | 0.024 | [0.631, 0.734] | 3e-5 | 0.667, 0.695, 0.655, 0.712, 0.712 |
| mmBERT | 0.582 | 0.023 | [0.518, 0.635] | 7e-5 | 0.590, 0.586, 0.538, 0.594, 0.604 |
| our architecture, untrained | 0.429 | 0.036 | [0.379, 0.468] | 1e-4 | 0.408, 0.372, 0.468, 0.430, 0.466 |
| XLM-R's architecture, untrained | 0.382 | 0.017 | [0.330, 0.425] | 3e-5 | 0.396, 0.377, 0.394, 0.395, 0.350 |
| XLM-R | 0.358 | **0.161** | [0.311, 0.398] | 3e-5 | 0.543, 0.386, 0.409, 0.396, **0.057** |

This supersedes the SIB-200 tables in [report 06](06-when-a-number-is-not-a-result.md) and
[report 08](08-what-the-tokenizer-costs.md) §2.

---

## 4. Three things moved

**The headline got bigger and crossed the floor.** Ours over mmBERT is **+0.106**, against 0.059
in report 06 and 0.071 in report 08. For the first time it clears the project's own 0.06
resolution floor on 204 test items. The intervals still overlap — by 0.004 — so the wording does
not change: **ahead, not beats.** A 33.8M model trained on 64M tokens of one language is ahead of
one trained on 3 trillion tokens across 1,800.

**XLM-R fell below its own untrained architecture.** Report 08 had it +0.039 above and said that
was an upper bound. Symmetric selection gives **−0.024**.

The useful part is that this survives the obvious objection. One of XLM-R's five seeds collapses
to 0.057, below chance, so 0.358 is a mixture rather than a mean. Discard that seed outright — the
most generous treatment anyone could ask for — and the remaining four average **0.434**, which is
+0.052 over the control and *still* inside the 0.06 floor. The conclusion does not depend on how
the failed seed is handled:

| treatment | XLM-R | control | gap |
|---|---|---|---|
| all five seeds | 0.358 | 0.382 | −0.024 |
| the four that trained | 0.434 | 0.382 | +0.052 |

Whatever XLM-R learned from 100 languages does not reach Yoruba topic classification.

**Selection inflated exactly one row.** Comparing each arm's dev-selected cell against the best
cell on test:

| | dev-selected | best-on-test | delta |
|---|---|---|---|
| from-scratch ours | 0.688 | 0.688 | +0.000 |
| mmBERT | 0.582 | 0.582 | +0.000 |
| our architecture, untrained | 0.429 | 0.429 | +0.000 |
| XLM-R's architecture, untrained | 0.382 | 0.382 | +0.000 |
| XLM-R | 0.358 | 0.408 | **+0.049** |

Four arms agree on the argmax between dev and test; XLM-R does not. So the practice this report
exists to correct was, in the end, harmless everywhere except the single row whose conclusion
depended on it — which is the least comfortable place for it to have mattered.

Read that delta as *"did dev and test pick the same cell"* rather than as a clean estimate of
selection inflation. It is bounded below by zero only because the dev-selected cell is itself one
of the candidates it is compared against.

---

## 5. The grid edge, and why the last two rates exist

The sweep first ran seven rates, stopping at 1e-4. A check written before the data existed fired:
`untrained ours` selected **1e-4, the top of the grid, and was still climbing** — 1e-4 at 0.415
against 7e-5 at 0.405. A winner at the boundary is not a winner. It says the optimum is outside
the range, which makes that arm a lower bound and every gap measured over it an upper bound.

Three of this project's sweeps had already ended that way. The grid was extended to 2e-4 and 3e-4
**for all five arms**, not only the one that ran out of room: extending just the arm that hit the
boundary is the same selection asymmetry this report is about, one level up.

Nothing moved. Every arm kept its pick, and `untrained ours` is now properly bracketed —
2e-4 falls to 0.361 and 3e-4 to 0.154. The extension bought a deleted caveat rather than a changed
number, which is the outcome a boundary check should usually have.

---

## 6. The seed that was already on disk

XLM-R's collapsed seed is not new. The record `lr3e-05` with scores
`[0.543, 0.386, 0.409, 0.396, 0.057]` was committed on **7 August** and sat in the repository for
two days. Nothing was broken and nothing was hidden — report 08 quoted XLM-R's 1e-5 cell, so this
one was never the reported number, and the row was only ever read as a mean.

What surfaced it was not the sweep. It was a chance check: any cell whose seeds straddle 1/k is a
mixture, and its mean describes no run that happened. Worth noting which instruments *cannot* see
this. The 95% CI on that cell is [0.311, 0.398] — **narrower** than the seed spread implies,
because the bootstrap resamples test items with predictions pooled across seeds and is blind to
variation between them. A tight interval is not evidence of a stable number.

XLM-R is also the only arm that is unstable *within* its working mode: its four trained seeds span
0.157, against 0.046–0.096 for every other arm. And the most reproducible row in the whole table
is the untrained control, at sd 0.017. A randomly initialised encoder is a steadier predictor of
Yoruba topic than XLM-R is.

---

## 7. What this does to the study's claims

| claim | status |
|---|---|
| A 33.8M from-scratch Yoruba model is ahead of mmBERT on topic classification | **Hold, and stronger.** +0.106 under symmetric dev-split selection, clearing the 0.06 floor for the first time. Intervals still overlap by 0.004, so still *ahead*, not *beats*. |
| XLM-R's pretraining is worth +0.039 over the same architecture untrained | **Withdrawn — sign reversed.** −0.024 with all seeds, +0.052 discarding the collapsed one. Inside the floor either way. |
| Test-set selection was inflating the study's downstream numbers | **Partly.** Exactly one row, XLM-R's, by 0.049. The other four arms pick the same cell on dev and on test. |
| The from-scratch win is an artifact of unequal learning-rate tuning | **Withdrawn.** Every arm now gets the same nine rates, chosen the same way, on data it is not scored on. |
| SIB-200 is substantially solvable from its 701 labels alone | **Hold, and sharpened.** A properly tuned untrained encoder reaches 0.429, which is 62% of the best score in the table. |

---

## 8. What is still open

The 0.06 floor is doing a lot of work in this report and it is an estimate, not a measurement —
204 test items is simply not many. Three of the five gaps here sit near it.

The untrained control at 0.429 is now a dev-selected optimum rather than a default, which makes it
usable as the SIB-200 floor.

> **Closed on 12 August, and this paragraph is kept because the fix is the interesting part.** When
> this was written the MasakhaNER floor was *not* in that state: mmBERT, XLM-R and the from-scratch
> model were all best-of-a-sweep there while the control was a single cell at 3e-5 — the same
> asymmetry this report had just removed from SIB-200, still present on the other task. It has since
> been swept over **twelve learning rates** (`study_ner_control_sweep.py`,
> `runs/ner_control_sweep.json`) and peaks at **0.6261 at 3e-4**, so the control is now selected the
> way its baselines are. The **0.4140** this report and the summary table carried for a fortnight was
> that sweep's 3e-5 cell — a rate one tenth of the best, quoted only because nobody had swept it.
> Fixing it **halves the from-scratch model's gap to the floor, 0.423 → 0.211**, which is the sort of
> movement that should make anyone nervous about a number nobody has swept.
