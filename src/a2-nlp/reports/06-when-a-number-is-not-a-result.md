# When a number is not a result

*A2-NLP · August 2026 · the first downstream runs, and two numbers that were not measuring*

Forty-three pretraining runs exist in `runs/`. Until this week, zero downstream runs did — and the
group's claim is a downstream score. Every finding in
[report 04](04-the-language-gradient.md) and [report 05](05-when-data-stops-mattering.md) lives in
loss-space, on a project that has twice caught validation loss failing to predict fine-tuned
quality.

This note is the first pass at closing that gap. Five downstream cells are now recorded, with
seeds and pooled bootstrap intervals, in `runs/ft_*.json`. But the headline is not the new
numbers — it is that **two of the numbers the study already quotes downstream were not
measurements of what they appear to measure.** One is a baseline that never trained. The other is
a tokenizer comparison run on text in the wrong Unicode normalisation form, which reversed its own
conclusion.

Both are fixed. The thesis survives one of them and is strengthened by the other.

---

## 1. The harness moved into a module

The fine-tuning half lived inline in `POC_v4_factory.ipynb` — `macro_f1`, `pooled_ci`,
`finetune_once`, the SIB-200 and MasakhaNER loaders. The open questions are all *comparisons*, and
each wants its own small notebook. Copied into three notebooks, that becomes three harnesses
inside a week, and a seed count changed in one silently makes its numbers incomparable with the
others.

It is now `ft_api.py`, alongside `mlm_api.py` and mirroring its shape. One behavioural change was
made deliberately, and it matters more than it looks:

**The epoch loop became a fixed step budget.** The notebook trained `for _ in range(FT_EPOCHS)`.
On 701 SIB-200 examples at batch 16 that is 352 updates; on 6,876 MasakhaNER sentences it is
3,440. So the existing SIB-200-versus-NER comparison varied the update budget by 10× at the same
time as the task and the label count — and the planned experiment that subsamples MasakhaNER to
701 sentences would, under that harness, have cut its budget by 10× as well, then reported the
budget cut as a label-count effect.

The defaults (352 and 2,150) equal what the old loop spent on the full splits, so existing numbers
reproduce; what changes is only the case the notebook got wrong. There is no early stopping: the
smallest conditions have no dev set to stop on.

Every cell now writes one record per `(model, task, lang, n_train, lr, steps, max_length,
normalize, seed-set)` to `runs/ft_*.json`. The last three are in the key because each of them
moves the number, and a key missing one does not collide so much as silently return the wrong
run's result.

---

## 2. XLM-R never learned SIB-200

The study quotes XLM-R at **0.127** macro-F1 on Yoruba topic classification against mmBERT's
0.537. [Report 05](05-when-data-stops-mattering.md) already flagged it as *"almost certainly a
fine-tuning failure"* on the grounds
that the same model scores 0.843 on Yoruba entity recognition, and a model with no usable Yoruba
cannot do that. Re-run through the extracted harness with three seeds, it is worse than a failure.

| model | mean | seed sd | per-seed |
|---|---|---|---|
| XLM-R base | 0.073 | 0.022 | 0.057, 0.057, 0.105 |
| mmBERT base | 0.529 | 0.021 | 0.523, 0.507, 0.558 |

Now the reference points. SIB-200's Yoruba test split is 204 items across 7 classes, distributed
19 / 17 / 22 / 30 / 51 / 25 / 40:

| behaviour | macro-F1 |
|---|---|
| collapse to a single class | 0.022 – **0.057** |
| collapse to two classes | 0.038 – 0.099 |
| uniform random guessing | 0.133 ± 0.022 |
| perfectly balanced chance | 0.143 |

**0.057 is exactly the score for predicting the majority class on every item** — precision
51/204, recall 1.0, F1 0.400, averaged across seven classes. Two of three seeds hit it to three
decimal places. The third, at 0.105, sits in the two-to-three-class range.

All three seeds score below uniform random guessing. Nothing was learned in any of them.

**It also scores below the untrained control.** Random init on the same task, same harness, three
seeds, is **0.107** — reproducing the ~0.10 already on record. A fully pretrained multilingual
encoder is finishing below an untrained one, which is as clear as this gets: 0.127 was never a
measurement of XLM-R's Yoruba coverage.

This is not a property of the harness. mmBERT, through the identical code path, lands at
0.507–0.558 across the same seeds, and reproduces the notebook's 0.537 to within 0.011. The
failure is model-specific and looks like the textbook degenerate fine-tune, where a freshly
initialised classification head never receives usable gradient.

### The confidence interval does not catch this

The project's stated practice is to lean on pooled bootstrap CIs. On this failure mode they are
worse than useless:

| cell | 95% CI | width |
|---|---|---|
| XLM-R, SIB-200 — collapsed | [0.062, 0.084] | **0.022** |
| mmBERT, SIB-200 — learned | [0.461, 0.581] | **0.120** |

The collapsed run's interval is **5.5× narrower** than the working one's. A model that answers
"class 4" for every item is consistently wrong, so resampling test items finds almost no
variation, and the interval reads as precision. Nothing about [0.062, 0.084] signals that the
number inside it is an artifact.

Seed spread does not catch it either — the sd is 0.022, comparable to mmBERT's 0.021. Only the
comparison against chance does, which is why `ft_api` now derives it rather than leaving it to
whoever reads the table.

It is also not stable enough to be quoted as anything. The same configuration has now produced
0.127, 0.128 and 0.073 on separate runs — a spread wider than the ~0.06 macro-F1 floor the project
uses to decide whether a difference on 204 items is real.

`ft_api` now derives `chance = 1/k` for classification cells, records `chance` and `degenerate`
beside the score, and says so on stdout with the per-seed values, so a collapsed run is read as an
absent number rather than a small one.

---

## 3. MasakhaNER ships decomposed, and that reversed the tokenizer result

[Report 04](04-the-language-gradient.md)'s central measurement is that a multilingual vocabulary
is not inherently worse — it
costs English 1.04×, Indonesian 1.00×, Mandarin 0.95× — but costs Yoruba **1.65×**. That was
measured on FineWeb-2, the pretraining corpus. The obvious next step is to check it on the
datasets the study actually evaluates on.

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
its fingerprint off `15abd33de5af` and make every one of the 43 pretraining runs incomparable.
`ft_api` normalises to NFC on the way in, records the setting on every result, and puts it in the
record key so an NFC run and a raw run cannot land on the same file.

**The thesis comes out stronger.** With the encoding corrected, XLM-R costs:

| corpus | XLM-R costs |
|---|---|
| SIB-200 Yoruba | 1.60× |
| MasakhaNER Yoruba | 1.74× |
| FineWeb-2 Yoruba (report 04) | 1.65× |

The tokenizer gradient now holds on the *evaluation* data, not only on the pretraining corpus.
That is a better-founded claim than report 04 could make on its own.

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

That is worth stating precisely because it cuts both ways. It confirms the window is not the
mechanism — and it also means re-running the NER table at `max_length=256` buys accuracy of about
0.003. Worth doing for tidiness, not for correctness.

The same fact stated as capacity survives, and is what the poster should say instead:

> In a 128-token window, a Yoruba-fitted vocabulary holds **77 Yoruba words**. XLM-R holds **44**.
> mmBERT holds 48.

One observation that cuts the other way and should not be buried. At `max_length=128`, XLM-R loses
the tail of 13.3% of MasakhaNER sentences and the from-scratch model loses 0.2% — and XLM-R still
wins entity F1 0.843 to 0.698. That comparison is confounded (the models differ in parameter
count, pretraining corpus and budget, not only in vocabulary), but it is the closest thing to a
direct test the project currently has, and it does not support tokenizer fit determining the
downstream score.

---

## 5. The downstream table, and what it does to the study's claims

Every cell below is three seeds, NFC text, fixed step budget, no early stopping, with a 95%
interval bootstrapped over test items and pooled across seeds. This is the canonical table: the
proposal and the poster should quote it rather than any number printed in a notebook.

**SIB-200 Yoruba topic classification** — 701 train / 204 test, 7 classes, 352 steps, batch 16.
Chance is 0.143 and the untrained control is 0.107, so both are printed as rows rather than left
to the reader:

| model | lr | macro-F1 | seed sd | 95% CI | |
|---|---|---|---|---|---|
| mmBERT base | 2e-5 | **0.529** | 0.021 | [0.461, 0.581] | |
| from-scratch 33.8M | 5e-5 | *not re-measured* | | | checkpoint not available off the workstation |
| random init (control) | 5e-5 | 0.107 | 0.011 | [0.086, 0.128] | at chance, as intended |
| XLM-R base | 2e-5 | 0.073 | 0.022 | [0.062, 0.084] | **collapsed — not a result** |

**MasakhaNER 2.0 Yoruba entity recognition** — 6,876 train / 1,964 test, 2,150 steps, batch 16,
lr 3e-5:

| model | entity F1 | seed sd | 95% CI | |
|---|---|---|---|---|
| mmBERT base | **0.851** | 0.007 | [0.837, 0.865] | |
| XLM-R base | 0.841 | 0.008 | [0.827, 0.856] | |
| from-scratch 33.8M | *not re-measured* | | | the one row the encoding fix can have moved |

Two things to read off it. The two multilingual baselines are **indistinguishable on NER** —
0.841 against 0.851, intervals overlapping across most of their width — so the study's NER task
does not separate them. And the from-scratch row is missing from both tables for the same reason:
no checkpoint exists outside the workstation.

### What this does to the study's claims

| claim | status |
|---|---|
| XLM-R scores 0.127 on Yoruba topic classification | **Withdraw.** A majority-class predictor scores 0.057 and the untrained control scores 0.107; the run never trained. |
| mmBERT beats XLM-R by 0.41 on topic classification | **Withdraw.** mmBERT trained and XLM-R did not. |
| The from-scratch model beats XLM-R on topic classification | **Withdraw** as evidence of tokenizer fit. It beat a baseline that never ran. |
| A multilingual vocabulary costs Yoruba ~1.65× | **Hold, strengthened.** 1.60× and 1.74× on the two evaluation sets. |
| 65% of the context window is spent on fragments | **Withdraw.** Nothing is truncated, and a 14.5-point change in truncation is worth 0.003 F1; use the 77-vs-44-words framing. |
| MLM loss does not predict downstream quality | **Hold, but one leg is weaker.** One of its two supports was the SIB-200/NER divergence, and the SIB-200 half of that is the collapsed run above. |
| XLM-R and mmBERT both score ~0.84 on Yoruba NER | **Hold.** 0.841 and 0.851 under NFC, three seeds, overlapping CIs — the two are indistinguishable on this task. |
| The from-scratch model loses NER by 0.145 | **Still open, but narrower.** Both baselines are now confirmed insensitive to the encoding fix (±0.003). The from-scratch tokenizer is the one whose fertility moved 47%, so its row is the only one the fix can plausibly have changed — and it is the one that needs a checkpoint to re-measure. |

---

## 6. What is not settled

- **Whether XLM-R can be fine-tuned on 701 examples at all.** The sweep should extend the learning
  rate *downward* from 2e-5, vary the step budget alongside it, use five seeds, and report the
  fraction of seeds clearing 0.133 rather than the mean — averaging collapsed and converged seeds
  describes neither. A negative result across the whole grid is still a finding, and a more honest
  poster line than 0.127.
- **The from-scratch row on MasakhaNER under NFC.** The two baselines have now been re-measured and
  did not move (0.841 and 0.851, within 0.003 of their pre-fix values). The from-scratch model is
  the one whose tokenizer changed by 47%, and it is the one still unmeasured, because no
  from-scratch checkpoint exists outside the workstation.
- **Whether the tokenizer penalty causes anything.** Nothing in the project establishes this. The
  swap experiment — same architecture, same corpus, same budget, one model on `yor-bpe16k` and one
  on XLM-R's vocabulary — is the only design that isolates it, and §4 above raises rather than
  lowers the stakes on running it.
- **The 86M checkpoints downstream.** Still untested, and now gating two questions rather than one.

---

## 7. Reproducing

The score numbers come from `ft_api.evaluate`, which writes `runs/ft_*.json`; `ft_api.results()`
and `ft_api.table()` read them back. The collapse reference points in §2 are computed directly
from the SIB-200 test labels. The tokenizer measurements in §3 and §4 are deterministic CPU
arithmetic over the committed `tokenizers/yor-bpe16k` and the two hub tokenizers, and need no GPU;
`ft_api.token_lengths` and `ft_api.decomposition_report` produce them.

Fine-tuning here ran on a Colab runtime with an **NVIDIA A100-SXM4-80GB**, not on the CSED 504
workstation's RTX PRO 6000 Blackwell. Wall-clock figures therefore do not transfer from reports
03–05, which are all Blackwell measurements; the working multiplier is roughly 2–3× slower here.
The card is recorded in the `gpu` field of every result record, and it is worth checking rather
than assuming — Colab does not always allocate the same hardware, and this session did not get
what the handoff notes predict.

Per seed, three seeds per cell:

| cell | steps | s/seed |
|---|---|---|
| MasakhaNER, mmBERT base | 2,150 | 254 |
| MasakhaNER, XLM-R base | 2,150 | 196 |
| SIB-200, mmBERT base | 352 | 41 |
| SIB-200, XLM-R base | 352 | 32 |
| SIB-200, random init (33.8M) | 352 | 10 |

The whole table is about 27 minutes of fine-tuning. Nothing in §2–§5 depends on the hardware — the
tokenizer measurements are CPU arithmetic and the scores are comparisons within a fixed budget —
but any *projection* from these timings needs the A100 figure, not a Blackwell one.
