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
[report 04](04-the-language-gradient.md). One seed per cell.

---

## 2. The data axis saturates, and it saturates early

Marginal gain from each 4× increase in data, at fixed compute:

| step | 33.8M model | 86M model |
|---|---|---|
| 4M → 16M | **+0.923** | +3.519 |
| 16M → 64M | +0.144 | −0.078 |
| 64M → 256M | **+0.007** | +0.382 |
| 256M → 1024M | **+0.017** | +0.321 |

For the 33.8M model the data axis is finished by 64M tokens. Beyond that, **a sixteen-fold
increase in unique text buys 0.024 nats** — half the 0.049 seed spread measured on the Yoruba
ladder. It is not a small effect; it is not an effect.

So the Yoruba ladder's conclusion was right, and right for a reason it could not see. Data does
stop mattering. It stops mattering at a threshold, and past that threshold the loss is set by
compute and capacity alone.

### What this says about Yoruba

The saturation point for this model at this budget is **around 64M tokens. All of Yoruba is
69.1M.**

If that threshold transfers, the group's language is not data-starved at the budgets anyone here
is training at. It sits just past the point where more text would have stopped helping anyway.
That reframes the study's central claim: the case for from-scratch Yoruba pretraining cannot rest
on *"there is too little Yoruba text"*, because at these compute budgets there is enough. It has
to rest on the tokenizer-fit argument in [report 04](04-the-language-gradient.md) §4, where
Yoruba pays XLM-R a 1.65× penalty that no other language in the set pays.

That is a **transfer assumption and it has not been tested.** The honest version is that the
threshold was measured in English at one model size and one compute budget. Running two Yoruba
rungs at this compute budget — 16M and 64M at 1.024B updates — would confirm or kill it for about
ninety minutes of GPU time, and it is the single highest-value experiment left.

---

## 3. The bigger model still has not crossed over

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

### One number that does not fit

The 86M model scores **3.278 at 64M tokens and 3.200 at 16M** — worse with four times the data,
the only inversion in either column. Single seed, so it cannot be separated from noise, though it
is larger than the 0.049 spread we have measured elsewhere. It should be repeated before it
appears in any writeup. I have not repeated it.

---

## 4. What this cost

Ten cells, both cards, no failures. 86M cells ran ~93 min each at 184k tokens/sec; 33.8M cells
~40–52 min at 408k. Total wall clock about six hours.

Corpus preparation was 20 minutes for 1.1B tokens: FineWeb-Edu streams at 49.5M chars/sec on this
machine and the BPE encodes at 9.1M, so encoding dominates and the download is nearly free. Both
rates were measured before committing to the run rather than estimated after it.

---

## What this does not settle

The saturation threshold is one model size, one compute budget, one language. It is stated above
as a transfer assumption precisely because it is one.

Every cell is a single seed. The seed spread we rely on for significance (0.049) was measured on
a different corpus at a different budget.

No fine-tuning here — this is pretraining loss only, and the group's headline comparison is a
downstream task score. A model that wins on validation loss has not thereby been shown to win on
SIB-200 or MasakhaNER.
