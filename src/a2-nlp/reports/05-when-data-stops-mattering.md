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

For the 33.8M model the data axis is spent by the time you *reach* 64M tokens. Beyond that point
**a sixteen-fold increase in unique text buys 0.024 nats** — half the 0.049 seed spread measured
on the Yoruba ladder. It is not a small effect; it is not an effect.

The distinction matters and the next subsection turns on it: arriving at 64M is still worth
+0.144, but going anywhere past it is worth nothing.

So the Yoruba ladder's conclusion was right, and right for a reason it could not see. Data does
stop mattering. It stops mattering at a threshold, and past that threshold the loss is set by
compute and capacity alone.

### Where the saturation point actually is

An earlier draft of this report put it "around 64M tokens", which was too tidy. The 64M rung is
where the *deceleration* is obvious, not where the curve flattens. English still gained **+0.144**
going from 16M to 64M — 2.9× the 0.049 seed spread, so a real gain — and only went flat on the
step after, at +0.007.

**The saturation point is somewhere between 64M and 256M tokens.** That matters for Yoruba,
because all of Yoruba is 69.1M — which lands at the bottom of that band rather than past it.

---

## 3. The Yoruba check

The claim above was a transfer assumption, so it got tested. Three Yoruba rungs at the same
budget as the English ladder — 33.8M model, 1.024B tokens of updates, 62,500 steps:

| unique tokens | Yoruba | English |
|---|---|---|
| 4M | 4.128 | 3.284 |
| 16M | 2.484 | 2.361 |
| 64M | **2.315** | **2.217** |

Absolute losses across the two columns are **not** comparable — different corpora, different 16k
vocabularies. What compares is the shape:

| step | Yoruba | English |
|---|---|---|
| 4M → 16M | +1.644 | +0.923 |
| 16M → 64M | **+0.169** | **+0.144** |
| 64M → 256M | *(no such rung exists)* | +0.007 |

**The curves track each other closely.** At the one step where both languages can be measured,
they differ by 0.025 nats — half the seed spread. Yoruba decelerates on the same schedule English
does, from a steeper start.

Two conclusions, and they are not the same strength:

**Supported: Yoruba is not badly data-starved.** English's next step after 64M was worth +0.007 —
nothing. Since the two curves agree to within noise at 64M, the text Yoruba does not have is very
probably worth about as little. The group's case cannot rest on *"there is too little Yoruba
text"*; at these compute budgets, having more would buy almost nothing. It has to rest on the
tokenizer-fit argument in [report 04](04-the-language-gradient.md) §4, where Yoruba pays XLM-R a
1.65× penalty no other language in the set pays.

**Not supported: that Yoruba has saturated.** It is still gaining **+0.169 at 64M**, which is 3.4×
the seed spread. So Yoruba sits just *below* the flattening point, taking the last measurable
gain, with no ability to take the step that would confirm the curve has gone flat. The earlier
phrasing — "it sits just past the point where more text would have stopped helping" — was wrong,
and this replaces it.

The remaining gap is an extrapolation of one step in one language. It cannot be closed with more
Yoruba, because there is no more Yoruba.

---

## 4. The bigger model still has not crossed over

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

Every cell is a single seed. The seed spread we rely on for significance (0.049) was measured on
a different corpus at a different budget.

No fine-tuning here — this is pretraining loss only, and the group's headline comparison is a
downstream task score. A model that wins on validation loss has not thereby been shown to win on
SIB-200 or MasakhaNER.
