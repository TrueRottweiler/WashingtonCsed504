# Build sheet: the bottom board

*What goes in each cell, what it says, and which figure to drop in. Report 09 is the long form.
This is the board — nine cells and a strip. One is not a subset of the other, so this sheet is the
translation, and it is the thing to print and keep beside you while laying out.*

Each week below is written to be **read by a student who has taken one or two ML courses and has
never run a hundred models**. That is the audience the board is for. Where a cell states a number,
this sheet also says why anybody would ask the question in the first place, what we expected, and
what the answer changed — because a number without the question that produced it is trivia, and
the point of the board is that the questions came in an order.

---

## Format

Nine cells in a three-by-three grid, plus a full-width strip along the bottom, title block above.
The board is **3 ft × 4 ft — 91.4 × 121.9 cm**, portrait. Not A0, which this sheet used to say:
A0 is 84.1 × 118.9, so the real board is 7 cm wider and 3 cm taller.

With 6 cm margins, a 14 cm title block, a 20 cm bottom strip and 2.5 cm gutters, each grid cell is
**24.8 × 22.8 cm** and the strip is **79.4 × 20 cm**. The cell size the earlier version quoted —
25 × 22 — survives the change, so nothing already laid out to it has to move.

**Type.** At 24.8 cm wide a cell holds about 90–120 words at 24 pt, which is the smallest anyone
will read standing at arm's length. The big number wants 90–110 pt. Treat the word budget as a
hard limit rather than a target: the failure mode for a board like this is a cell nobody finishes.

**The prose below is not what goes on the board.** It is the source you cut the 90–120 words from,
and the thing you will want in your head when somebody stands in front of the cell and asks a
question. Write the panel from it; do not print it.

---

## Title block

> ## CSED 505: Building a Model Factory
> *the course that would come after 504 — ten weeks, derived from one term of building one*
>
> **501** the statistics · **502** the mechanics · **503** the language stack · **504** scale
> **All four end when a model finishes training. None covers needing a hundred models and having
> to believe the differences between them.**

Bottom-right of the title block, small: *Jeffrey Stall · A2-NLP · the upper board is the
experiment this machinery served.*

---

## The arc, in one paragraph

Read the nine cells left to right, top to bottom, and they are one argument. The first three build
a factory: a unit to measure runs in, the speed that makes running them repeatedly affordable, and
records that survive the person who made them. The next three turn that factory on its own
measurements and find three of them wanting — a significance rule that was half a rule, a metric
that predicts one task and not another, and a unit that was not a unit. The last three spend the
corrected instruments on real questions, and two of the three answers are *no*. That shape is the
board's actual claim: **the machinery is not overhead on the science, it is what makes the science
falsifiable.**

---

## Week 1 — What does a run cost, and in what unit?

**Big number:** `62,500 steps = 1.024B tokens` · **Figure:** *needs one — see Gaps*

**Problem.** Every course up to this point measured training in epochs, and an epoch is a
perfectly good unit as long as the dataset is a constant. The moment the dataset becomes a
variable — which is the first thing any scaling study does — an epoch stops meaning anything
comparable between two runs.

**Hypothesis.** We assumed, without stating it, that "train for N epochs" was a fair way to give
two models the same amount of work. That assumption is invisible precisely because every prior
course held the dataset fixed, so nobody had ever needed to question it.

**Approach.** Vary the corpus from 4M to 1,024M tokens and hold the step budget fixed, then look
at what an epoch has become. At 4M tokens the model sees the corpus sixty times; at 1,024M it sees
a quarter of it once. The unit that stayed meaningful across all of it is **tokens of updates**:
steps × batch × sequence length, which is 62,500 × 128 × 128 = 1.024 billion regardless of what
corpus is behind it. That number is not a choice anybody made — it is a compute budget divided by
a batch size, and it was inherited from a notebook before anyone checked what it implied.

**Results.** Every run in this project is quoted in tokens of updates. The step count is a
consequence, not a setting, and the two are only interchangeable while the batch size is frozen —
which is why the batch is 128 in 99 of 105 runs and why the fleet queues are written in update
tokens rather than steps.

**Learning.** Epochs stop being a unit the moment the dataset is a variable, and scaling studies
make it a variable by definition. Pick the unit that survives the thing you intend to vary, and
write it into the tooling so nobody has to remember. The cost of getting this wrong is not a bad
number — it is two numbers that look comparable and are not.

**What this makes possible.** You cannot say a run got faster until you can say what a run *is*.
Week 2 is the measurement, and it needs this unit to be meaningful.

---

## Week 2 — Why optimize before anything needs it?

**Big number:** `2.07×` — of which only **1.32×** is efficiency · **Figure:**
`14-where-the-speedup-came-from.svg`

**Problem.** Nothing needed to be fast yet. The study had a handful of runs, they finished
overnight, and the received wisdom — profile when it hurts — says optimizing at that point is
premature.

**Hypothesis.** We expected the speed work to buy hours, and hours were not scarce. The
counter-argument, which turned out to be the real one, is that speed does not only shorten the
work you already planned; it changes which work you are willing to begin.

**Approach.** The 2.07× on the board is the *last two steps* of about a year, and quoting it
alone badly undersells what a student would have to learn. The path ran from hand-written NumPy in
CSED 502, through PyTorch in 503, to a workstation whose two cards sit at 91% and 93% under load.
Almost none of it was clever; nearly all of it was measuring instead of assuming, and roughly a
third of what was tried turned out not to be worth keeping. The full list is below because the
*shape* of it is the teaching point: the wins are concentrated in the data path and in precision,
not in the model.

**Results.** 25.2 minutes → 12.2 minutes on the same four cells, decomposed honestly: **1.32× is
real efficiency** — less GPU-time for the same work — and the rest is a second card doing the same
GPU-minutes in parallel. The project has since spent 143.3 GPU-hours; without the throughput work
it would have been roughly double, which on realistic evenings is weeks rather than nights.

**Learning.** The hours saved are not the point; the experiments you become willing to start are —
four of the five corrections on this board came from a cheap re-run somebody did on a hunch, and a
hunch is not worth acting on at 25 minutes a cell. Conflating efficiency with parallelism is how a
project claims 2× and bought 1.3×, so decompose any speedup before quoting it. And optimize early
when what you are buying is *iteration* rather than throughput.

**What this makes possible.** Weeks 4 through 9 each rest on running one cell three to fifteen
times. At 25 minutes a run nobody would have done that, and every result below would have been a
single seed with an anecdote attached.

### The year, in twenty changes

Measured on this hardware unless marked. "Kept?" matters as much as the number — a third of these
were tried and rejected, and a list of only the wins would be marketing rather than a log.

#### The data path — where most of the win actually is

| # | change | measured impact | kept? |
|---|---|---|---|
| 1 | **Delete the DataLoader; hold the dataset resident in GPU memory.** One CPU core augments ~4,000 img/s while the card trains at ~13,000 — the CPU was starving the GPU 3× and the card spent its life waiting on Python. | CIFAR: 13.8k img/s (workers, bs128) → **18.5k** (resident, bs512), at far higher utilization | yes |
| 2 | **Store images as uint8, convert to float per batch.** Float residency would be 4× the memory for no gain; the conversion is nearly free on-device. | CIFAR-100 train = **147 MB**; ImageNet-32 = 3.9 GB on a 96 GB card | yes |
| 3 | **Augment on the GPU** — crop, flip, erase, normalize as tensor ops on data already in VRAM. | no PIL, no JPEG decode, no Python in the inner loop, no per-batch host→device copy | yes |
| 4 | **Tokenize once, up front, into a flat array.** Training never touches raw text, the tokenizer, or the datasets library again — it indexes. | prep is **22 s** against 85 min of training: a 232× ratio | yes |
| 5 | **Choose the token dtype from the vocabulary**, not a fixed uint16. | yor at 16k vocab: 2 bytes/token, **0.15 GB**. yor_xlmr at 250k: 4 bytes, **0.49 GB** — same text | yes |

#### Precision and kernels

| # | change | measured impact | kept? |
|---|---|---|---|
| 6 | **bf16 autocast instead of fp16.** Same speed, but bf16 keeps fp32's exponent range — no GradScaler, no loss-scale underflow to diagnose at 3 a.m. | **1.34×** on a power-capped sm_89 laptop (with #7); ~**+2%** on Blackwell | yes |
| 7 | **channels_last (NHWC) for CNNs — bf16 only.** | see #6 — and the trap: fp16 + channels_last drops into a pathological cuDNN path, **3.5× slower** on sm_89 | yes, guarded |
| 8 | **channels_last for ViTs** | a no-op for a transformer; pure overhead | **no** |
| 9 | **TF32 for matmul and cuDNN** | free on Ampere and later | yes |
| 10 | **`cudnn.benchmark = True`** — let cuDNN autotune per shape | worth it when shapes are fixed, which ours are | yes |
| 11 | **Fused optimizers** — `SGD(nesterov, fused=True)`, `AdamW(fused=True)` | one kernel instead of a chain of elementwise ops | yes |
| 12 | **`torch.compile`** | **unavailable**: Windows PyTorch wheels ship without Triton, and `triton-windows` is unreliable on Blackwell | **no** |
| 13 | **Attention backend** | already `sdpa`; nothing to win | **no change** |

#### Host-sync discipline

| # | change | measured impact | kept? |
|---|---|---|---|
| 14 | **Every `.item()` is a GPU→CPU sync.** Accumulate loss and top-k counts in on-device tensors; pay the sync once per epoch, not once per batch. | **~7%** on CIFAR | yes |
| 15 | **The same discipline at MLM scale** | only **2%** — an MLM step is ~23 ms against a CNN step's fraction of a millisecond, so the sync is genuinely in the noise. **Kept the `.item()`**: a readable live loss is worth 2% | **rejected** |
| 16 | **`zero_grad(set_to_none=True)`** | skips a full zero-fill of every gradient buffer | yes |

#### Batching and scheduling

| # | change | measured impact | kept? |
|---|---|---|---|
| 17 | **Batch 64 → 128** | **1.33×** predicted from the sweep, **1.31×** measured. Throughput *peaks* at 128–256 and then falls — batch 2048 is slower than 128 while using 89.7 GB. There is no reward for filling the card | yes |
| 18 | **Memory-saving tricks** (gradient accumulation, checkpointing) | peak allocation at the shipped config is **3.3 GB of 96**. Nothing here is memory-bound; this would have been wasted effort | **no** |
| 19 | **Use the second card** | utilization **91% and 93%** at 300 W each, against one card busy and one idle | yes |
| 20 | **Order the queue longest-budget-first.** Card 1 finished its long cell at 7.9 min then sat at 1–3% for four minutes while card 0 worked the tail — both short cells had landed behind a long one. | projected **2.55×** against the observed 2.07× | yes |

**Two more that are not speed-ups and matter more than several that are.** The compute axis moved
from *optimizer steps* to *tokens of updates*, because steps are not comparable across batch sizes
— raising the batch while holding steps fixed would have been a different, larger experiment that
scores better, and reporting it as a speed-up would have been wrong. And `pretrain(reuse=True)`
means re-running a notebook to iterate on the fine-tuning gates no longer spends twenty minutes
reproducing checkpoints that are already on disk.

**What the shape of that list says.** Fourteen kept, six rejected or unavailable. The largest
single win was deleting a component (#1) rather than tuning one, the second largest was a dtype
choice (#6), and the model architecture was never touched. Also worth a student's attention: #7
and #15 are the same idea evaluated on two workloads and answered differently, which is the whole
reason the list has a "kept?" column.

---

## Week 3 — What makes a record survive you?

**Big number:** `fingerprint 15abd33de5af` · **Figure:** `07-dashboard.png`

**Problem.** Two runs silently overwrote each other because their filenames did not encode a
setting that had changed between them. Separately, two people prepared "the same" corpus and got
different vocabularies, and nothing in either record said so.

**Hypothesis.** We believed a descriptive filename was enough — that `yor_64M_62.5k_s0` told you
what a run was. It does, right up until you vary something the name does not mention, at which
point the name becomes a collision rather than an identity.

### What there is to keep track of

This is the part that does not fit in anyone's head, and the reason the cell exists. As of
2026-08-10:

| | count |
|---|---|
| **pretraining runs** | **197** |
| **fine-tuning records** | **278** |
| **individual fine-tuning runs behind them** | **892** |
| distinct models fine-tuned | 28 |
| corpora prepared | 22, across 17 languages |
| figures generated from those records | 16 |

The pretraining runs, by corpus — and note that the shape is lopsided on purpose, because English
is the ruler the Yoruba results are read against:

```
eng_1b  70    yor  48    hau 13    ibo 13    nya 13    swh 13    yor_xlmr 7    other 20
```

And by axis: **two model presets** (33.8M `poc`, 86M `afriberta`), **twelve seeds** (0–11),
**ten distinct step budgets** from 2,930 to 62,500, and downstream, **twelve learning rates** swept
across **two tasks** — 172 SIB-200 records and 106 MasakhaNER.

Nine hundred runs is not a number you manage by being careful. Every one of them has a corpus, a
vocabulary, a token budget, a step budget, a seed, a learning rate, a gradient-clipping value, a
normalization form, a maximum sequence length and an evaluation split — and **any one of those
silently deciding a result is a real event that has happened to this project five times.** That is
the five-constants table on the bottom strip.

**Approach.** Put **every setting that moves the number** into the record's name, and hash the
vocabulary itself rather than trusting the corpus label. Normalization goes in the tag because it
moves MasakhaNER fertility by 47%; `max_length` goes in because at 128 it truncates 13.3% of
MasakhaNER under XLM-R and at 256 it truncates none. The corpus gets a SHA-256 fingerprint of its
vocabulary — `15abd33de5af` for Yoruba — so "the same corpus" is checkable rather than
remembered. And a reuse guard refuses to hand back a cached run whose stored settings differ from
the ones asked for.

**Results.** A record is now an identity you can verify. The guard has already refused a request
that would have silently returned a clip-1.0 run to a caller asking for clip-0.5 — same cell tag,
different experiment, and the first version of the guard missed it because it treated an absent
field as agreement rather than as "cannot confirm".

**Learning.** A filename is an identity, and an identity has to include everything that changes
the answer. A vocabulary needs a hash, because two people following the same recipe do not
produce the same tokenizer. And a dashboard that shows an empty queue while both cards are at 85%
is worse than no dashboard, because it answers the question wrongly instead of declining to.

**What this makes possible.** You cannot ask whether two numbers differ until you are certain they
came from two different experiments. Week 4 is that question, and it is meaningless without this.

---

## Week 4 — Is this difference real?

**Big number:** `2.27×, not 1.0×` · **Figure:** `13-how-many-seeds.svg`
*(pair with `04-two-outcomes.svg` if the cell has room)*

**Problem.** This project's rule for believing a difference was "bigger than the cell's own seed
spread." That rule is doing something real — it correctly rejects anything smaller than the noise
— but it had never been checked against what a significance test actually requires.

**Hypothesis.** We treated the rule as symmetric: below the spread, reject; above it, accept.
Nobody had asked what multiple of the spread a difference must clear before the two arms are
genuinely separated at three seeds a side.

**Approach.** Derive the bar rather than look it up, because every number in this panel is one a
reader would otherwise have to take on faith. Four of them, in order.

**Where 0.05 comes from.** It is a convention, not a fact about the world — the probability we are
willing to accept of announcing a difference that is not there. It comes from **Ronald A. Fisher**
(1890–1962), the British statistician who built most of the machinery this panel uses, in
*Statistical Methods for Research Workers* (1925) — where he suggested one-in-twenty as a
convenient line and said so in about that many words [[1](#references)]. It stuck because it was
convenient, and nothing in the mathematics requires it. It matters here only because everything
below is calibrated to it: change α to 0.01 and every threshold in this cell moves. Stating it as
a choice rather than a law is the first honest thing a panel about significance can do.

**Where 2.27× comes from.** A two-sample t-test asks whether the difference between two means is
large compared with the noise in those means. Written as a multiple of the pooled standard
deviation, the threshold is `t* × √(2/n)`, where `t*` is the critical value of the t-distribution
at your chosen α and `n` is the seeds per arm. At three seeds a side there are `2n − 2 = 4` degrees
of freedom, and `t*` at the 97.5th percentile — two-sided 0.05 — is **2.776**. Then
`√(2/3) = 0.8165`, and `2.776 × 0.8165 = 2.267`. That is the whole derivation: **2.27**, and it
falls as seeds are added because `t*` shrinks and `√(2/n)` shrinks with it.

**Where 0.10 comes from, and why it is the sharper limit.** A permutation test makes no
distributional assumption at all — the idea is Fisher's again, from *The Design of Experiments*
(1935), and was put on a formal footing by Pitman two years later [[2](#references),
[3](#references)]. It pools the six numbers, tries *every* way of splitting them into two groups of
three, and asks how often a split as extreme as the one you observed comes up by chance. There are `C(6,3) = 20` such splits. If your actual arrangement is the most extreme
possible, exactly one split beats it in each direction, so the two-sided p is `2/20 = 0.10`. **No
arrangement of six numbers can do better.** Even if every seed of one arm beats every seed of the
other by a mile, the test reports 0.10 — and 0.10 is above 0.05, so a three-seed experiment cannot
reach significance *at all* on this test. The floor is `2/C(2n, n)`:

| seeds a side | ways to split | smallest reachable p |
|---|---|---|
| 3 v 3 | C(6,3) = 20 | **0.100** — above α, so unreachable |
| 4 v 4 | C(8,4) = 70 | **0.029** |
| 5 v 5 | C(10,5) = 252 | **0.0079** |
| 6 v 6 | C(12,6) = 924 | **0.0022** |

That table is why the swap experiment got a fourth seed: going from three to four is the cheapest
change available to what the experiment is *allowed to claim*, and it costs about 50 minutes.

**Where the 22% comes from.** There are two standard deviations. The *population* sd divides by
`n`; the *sample* sd divides by `n − 1`, because estimating the mean from the same data uses up a
degree of freedom. The ratio is `√(n/(n−1))`, which at n = 3 is **1.2247** — so the sample sd is
22% larger, and at n = 5 it is 12% larger. Our records store the population form, and every
threshold above is derived for the sample form. Feeding one to the other silently inflates every
"× the spread" figure by that factor, which is more than enough to move a claim across a line.

**Results.** Two of our own claims were sitting in the gap between 1.0× and 2.27×. The tokenizer
penalty passed the old rule at 1.4× and fails the real one. The clipping ladder's celebrated "38×
spread reduction" came from three-seed rows that could not have established anything in either
direction.

**Learning.** Our rule was half a rule: sound at rejecting things below the noise, silent about
things just above it. A pre-registered sample size sets a floor on the p-value you are allowed to
quote, and that floor is often larger than people expect. This is the instrument the rest of the
board is read through, and the first thing it did was retire two of our own numbers.

**What this makes possible.** Weeks 5 through 9 are all comparisons. Without this they are
anecdotes with decimal points.

---

## Week 5 — Is your metric the right metric, and why does one task refuse to be predicted?

**Big number:** `−0.888 against +0.303` · **Figures:** `11-metric-validity.svg` + `12-floors.svg`

**Problem.** We minimized validation loss for an entire term without ever checking that it
predicts anything we care about. Loss is the thing training optimizes, which makes it very easy to
mistake for the thing we want.

**Hypothesis.** The working assumption was the obvious one: lower validation loss means a better
model, and a model with better loss will be better downstream. If that held, one number could
stand in for two expensive evaluations.

**Approach.** Take the sixteen checkpoints that actually trained — the `val_loss < 3.1` cut,
which matters and is stated rather than hidden — and correlate final loss against both downstream
tasks. Topic classification: **r = −0.888**, p < 0.001, a strong relationship in the expected
direction. Entity recognition: **r = +0.303**, p = 0.25, nothing. Do not read the sign: the claim
is *absence*, and at n = 16 that correlation is indistinguishable from zero. The aggregate across
all nineteen models is −0.935, which looks decisive and is three under-trained models holding up
a line.

**Results.** Then the harder question — why one task and not the other. The first answer we
published was that the floors differ, and it was wrong: the floors are near-identical as a share
of achievable, and were retracted. What separates them is the **variability of the gain**. Entity
recognition hands every working model a score between 0.754 and 0.798 — a band **0.044** wide.
Topic classification spreads the same models over **0.143**, more than three times as far.

**Learning.** A benefit that every model receives equally cannot be predicted by anything, because
there is nothing left to predict. That is why loss tracks one task and not the other, and it is a
fact about the *task*, not about the models or the floor. Check that your metric predicts the
thing you care about before you spend a term minimizing it.

**What this makes possible.** If the metric can be wrong, so can the unit underneath it. Week 6
goes one level down.

---

## Week 6 — Which of your units are not units?

**Big number:** `5.1× the compute, at "matched" steps` · **Figure:**
`03-matched-steps-vs-compute.svg`

**Problem.** Two vocabularies produce two validation losses that are not on the same scale,
because a loss per token means something different when the tokens are different sizes. Comparing
them directly is a category error that looks like arithmetic.

**Hypothesis.** We believed "same number of steps" was the fair way to hold compute constant
between the two arms. It is the most natural reading of a controlled comparison, and it is wrong
here for a reason that is invisible until you count.

**Approach.** Convert to **bits per character**, which is vocabulary-independent: nats per token,
divided by ln 2, divided by characters per token. Then count what matched steps actually bought
each arm. A 250k output projection at hidden size 512 is a 128M-parameter head against 8.2M, and
its matmul dominates the forward pass — measured, the large-vocabulary arm runs at 80k tokens/sec
against 408k. So "12,000 steps each" handed one arm **5.1× the compute** of the other.

**Results.** Read as matched steps the two vocabularies looked indistinguishable, and the study's
central argument was in trouble. Read as matched compute — the question anyone actually has,
because nobody chooses a vocabulary and then buys whatever hardware it needs — the answer
reversed. Both readings come from the same three runs.

**Learning.** Fairness is a property of the unit, not of the intention behind it. Any time two
arms differ in something that changes cost per step, "same steps" is a budget transfer rather than
a control. Convert to a unit that does not depend on the thing you are varying, and state which
axis you held fixed, because the reader cannot tell from the number.

**What this makes possible.** Only now can the tokenizer question be asked in a unit capable of
answering it. Week 7 asks it properly.

---

## Week 7 — Is the tokenizer a cost, or a coin flip?

**Big number:** `3.9× the spread` *(p = 0.0098)* · **Figure:** `17-tokenizer-lottery.svg`

**Problem.** Report 08's headline was that XLM-R's 250k vocabulary costs Yoruba **0.144 bits per
character** at matched compute. That number came from three seeds a side and passed the old
"bigger than the spread" rule at 1.4× — which week 4 had just established is not a test.

**Hypothesis.** We expected more seeds to confirm the penalty and tighten it. The power
calculation said six a side would reach p = 0.039 at the observed effect size, so six was fixed in
advance and written into the script **before the runs started** — precisely so that "run seeds
until it works" was not available as an option.

**Approach.** Three more seeds per arm, same corpus, same settings, same everything: anything that
differed would make them a new cell rather than more of the same one. Then test both what we
predicted and what we had not thought to look at — location *and* spread. The location result:
the penalty **shrank** to 0.059 with p = 0.37, and the two arms interleave, with three of the six
large-vocabulary runs landing *below* the small vocabulary's median. There is no direction left to
report.

**Results.** The spread is a different story: **0.145 against 0.037, F = 15.1, p = 0.0098.** The
large-vocabulary arm is not merely wider, it is in two clusters — three runs better than anything
the small vocabulary produced, three much worse. The same shape appears downstream on entity
recognition (8.6× wider, p = 0.005) though not on topic classification, where our own arm produced
a weak seed and the ratio reverses.

**Learning.** It is not a tax, it is a lottery: a badly-fitting vocabulary does not reliably cost
you bits per character, it decides how much of a gamble the run is. Testing only the mean would
have recorded a null and thrown the finding away — the mirror image of the error the board is
about. And a pre-registered sample size is only pre-registered if the analysis cannot quietly
grow; ours nearly became n = 7 when a different study built a cell matching the same glob.

**What this makes possible.** A setting whose effect is on variance rather than on the mean is
exactly the kind that will not transfer to a new situation. Week 8 tests transfer directly.

---

## Week 8 — Does a tuned setting transfer?

**Big number:** `7e-4: best for three languages, fatal to a fourth` · **Figure:**
`16-lr-transfer.svg`

**Problem.** The factory's selling point is that adding a language is one function call. That is
only true if the *settings* come with it, and we had never checked whether they do.

**Hypothesis.** We expected the best learning rate to be roughly stable across languages of
similar size and script, so that tuning once and reusing was a reasonable saving. The worst case
we imagined was leaving a little performance on the table.

**Approach.** Five languages — Hausa, Igbo, Nyanja, Swahili, Yoruba — six learning rates, two
seeds, sixty runs. Plot them as small multiples rather than five curves on one axis, because the
heights are not comparable across languages with different corpora and the *shape* is what carries
the answer. Draw every seed, not the cell mean: three cells here are split, with one seed training
and the other collapsing, and a mean over those describes no run that happened.

**Results.** Hausa, Nyanja and Swahili all peak at **7e-4**. Igbo **collapses** at 7e-4 and at
every rate above it — 2.889 at 5e-4 against 5.638 at 7e-4, and it stays there. Yoruba peaks at
5e-4 and degrades at 1e-3. So the rate that is optimal for three of the five destroys a fourth.

**Learning.** The risk of transferring a tuned setting is not a slightly worse model; it is a
wasted night that looks like a result until you check it against something. A setting is part of
the experiment, not part of the tooling, and "one function call" is a claim about the code rather
than about the science. When a hyperparameter can fail catastrophically rather than gradually,
sweep it per language or do not claim transfer.

**What this makes possible.** If runs can fail this way, the obvious engineering response is to
catch them early and stop paying for them. Week 9 tries exactly that.

---

## Week 9 — Detect the failure, or prevent it?

**Big number:** `0 of 11 checkpoints separate them` · **Figure:** `10-early-signal.svg`

**Problem.** Across 195 runs, **35 never learned** — 17.9%, wasting 25.5 GPU-hours. That is real
money on a shared machine and the obvious fix is to notice early and kill the run.

**Hypothesis.** We expected a doomed run to be visible in its first few thousand steps, and an
abandonment rule to pay for itself. The intuition is strong: failed runs look bad early, so a
threshold should separate them.

**Approach.** Score every run against **its own untrained baseline** rather than against a fixed
loss — raw loss is not comparable between a 16k and a 250k model, since an untrained 250k model
starts 2.7 nats higher and a fixed threshold would condemn every large-vocabulary run at step one.
Measure nats gained at eleven checkpoints from step 500 to step 24,000. Then price every
abandonment rule: what it saves in abandoned compute against what it costs in healthy runs killed
by mistake.

**Results.** At **all eleven checkpoints the two outcomes overlap** — the best doomed run has
always gained more than the worst healthy run. Only one rule in the entire grid ever fires, and it
kills two healthy runs and zero dead ones, netting −0.5 GPU-hours. The patience-based rules lose
up to 84. There is no threshold, because there is no separation to threshold.

**Learning.** Do not build the detector; the signal it needs is not there. Spend the effort on
prevention instead — and then measure that too, because tighter gradient clipping, tested at
fifteen seeds a side, does **not** prevent divergence either (4 of 15 against 3 of 15, Fisher
p = 1.00). What clipping does do is improve the runs that survive (2.825 → 2.530, exact p = 0.0003)
and tighten them (sd 0.256 → 0.104, p = 0.0065), which is a real benefit and not the one it was
sold on.

---

## The bottom strip — Week 10: writing it down so it stays true

Full width, three columns.

**Left — What it cost.** `06-what-it-cost.svg`, plus the three routes to reproducing it: a
workstation at roughly **$24,000**, Colab Pro at about **$120**, or a plugged-in laptop over a
month of nights, free. The line that matters: *the workstation bought latency, not access.* A
student who runs one cell a night for a month gets the same board; they just wait longer for it.
The hardware figure belongs here too once the measurements exist — see the Appendix.

**Middle — The five constants.** The table from report 09 §10, verbatim. Each was a value chosen
sensibly for one purpose that then silently decided an answer somewhere else: a seed spread
measured on one grid and applied everywhere, an `epoch` field kept for dashboard compatibility, a
400-document sample nobody revisited, an `FT_STEPS = 352` inherited from an extraction, and
matched steps as the obvious way to hold compute fixed. None was a bug. Each stopped being
defensible somewhere it had travelled to without anyone noticing it had moved.

**Right — How this stays honest.** Numbers generated from records, not typed — and a gate that
tests each comparative claim against its null rather than checking that the digits still match.
It currently reports **9 claims: 6 supported, 2 not supported, 1 underpowered**, and two of the
three failures are ours. That is the claim worth making: not that we were careful, but that we
ran the test, published the count of our own failures, and did not quietly drop the two claims
that failed — a tokenizer penalty and a floor explanation, both already written up, one already
emailed.

---

## Gaps

**Week 1 has no figure, and it is the only cell that is blocked.** It wants the hardware
comparison as a graphic: the machines, their generation, whether the model fits in memory, and
what one run costs on each. Everything needed to produce it is written and self-contained —
`bench_portable.py` — but the measurements can only come from the machines themselves. **See the
Appendix.**

**Weeks 5 and 6 were one investigation split across two cells** in an earlier draft, which is why
the board had nothing to put between them. They are now the metric question and the unit question,
which is a real distinction: one asks whether the number you optimize predicts anything, the other
asks whether the number you compare is on a comparable scale.

**A note on week 5's figure.** `12-floors.svg` replaced `01-headline.svg` here, because figure 01
went to Patrick — it is his comparison, and whoever owns the selection rule owns the numbers.
Building the replacement caught an error in the writeup: the first version claimed the difference
between the floors explained the task divergence. It does not, and the retraction was originally
argued from the two shares being near-identical at 57% and 52%. **Do not reuse that sentence
either.** Sweeping the MasakhaNER floor moved it from one unswept cell at 0.4140 to **0.6261**
across twelve rates, and the shares are now 61% and 78%. The retraction still stands, for the
better reason: near or far, a floor is a fact about the task and cannot explain why one task's
scores are predictable and the other's are not. The band widths do that, and they are unaffected
by where the floor sits. It is worth holding both versions in mind while laying the cell out,
because it is the same shape as the mistake the cell is about — an explanation that sounded right,
was checked, and turned out to rest on a coincidence that has since evaporated.

---

## Order of work

1. ~~Figure for week 5~~ — **done**, `12-floors.svg`.
2. ~~Figure for week 8~~ — **done**, `16-lr-transfer.svg`, on the completed 60-run grid.
3. ~~Figure for week 7~~ — **done**, `17-tokenizer-lottery.svg`.
4. **Figure for week 1** (hardware). Blocked on the measurements in the Appendix — the only item
   on the board that cannot be produced from this machine.
5. **Regenerate everything and run the staleness check.** Do this *last*, once no study is still
   writing, and do it once.

---

## What the board must not do

Quote a count typed by hand. Every number on it comes from `poster_bottom.ipynb`, which recomputes
from the records and flags any sentence in report 09 the data no longer supports.

The counts have moved twice while this was being written: 105 pretraining runs became 156, and 156
became **213**. Fine-tuning records stand at **215**, of which 99 are dev-scored and therefore not
reportable numbers.

None of that is an erratum, and the reason is the only process claim this board makes: **nothing
was typed twice.**

---

# Appendix — measuring what a run costs on your machine

This is the missing week 1 figure. Everything below is self-contained: `bench_portable.py` needs
no corpus, no tokenizer and no repository data, because its token stream is random integers. That
is worthless for learning and identical for timing — a transformer's cost per step does not depend
on which token ids arrive — which means it can be pasted straight into a fresh Colab cell.

**What we are collecting.** For each machine: tokens per second on both model shapes, whether the
model fits in memory, and the extrapolated wall-clock for one full 62,500-step run. The reference
row already exists — the workstation sustains **381,817 tok/s** on the 33.8M `poc` preset and
**184,329 tok/s** on the 86M `afriberta` preset, medians over 96 and 55 real runs.

**Before you start, on every machine:** close other GPU work. The script warns when it can detect
contention, but a number taken while something else holds the card is simply wrong, and it is the
most common way these tables end up misleading.

---

## A. The workstation (reference row — already have it, re-run to confirm)

```bash
cd /o/Sources/GitHub/TrueRottweiler/WashingtonCsed504/src/a2-nlp
CUDA_VISIBLE_DEVICES=0 bash py.sh bench_portable.py --out runs/hardware.json --note "Toothless, RTX PRO 6000 Blackwell Max-Q, 1 card"
```

Takes about two minutes. Use one card deliberately — the figure compares *a card* against other
machines, and a two-card number is not comparable to a MacBook.

---

## B. Surface Studio Laptop (RTX 2000 Ada Mobile)

The interesting row, because it is the machine a student is most likely to actually own.

1. Clone the repository, or copy just `src/a2-nlp/bench_portable.py` — it needs nothing else.
2. Install PyTorch with CUDA if it is not already there:
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
3. Run it:
   ```
   python bench_portable.py --out hardware.json --note "Surface Studio Laptop, RTX 2000 Ada Mobile, plugged in"
   ```
4. **Then run it again on battery**, and note that in `--note`. Mobile GPUs throttle hard on
   battery, and "can a student do this overnight on a laptop" is a different question plugged in
   than unplugged. If the two differ by more than about 20%, the board should say so.

If the 86M `afriberta` preset runs out of memory, that is a result rather than a failure — record
it. Knowing which machines cannot hold the larger model is exactly what the figure is for.

---

## C. MacBook Pro (M4 Pro, MPS)

```bash
python bench_portable.py --out hardware.json --note "MacBook Pro M4 Pro, MPS"
```

The script detects MPS automatically. Two things to expect: MPS does not support the same
mixed-precision path as CUDA, so the number is honest but not directly comparable in *kind*; and
Apple's unified memory means the larger preset may fit where a discrete GPU of nominally similar
size would not. Both are worth a footnote on the figure.

---

## D. Google Colab — the important set

This is the row that decides whether the poster's "$500 and you can do this" claim survives, so it
is worth doing properly across tiers.

**Setup, once per runtime.** New notebook → Runtime → Change runtime type → pick the accelerator →
then in the first cell:

```python
!pip -q install torch --upgrade
!wget -q https://raw.githubusercontent.com/TrueRottweiler/WashingtonCsed504/main/src/a2-nlp/bench_portable.py
!python bench_portable.py --note "Colab T4"
```

Change the `--note` for each. Run these four:

| runtime | why it is on the list |
|---|---|
| **T4** (free tier) | the honest floor — what somebody with no budget gets |
| **L4** | the cheap paid tier, and probably the best value row |
| **A100 40GB** | the fast tier, for the "under two hours" claim |
| **TPU** | *skip it* — the script refuses, deliberately. It needs a different training loop and a TPU row would not be comparable to the others anyway |

**Copy the printed JSON out of each session before it disconnects.** A Colab runtime's disk is
disposable and these numbers are the whole point of the exercise; paste them into
`runs/hardware.json` on the workstation, or just send me the four blocks.

**Also record, for each tier:** the actual cost. Compute units per hour and what they cost, so the
strip's "$120" is a measured claim rather than a remembered one. If an A100 session gets
interrupted before the benchmark finishes, that is worth noting too — session limits are part of
what a student is buying.

---

## E. What I will do with them

Drop the collected rows into `runs/hardware.json` and I will generate the week 1 figure —
machines on one axis, wall-clock for one full run on the other, with a memory-fit marker and the
three cost routes annotated. Then the last blocked cell is unblocked, and the board is complete.

**The sentence the figure has to earn:** *you do not need the workstation.* It is only true if the
numbers say so, and right now we have one machine's worth of evidence for a claim about every
machine.

---

# References

Everything the board leans on that we did not measure ourselves. The poster should carry a
shortened version of this — a board that quotes a significance threshold and a dataset without
saying where either came from is asking to be taken on trust, which is the one thing this board
argues against.

### Statistics

1. **Fisher, R. A.** (1925). *Statistical Methods for Research Workers.* Oliver & Boyd,
   Edinburgh. — the source of the 0.05 convention, offered there as a convenience rather than a
   law. [Archive copy](https://archive.org/details/statisticalmethod031898mbp)
2. **Fisher, R. A.** (1935). *The Design of Experiments.* Oliver & Boyd, Edinburgh. — randomization
   as the basis of inference; the ancestor of the permutation test used throughout this board.
3. **Pitman, E. J. G.** (1937). "Significance tests which may be applied to samples from any
   populations." *Supplement to the Journal of the Royal Statistical Society* 4(1), 119–130. — the
   formal treatment of permutation tests. [DOI: 10.2307/2984124](https://doi.org/10.2307/2984124)
4. **Welch, B. L.** (1947). "The generalization of Student's problem when several different
   population variances are involved." *Biometrika* 34(1–2), 28–35. — the unequal-variance t-test
   `claims_audit.py` uses by default.
   [DOI: 10.1093/biomet/34.1-2.28](https://doi.org/10.1093/biomet/34.1-2.28)
5. **Levene, H.** (1960). "Robust tests for equality of variances." In *Contributions to
   Probability and Statistics*, Stanford University Press, 278–292. — the variance test behind the
   tokenizer-lottery result in Week 7.

*A note worth making to a class rather than hiding in a bibliography:* Fisher was also a prominent
eugenicist, and that is a matter of record rather than a matter of opinion. It does not make the
arithmetic wrong. It is a useful reminder that a field's foundational tools arrive attached to the
people who built them, and that using the tool is not the same as endorsing the person.

### Models

6. **Conneau, A. et al.** (2020). "Unsupervised Cross-lingual Representation Learning at Scale."
   *ACL 2020.* — XLM-R, our 277M-parameter multilingual baseline, loaded as
   `FacebookAI/xlm-roberta-base`. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116)
7. **mmBERT** — loaded as [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base);
   246M parameters, reported by its authors as trained on roughly three trillion tokens across
   1,800 languages. *Cite the model card directly — check the current card for the paper reference
   before the poster is printed.*
8. **Liu, Y. et al.** (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." — the
   architecture our from-scratch models use.
   [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
9. **Devlin, J. et al.** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for
   Language Understanding." *NAACL 2019.* — the 80/10/10 masking scheme every pretraining run here
   follows. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
10. **Ogueji, K., Zhu, Y. and Lin, J.** (2021). "Small Data? No Problem! Exploring the Viability of
    Pretrained Multilingual Language Models for Low-resourced Languages." *MRL Workshop, EMNLP
    2021.* — AfriBERTa, after which our 86M `afriberta` preset is shaped and named.
    [ACL Anthology](https://aclanthology.org/2021.mrl-1.11/)

### Data

11. **Adelani, D. et al.** (2024). "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for
    Topic Classification in 200+ Languages and Dialects." *EACL 2024.* — our topic-classification
    task, loaded as `Davlan/sib200`. 701 train / 99 validation / 204 test for Yoruba.
    [arXiv:2309.07445](https://arxiv.org/abs/2309.07445)
12. **Adelani, D. et al.** (2022). "MasakhaNER 2.0: Africa-centric Transfer Learning for Named
    Entity Recognition." *EMNLP 2022.* — our entity-recognition task. Read from
    [the CoNLL files in the masakhane-ner repository](https://github.com/masakhane-io/masakhane-ner),
    **not** via `load_dataset` — the HuggingFace copy ships a custom loading script and that path
    is no longer executed. [arXiv:2210.12391](https://arxiv.org/abs/2210.12391)
13. **FineWeb-2** — [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
    the source of every corpus in the language gradient, including all 69.1M tokens of Yoruba that
    exist there. English rungs come from `fineweb-edu` and `fineweb`.

### Method

14. **Smith, L. N.** (2018). "A disciplined approach to neural network hyper-parameters." — the
    one-cycle schedule every run here anneals under, which is why a run cannot be truncated and
    still compared. [arXiv:1803.09820](https://arxiv.org/abs/1803.09820)

**Check before printing.** Reference 7 is the one to verify — mmBERT is recent enough that the
canonical citation may have changed since this was written, and a poster is a bad place to be
wrong about somebody else's model.
