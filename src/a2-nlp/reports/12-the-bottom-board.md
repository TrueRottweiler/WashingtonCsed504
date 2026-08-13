# Build sheet: the bottom board — v2

*What goes in each cell, what it says, and which figure to drop in. Report 09 is the long form.
This is the board — nine cells and a strip. One is not a subset of the other, so this sheet is the
translation, and it is the thing to print and keep beside you while laying out.*

> **What changed from [v1](12-poster-build-sheet-v1.md), and why.** v1 had drifted into being nine
> findings. Good findings, but four of them arrived in the last week of a year-long project, and
> two — metric validity and the tokenizer lottery — needed almost none of the machinery this board
> is supposed to be about. Meanwhile four panels from the original outline were missing entirely,
> and all four were about the factory as a *thing other people use*.
>
> So two cells are cut and two come back:
>
> | out | in |
> |---|---|
> | *Is your metric the right metric?* (the −0.888 / +0.303 correlation study) | **What belongs in a notebook, and what belongs in a queue?** |
> | *Is the tokenizer a cost, or a coin flip?* (the six-seed variance result) | **What does someone else have to be able to call?** |
>
> The tokenizer cell is not deleted so much as **handed upstairs**. The tokenizer is what Patrick
> and Leon's board argues about; whoever owns the question owns the panel, which is the same call
> already made with figures 01 and 02. Both cut findings survive in full in report 09 and in
> `poster_bottom.ipynb` — they leave the board, not the project.
>
> The test v2 has to pass that v1 did not: **does this board demonstrate the factory worked for
> anyone other than the person who built it?** v1 never showed that. v2 spends two of nine cells
> on it.

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

Read the nine cells left to right, top to bottom, and they are one argument.

**The top row builds a factory.** A unit to measure runs in, the speed that makes running them
repeatedly affordable, and the split between work that belongs in a notebook and work that belongs
in a queue.

**The middle row makes it survivable, shareable, and trustworthy.** Records that outlive the
person who made them, an interface somebody else can actually call, and — because people who can
call it start challenging your numbers — a rule for when a difference is real.

**The bottom row spends the instruments.** A unit that turned out not to be a unit, a tuned
setting that does not transfer, and a failure that cannot be detected. Two of those three answers
are *no*, which is the point.

That shape is the board's claim: **the machinery is not overhead on the science, it is what makes
the science falsifiable — and the proof is that somebody else used it to prove me wrong.**

The grid as it goes on the wall, with each cell's big number, so the layout can be checked at a
glance before any of the prose below is cut down:

| | | |
|---|---|---|
| **1** What does a run cost, and in what unit?<br>`62,500 steps = 1.024B tokens` | **2** Why optimize before anything needs it?<br>`2.07×, of which 1.32× is efficiency` | **3** What belongs in a notebook, and what belongs in a queue?<br>`53 s against 85 min — 96×` |
| **4** What makes a record survive you?<br>`fingerprint 15abd33de5af` | **5** What does someone else have to be able to call?<br>`9 functions` | **6** Is this difference real?<br>`2.27×, not 1.0×` |
| **7** Which of your units are not units?<br>`5.1× the compute at "matched" steps` | **8** Does a tuned setting transfer?<br>`7e-4: best for three languages, fatal to a fourth` | **9** Detect the failure, or prevent it?<br>`0 of 11 checkpoints separate them` |

Three of the nine big numbers are ratios of a thing to itself measured two ways — 96×, 5.1×,
2.27×. That is not a stylistic tic. It is what the whole board is about: the same work, counted in
two units, giving two different answers.

**Every cell now carries its own figure except one.** Week 1 is still waiting on the hardware
measurements, which are the only thing on this board that cannot be produced from this machine.
Everything else is drawn and committed, and `check_boards.py` confirms no figure is claimed by
both posters.

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
which is why the batch is 128 in **191 of 197** pretraining runs and why the fleet queues are
written in update tokens rather than steps.

**Learning.** Epochs stop being a unit the moment the dataset is a variable, and scaling studies
make it a variable by definition. Pick the unit that survives the thing you intend to vary, and
write it into the tooling so nobody has to remember. The cost of getting this wrong is not a bad
number — it is two numbers that look comparable and are not.

**Source.** Every pretraining record, via `mlm_api.results('*')`; the step and batch fields are
written by `mlm_train.py` at the end of each run. The 4M-to-1,024M ladder is `runs/scaling_law.json`
(`scaling_law.py`).

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

**Source.** `runs/pipeline_bench.json` (`pipeline_bench.py`) for the stage timings in change 4;
the 2.07× decomposition is report 03, measured on the four-cell comparison it names. Rows without a
record file were measured once at the time and are marked as such in the table.

**What this makes possible.** Weeks 6 through 9 each rest on running one cell three to fifteen
times. At 25 minutes a run nobody would have done that, and every result below would have been a
single seed with an anecdote attached. Week 3 is where that affordability stops being about the
clock and starts being about where the work runs.

### The year, in twenty changes

Measured on this hardware unless marked. "Kept?" matters as much as the number — a third of these
were tried and rejected, and a list of only the wins would be marketing rather than a log.

#### The data path — where most of the win actually is

| # | change | measured impact | kept? |
|---|---|---|---|
| 1 | **Delete the DataLoader; hold the dataset resident in GPU memory.** One CPU core augments ~4,000 img/s while the card trains at ~13,000 — the CPU was starving the GPU 3× and the card spent its life waiting on Python. | CIFAR: 13.8k img/s (workers, bs128) → **18.5k** (resident, bs512), at far higher utilization | yes |
| 2 | **Store images as uint8, convert to float per batch.** Float residency would be 4× the memory for no gain; the conversion is nearly free on-device. | CIFAR-100 train = **147 MB**; ImageNet-32 = 3.9 GB on a 96 GB card | yes |
| 3 | **Augment on the GPU** — crop, flip, erase, normalize as tensor ops on data already in VRAM. | no PIL, no JPEG decode, no Python in the inner loop, no per-batch host→device copy | yes |
| 4 | **Tokenize once, up front, into a flat array.** Training never touches raw text, the tokenizer, or the datasets library again — it indexes. | prep is **53 s** against 85 min of training: a 96× ratio | yes |
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

## Week 3 — What belongs in a notebook, and what belongs in a queue?

**Big number:** `53 seconds against 85 minutes` — a ratio of **96×** · **Figure:**
`15-what-a-run-is-made-of.svg`

**Problem.** The group's proof of concept was a notebook, and it worked — that is not the
criticism. The criticism is that a notebook is a bad place to *keep* work: close the laptop and
the state is gone, run the cells in a different order and get a different answer, train for three
hours and lose it when the kernel restarts.

**Hypothesis.** The obvious response is "move everything into scripts," and it is wrong. A
notebook is the best tool anyone has for the part of the work that is *figuring out what you
want* — look at the data, try one small model, plot something, change your mind. Deleting it
would trade a real problem for a different real problem.

**Approach.** So measure where the time actually goes and let the ratio decide the split, rather
than taste. `pipeline_bench.py` times every stage on the real Yoruba corpus — 79,999 documents,
260M characters, 69.1M training tokens. Reading all of it takes about **1 s**. Training the 16k
BPE takes **20.7 s** on the 12.7M-character sample it is fitted to, which is the real cost because
that sampling is what production does. Encoding the whole corpus to token ids is **21 s**, and
moving the resulting store onto the card is **11.4 s** — done once, because there is no DataLoader
in the loop. Preparation, all in, is **53 seconds**. One pretraining run at 62,500 steps is
**85 minutes** for the 98M model and 37 for the 33.8M. That is **96×**, and a gap that size is not
a judgment call.

**Results.** The architecture follows from the ratio. **Everything cheap stays interactive** —
preparation, inspection, plotting, one small model to see whether the idea is worth a night.
**Everything expensive moves to a file you can start and walk away from**, because a three-hour
job must survive a closed laptop. And the part that makes the split work rather than merely
tidy: **both paths write the same record.** A result from a notebook cell and a result from an
overnight queue land in the same folder in the same format, comparable without translation. We
also rewrote the group's original notebook to call the factory instead of doing everything
itself, with each change marked and the code it replaced left visible underneath — so they could
see what had changed rather than being handed something unfamiliar.

**Learning.** Do not choose between notebooks and scripts on principle; measure the ratio and let
it choose. The failure mode of a notebook is not slowness, it is that its state is invisible and
unreproducible, so the rule is that anything you would be upset to lose does not live in one.
And a split only helps if both halves produce the same artifact — two formats would have moved
the problem rather than solved it.

**Source.** `runs/pipeline_bench.json` (`pipeline_bench.py`) — the 20.7 s is stage *"2. train 16k
BPE tokenizer"*, `trained_on_chars` 12,745,110; the 1 s read and the 21 s encode are the `extrapolated_full_s`
fields of stages 1 and 3, not the sampled `seconds` beside them. The 85 minutes is stage *"6. train step,
98M model"*, `hours_for_62500_steps` × 60.

**What this makes possible.** Work that runs unattended overnight has nobody watching it, which
means it has to describe itself. Week 4 is what that costs.

---

## Week 4 — What makes a record survive you?

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

**Source.** `mlm_api.results('*')` and `ft_api.results('*', eval_split=None)` for the inventory;
the fingerprint is `corpus_info('yor')['vocab_fingerprint']`.

**What this makes possible.** A record that describes itself is the smallest unit of work somebody
else can pick up without asking you what it is. Week 5 is that same property at the scale of a
whole interface — and Week 6 needs both, because you cannot ask whether two numbers differ until
you are certain they came from two different experiments.

---

## Week 5 — What does someone else have to be able to call?

**Big number:** `9 functions` · **Figure:** `19-the-interface.svg`

*The second half of that big number used to read `· 12,861 lines they never open`, and it is not
printed here any more because it moved twice while this sheet was being written — 12,861, then
14,409, then 14,537 — as four checks and two studies were added, none of which touched the
interface. It is **9 functions against whatever the figure prints on the day you export it**. The
figure counts the folder at render time for exactly that reason, and the moving number is itself
the point: the surface stayed at nine while the body grew 13%.*

**Problem.** A factory only its author can operate is a hobby with a queue. Two other people had a
study to run and a deadline, and neither of them was going to read fourteen thousand lines of
somebody else's Python to find out how to train a model.

**Hypothesis.** We assumed the hard part was making the machinery correct, and that a reasonable
interface would follow from reasonable code. It does not. Correctness and callability are separate
problems with different failure modes: a wrong answer announces itself eventually, whereas a
person who cannot work out how to call your code quietly writes their own and never mentions it.

**Approach.** Cut the surface until it fits on one screen, and hide the rest. **Nine functions in
`mlm_api` — prepare, inspect, estimate, pretrain, the untrained control, and three ways to read
results back — sitting on 1,634 lines of pretraining machinery inside a folder that was 12,861
lines of Python when this was written and is more now, none of which they have to open.** Three of those nine exist only because a
mistake made them necessary: `estimate()` measures twenty real steps on the actual card instead of
consulting a throughput table that was wrong the first time something else held the GPU;
`random_init()` is one line because a control that costs an afternoon does not get run; and the
run tag encodes every setting, because two runs once quietly overwrote each other. Then hold the
interface against the three things it has to survive. *Sharing:* every corpus, vocabulary and
result committed, so they can plot our findings without re-running anything. *Their machines:* one
card, free Colab included — nothing needs the two-card workstation. *Argument:* reproducible
enough that a number can be challenged rather than accepted.

**Results.** Those three held. A fourth requirement we never wrote down did not, and it is the one
worth the wall space. **Something they could find — we got that wrong.** Leon cloned the
repository, read the documentation, and asked whether there was an interface he was supposed to be
using. There was. He could not find it, because the folder's front page was titled with a
*different study* — "When Does Attention Beat Recurrence?" — and `mlm_api` first appeared on line
25, as one row of a second table, under a heading about "the masked-LM half". Every word of that
was accurate. It was still unfindable, and that is a failure of my half rather than his: **a tool
nobody can find does not exist.** The fix took twenty minutes — a two-row table at the top saying
which study you are here for — and it should have been the first thing written, not the last.

And the result that justifies the whole board: **the most valuable thing the factory produced was
not a model — it was Patrick being able to check a number I was confident about and show it was
wrong. Twice.** Once, a baseline everyone had quoted for weeks that turned out never to have
trained. Once, an evaluation dataset using a different text encoding from our vocabulary, which
had silently reversed a comparison. Both came from him re-running things rather than trusting
them.

**Learning.** The measure of a factory is not what it computed; it is whether two other people
could use it, and the honest test of that is the poster hanging above this one. Discoverability is
part of the interface and fails silently — nobody files a bug saying they could not find your API.
And the property to defend hardest is not accuracy but *challengeability*: the collaboration
worked because results were reproducible enough to be argued with, and being proved wrong twice by
a partner is evidence the tooling succeeded, not evidence it failed.

**Source.** Counted at render time from `mlm_api.py` and the folder by `poster_figures._api_surface()`,
which is why the figure and this sheet cannot disagree about a number that moves weekly.

**What this makes possible.** Once other people can run your comparisons and disagree with your
numbers, "is this difference real" stops being a private question. Week 6 is the rule we needed.

---

## Week 6 — Is this difference real?

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

**Source.** `runs/claims_audit.json` (`claims_audit.py`) for which claims clear the bar. The four
derivations above are arithmetic, not measurements — they are reproduced here so a reader never has to
take a threshold on faith.

**What this makes possible.** Weeks 7 through 9 are all comparisons, and so is every number on
the board above this one. Without this they are anecdotes with decimal points.

---

## Week 7 — Which of your units are not units?

**Big number:** `5.1× the compute, at "matched" steps` · **Figure:** *none — set the table below
as type.* The matched-steps chart went to the top board, which needs it for its own
matched-compute panel and reads first; see Gaps for which figure that is and why.

Losing the chart is a gain rather than a concession, and it is worth saying why: the finding is
that **one experiment read two ways gives opposite answers**, and a bar chart has to work against
itself to show that — it draws one set of heights and then annotates its way to the second
reading. Two rows of type do it directly, at a size legible from three metres, which a five-bar
chart never is:

| 12,000 steps each, read two ways | 16k vocabulary | 250k vocabulary |
|---|---|---|
| **what it scored** — bits per character | 1.131 | **0.989** *(looks better)* |
| **what it cost** — minutes per seed | 8 | **42** |

The cell's whole argument is that the second row is missing from the first reading. Set the `5.1×`
at 90–110 pt, the table beneath it, and nothing else.

*Recomputed 12 August: n = 4 and n = 6, the 250k arm taken through `arm_records()` rather than the
glob. The wall-clock ratio is 5.24× on these cells; **5.1×** is the throughput measurement — 408k
tokens/sec against 80k — and is the number to print, because it is the property of the two
architectures rather than of one night's scheduling. An earlier draft of this table quoted 1.135
and 1.056 at 8 and 41 minutes, which was the three-seed version from the README. Same conclusion,
and it should still have been rederived rather than copied.*

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
reversed. Both readings come from the same runs; only the axis held fixed changed.

**Learning.** Fairness is a property of the unit, not of the intention behind it. Any time two
arms differ in something that changes cost per step, "same steps" is a budget transfer rather than
a control. Convert to a unit that does not depend on the thing you are varying, and state which
axis you held fixed, because the reader cannot tell from the number.

**Source.** `runs/tokenizer_seeds.json` (`study_tokenizer_seeds.py`), selected through `arm_records()`
rather than a glob; throughput from the `tokens_per_s` field of the runs themselves.

**What this makes possible.** The tokenizer question can now be asked in a unit capable of
answering it — and it is asked on the board above this one, where it belongs. Week 8 takes the
same lesson from a unit to a setting.

---

## Week 8 — Does a tuned setting transfer?

**Big number:** `7e-4: best for three languages, fatal to a fourth` · **Figure:**
`16-lr-transfer.svg`
*(alternative, if the cell should carry the budget lesson instead of the transfer one:*
`12× — what length buys against what seeds buy`*. Both are below; only one fits.)*

**Problem.** The factory's selling point is that adding a language is one function call. That is
only true if the *settings* come with it, and we had never checked whether they do.

**Hypothesis.** We expected the best learning rate to be roughly stable across languages of
similar size and script, so that tuning once and reusing was a reasonable saving. The worst case
we imagined was leaving a little performance on the table.

**Approach.** Five languages — Hausa, Igbo, Nyanja, Swahili, Yoruba — six learning rates, two
seeds, sixty runs. Plot them as small multiples rather than five curves on one axis, because the
heights are not comparable across languages with different corpora and the *shape* is what carries
the answer. Draw every seed, not the cell mean: two cells here are split, with one seed training
and the other collapsing, and a mean over those describes no run that happened.

**Results.** Hausa, Nyanja and Swahili all peak at **7e-4**. Igbo **collapses** at 7e-4 and at
every rate above it — 2.889 at 5e-4 against 5.638 at 7e-4, and it stays there. So the rate that is
optimal for three of the five destroys a fourth.

Yoruba's own top end is the more instructive row, and an earlier draft of this cell got it wrong.
It said *"Yoruba peaks at 5e-4 and degrades at 1e-3."* The 1e-3 cell does not degrade. It is
**split**: one seed collapsed at 5.540 and the other trained to 3.326, which is within 0.32 of
Yoruba's best rate. "Degrades" is what the average of those two numbers looks like, and it
describes neither run. That is the mistake the Approach paragraph above warns about, made in the
Results paragraph directly underneath it, which is roughly how these errors actually happen.

**Learning.** The risk of transferring a tuned setting is not a slightly worse model; it is a
wasted night that looks like a result until you check it against something. A setting is part of
the experiment, not part of the tooling, and "one function call" is a claim about the code rather
than about the science. When a hyperparameter can fail catastrophically rather than gradually,
sweep it per language or do not claim transfer.

### Luck, skill, and what to spend the next hour on

The honest question underneath this cell is one nobody asks out loud: **how much of a tuned result
is knowing what you are doing, and how much is having run enough things that one of them came out
well?** Sixty runs is enough to answer that, and the answer is uncomfortable.

**1. The winning rate is not identified — in any of the five languages.** For each language, the
gap between the best rate and the runner-up is *smaller than the gap between two seeds at the same
rate*. Every "peak" quoted above is a range, not a point.

| language | best | runner-up | gap between them | seed sd at one rate | identified? |
|---|---|---|---|---|---|
| Hausa | 7e-4 · 3.139 | 8.5e-4 · 3.160 | 0.021 | 0.022 | no |
| Igbo | 5e-4 · 2.889 | 3e-4 · 2.956 | 0.067 | 0.127 | no |
| Nyanja | 7e-4 · 2.734 | 8.5e-4 · 2.782 | 0.048 | 0.354 | no |
| Swahili | 7e-4 · 2.853 | 8.5e-4 · 2.916 | 0.063 | 0.070 | no |
| Yoruba | 5e-4 · 3.006 | 7e-4 · 3.026 | 0.020 | 0.090 | no |

What the experiment *did* establish is real and worth having: which rates are fatal, and roughly
where the usable band sits. What it did not establish is which rate inside that band is best.
Those are different claims, and only the first one survives Week 6's bar.

A footnote for whoever checks this against `claims_audit.py`, which flags Swahili as separated
where the table above does not. The audit compares the gap against the larger of the two cells'
own seed gaps (0.063 against 0.050); this compares it against the sd estimated over every
surviving rate (0.063 against 0.070). Both are heuristics, and the reason they can disagree is
that neither is a test: **at two seeds a side the exact permutation floor is 2 / C(4,2) = 0.333**,
so nothing in this grid could have reached 0.05 whatever it showed. Week 6's arithmetic applies to
our own sweep exactly as it applies to Patrick's, which is the point of having built it.

**2. What identifying it would cost.** Apply the same arithmetic as Week 6 in reverse — how many
seeds a side to resolve a gap that small against that much seed noise. Hausa needs 9 and Swahili
10, which are affordable. Yoruba needs **164 seeds per rate** and Nyanja **413**. Across the five
languages and six rates that is **3,744 runs — 565 GPU-hours, about four times this entire
project**, spent to move a validation loss by two hundredths of a nat. The correct response to
that number is not to run it. It is to stop describing the winner as the winner.

**3. Longer, or more seeds?** The one place we paid for both arms: Yoruba's full corpus at 5e-4,
run at 12,000 steps for four seeds and 62,500 steps for six. A long run costs **5.4×** a short
one, so the comparison is one long run against five short ones with the best kept.

| what you buy with the same compute | expected validation loss |
|---|---|
| one run at 12,000 steps | 2.9274 |
| best of 2 at 12,000 steps | 2.8879 |
| best of 4 at 12,000 steps | 2.8855 |
| **one run at 62,500 steps** | **2.4067** |

**Length buys 0.521 nats. Best-of-four seeds buys 0.042 — twelve times less.** And it is not
close in the tail either: the *worst* of the six long runs (2.5326) beats the *best* of the four
short ones (2.8855) by 0.35 nats, with no overlap between the two sets at all. Given a fixed
budget and a rate you already believe in, **train longer**.

Two honest caveats on that table. Only four short seeds exist, so best-of-*five* — the true
like-for-like against one long run — cannot be priced; but going from three to four bought
0.0008 nats, so the missing row would not change the conclusion by anything visible. And this
holds *at a rate that works*. The trade reverses completely at a rate near the cliff, where the
long run is a coin flip and the short runs are cheap information about which way it lands.

**4. Then why run more than one seed at all?** Not for quality — for finding out which kind of
cell you are standing in. Collapse is partly a seed event: of the nine cells in the grid where
collapse happened at all, **two collapsed for one seed and not the other**. A single run at
Yoruba's 1e-3 tells you either "this rate is fatal" or "this rate is fine", with a coin deciding
which, and a sweep built from single runs inherits that coin at every rate. Seeds are not how you
buy a better number. They are how you find out whether your number was a number.

**The order to spend in, measured rather than asserted.** A wrong rate costs about **3 nats** —
Hausa at 1.5e-4 is 6.204 against 3.139 at its best. Five times the training length buys **0.52**.
Four seeds buy **0.042**. That is roughly **75 : 13 : 1**, and it is the whole budget policy:
find the usable band first, then spend everything left on length, and run a second seed for
information rather than for a better score.

And the sharpest way to say it: **inside the usable band, choosing the rate is worth about as much
as choosing the seed.** Comparing the spread across rates against the spread across seeds at one
rate, over trained runs only, the median ratio across the five languages is **1.2×** — Hausa 7.1,
Swahili 3.1, Yoruba 1.2, Nyanja 1.0, Igbo 0.4, where below 1.0 means the seed mattered *more* than
the rate did. Almost the entire measured effect of the learning rate is the cliff, not the slope.
Which means the sweep was worth running and the winner was not worth quoting, and those two
sentences have to be said together or the second one sounds like modesty.

**And the reproducibility consequence.** "We tuned the learning rate on a held-out set" is a
sentence in almost every paper, this one's drafts included, and on this evidence it usually means
*we ran a grid and reported the argmin.* With one seed per cell the argmin is substantially a draw
from the seed distribution, which is why a tuned setting reproduces so badly on somebody else's
machine: they are not failing to reproduce your model, they are failing to reproduce your luck.
The defensible version is narrower and costs nothing extra to say — **report the band, name the
rates that failed, and say how many seeds each cell got.** Every number in this section came out
of records that were already on disk, which is the only reason the question could be asked at all
after the fact rather than re-run from scratch — not one new model was trained to produce any of
it. `study_budget.py` recomputes all five parts and writes `runs/budget.json`; quote it from there
rather than from this sheet.

**Source.** `runs/lr_transfer.json` (`study_lr_transfer.py`) for the 60-run grid, and `runs/budget.json`
(`study_budget.py`) for every number in the luck-and-skill section below — that script recomputes all five
parts and trained nothing new to do it.

**What this makes possible.** If runs can fail this way — and if a second seed is the cheapest
instrument for noticing — the obvious engineering response is to catch the failures early and stop
paying for them. Week 9 tries exactly that, and fails.

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

**Source.** `runs/early_signal.json` (`early_signal.py`) for the eleven checkpoints and the abandonment
grid; `runs/clip_prevention.json` (`study_clip_prevention.py`) for the fifteen-seeds-a-side clipping test.

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

The AI disclosure belongs in this column too, in one line, because it is the same kind of claim
and it is derived the same way: **120 of the repository's 144 commits carry a `Co-Authored-By:
Claude` trailer.** That is `git log`, not a promise. Reference 19 has the rest, and report 09 §13
has the version worth actually reading — including the three occasions it was confidently and
completely wrong.

---

## Gaps

**Week 1 has no figure, and it is the only cell blocked on something outside this machine.** It wants the hardware
comparison as a graphic: the machines, their generation, whether the model fits in memory, and
what one run costs on each. Everything needed to produce it is written and self-contained —
`bench_portable.py` — but the measurements can only come from the machines themselves. **See the
Appendix.**

**Week 5's figure is drawn: `19-the-interface.svg`.** This sheet used to say it was "an hour in a
vector editor, not a script" — and hand-setting it would have been a mistake, for the reason the
figure itself demonstrates. Every quantity on it is counted at render time: the nine signatures
are read out of `mlm_api` with `ast`, and the line counts off the files. While it was being built
the folder went from 12,861 lines to 14,537, because we added four checks and two studies in three
days. **A hand-set panel would have been wrong within a week and would not have said so.**

Two design notes worth keeping, since it is the only figure on the board whose content is
typography. It is *emphasis*, not a chart: one accent for the surface a caller touches, grey for
everything behind it, and the segments keyed below the track rather than labelled with leader
lines — the first draft used `annotate()` and the leader landed on top of the subtitle. And the
function count is set beside the track rather than as a segment in it, because nine functions and
14,537 lines are not the same quantity and stacking them would be the dual-axis mistake in
disguise. Both defects were caught by rendering the figure and looking at it, which is the only
way that class of defect is ever caught — including `results(, …)`, which is what the obvious
f-string prints for a function whose arguments are all optional.

**The three released figures: two taken, one still free.** Cutting the metric-validity and
tokenizer cells released `11-metric-validity.svg`, `12-floors.svg` and `17-tokenizer-lottery.svg`.
[Report 13](13-the-top-board.md) has since claimed **12** and **17**, which is the right home for
both — the tokenizer lottery is their argument, and the same rule already applied to figures 01
and 02. **`11-metric-validity.svg` is still unclaimed by either board**, as are
`08-why-not-shorter.svg` and `09-scaling-with-cards.svg`, which no cell in either document
currently carries.

**Figure 03 was claimed by both boards. Settled — it is the top board's.**
`03-matched-steps-vs-compute.svg` was Week 7's figure here and is also the figure under report
13's matched-compute panel. Two posters side by side carrying the identical chart reads as a
mistake even when it is not, and neither document could see the collision from the inside — the
same shape as the index row that broke in the gap between two pull requests.

It goes upstairs, where it is load-bearing for their thesis and their board is read first: the
same rule that sent 01, 02, 12 and 17. **Week 7 carries no figure and sets its table as type
instead**, which is the better outcome and not a consolation — the finding is that one experiment
read two ways gives opposite answers, and a bar chart has to fight itself to show that. Two rows
of type at a size legible from three metres do it directly. `check_boards.py` exits 0 once this
lands.

**The retracted floor sentence goes with them, but keep it in mind.** The earlier draft claimed
the difference between the two tasks' floors explained why one task's scores were predictable and
the other's were not, argued from two shares being near-identical at 57% and 52%. Sweeping the
MasakhaNER floor moved it from one unswept cell at 0.4140 to **0.6261** across twelve rates, and
the shares are now 61% and 78% — so the sentence was wrong *and* its supporting numbers have since
evaporated. **Do not reuse it anywhere.** It is worth remembering while laying out the rest,
because it is the shape of mistake this whole board is about: an explanation that sounded right,
survived a reading, and rested on a coincidence.

---

## Order of work

1. ~~Figure for week 8~~ — **done**, `16-lr-transfer.svg`, on the completed 60-run grid.
2. ~~Figure for week 3~~ — **done**, `15-what-a-run-is-made-of.svg`, from the pipeline benchmark.
3. ~~Figure for week 5~~ — **done**, `19-the-interface.svg`, generated rather than hand-set.
4. ~~Offer figures 11, 12 and 17 to Patrick and Leon~~ — **done**; report 13 took 12 and 17.
5. ~~Settle figure 03~~ — **done**; it is the top board's, and Week 7 sets its table as type.
6. **Figure for week 1** (hardware). Blocked on the measurements in the Appendix — the only item
   on the board that cannot be produced from this machine.
7. **Regenerate everything and run the staleness check.** Do this *last*, once no study is still
   writing, and do it once. `check_links.py` belongs in the same pass, and so does the
   figure-collision check above: neither board can see either problem from the inside.

---

## What the board must not do

Quote a count typed by hand. Every number on it comes from `poster_bottom.ipynb`, which recomputes
from the records and flags any sentence in report 09 the data no longer supports.

The counts have moved three times while this was being written: 105 pretraining runs became 156,
then 197, checked today against `mlm_api.results('*')`. Fine-tuning records stand at **278**, of
which **161** are test-scored and **117** dev-scored — the dev rows exist to pick a learning rate,
are chosen on the items they are scored on, and are therefore not reportable numbers. `results()`
excludes them by default, which is the only reason none has ever reached a figure.

Anything quoted here is a value on the day it was written and will have moved by the time the
board is printed. **Recompute, do not copy from this sheet.**

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

# Provenance — where every number on this board comes from

Each cell above ends with a **Source** line naming the record a reader can open. This is the index
of those records. The layer being cited is deliberate: a report cites the **study record**, the
study record is computed from the ~700 per-run files beside it, and those are what the cards
actually wrote. The middle layer is the useful altitude — small enough to read, specific enough to
check.

| record | written by | what it holds |
|---|---|---|
| `runs/pipeline_bench.json` | `pipeline_bench.py` | what each stage of a run costs in wall-clock |
| `runs/budget.json` | `study_budget.py` | luck vs skill vs search budget on the 60-run grid |
| `runs/lr_transfer.json` | `study_lr_transfer.py` | five languages x six rates x two seeds |
| `runs/early_signal.json` | `early_signal.py` | can a doomed run be detected early |
| `runs/clip_prevention.json` | `study_clip_prevention.py` | does tighter clipping prevent divergence |
| `runs/swap_downstream.json` | `study_swap_downstream.py` | the vocabulary swap, carried downstream |
| `runs/tokenizer_seeds.json` | `study_tokenizer_seeds.py` | six pre-registered seeds a side |
| `runs/label_quantity.json` | `study_label_quantity.py` | the decisive labelled-data experiment |
| `runs/ner_control_sweep.json` | `study_ner_control_sweep.py` | the untrained NER floor, twelve rates |
| `runs/downstream_correlation.json` | `study_downstream_correlation.py` | does validation loss predict downstream score |
| `runs/gradient_table.json` | `gradient_table.py` | the vocabulary penalty across seventeen languages |
| `runs/gradient_languages.json` | `prepare_gradient_languages.py` | which languages were prepared |
| `runs/scaling_law.json` | `scaling_law.py` | the fitted data/compute surface |
| `runs/claims_audit.json` | `claims_audit.py` | every comparative claim against its null |
| `runs/hardware.json` *(not yet on disk)* | `bench_portable.py` | one run costed on each machine — NOT YET COLLECTED |

`check_provenance.py` resolves every `runs/*.json` named in any report against what is on disk, and
reports the other direction too: a record that exists and no report points at is an experiment that
ran and never reached a reader. It exits non-zero on a dangling citation, so it belongs in the print
gate beside `check_links.py` and `check_boards.py`.

**One citation dangles on purpose.** `runs/hardware.json` is named by Week 1 and the Appendix and
does not exist, because the measurements can only come from machines that are not this one. That is
the blocked cell, and the check stops failing the moment the benchmark is run.

---

# References

Everything the board leans on that we did not measure ourselves — **including the assistant**,
which belongs here rather than in a footnote, because a tool that wrote code you are now reading
results from is a source in exactly the sense the other eighteen entries are. The poster should
carry a shortened version of this. A board that quotes a significance threshold, a dataset and a
model without saying where any of them came from is asking to be taken on trust, which is the one
thing this board argues against.

Five entries left with the two cut cells — Levene, Pearson, Spearman, Simpson and Anscombe, which
between them supported the metric-validity and tokenizer-variance results. They are still cited in
report 09, where those results now live in full. If the tokenizer panel goes to the top board,
Levene goes with it.

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

*A note worth making to a class rather than hiding in a bibliography:* Fisher was also a prominent
eugenicist, and that is a matter of record rather than a matter of opinion. It does not make the
arithmetic wrong. It is a useful reminder that a field's foundational tools arrive attached to the
people who built them, and that using the tool is not the same as endorsing the person.

### Models

5. **Conneau, A. et al.** (2020). "Unsupervised Cross-lingual Representation Learning at Scale."
   *ACL 2020.* — XLM-R, our 277M-parameter multilingual baseline, loaded as
   `FacebookAI/xlm-roberta-base`. [arXiv:1911.02116](https://arxiv.org/abs/1911.02116)
6. **mmBERT** — loaded as [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base);
   246M parameters, reported by its authors as trained on roughly three trillion tokens across
   1,800 languages. *Cite the model card directly — check the current card for the paper reference
   before the poster is printed.*
7. **Liu, Y. et al.** (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." — the
   architecture our from-scratch models use.
   [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
8. **Devlin, J. et al.** (2019). "BERT: Pre-training of Deep Bidirectional Transformers for
   Language Understanding." *NAACL 2019.* — the 80/10/10 masking scheme every pretraining run here
   follows. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
9. **Ogueji, K., Zhu, Y. and Lin, J.** (2021). "Small Data? No Problem! Exploring the Viability of
    Pretrained Multilingual Language Models for Low-resourced Languages." *MRL Workshop, EMNLP
    2021.* — AfriBERTa, after which our 86M `afriberta` preset is shaped and named.
    [ACL Anthology](https://aclanthology.org/2021.mrl-1.11/)

### Data

10. **Adelani, D. et al.** (2024). "SIB-200: A Simple, Inclusive, and Big Evaluation Dataset for
    Topic Classification in 200+ Languages and Dialects." *EACL 2024.* — our topic-classification
    task, loaded as `Davlan/sib200`. 701 train / 99 validation / 204 test for Yoruba.
    [arXiv:2309.07445](https://arxiv.org/abs/2309.07445)
11. **Adelani, D. et al.** (2022). "MasakhaNER 2.0: Africa-centric Transfer Learning for Named
    Entity Recognition." *EMNLP 2022.* — our entity-recognition task. Read from
    [the CoNLL files in the masakhane-ner repository](https://github.com/masakhane-io/masakhane-ner),
    **not** via `load_dataset` — the HuggingFace copy ships a custom loading script and that path
    is no longer executed. [arXiv:2210.12391](https://arxiv.org/abs/2210.12391)
12. **FineWeb-2** — [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
    the source of every corpus in the language gradient, including all 69.1M tokens of Yoruba that
    exist there. English rungs come from `fineweb-edu` and `fineweb`.

### Method

13. **Smith, L. N.** (2018). "A disciplined approach to neural network hyper-parameters." — the
    one-cycle schedule every run here anneals under, which is why a run cannot be truncated and
    still compared. [arXiv:1803.09820](https://arxiv.org/abs/1803.09820)

### Search, seeds, and reproducibility

Week 8's luck-versus-skill section is the part of this board most likely to be met with "surely
somebody has already shown that." They have, repeatedly, and for longer than this project has
existed. Our contribution there is not the finding; it is that we measured it on our own grid
instead of citing it and carrying on.

14. **Bergstra, J. and Bengio, Y.** (2012). "Random Search for Hyper-Parameter Optimization."
    *Journal of Machine Learning Research* 13, 281–305. — why a grid spends most of its budget
    re-measuring dimensions that do not matter. Our sweep is a grid, which is defensible only
    because it varies one parameter.
    [JMLR](https://jmlr.org/papers/v13/bergstra12a.html)
15. **Dodge, J., Gururangan, S., Card, D., Schwartz, R. and Smith, N. A.** (2019). "Show Your
    Work: Improved Reporting of Experimental Results." *EMNLP-IJCNLP 2019.* — the expected-maximum
    curve as a function of search budget, which is exactly the best-of-*k* column in Week 8's
    second table, and the argument that a result is not reportable without the budget that
    produced it. [arXiv:1909.03004](https://arxiv.org/abs/1909.03004)
16. **Melis, G., Dyer, C. and Blunsom, P.** (2018). "On the State of the Art of Evaluation in
    Neural Language Models." *ICLR 2018.* — a plain LSTM, tuned with the same budget as its
    challengers, beats several architectures published as improvements on it. The cleanest
    demonstration that search budget masquerades as method.
    [arXiv:1707.05589](https://arxiv.org/abs/1707.05589)
17. **Reimers, N. and Gurevych, I.** (2017). "Reporting Score Distributions Makes a Difference:
    Performance Study of LSTM-networks for Sequence Tagging." *EMNLP 2017.* — report the
    distribution over seeds rather than a single number. Written about the same task family as our
    MasakhaNER half. [arXiv:1707.09861](https://arxiv.org/abs/1707.09861)
18. **Picard, D.** (2021). "torch.manual_seed(3407) is all you need." — the seed lottery measured
    at scale on vision models; the title is a joke and the measurement is not.
    [arXiv:2109.08203](https://arxiv.org/abs/2109.08203)

### The assistant, disclosed

19. **Anthropic Claude**, used through the Claude Code command-line tool — Opus 4.8, Opus 5 and
    Fable 5 over the term. Scaffolding, analysis code, written explanations, and the group's
    coordination email. **120 of this repository's 144 commits carry a `Co-Authored-By: Claude`
    trailer** — 83%, and that figure is `git log` rather than a sentence about honesty, which is
    the only form of disclosure this board is entitled to make. What it did not do: no model ran
    unattended overnight, no result was analyzed without a person reading the records, and the
    scheduler that spent every one of the 143.3 GPU-hours has no model in it. It was also
    confidently and completely wrong three times, in ways worth reading rather than summarizing.
    **The full account is [report 09 §13](09-the-bottom-report.md), "How we used AI, honestly."**

**Check before printing.** Reference 6 is the one to verify — mmBERT is recent enough that the
canonical citation may have changed since this was written, and a poster is a bad place to be
wrong about somebody else's model. Reference 19's commit counts move every time anybody pushes;
recompute them rather than reprinting these.
