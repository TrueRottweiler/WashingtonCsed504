# Build sheet: the bottom board

*What goes in each cell, what it says, and which file to drop in. Report 09 is the long form —
fifteen written panels. This is the board — ten cells. One is not a subset of the other, so this
sheet is the translation, and it is the thing to print next to you while laying it out.*

**Format.** Nine cells in a three-by-three grid, plus a full-width strip along the bottom, title
block above. The board is **3 ft × 4 ft — 91.4 × 121.9 cm**, portrait. Not A0, which this sheet
used to say: A0 is 84.1 × 118.9, so the real board is 7 cm wider and 3 cm taller.

With 6 cm margins, a 14 cm title block, a 20 cm bottom strip and 2.5 cm gutters, each grid cell
is **24.8 × 22.8 cm** and the strip is **79.4 × 20 cm**. The cell size the earlier version quoted
— 25 × 22 — survives the change, so nothing already laid out to it has to move; only the sheet
size and the strip width were wrong.

**What that means for type.** At 24.8 cm wide a cell holds about 90–120 words at 24 pt, which is
the smallest anyone will read standing at arm's length. The big number wants 90–110 pt. Read the
word budget as a hard limit rather than a target: the failure mode for a board like this is a
cell nobody finishes.

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

## The nine cells

Each cell gets: a **question** as the heading, one **figure**, one **number** set large, and
about 90–120 words. Anything longer will not be read standing up.

Every number below was recomputed from the records on 2026-08-10, not carried over. Four had
moved and one had never been in the data at all; those are marked.

| | question (the heading) | the number, set large | figure | checked |
|---|---|---|---|---|
| **1** | What does a run cost, and in what unit? | **62,500 steps = 1.024B tokens** | *needs one — see gaps* | exact |
| **2** | Why optimize before anything needs it? | **2.07×**, of which only **1.32×** is efficiency | `14-where-the-speedup-came-from.svg` | 25.2 → 12.2 min |
| **3** | What makes a record survive you? | **fingerprint `15abd33de5af`** | `07-dashboard.png` | live |
| **4** | Is this difference real? | **2.27×, not 1.0×** | `13-how-many-seeds.svg` | exact |
| **5** | Is your metric the right metric — and why does one task refuse to be predicted? | **−0.888 against +0.303** | `11-metric-validity.svg` + `12-floors.svg` | exact |
| **6** | Which of your units are not units? | **5.1× the compute, at "matched" steps** | `03-matched-steps-vs-compute.svg` | — |
| **7** | Is the tokenizer a cost, or a coin flip? | **3.9× the spread** *(p = 0.0098)* | *needs one — see gaps* | **changed** |
| **8** | Does a tuned setting transfer? | **7e-4: best for three languages, fatal to a fourth** | `16-lr-transfer.svg` | **new** |
| **9** | Detect the failure, or prevent it? | **0 of 11 checkpoints separate them** | `10-early-signal.svg` | **changed** |

### The five numbers that moved, and why

- **Cell 7 was `0.144 bits/char`.** At six seeds the penalty shrank to 0.059 with p = 0.37, and
  the two arms interleave — three of six large-vocabulary runs land below the small vocabulary's
  median. There is no location effect to report. What survives is the **spread**, 0.145 against
  0.037, F = 15.1, p = 0.0098, and the same shape downstream (4.0× wider on topic, 7.7× on
  entities). The cell's question changes from *how much does it cost* to *is it a cost at all*.
- **Cell 9 was `−24 GPU-hours`.** That number is in no record I can find. What the data says is
  worse for early stopping and better as a panel: across all eleven checkpoints the two outcomes
  **never separate** — the best doomed run has always gained more nats than the worst healthy
  one. Only one rule in the whole grid ever fires, and it kills two healthy runs and zero dead
  ones. The base rate is 35/195 = 17.9%, and dead runs waste 25.5 GPU-hours that no rule recovers.
- **Cell 5 absorbs the old cell 6.** They were one investigation split across two cells, which
  the coherence pass flagged and which is why the board had nothing to put between them.
- **Cell 2 gains its own qualifier.** 2.07× is real but only 1.32× of it is efficiency; the rest
  is a second card doing the same GPU-minutes in parallel. Conflating them is how a project
  claims 2× efficiency when it bought 1.3×.
- **Cell 8 has a figure now** — `16-lr-transfer.svg`, on the completed 60-run grid.

### Each cell in the five parts

Set these as five short labelled blocks. **Problem** and **Learning** get the most space; the
middle three can be one line each. The word budget is the whole cell, not each part.

**1 — What does a run cost, and in what unit?**
*Problem.* Every course so far measured training in epochs. **Hypothesis:** epochs are a unit.
*Approach.* Vary the dataset size and watch what an epoch means. *Results.* It stops meaning
anything: 62,500 steps × 128 × 128 = 1.024B tokens of updates regardless of corpus.
*Learning.* Epochs stop being a unit the moment the dataset is a variable. 62,500 is not a
choice, it is a budget divided by a batch. **Enables cell 2:** you cannot say a run got faster
until you can say what a run is.

**2 — Why optimize before anything needs it?**
*Problem.* Nothing needed to be fast yet. **Hypothesis:** optimizing early is premature.
*Approach.* Raise the batch 64→128, use the second card, measure the same work both ways.
*Results.* 25.2 → 12.2 minutes. 1.32× is real efficiency; the rest is parallelism.
*Learning.* The hours saved are not the point. Speed changes which experiments you are willing to
**start**. **Enables cells 4–9:** every result below rests on running the same cell three to
fifteen times, which at 25 minutes a run nobody would have done.

**3 — What makes a record survive you?**
*Problem.* Two runs silently overwrote each other; two people built "the same" corpus and got
different vocabularies. **Hypothesis:** a descriptive filename is enough.
*Approach.* Put every setting that moves the number into the record's name, and hash the
vocabulary. *Results.* `15abd33de5af` — a corpus is now a thing you can check rather than a thing
you remember. *Learning.* A filename is an identity. **Enables cell 4:** you cannot ask whether
two numbers differ until you are certain they came from two different experiments.

**4 — Is this difference real?**
*Problem.* Our rule was "bigger than the cell's own seed spread." **Hypothesis:** that is a test.
*Approach.* Derive what the bar actually is at three seeds. *Results.* **2.27×**, not 1.0× — and
the exact permutation test cannot return a p below 0.10 at three a side no matter how far apart
the arms land. *Learning.* Our rule was half a rule: sound at rejecting things below the noise,
silent about things just above it. **Enables 5–9:** this is the instrument the rest of the board
is read through, and it retired two of our own claims.

**5 — Is your metric the right metric, and why does one task refuse to be predicted?**
*Problem.* We minimized validation loss for a term without checking it predicts anything.
**Hypothesis:** lower loss means a better model. *Approach.* Correlate final loss against both
downstream tasks across the 16 models that trained. *Results.* **−0.888** on topic classification
(p < 0.001), **+0.303** on entities (p = 0.25). The aggregate −0.935 was three under-trained
models holding a line. *Learning.* Entity recognition hands every working model 0.340–0.384, a
band **0.044** wide; topic classification spreads them over **0.143**. A benefit everyone
receives equally cannot be predicted by anything. **Enables cell 6:** if the metric can be wrong,
so can the unit under it.

**6 — Which of your units are not units?**
*Problem.* Two vocabularies produce two losses that are not on one scale. **Hypothesis:** matched
steps is a fair comparison. *Approach.* Convert to bits per character, and count what matched
steps actually bought each arm. *Results.* A 250k output head is **5.1×** the compute per step,
so "matched steps" handed one arm five times the budget and reversed the answer.
*Learning.* Fairness is a property of the unit, not of the intention. **Enables cell 7:** only
now can the tokenizer question be asked in a unit that can answer it.

**7 — Is the tokenizer a cost, or a coin flip?**
*Problem.* Report 08's headline was a 0.144 bits/char penalty at three seeds.
**Hypothesis:** the large vocabulary costs a fixed amount per character. *Approach.* Six seeds a
side, fixed in advance from the power calculation, written into the script before the runs
started. *Results.* The penalty **shrank** to 0.059 (p = 0.37) and the arms interleave. The
spread does not: 0.145 against 0.037, F = 15.1, **p = 0.0098** — and 4.0× / 7.7× wider downstream.
*Learning.* It is not a tax, it is a lottery. Testing only the mean would have recorded a null
and thrown the finding away. **Enables cell 8:** a setting whose effect is on variance is exactly
the kind that will not transfer.

**8 — Does a tuned setting transfer?**
*Problem.* We claimed a tuned learning rate was reusable across languages.
**Hypothesis:** the best rate is roughly the same everywhere. *Approach.* Five languages, six
rates, two seeds, 60 runs. *Results.* Hausa, Nyanja and Swahili all peak at **7e-4**. Igbo
collapses at 7e-4 and at every rate above it. *Learning.* The risk is not a slightly worse model,
it is a wasted night that looks like a result. **Enables cell 9:** if runs fail this way, the
obvious response is to catch them early.

**9 — Detect the failure, or prevent it?**
*Problem.* 17.9% of runs die and waste 25.5 GPU-hours. **Hypothesis:** a doomed run is visible
early. *Approach.* Score every run against its own untrained baseline at eleven checkpoints and
price every abandonment rule. *Results.* The two outcomes **overlap at all eleven** — the best
doomed run always looks better than the worst healthy one. One rule in the grid ever fires; it
kills two healthy runs and zero dead ones. *Learning.* Do not build the detector. Spend the
effort on prevention, and measure that too — clipping, tested at fifteen seeds a side, does
**not** prevent divergence (Fisher p = 1.00) though it does improve the runs that survive.

---

## The bottom strip — week 10

**Writing it down so it stays true.** Full width, three columns inside it:

| left | middle | right |
|---|---|---|
| **What it cost.** `06-what-it-cost.svg`, plus the three routes: workstation ~$24,000 / Colab ~$120 / a plugged-in laptop over a month of nights, free. *The workstation bought latency, not access.* | **The five constants.** The table from §10 verbatim — a setting chosen for a good reason in one context, silently deciding a result in another. | **How this stays honest.** Numbers generated from records, not typed, and a gate that tests each claim against its null rather than checking the digits still match. It currently reports **9 claims: 6 supported, 2 not, 1 underpowered** — and two of the three failures are ours. |

The right-hand column is the one to get right, because it is the only cell that says what this
board is *for*. The draft above is better than the earlier version, which boasted that a
staleness check caught a count moving from 105 to 156. That is a spellchecker's virtue. The
harder and more useful claim is that the project runs a test against the null for every
comparative sentence it makes, publishes the count of its own failures, and did not quietly drop
the two claims that failed — a tokenizer penalty and a floor explanation, both of which had
already been written up and one of which had been emailed.

---

## Gaps — three cells have no figure yet

**Cell 1 needs one.** The hardware comparison table would work as a graphic: the cards, their
generation, whether the model fits, and what a run costs on each. Blocked on the Colab and laptop
measurements.

**Cell 5 is done — `12-floors.svg`.** It was going to be `01-headline.svg`, but that figure has
gone to Patrick: it is his comparison, its selection rule is best-on-test, and whoever fixes the
rule has to own the numbers.

Building the replacement caught an error in the writeup. The first version showed the floor as a
share of achievable and claimed the difference between those shares explained why the two tasks
disagree. It does not. What separates them is the *variability* of the gain: entity recognition
hands every working model between 0.340 and 0.384, a band 0.044 wide, while topic classification
spreads them over 0.143. A benefit everybody receives equally cannot be predicted, which is why
loss tracks one task and not the other. The figure now draws the band rather than a single best.

The retraction was originally argued from the shares being near-identical — 57% and 52%. **Do not
reuse that sentence.** Sweeping the MasakhaNER untrained floor moved it from one cell at 0.4140
to 0.6261 across twelve rates, and the shares are now 61% and 78%, seventeen points apart. So the
premise of the original retraction is gone while the retraction itself still stands, for the
better reason: near or far, a floor is a fact about the *task*, and it cannot explain why one
task's scores are predictable and the other's are not. The band widths do that, and they are
unaffected by where the floor sits.

It is worth keeping both versions of this in mind while laying the cell out, because it is the
same shape as the mistake the cell is about — an explanation that sounded right, was checked, and
turned out to be resting on a coincidence that has since evaporated.

**Cell 8 is done — `16-lr-transfer.svg`**, on the completed 60-run grid. Five small multiples,
loss against rate, one panel per language.

Two drafts of it were wrong, and both mistakes are the board's own subject matter, so they are
worth knowing before anyone edits the figure. The first joined Igbo's collapsed cells into the
curve, which its own docstring said not to do: the failure threshold was anchored on
random-guessing loss, and the collapses sit nowhere near random, so nothing tripped it. It is
anchored on each language's own best now. The second plotted cell means — and three cells in this
grid are split, one seed training and the other collapsing. Yoruba at 1e-3 is 3.326 and 5.540, so
plotting 4.433 as a point on a curve is the same error the clipping ladder had. Every seed is
drawn now, and a split cell gets a vertical tie and no mean marker.

**Cell 7 needs one, and it is new.** The tokenizer cell used to be a bar with one number on it.
Now that the finding is about spread rather than location, it wants a strip plot: twelve dots,
six per arm, so the reader sees the large-vocabulary arm splitting into two clusters while the
small-vocabulary arm stays tight. `04-two-outcomes.svg` is the closest existing template.

---

## Order of work

1. ~~Figure for cell 5~~ — **done**, `12-floors.svg`.
2. ~~Figure for cell 8~~ — **done**, `16-lr-transfer.svg`.
3. **Figure for cell 7** (the tokenizer strip plot). Nothing blocks it; the data is on disk.
4. **Figure for cell 1** (hardware). Still blocked on the Colab and laptop measurements — the
   only cell on the board waiting on something outside this machine.
5. **Regenerate everything and run the staleness check.** Do this *last*, once no study is still
   writing, and do it once.

## What the board must not do

Quote a count typed by hand. Every number on it comes from `poster_bottom.ipynb`, which recomputes
from the records and flags any sentence in report 09 the data no longer supports.

The counts have now moved twice while this was being written: 105 pretraining runs became 156,
and 156 became **213**. Fine-tuning records stand at **215**, of which 99 are dev-scored and
therefore not reportable numbers. Nine runs now carry per-item predictions, which is what makes a
paired test possible on anything trained after 10 August and impossible on everything before it.

None of that is an erratum, and the reason is the only process claim this board makes: nothing
was typed twice.
