# Build sheet: the bottom board

*What goes in each cell, what it says, and which file to drop in. Report 09 is the long form —
fifteen written panels. This is the board — ten cells. One is not a subset of the other, so this
sheet is the translation, and it is the thing to print next to you while laying it out.*

**Format.** Nine cells in a three-by-three grid, plus a full-width strip along the bottom.
Title block above. Assume A0 portrait, so each grid cell is roughly 25 × 22 cm and the strip is
about 84 × 18 cm.

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

| | question (the heading) | the number, set large | figure | source |
|---|---|---|---|---|
| **1** | What does a run cost, and in what unit? | **62,500 steps = 1.024B tokens** | *needs one — see gaps* | §5 |
| **2** | Why optimize before anything needs it? | **2.07×** | `09-scaling-with-cards.svg` | §4 |
| **3** | What makes a record survive you? | **fingerprint `15abd33de5af`** | `07-dashboard.png` | §6–8 |
| **4** | Is this difference real? | **2.27×, not 1.0×** | `13-how-many-seeds.svg` (+ `04-two-outcomes.svg` if room) | §10 |
| **5** | Why can one task be predicted and not the other? | **0.044 wide vs 0.143** | `12-floors.svg` | §14 |
| **6** | Which of your units are not units? | **0.144 bits/char** | `03-matched-steps-vs-compute.svg` | §10 |
| **7** | Is your metric the right metric? | **−0.888 vs +0.303** | `11-metric-validity.svg` | §14 |
| **8** | Does a tuned setting transfer? | **5e-4 or 7e-4, depending** | *needs one — see gaps* | §14 |
| **9** | Detect the failure, or prevent it? | **−24 GPU-hours** | `10-early-signal.svg` | §14 |

### What each cell has to land, in one sentence

1. Epochs stop being a unit the moment the dataset is a variable; 62,500 is not a choice, it is a
   budget divided by a batch.
2. Speed bought before it was needed is what made the corrections affordable — four of the five
   came from a cheap re-run somebody did on a hunch.
3. Two runs silently overwrote each other and two people built "the same" corpus differently; a
   filename is an identity and a vocabulary needs a hash.
4. Spread is a property of the cell, not a constant — and "bigger than the spread" is half a
   rule: at three seeds a difference needs to be 2.27x it, which our own headline number was not.
5. A benefit every model receives equally cannot be predicted by anything — which is why loss
   tracks topic classification and not entity recognition, and it is not about where the floor is.
6. Two vocabularies give two losses that are not on one scale, and matched *steps* handed one arm
   5.1× the compute and reversed the answer.
7. We minimized validation loss for a term; it predicts one task strongly and the other not at all.
8. The best rate is not the same across languages, and one diverges exactly where others improve.
9. Every abandonment rule loses money, because a 12% base rate against an asymmetric penalty
   cannot close.

---

## The bottom strip — week 10

**Writing it down so it stays true.** Full width, three columns inside it:

| left | middle | right |
|---|---|---|
| **What it cost.** `06-what-it-cost.svg`, plus the three routes: workstation ~$24,000 / Colab ~$120 / a plugged-in laptop over a month of nights, free. *The workstation bought latency, not access.* | **The five constants.** The table from §10 verbatim — a setting chosen for a good reason in one context, silently deciding a result in another. | **How this stays honest.** Numbers generated from records, not typed. A staleness check that names any sentence the data no longer supports. The count moved 105 → 156 while this poster was being written, and the check caught it. |

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
disagree. It does not — the shares are 57% and 52%, near enough identical. What separates them is
the *variability* of the gain: entity recognition hands every working model between 0.340 and
0.384, a band 0.044 wide, while topic classification spreads them over 0.143. A benefit everybody
receives equally cannot be predicted, which is why loss tracks one task and not the other. The
figure now draws the band rather than a single best.

**Cell 8 needs one.** The five-language learning-rate table wants to be a small-multiples line
plot — five panels, loss against rate, with the divergence at Igbo visible. Waiting on the
extended sweep to 1e-3, which is queued behind study 4 on card 0.

---

## Order of work

1. ~~Figure for cell 5~~ — **done**, `12-floors.svg`.
2. **Figure for cell 8** (learning-rate transfer). Needs the extension to finish tonight.
3. **Figure for cell 1** (hardware). Needs the Colab and laptop numbers.
4. **Week 9 write-up** once study 4 lands, ~23:40 tonight.
5. **Regenerate everything and run the staleness check.** Do this *last*, once no study is still
   writing, and do it once.

## What the board must not do

Quote a count typed by hand. Every number on it comes from `poster_bottom.ipynb`, which recomputes
from the records and flags any sentence in report 09 the data no longer supports. The counts moved
by half while this was being written — 105 pretraining runs became 156 — and the only reason that
is a footnote rather than an erratum is that nothing was typed twice.
