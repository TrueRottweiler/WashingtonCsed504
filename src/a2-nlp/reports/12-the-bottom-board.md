# The bottom board — what to print

*The poster itself: the grid, the measurements, and the words that go in each cell. This is the
thing to hold while laying out. [Report 09](09-the-bottom-report.md) is the long form of the same
ten panels — go there when a cell makes somebody want the whole argument.*

> **Read this first if you laid anything out before 12 August.** Every measurement in the previous
> version was wrong, because it assumed a 3 ft × 4 ft board and the UW template is **2 ft × 3 ft**.
> The real numbers were read out of `ResearchPoster_Template_Vertical_2023.pptx` rather than
> assumed. The board is 2.25× smaller in area than planned and the body type is 18 pt rather than
> 24, which halves the words per cell from 90–120 to **55**.

---

## Format — measured from the template, not assumed

| | value |
|---|---|
| board | **24 × 36 in** · 61.0 × 91.4 cm, portrait |
| columns | **three**, at x = 1.50 / 8.75 / 16.13 in, each **6.35 in** wide |
| header band | 8.5 in deep, full width |
| body area | y = 9.25 → 34.60 in, so **25.35 in** of column |
| title | **115 pt** · author line 24 pt |
| section header | **40 pt** |
| body | **18 pt** |
| photo caption | 12 pt |

**The grid this board uses — v2 of the redesign.** The **top board's own three columns**: x =
1.50 / 8.75 / 16.13, each **6.35 in**, 0.90 gutters. Two rows of three panels, a full-width band
between them, then two rows of three. **Five charts at one column each**, drawn at the exact size
they print, so in-chart type is real 12–23 pt rather than a reduction of a report figure. No panel
borders: a heading, a gold rule, whitespace. Geometry in `build_posters.FACTORY_GEO`.

**Word budgets.** A chart panel's caption is **45–60 words** — roughly six lines in a 6.35 in
column at 18 pt, twice what the first redesign allowed. That is deliberate and it is Jeffrey's
call: a caption of fifteen words next to a chart nobody can read closely is two things failing
together. A prose panel runs **45–60 words** behind a heading that IS the finding. The
55-words-per-cell arithmetic above described the old uniform nine-cell grid and survives in git
history.

**Headings are statements, not questions**, and they are set at **32 pt** rather than the
template's 40. The top board's section headers are single words — ABSTRACT, RESULTS — and 40 pt
suits them; a whole sentence at 40 pt in a 6.35 in column wraps to three lines and eats the chart.
The **title** takes the size back, at 72 pt in the theme's own `Encode Sans Normal Black`, which is
the face the top board's title is set in and not the Normal weight this board had been bolding.

---

## Title block

> ## CSED 505: Building a Model Factory
> *the course that would come after 504 — ten weeks, derived from one term of building one*
>
> **501** the statistics · **502** the mechanics · **503** the language stack · **504** scale —
> **All four end when a model finishes training. None covers needing a hundred models and having
> to believe the differences between them.**

Author line, in the header band: *Jeffrey Stall · Patrick Kwok · Leon Wan — CSED 504. This
board is the machinery; the upper board is the experiment it served.*

**Goals**

> Build the machinery a hundred-model study needs — one training loop, one record format, one
> benchmark — and measure it honestly enough that the numbers survive being checked.

The assignment asks for team member names. Both boards carry all three in full — **Jeffrey Stall,
Patrick Kwok, Leon Wan** — with the role split named once, on this board, in the strip.

*(17 Aug: "A2-NLP" is off both author lines — it is our internal project code and means nothing to
a reader, where the course number identifies the work. "Leon" is "Leon Wan" wherever the board
lists the team, so the two boards do not render a teammate's name two ways side by side. The
first-name reference in cell 5 is narrative and stays as it is.)*

**Citations**, set as the footer line across the foot of the board. This board is the factory, so
its footer names the stack the factory is built on; the top board's names the one the experiment
used. Neither repeats the other:

> **Built with** PyTorch and HuggingFace `tokenizers`, plus NumPy and matplotlib; no training
> framework above them — the loop, the masking, the scheduler and the run records are this
> project's own. Long jobs ran on a 2 × RTX PRO 6000 Blackwell workstation, benchmarked against
> Colab A100 / L4 / T4, a MacBook Pro (M4 Pro, MPS) and an 8 GB laptop. · Nineteen references,
> in full in report 09 §References.

---

## The cells — v2 of the redesign, 18 August

> **What changed from the 17 August version, and why.** Printed at 24 × 36 the charts were too
> large: a 10.33 in figure is most of a two-column row, and five of them made the board a slideshow
> of charts with captions. Jeffrey's read, and it is the design now: **the same charts at one
> column, 6.35 in, about two thirds of the area** — at which width five of them fit three-across,
> two chart rows do the work three were doing, and the room that frees pays for a full-width
> speedups table and captions twice as long.
>
> Three consequences worth writing down because they are rules now rather than choices.
>
> **The grid is the top board's grid** — columns at 1.50 / 8.75 / 16.13, 6.35 in wide. The two
> halves hang one above the other and a reader's eye runs down a column across both, which is worth
> more than any styling decision either board makes alone.
>
> **No containers.** The top board has none: headings, a rule, whitespace. A rounded rectangle
> around every panel was the single thing that read as belonging to a different poster. The stat
> rail keeps its purple fill, because that is their KEY DETAILS device and matching it is the point.
>
> **A 6.35 in chart holds axis labels and value labels and nothing longer.** Every sentence that
> used to sit inside a drawing — figure 25's four-part key, figure 26's callout, figure 26's two
> footnotes — is in the caption now, set in the board's own 18 pt rather than in whatever a
> matplotlib text call happened to be scaled to. The chart carries the picture; the caption carries
> the claim.

Layout, top to bottom: a chart row, the full-width speedups table, a second chart row, then the
two prose rows. That is the order the work happened in, and the full-width band between the chart
rows keeps them from reading as one six-panel block. Geometry lives in
`build_posters.FACTORY_GEO`; the words below are the words on the wall, as ever.

### 1 · You do not need the workstation

**Figure:** `26-poster-hardware.svg`

> An **8 GB laptop GPU beats the free tier, even unplugged**. The two paid Colab tiers bill the
> same — **$111 for 7 days** against **$112 for 30** — so the faster tier buys time, not money.
> Electricity for all 148 GPU-hours was **$7**; the workstation that ran them, **$24,000**.

*50 words · `runs/hardware.json` · `bench_portable.py`*

### 2 · Cheap stays interactive; expensive goes in a queue

**Figure:** `23-poster-run.svg`

> Preparing the corpus — train the tokenizer, encode it, load it onto the card — takes **53
> seconds**. One pretraining run takes **37 to 85 minutes** depending on the model. A gap of **41×
> to 96×** is not a judgment call: everything cheap stays in the notebook, everything expensive
> goes in a queue, and **both paths write the same record**.

*60 words · `runs/pipeline_bench.json`*

### 3 · The factory, in numbers

> **197** pretraining runs, every one from scratch · **892** fine-tuning runs on top of them,
> across two tasks · **17** languages in four writing systems · **148** GPU-hours on two cards,
> 71 kWh at the wall · **62,500** steps a run — 1.024B tokens of updates, the unit that makes two
> machines comparable · **0** identity collisions, because a vocabulary carries a hash

*The rail: six value–label rows, split on the middots by the builder. The labels run two to three
times longer than the first version's — a number nobody can read the units of is decoration, and
"197 pretraining runs" does not say that every one of them started from random weights.*

### 4 · What we made faster

> **2.07× on the same four cells** — only **1.32×** of it is efficiency; the rest is a second card
> nobody had used. Every row had to make the *same* experiment faster: raising the batch while
> holding **steps** fixed would have been a bigger run that scores better, not a speed-up.

| what changed | measured | kept? |
|---|---|---|
| dataset resident on the card, no DataLoader | 13.8k → **18.5k img/s** | yes |
| tokenize once, up front, into a flat array | 53 s against 85 min — **96×** | yes |
| batch 64 → 128 | **1.31×** — 2048 is *slower* | yes |
| both cards, longest-budget-first | **1.57×**, 91% and 93% busy | yes |
| bf16 instead of fp16 | **1.34×**, and no loss-scale | yes |
| fp16 **+** channels_last | **3.5× slower** on sm_89 | no |
| drop the per-step `.item()` | **2%** — a live loss is worth it | no |
| `torch.compile` | no Triton on Windows | can't |

*50 words · report 09 §Optimization · `runs/pipeline_bench.json`*

### 5 · You cannot judge a run early

**Figure:** `27-poster-cliffs.svg`

> Every run that learns sits flat and then falls off a cliff — and **no two fall at the same
> step**. The earliest is **2,200**, the latest **48,500**: 15% to 90% of the budget. So "still
> flat at step k" is evidence of nothing. **35 of 195 runs never learned**, and the seven that
> blew up did not do it before a quarter of the budget was gone.

*68 words · `runs/early_signal.json`*

### 6 · The setting four languages want destroys the fifth

**Figure:** `25-poster-transfer.svg`

> Five languages, six learning rates, sixty runs. Every cell is scored against **that language's
> own best**: ★ its winner, ✕ more than 1.5 nats behind it, ◐ one seed of two, and a number for how
> far behind. **Four languages land at their best in the outlined column. Igbo collapses there** —
> and no winner clears its own seed noise.

*60 words · `runs/lr_transfer.json` · `runs/budget.json`*

### 7 · One screen, both cards, a rush order at 9 p.m.

**Figure:** `28-poster-dashboard.svg`

> On 9 August there were **44.7 GPU-hours committed across 69 runs** and Patrick needed a
> learning-rate sweep before he could write anything. A fleet pins to one card, so the urgent work
> took **card 1** while the overnight queue kept card 0. Twenty minutes to running — and the
> promise was a measured **"fifty minutes"**, not "sometime tonight".

*58 words · report 09 §The rush order · `dashboard.py`*

### 8 · An identity is everything that changes the answer

> Two runs silently overwrote each other — same filename, different vocabularies, and neither
> record said so. A filename is an identity, so the vocabulary carries a **hash** rather than a
> label: `15abd33de5af`. **197 pretraining and 892 fine-tuning runs later, no collision.**

*41 words · `mlm_api.results()` · `ft_api.results()`*

### 9 · What "three seeds" really means

> **Run it three times against three, and the best you can ever claim is "one chance in ten."**
> There are only 20 ways to sort six results into two groups of three, and two look this good by
> luck alone. We required a gap **2.27× the seed spread** instead, and **retired two of our own
> claims** that failed it.

*59 words · `runs/claims_audit.json`*

### 10 · A factory that makes one thing is not a factory

> Seventeen languages in four scripts, because one language is an anecdote and seventeen is a
> slope. English runs a ladder from **4M to 1,024M tokens**, so the Yoruba result has a **ruler**
> to measure against. **Twelve languages trained in 48 minutes** — 96 on one card. Adding one is a
> function call.

*52 words · `runs/gradient_table.json` · report 09 §Why seventeen*

---

## The strip — the bottom row, which is what a reader leaves with

*Set at the same 18 pt as every other panel now. The strip ran two steps down when it lived in a
4.45 in box; with the boxes gone it is three more columns on the same grid, and setting it smaller
was only ever compensating for a box that was too short.*

### Two words that sound alike and do opposite jobs

> **Accumulation** runs one 16,384-token step as several smaller passes. It fixes **memory** — it
> is why an 8 GB laptop and a 24 GB Mac train the identical step, at **6.1** and **16.2 GB**.
> **Clipping** caps the gradient norm and is supposed to fix **divergence** — at fifteen seeds a
> side it **did not**: 4 of 15 against 3, Fisher **p = 1.00**.

*63 words · `runs/clip_prevention.json` · report 09 §Memory*

> **Why this replaced "nine functions, nothing else to import" on 18 August.** Jeffrey's call:
> the interface panel is a story about our repository, and this is the tradeoff a student with
> 8 GB of VRAM actually hits. It also keeps a measured *negative* on the board — clipping does
> not do the job its reputation gives it. The layering argument the interface panel carried
> survives in cell 10, where adding a language is one function call with a measurement attached.
> Report 09 §The interface keeps the nine signatures and figure 19.

### What we do not claim

> **Nobody on this team reads Yoruba** — every judgement here is a benchmark number, every claim
> relative. The corpus is scraped web text nobody consented to. And the audit gate prints
> **9 claims: 6 supported, 2 not, 1 underpowered** — two of the failures ours, printed rather than
> dropped.

*49 words · `runs/claims_audit.json` · report 09 §Ethics*

### Next · sources · AI

> **Next:** a wall meter, a corpus audit, a Yoruba-speaking evaluation we could not do.
> **Sources:** RoBERTa (Liu 2019), XLM-R (Conneau 2020), FineWeb-2 (Penedo 2025), MasakhaNER and
> SIB-200 (Adelani 2022, 2024) — nineteen in report 09. **AI: Claude Code** throughout; it made
> building faster and **verifying no faster at all**.

*49 words · report 09 §Panel 10*

> **The commit ratio is off the board as of 17 August, and getting it wrong three times is the
> reason.** It said 120 of 144, then 96 of 168, and a recount on the 17th made it 101 of 173 —
> squash-merging collapses a branch into one commit, so the ratio falls as the history is tidied.
>
> Worse, the method was wrong. **`git log --format='%b' | grep -c` returns 228, more trailers
> than there are commits**, because a squashed commit carries one trailer per commit it absorbed:
> counting *lines* counts co-authorships where the sentence claims *commits*. A precise-looking
> number that has been wrong at every telling is worth less than the plain statement, and the
> plain statement is what the assignment asks for. Report 09 keeps the derivation.

> **The team names came off this block on 18 August.** They are in the header band, on the author
> line, where a reader looks for them — and the assignment asks for them once, not twice.

---

## Does the board answer the assignment?

Checked against the rubric rather than assumed, because a panel that satisfies nothing required is
expensive wall space. The pair of boards is what gets marked; this column is only the bottom one.

| required | where it is on this board |
|---|---|
| Team member names | header band, on the author line — once, not twice |
| Problem / motivation | title block, and cells 1 and 2 |
| Goals | title block, and report 09 §Goals in full |
| Tool stack | cells 2, 4, 7 and strip 1 |
| Pre-existing vs from scratch | cell 10; the top board's panels 3 and 5 |
| How performance was explored | cells 1, 4, 5, 6 |
| Dataset EDA | cell 3's inventory and cell 10; the top board's panel 2 carries the corpus EDA |
| Results / summary statistics | cells 1–10, every one a measured number |
| Discussion / limitations | cells 5, 9 and strip 2 |
| **Ethical impact** | **strip 2**, with the full account in report 09 |
| Next steps | strip 3 |
| Citations + AI statement | strip 3, and the footer, with all 19 references in report 09 |

---

## Notes on the hardware panel

**Cell 1 draws `26-poster-hardware.svg`, not `21-hardware.svg`.** The report keeps figure 21 —
eight columns, both model shapes, a capped axis and a cost table beside it — because a reader at
40 cm can work through all of that. The poster version drops to seven rows and one model shape,
puts each machine's project cost in its own row label, and states the one thing a reader can act
on: **an 8 GB laptop GPU beats the free tier, even unplugged.** Both draw from the same reduced
records, so they cannot disagree about a number.

**A Colab A100 comes within a fifth of one Blackwell card** — 46 minutes a run against 37. The L4
is **4.9× slower, 30 days of card time, $112**. The free T4 is the floor of what somebody with no
budget gets — **8.3×, 53 days, and the 98M model fits there too**. The laptop a student most
likely already owns runs **4.3 h / 10.2 h per run, 56× its own CPU, and beats the free T4 on both
model sizes even on battery.** More rows drop into `runs/hardware.json` without touching either
figure.

> **Read this paragraph before quoting any of it.** On 14
> August the benchmark stopped timing a stripped-down step and started timing the one
> `mlm_train.pretrain()` actually runs — batches built and masked on-device, gradients clipped,
> the loss read back every step. **Every GPU on this figure is now measured that way, and every
> one of them at least twice.** No `‡` remains on any bar.
>
> **One row is not, and the `‡` logic cannot see it.** The `--cpu` baseline sits in the side
> panel rather than the bar panel, so it is not in the list the mark is computed from — it is
> still a bare-step reading, and this paragraph said "every machine" until somebody checked.
