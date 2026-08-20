# The bottom board — what to print

*The poster itself: the grid, the measurements, and the words that go in each cell. This is the
thing to hold while laying out. [Report 09](09-the-bottom-report.md) is the long form of the same
material, organized as the **ten weeks** of the course this board proposes — the report runs in
week order, the order the traps arrive; the board is arranged by importance instead. A caption's
§ reference names the report section that carries the full argument.*

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

*Those are the template's own measurements — what the page ships with, kept for reference. The
board deliberately departs from three of them: body **17 pt**, section headings **28 pt**, title
**72 pt**. The paragraphs below say why; a layout pass should follow those, not this table.*

**The grid this board uses — final, 18 August.** The **top board's own three columns**: x =
1.50 / 8.75 / 16.13, each **6.35 in**, 0.90 gutters. Five rows: hardware chart + memory + the
rail · budgeting + the speedups table (two columns) · three charts · three prose panels ·
languages (two columns) + sources. **Four charts**, each drawn at the exact size it prints — the
hardware chart bleeds 0.55 in into the page margin and the gutter (7.45 × 3.09 in); the other
three hold one column (6.35 × 2.60 in). No panel borders: a heading, a gold rule, whitespace.
Geometry in `build_posters.FACTORY_GEO`; figure sizes in `poster_figures.POSTER_BOX`.

**Word budgets and type.** Body text is **17 pt everywhere** — captions, prose panels and list
rows alike (Jeffrey's number, one size across the board); list rows in the memory and budgeting
panels set at 17 with their bold claim leading, the languages rows a step airier at 8 pt between
paragraphs, sources at **13 pt** in full citation form, the footer at 12. A chart panel's caption
is **45–60 words**; a prose panel about the same behind a heading that IS the finding. Paragraphs
sit **6 pt apart**, and a bare `>` line inside a blockquote is the sheet's paragraph-break marker
— both from Jeffrey's hand-edit pass on the printed board, transcribed here so builds reproduce
his file. The 55-words-per-cell arithmetic above described the old nine-cell grid and survives in
git history.

**Headings are statements, not questions**, and they are set at **28 pt** rather than the
template's 40. The top board's section headers are single words — ABSTRACT, RESULTS — and 40 pt
suits them; a whole sentence at that size in a 6.35 in column wraps to three lines and eats the
chart, and the 32/13 pairing of an earlier pass put a canyon between a heading and its own text.
The **title** takes the size back, at 72 pt in the theme's own `Encode Sans Normal Black`, which
is the face the top board's title is set in and not the Normal weight this board had been
bolding. The author line takes the subtitle's face and size — 19 pt of the same family — because
19 pt of Open Sans beside 19 pt of Uni Sans read as two different sizes.

---

## Title block

> ## CSED 505: Building a Model Factory
> *501 the statistics · 502 the mechanics · 503 the language stack · 504 scale — All four end when a model finishes training. None covers needing a hundred models and having to believe the differences between them.*

Author line, in the header band: *Jeffrey Stall · Patrick Kwok · Leon Wan — CSED 504. This board
is the machinery; the upper board is the experiment it served.*

**The subtitle slot carries the thesis**, which is Jeffrey's arrangement and the right one: the
501/502/503/504 sentence is what this board is *for*, and it had been sitting in a gold box below
the band while a tagline about a course number that does not exist held the slot above it. The
author line runs under the template's gold rule and carries the course and the two-board note.

The gold box is **empty and not rendered** as of the 18 August proof pass. It briefly carried
Goals, and Jeffrey's read was right: a goals line under a thesis line is two sentences doing one
job, and the header was cluttered. The requirement check: the assignment asks for Goals on the
submission, the **top board carries Patrick's Goals block in full**, and the pair of boards is
what gets marked — this board's own rubric table has said so from the start. One goals statement
across the pair; report 09 §Goals holds the long form.

*(18 Aug, second pass: the names went above the gold rule for one build and came back below it.
Jeffrey had already arranged this band by hand in PowerPoint and the build overwrote his file —
which is the argument for the build sheet in one line. The arrangement lives here now.)*

The assignment asks for team member names. Both boards carry all three in full — **Jeffrey Stall,
Patrick Kwok, Leon Wan** — once each, in the header band.

*(17 Aug: "A2-NLP" is off both author lines — it is our internal project code and means nothing to
a reader, where the course number identifies the work. "Leon" is "Leon Wan" wherever the board
lists the team, so the two boards do not render a teammate's name two ways side by side. The
first-name reference in cell 5 is narrative and stays as it is.)*

**Citations**, set as the footer line across the foot of the board. This board is the factory, so
its footer names the stack the factory is built on; the top board's names the one the experiment
used. Neither repeats the other:

> **AI statement.** We used **Claude Code** throughout to turn ideas into working code. It made
> the building faster and the **verifying no faster at all** — every number on this board is
> regenerated from committed records by a gate that fails the build when a claim and a record
> disagree. · **Built with** PyTorch and HuggingFace `tokenizers`, plus NumPy and matplotlib; no
> training framework above them — the loop, the masking, the scheduler and the run records are
> this project's own. Long jobs ran on a 2 × RTX PRO 6000 Blackwell workstation, benchmarked
> against
> Colab A100 / L4 / T4, a MacBook Pro (M4 Pro, MPS) and an 8 GB laptop. · Nineteen references,
> in full in report 09 §References.

---

## The cells — v3 of the redesign, 18 August

> **The standing rules, distilled from five review passes.** A panel that needs the reader to
> already know a piece of jargon does not work at two meters — XLM-R is "the borrowed
> vocabulary", a Fisher test is "no test can call anything significant", and every scale is
> defined in the row that uses it, anchored to the study (Yoruba pays 1.76). A number nobody can
> read the units of is decoration. And a chart holds axis and value labels only; every sentence
> lives in a caption, at the board's own type size. Report 09 keeps every argument in full.

Layout, top to bottom: hardware chart · **memory** · the rail — then the speedups band beside
**budgeting** — then the three remaining charts — then two prose rows. Memory and budgeting
swapped slots on the 18 August proof pass, so memory sits beside the hardware chart it now
points at ("nobody on the chart beside this needed 80 GB"). The cells below keep their numbers;
`build_posters.FACTORY_PLAN` is where wall position lives, and the words below are the words on
the wall, as ever.

### 1 · You do not need the workstation

**Figure:** `26-poster-hardware.svg`

> An **8 GB laptop GPU beats the free tier, even unplugged**. The two paid Colab tiers bill the
> same — **$111 for 7 days** against **$112 for 30** — so the faster tier buys time, not money.
> Electricity for all 148 GPU-hours was **$7**; the workstation that ran them, **$24,000**.

*50 words · `runs/hardware.json` · `bench_portable.py`*

### 2 · Budget in tokens, not epochs

| | |
|---|---|
| **Epochs were right in 502–504: the dataset was fixed.** | Here the dataset *is* the experiment — "40 epochs each" gives one arm **256× the compute** of another. |
| **So fix the total number of tokens, and let epochs fall out.** | Every run is **1.024B tokens**: the 4M corpus is seen **256 times**, the 1,024M once. |
| **Tokens ÷ tok/s = the wall clock.** | 1.024B tokens ÷ 443k tok/s ≈ **38 min** — the top bar of the chart above. |
| **A truncated run is a different experiment.** | The schedule anneals to zero at the *planned* end — a stopped run is caught mid-schedule, not half as good. |
| **Our own records carried the bug.** | The `epoch` field reads **125**; the truth is **16 passes**. |

*Five rows, no prose · report 09 §Week 1 · `runs/hardware.json`*

*Why this replaced the pipeline chart on 18 August — set as plain commentary, not a blockquote,
because the parser reads a cell's first blockquote as its lead and this must not print: Jeffrey's
call — budgeting a run, the unit and the rate that turns it into a promise, was essential to the
whole term, and the report's strongest section had no panel. The pipeline chart was the board's
weakest: one ratio its caption already stated, surviving verbatim as speedups row 4 (53 s vs
85 min — 96×). Figure 23 is retired with the pipeline panel — deleted, not kept; the
queue-vs-notebook argument lives on in the dashboard panel.*

### 3 · The factory, in numbers

> **197** pretraining runs, every one from scratch · **892** fine-tuning runs on top of them,
> across two tasks · **17** languages measured, 12 of them pretrained from scratch · **148** GPU-hours, two cards,
> 71 kWh at the wall · **62,500** steps a run — 1.024B tokens of updates, the unit that makes two
> machines comparable · **0** identity collisions, because a vocabulary carries a hash

*The rail: six value–label rows, split on the middots by the builder.*

### 4 · What we made faster

> **2.07× on the four-cell benchmark** — only **1.32×** of it efficiency; the rest is a second card
> nobody had used. **A third of the twenty were rejected**, and a list of only the wins would be
> marketing rather than a log. Every row had to make the *same* experiment faster.

| what changed | measured | kept? |
|---|---|---|
| **the data path** | | |
| 1 · dataset on the card, no loader | 13.8k → **18.5k img/s** | yes |
| 2 · images as `uint8`, cast per batch | 4× less memory, free cast | yes |
| 3 · augment on the GPU | no PIL, no host copy | yes |
| 4 · tokenize once, up front | 53 s vs 85 min — **96×** | yes |
| 5 · token dtype from the vocabulary | 16k → **2 bytes a token** | yes |
| **precision and kernels** | | |
| 6 · bf16 autocast, not fp16 | **1.34×** on sm_89, no loss-scale | yes |
| 7 · channels_last for CNNs, bf16 only | fp16 + it is **3.5× slower** | guarded |
| 8 · channels_last for ViTs | a no-op on a transformer | no |
| 9 · TF32 for matmul and cuDNN | free on Ampere and later | yes |
| 10 · `cudnn.benchmark = True` | worth it when shapes are fixed | yes |
| 11 · fused SGD and AdamW | one kernel, not a chain | yes |
| 12 · `torch.compile` | no Triton on Windows | can't |
| 13 · attention backend | already `sdpa` | no change |
| **host-sync discipline** | | |
| 14 · drop the per-step `.item()`, CIFAR | **~7%** | yes |
| 15 · the same at MLM scale | **2%** — a live loss is worth it | no |
| 16 · `zero_grad(set_to_none=True)` | skips a full zero-fill | yes |
| **batching and scheduling** | | |
| 17 · batch 64 → 128 | **1.31×** — 2048 is *slower* | yes |
| 18 · checkpointing for speed | peak is **3.3 GB of 96** | no |
| 19 · use the second card | **91% / 93%** busy at 300 W | yes |
| 20 · longest-budget-first queue | **2.55×** projected, 2.07 seen | yes |

*50 words · report 09 §Week 2, "The year, in twenty changes"*

### 5 · You cannot judge a run early

**Figure:** `27-poster-cliffs.svg`

> Yoruba, English and ten more, all on one axis. Every run that learns sits flat and then falls
> off a cliff — and **no two fall at the same step**: earliest **2,200**, latest **48,500**.
>
> So "still flat at step k" is evidence of nothing. **35 of 197 never learned.**

*49 words · `runs/early_signal.json`*

### 6 · The setting four languages want destroys the fifth

**Figure:** `25-poster-transfer.svg`

> Five languages, six learning rates, sixty runs, each cell scored against **that language's own
> best**: ★ its winner, ✕ far behind it, ◐ one seed of two. **Four land at their best in the
> outlined column. Igbo collapses there** — the same setting, a wasted night.

*46 words · `runs/lr_transfer.json`*

### 7 · One screen, both cards, a rush order at 9 p.m.

**Figure:** `28-poster-dashboard.svg`

> The chart is five English corpus sizes separating while they run. On 9 August **44.7 GPU-hours
> sat across 69 runs** and Patrick needed a sweep: a fleet pins to one card, so it took **card 1**
> while the queue kept card 0. **Twenty minutes** to running, on a measured promise.

*50 words · report 09 §The rush order*

### 8 · When memory is a wall, and when it is only a speed limit

| | |
|---|---|
| **What must fit is the model** — about 12–16 bytes per parameter. | Our 98M needs **1.6 GB**; a 7B — a small chat assistant — needs **84–112 GB**, beyond any consumer card. |
| **Everything above the floor is batch, and batch is optional.** | Accumulation runs one big step as smaller passes — **the same update**, within 2%. |
| **So an 8 GB laptop trains the very step a 96 GB card trains.** | A 10 GB run folds into **6.1 GB at 28,027 tok/s** — the same model. |
| **Nobody on the chart beside this needed 80 GB.** | 96, 80, 24, 16, 8 GB — all ran the same step. The A100 sells **speed**, not room. |
| **And Windows does not warn you.** | An oversized batch silently spills into system RAM: **28,027 → 5,075 tok/s**. |

*Five rows, no prose · report 09 §Week 1's in-depth (the accumulation and memory-wall
discussion) · `runs/hardware.json`*

### 9 · A hundred runs is a filing problem first

> A study is declared as a **grid**, languages × rates × seeds, never typed run by run; the
> queue orders it **longest-budget-first**.
>
> Every run writes a record naming **everything that changes the answer**: data fingerprint,
> vocabulary, settings.
>
> "What have we tried?" is a **query**: **1,089 runs later, no collisions, no archaeology.**

*51 words · `mlm_api.results()` · `ft_api.results()` · report 09 §Week 4*

*(18 Aug: rewritten from "An identity is everything that changes the answer" — Jeffrey's call
that filenames are not the message for an AI-student audience; organizing the hyperparameter
space and choosing what to run is. The hash mechanism survives as one clause, the collision
count as the payoff.)*

### 10 · A result has to beat your own noise

> Run the same recipe twice and it lands in two different places. That is the seed, not the
> change you made.
>
> So we measured **our own noise first** and made every claim clear it by **2.27×**. At three
> runs a side, no test can call anything significant.
>
> The spread is a property of **each cell**, never a constant we reused. **Two of our own
> claims did not clear it. We retired them.**

*72 words · `runs/claims_audit.json`*

### 11 · A factory that makes one thing is not a factory

| | |
|---|---|
| **English is the ruler.** *Why:* the one language where data was never the constraint. | *Learned:* **more data stops helping past 64M tokens** — so Yoruba's 69M-token corpus is not the bottleneck it looks like. |
| **French, Indonesian, Mandarin are the control.** *Why:* the borrowed vocabulary knows these three, so they test mere unfamiliarity. | *Learned:* they pay **1.04, 1.01, 0.95** tokens where a language's own vocabulary pays one. **Yoruba pays 1.76** — second-highest of seventeen. |
| **Twelve African languages are the gradient.** *Why:* one language is an anecdote; seventeen make a slope. | *Learned:* the cost tracks **vocabulary coverage**, not geography — **Wolof, uncovered yet cheap at 1.31**, proves it. |
| **Nine functions**, one call per language — **twelve pretrained in 48 minutes**. |

*Four rows — three name why the language group is there and what it taught, and the last is
the factory's payoff. The "None of us speaks any of them" line was Jeffrey's own cut on the
printed board. The top board's learnability table shows ten languages, not twelve — Wolof and
Luganda pretrained but sat out that comparison for want of the full 50M tokens; report 07 §4
records the exclusion. · `runs/gradient_table.json` · `runs/gradient_languages.json` · report 09
§"The problem, in two halves"*

---

## The strip — Ethics and Sources

*("The strip" survives from the nine-cell layout as the name for the two compliance panels; on
the five-row wall they are not one row — Ethics sits in the prose row beside cells 9 and 10,
Sources in the last row beside the languages panel.)*

### Ethics: what we do not claim, and what is next

> **Nobody on this team reads Yoruba** — every judgement here is a benchmark number. And the
> corpus is scraped web text nobody consented to.
>
> The factory is itself an ethics tool: **sweeping seeds avoids one lucky draw**, each language
> at its own best.
>
> Our gate prints **9 claims: 6 supported, 2 not, 1 underpowered**. Next: a corpus audit, the
> evaluation we could not do.

*64 words · `runs/claims_audit.json` · report 09 §Ethics*

### Sources

> A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," ACL, 2020. · M. Marone et al., "mmBERT: A modern multilingual encoder," arXiv:2509.06888,
> 2025. · Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach,"
> arXiv:1907.11692, 2019. · D. I. Adelani et al., "SIB-200," EACL, 2024. ·
> D. I. Adelani et al., "MasakhaNER 2.0," EMNLP, 2022. · G. Penedo et al.,
> "FineWeb2: One pipeline to scale them all," arXiv:2506.20920, 2025.

> All nineteen, in full, in report 09 §References.

*Formatted as the top board formats its SOURCES, because the pair hangs as one document —
Jeffrey's instruction, group heads removed on his option B. **Six entries** at 13 pt in full
citation form, hanging-indented, one flowing block (a stack of per-reference boxes defeated the
overflow gate twice). Page ranges and the Devlin, Ogueji, Smith and Dodge entries live in the
report's nineteen — the panels still credit the masking scheme and the AfriBERTa preset by name,
and each departure bought row E the height Jeffrey's paragraph air needed.*

> **What moved off this block on 18 August.** The team names went to the header, where a reader
> looks for them and where the assignment asks for them once. "Next" folded into the panel beside
> it, because a limit and the thing that would fix it are the same sentence. The AI statement went
> to the footer, which is now 12 pt rather than 7.2 — a compliance line nobody can read is not a
> compliance line.

---

## Does the board answer the assignment?

Checked against the rubric rather than assumed, because a panel that satisfies nothing required is
expensive wall space. The pair of boards is what gets marked; this column is only the bottom one.

| required | where it is on this board |
|---|---|
| Team member names | header band, directly under the title |
| Problem / motivation | title block, and cells 1 and 2 |
| Budget and units | cell 2, and report 09 §Week 1 in full |
| Goals | the top board's Goals block (the pair is marked together); report 09 §Goals in full |
| Tool stack | cells 2, 4, 7, 8 and the footer |
| Pre-existing vs from scratch | cell 11; the top board's panels 3 and 5 |
| How performance was explored | cells 1, 4, 5, 6, 8 |
| Dataset EDA | cell 3's inventory and cell 11; the top board's panel 2 carries the corpus EDA |
| Results / summary statistics | cells 1–11, every one a measured number |
| Discussion / limitations | cells 5, 8, 10 and strip 1 |
| **Ethical impact** | **strip 1**, with the full account in report 09 |
| Next steps | strip 1, folded in beside the limits they answer |
| Citations | **strip 2**, in full bibliographic form, with all 19 in report 09 |
| **AI statement** | the footer, at 12 pt |

---

## Notes on the hardware panel

**Cell 1 draws `26-poster-hardware.svg`, not `21-hardware.svg`.** The report keeps figure 21 —
eight columns, both model shapes, a capped axis and a cost table beside it — because a reader at
40 cm can work through all of that. The poster version drops to seven rows and one model shape,
puts each machine's project cost in its own row label, and states the one thing a reader can act
on: **an 8 GB laptop GPU beats the free tier, even unplugged.** Both draw from the same reduced
records, so they cannot disagree about a number.

**A Colab A100 comes within a fifth of one Blackwell card** — 46 minutes a run against 37 (cell
2 rounds the same run to ≈38 via the 443k rate; different sittings of one measurement, not a
disagreement). The L4
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
