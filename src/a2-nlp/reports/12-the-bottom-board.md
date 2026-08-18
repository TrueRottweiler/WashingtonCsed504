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
> to believe the differences between them.** This board is the machinery; the board above it is
> the experiment it served.

Author line, in the header band: *Jeffrey Stall · Patrick Kwok · Leon Wan*

**The names sit directly under the title**, at 20 pt, which is where a reader of any research
poster looks for them. Until 18 August they were below the tagline and above a gold rule the
template master draws, in the smallest type in the band — three names in 9.5 pt on a board two
feet wide. The course, the institution and the term moved to the line under them; the clause about
which board is which moved into the takeaway, because it is context for somebody standing in front
of both and not part of an author line.

**Goals**

> Build the machinery a hundred-model study needs — one training loop, one record format, one
> benchmark — and measure it honestly enough that the numbers survive being checked.

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

> **What changed and why.** Six rows now rather than five: memory earns a full-width band of its
> own, because when a card is too small to *load* a model is a different question from when it is
> merely slow, and that distinction is the most transferable thing on this half of the poster.
>
> Three panels were rewritten because they did not land — Jeffrey read them cold and could not
> follow them, which is the only test that counts. The seeds panel had compressed a combinatorial
> argument nobody has met yet into two sentences; the accumulation panel leaned on "divergence"
> and a Fisher exact test; the seventeen-languages panel was a paragraph where it wanted a list.
>
> The rule those three share: **a panel that needs the reader to already know a piece of jargon is
> a panel that does not work at two meters.** Report 09 keeps every one of the arguments in full.

Layout, top to bottom: a chart row, the speedups band, a chart row, the memory band, then two
prose rows. Geometry lives in `build_posters.FACTORY_GEO`; the words below are the words on the
wall, as ever.

### 1 · You do not need the workstation

**Figure:** `26-poster-hardware.svg`

> An **8 GB laptop GPU beats the free tier, even unplugged**. The two paid Colab tiers bill the
> same — **$111 for 7 days** against **$112 for 30** — so the faster tier buys time, not money.
> Electricity for all 148 GPU-hours was **$7**; the workstation that ran them, **$24,000**.

*50 words · `runs/hardware.json` · `bench_portable.py`*

### 2 · Cheap stays interactive; expensive goes in a queue

**Figure:** `23-poster-run.svg`

> Preparing the corpus — tokenizer, encoding, loading — takes **53 seconds**. One pretraining run
> takes **37 to 85 minutes**. A gap of **41× to 96×** is not a judgment call: cheap work stays in
> the notebook, expensive work goes in a queue, and **both write the same record**.

*48 words · `runs/pipeline_bench.json`*

### 3 · The factory, in numbers

> **197** pretraining runs, every one from scratch · **892** fine-tuning runs on top of them,
> across two tasks · **17** languages in four writing systems · **148** GPU-hours on two cards,
> 71 kWh at the wall · **62,500** steps a run — 1.024B tokens of updates, the unit that makes two
> machines comparable · **0** identity collisions, because a vocabulary carries a hash

*The rail: six value–label rows, split on the middots by the builder.*

### 4 · What we made faster

> **2.07× on the four-cell benchmark** — and only **1.32×** of it is efficiency; the rest is a
> second card nobody had used. The list is longer than that one number, and two of its best rows
> are decisions **not** to do something.

| what changed | measured | kept? |
|---|---|---|
| dataset on the card, no loader | 13.8k → **18.5k img/s** | yes |
| images as `uint8`, cast per batch | 4× less memory, free cast | yes |
| augment on the GPU | no PIL, no host copy | yes |
| tokenize once, up front | 53 s vs 85 min — **96×** | yes |
| token dtype from the vocab | 16k → **2 bytes a token** | yes |
| batch 64 → 128 | **1.31×** — 2048 is *slower* | yes |
| both cards, longest first | **1.57×**, 91% / 93% busy | yes |
| bf16 instead of fp16 | **1.34×**, no loss-scale | yes |
| TF32 · benchmark · fused Adam | free on Ampere and later | yes |
| fp16 **+** channels_last | **3.5× slower** on sm_89 | no |
| drop the per-step `.item()` | **2%** — not worth it | no |
| `torch.compile` | no Triton on Windows | can't |

*42 words · report 09 §Optimization*

### 5 · You cannot judge a run early

**Figure:** `27-poster-cliffs.svg`

> Every run that learns sits flat and then falls off a cliff — and **no two fall at the same
> step**. The earliest is **2,200**, the latest **48,500**. So "still flat at step k" is evidence
> of nothing. **35 of 195 runs never learned**, and we could not have told which.

*51 words · `runs/early_signal.json`*

### 6 · The setting four languages want destroys the fifth

**Figure:** `25-poster-transfer.svg`

> Five languages, six learning rates, sixty runs, each cell scored against **that language's own
> best**: ★ its winner, ✕ far behind it, ◐ one seed of two. **Four land at their best in the
> outlined column. Igbo collapses there** — the same setting, a wasted night.

*46 words · `runs/lr_transfer.json`*

### 7 · One screen, both cards, a rush order at 9 p.m.

**Figure:** `28-poster-dashboard.svg`

> On 9 August, **44.7 GPU-hours were committed across 69 runs** and Patrick needed a sweep before
> he could write. A fleet pins to one card, so the urgent work took **card 1** and the queue kept
> card 0. Twenty minutes to running, and a measured **"fifty minutes"**.

*47 words · report 09 §The rush order*

### 8 · When memory is a wall, and when it is only a speed limit

> What has to **fit** is the model. Everything above that is a choice.

| | |
|---|---|
| **The floor is the model** — weights, gradients, optimizer state: 12–16 bytes a parameter. | Our 98M: **1.6 GB**. A 7B model: **84–112 GB**, which no consumer card has. *(arithmetic)* |
| **Everything above it is batch, and batch is optional.** | Accumulation runs a 16,384-token step as smaller passes — same update, **within 2%** of the cost. |
| **So an 8 GB laptop trains the step a 96 GB card trains.** | A 10 GB run folds into **6.1 GB at 28,027 tok/s**. |
| **More memory does not buy speed.** | Batch 2048 uses **89.7 GB** and is *slower* than 128. |
| **And Windows does not warn you.** | An oversized batch spills to system RAM: **5,075 tok/s, no error**, 6× under the truth. |

*13 words · report 09 §Memory · `runs/hardware.json`*

### 9 · An identity is everything that changes the answer

> Two runs silently overwrote each other — same filename, different vocabularies, and neither
> record said so. A filename is an identity, so the vocabulary carries a **hash** rather than a
> label. **197 pretraining and 892 fine-tuning runs later, no collision.**

*40 words · `mlm_api.results()` · `ft_api.results()`*

### 10 · A result has to beat your own noise

> Run the same recipe twice and it lands in two different places. That is the seed, not the change
> you made. So we measured **our own noise first** and made every claim clear it by **2.27×** — at
> three runs a side, no test can call anything significant. **Two of our own claims did not clear
> it. We retired them.**

*60 words · `runs/claims_audit.json`*

### 11 · A factory that makes one thing is not a factory

| | |
|---|---|
| **English is the ruler.** | A ladder from 4M to 1,024M tokens — something to measure Yoruba against. |
| **French, Indonesian, Mandarin are the control.** | They cost **1.04, 1.01, 0.95** — the penalty tracks *coverage*, not "African". |
| **Twelve African languages are the gradient.** | One is an anecdote; seventeen is a slope. |
| **Adding one is a function call.** | **Twelve trained in 48 minutes.** |

*Four rows, no prose — the heading is the claim · `runs/gradient_table.json` · report 09 §Why seventeen*

---

## The strip — the bottom row of panels

### Nine functions, nothing else to import

> More than sixteen thousand lines behind **nine calls**. Leon read the documentation and asked
> whether there was an interface he should be using — **a tool nobody can find does not exist**.
> Only **3%** of the factory is masking-specific: a second study, on next-token prediction,
> **reuses it unchanged**.

*48 words · counted at render time from `mlm_api.py` · `mlm_data.py`*

### What we do not claim, and what we would do next

> **Nobody on this team reads Yoruba** — every judgement here is a benchmark number. The corpus is
> scraped web text nobody consented to. The audit gate prints **9 claims: 6 supported, 2 not, 1
> underpowered**. Next: a wall meter, a corpus audit, the evaluation we could not do.

*48 words · `runs/claims_audit.json` · report 09 §Ethics*

### Sources

> Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692,
> 2019. · A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," ACL,
> 2020, pp. 8440–8451. · M. Marone et al., "mmBERT: A modern multilingual encoder," arXiv:2509.06888,
> 2025. · G. Penedo et al., "FineWeb2: One pipeline to scale them all," arXiv:2506.20920, 2025. ·
> D. I. Adelani et al., "MasakhaNER 2.0," EMNLP, 2022, pp. 4488–4508. · D. I. Adelani et al.,
> "SIB-200," EACL, 2024, pp. 226–245.

*Nineteen references in full in report 09 §References. Set as a reference list by the builder,
one entry a line, split on the middots.*

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
| Goals | title block, and report 09 §Goals in full |
| Tool stack | cells 2, 4, 7, 8 and strip 1 |
| Pre-existing vs from scratch | cell 11 and strip 1; the top board's panels 3 and 5 |
| How performance was explored | cells 1, 4, 5, 6, 8 |
| Dataset EDA | cell 3's inventory and cell 11; the top board's panel 2 carries the corpus EDA |
| Results / summary statistics | cells 1–11, every one a measured number |
| Discussion / limitations | cells 5, 8, 10 and strip 2 |
| **Ethical impact** | **strip 2**, with the full account in report 09 |
| Next steps | strip 2, folded in beside the limits they answer |
| Citations | **strip 3**, in full bibliographic form, with all 19 in report 09 |
| **AI statement** | the footer, at 12 pt |

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
