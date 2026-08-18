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

**The grid this board uses — v3.** A hero row (a 13.7 in chart panel beside an 8.25 in stat
rail), then two rows of two 10.95 in chart panels, a row of three statement cards, and the strip.
**Five charts**, each drawn at the exact size it prints so its in-chart type is real 13–24 pt, not
a 0.3× reduction of a report figure. Geometry in `build_posters.FACTORY_GEO`.

**Word budgets, v3.** A chart panel's caption is **15–25 words** with the key phrase bold — the
chart carries the argument, the caption points at it. A statement card runs **30–45 words** behind
a statement heading that IS the finding. The 55-words-per-cell arithmetic above described v2's
uniform nine-cell grid and survives in git history; the constraint that replaced it is blunter:
**if a panel needs a paragraph, its chart is not doing its job.**

**Headings are statements, not questions.** "Does a tuned setting transfer?" makes the reader do
the work at two meters. "The rate three languages love is fatal to a fourth" hands them the
answer and lets the chart prove it.

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

## The cells — v3, 17 August

> **Why the nine-cell grid went.** The proof PNGs said it before anyone did: this board's charts
> were report figures drawn 8–16 in wide and scaled to ~0.3× inside a 2.12 in slot, so their axis
> type printed at 3–5 pt beside 18 pt prose. Jeffrey's call, quoted because it is the design rule
> now: **a poster with three or four readable charts beats a poster with eight nobody can read,
> and 15–20 words under a chart beat 55.** The template's own sample slide agrees — its
> demonstration figure is 10 in wide and its example charts are 4.3 in tall, not 2.1.
>
> v3 keeps the material and spends it differently: **four charts drawn at the exact size they
> print** (in-chart type 13–24 pt, no legends — series are labelled where they sit), a **stat
> rail** carrying the scale of the factory in six numbers, and **three statement cards** whose
> point was always a sentence rather than a picture. Question-headings became **statements** — a
> reader two meters away takes the answer without stopping. The tokens-not-epochs lesson folds
> into the rail; the 2.07×/1.32× optimization story returns to report 09, where its argument
> lives in full; the rush-order story stays in report 03. Nothing was deleted — it moved to where
> it is read at reading distance.

Layout: a hero row (13.7 in chart panel beside an 8.25 in rail, 6.90 in deep), two rows of two
10.95 in chart panels, a row of three statement cards, then the strip. Geometry lives in
`build_posters.FACTORY_GEO`; the words below are the words on the wall, as ever.

### 1 · You do not need the workstation

**Figure:** `26-poster-hardware.svg`

> Seven machines, the same 62,500 steps. The **two paid tiers bill the same** and differ 4× in
> wait. **No machine here takes longer than a day.**

*26 words · `runs/hardware.json` · `bench_portable.py`*

### 2 · The factory, in numbers

> **197** pretraining runs · **892** fine-tuning runs · **17** languages, four scripts ·
> **148** GPU-hours, two cards · **62,500** steps — 1.024B tokens · **0** record-identity
> collisions

*The rail: six value–label rows, split on the middots by the builder. "Tokens of updates" is the
unit because an epoch stops meaning anything once dataset size is the variable — the full
sentence lives in report 09 §Units.*

### 3 · You buy latency, not access

**Figure:** `22-poster-cost.svg`

> **148 GPU-hours ≈ $7 of electricity**, against **≈$111** to rent the same work. The faster
> tier buys **7 days instead of 30**, not a cheaper bill.

*26 words · `runs/hardware.json` · `mlm_api.results()`*

> **One honesty note before it prints, carried over from v2.** *148 GPU-hours* is measured.
> *71 kWh* and *$7* are not: they are 148.0 × 300 W × 1.65 × $0.0986, and only the 300 W was ever
> observed. They could be a third out either way and the sentence would still be true — which is
> why the board writes them with "≈" rather than as readings. Report 09 §Cost has the derivation.

### 4 · Cheap stays interactive; expensive goes in a queue

**Figure:** `23-poster-run.svg`

> Prepare: **53 s**. One run: **37–85 min**, a gap of **41–96×**. Not a judgment call — and
> both paths write **the same record**.

*23 words · `runs/pipeline_bench.json`*

### 5 · Do not build the early-stopping detector

**Figure:** `24-poster-earlystop.svg`

> **35 of 195 runs never learned** — yet at **every checkpoint** the best doomed run looks
> better than the worst healthy one. Acting on the signal loses GPU-hours on net.

*30 words · `runs/early_signal.json` · the figure draws the eight checkpoints every run
reached; the overlap holds at all eleven*

### 6 · The setting four languages want destroys the fifth

**Figure:** `25-poster-transfer.svg`

> Five languages, six learning rates, sixty runs. **Four land at their best on one setting.
> Igbo collapses there** — and no language's winner clears its own **seed noise**.

*28 words · `runs/lr_transfer.json` · `runs/budget.json`*

### 7 · Three seeds cannot say p < 0.05

> Three against three is **twenty possible shufflings**, so the smallest p an exact test can
> return is **0.10**. Our rule: real means clearing **2.27× the seed spread**. **Two of our own
> claims sat in that gap — both retired.**

*39 words · `runs/claims_audit.json`*

### 8 · An identity is everything that changes the answer

> Two runs silently overwrote each other — same filename, different vocabularies, and neither
> record said so. The vocabulary gets a **hash**, not a label: **197 pretraining and 892
> fine-tuning runs later, no collision.**

*33 words · `mlm_api.results()` · `ft_api.results()`*

### 9 · "Matched steps" handed one arm 5.1× the compute

> Convert to **bits per character** and the vocabulary divides back out. At 12,000 steps each:
> 16k vocab **1.131 bpc, 8 min**; 250k vocab **0.989 bpc, 42 min** — the better score cost five
> times the budget. **Fairness is a property of the unit.**

*43 words · `runs/tokenizer_seeds.json`*

---

## The strip — three blocks across the foot

### Nine functions, nothing else to import

> More than sixteen thousand lines behind **nine calls**. Leon read the documentation and asked whether
> there was an interface he should be using — **a tool nobody can find does not exist**. And only
> **3%** of the factory is masking-specific: a second study, on next-token prediction, already
> **reuses it unchanged**.

*50 words · counted at render time from `mlm_api.py` · `mlm_data.py`*

### What we do not claim

> **Nobody on this team reads Yoruba** — every judgement here is a benchmark number, every claim
> relative. The corpus is scraped web text nobody consented to. And the audit gate prints
> **9 claims: 6 supported, 2 not, 1 underpowered** — two of the failures ours, printed rather
> than dropped.

*49 words · `runs/claims_audit.json` · report 09 §Ethics*

### Next · sources · AI

> **Next:** a wall meter, a corpus audit, a Yoruba-speaking evaluation this project could not do.
> **Sources:** nineteen references, in full in report 09. **AI:** we used **Claude Code**
> throughout to turn ideas into working code — it made the building faster and the **verifying no
> faster at all**. Team: **Jeffrey Stall · Patrick Kwok · Leon Wan**.

*57 words · report 09 §Panel 10*

> **The commit ratio is off the board as of 17 August, and getting it wrong three times is the
> reason.** It said 120 of 144, then 96 of 168, and a recount on the 17th made it 101 of 173 —
> squash-merging collapses a branch into one commit, so the ratio falls as the history is tidied.
>
> Worse, the method was wrong. **`git log --format='%b' | grep -c` returns 228, more trailers
> than there are commits**, because a squashed commit carries one trailer per commit it absorbed:
> counting *lines* counts co-authorships where the sentence claims *commits*. A precise-looking
> number that has been wrong at every telling is worth less than the plain statement, and the
> plain statement is what the assignment asks for. Report 09 keeps the derivation.

---

## Does the board answer the assignment?

Checked against the rubric rather than assumed, because a panel that satisfies nothing required is
expensive wall space. The pair of boards is what gets marked; this column is only the bottom one.

| required | where it is on this board |
|---|---|
| Team member names | header band, and strip 3 |
| Problem / motivation | title block, and cells 1 and 3 |
| Goals | title block, and report 09 §Goals in full |
| Tool stack | strip 1, and cells 4 and 8 |
| Pre-existing vs from scratch | strip 1; the top board's panels 3 and 5 |
| How performance was explored | cells 1, 3, 4, 5, 6 |
| Dataset EDA | cell 2's inventory; the top board's panel 2 carries the corpus EDA |
| Results / summary statistics | cells 1–9, every one a measured number |
| Discussion / limitations | cells 5, 7, 9 and strip 2 |
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
