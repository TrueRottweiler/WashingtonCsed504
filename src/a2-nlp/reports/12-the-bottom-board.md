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

**The grid this board uses.** Three rows of **6.70 in** with 0.25 in gutters, then a **4.45 in**
strip across the foot. Nine cells of 6.35 × 6.70 in, three strip blocks of 6.35 × 4.45.

**Word budgets, which are hard limits rather than targets.** A cell spends 0.6 in on its header,
1.4 in on the big number, 2.3 in on a figure, and has **1.9 in left — about 55 words at 18 pt.**
A cell with no figure gets about 110. A strip block gets about 100.

**What 55 words means for the writing.** It is three sentences. The
Problem / Hypothesis / Approach / Results / Learning structure **cannot fit on a panel** and does
not belong here — it is the report's spine. A cell states the question, the answer, and why it
matters, and the figure does the rest.

**The big number must fit about 18 characters per line.** This is the constraint that catches
people out: `2.07×, of which 1.32× is efficiency` overflows a 6.35 in column and has to become
`2.07×` over `1.32× real`. Check every one against the measure before setting it.

![The bottom board, drawn to scale on the real template](figures/20-board-layout.png)

---

## Title block

> ## CSED 505: Building a Model Factory
> *the course that would come after 504 — ten weeks, derived from one term of building one*
>
> **501** the statistics · **502** the mechanics · **503** the language stack · **504** scale
> **All four end when a model finishes training. None covers needing a hundred models and having
> to believe the differences between them.**

Author line, in the header band: *Jeffrey Stall · A2-NLP · CSED 504 · the upper board is the
experiment this machinery served.*

The assignment asks for team member names. Both boards carry all three — **Jeffrey Stall, Patrick
Kwok, Leon** — with the role split named once, on this board, in the strip.

---

## The nine cells

> **A decision three cells need before setting.** Cells 2, 3 and 5 each carry a **second block**
> below, added on 12 August. A cell holds ~55 words *with* a figure or ~110 *without*, and two
> blocks is 120. So each of those three is a straight choice: **keep the figure and cut the second
> block, or drop the figure and run both blocks as type.** My recommendation in each case —
>
> - **Cell 2:** drop `14-where-the-speedup-came-from.svg`. The seventeen-language argument is the
>   stronger content and the speedup decomposition survives as a big number.
> - **Cell 3:** keep `15-what-a-run-is-made-of.svg`; the rush-order block moves to **cell 4**,
>   which is about records and queues and is currently the lightest cell on the board.
> - **Cell 5:** keep `19-the-interface.svg` and cut the masking block to one sentence — *"35 of
>   1,114 lines touch masking; a second study on a different objective already shares this
>   factory."* It is a footnote-sized point and the figure is doing more work.
>
> None of this is settled. It is the last real layout decision on the board.

Read left to right, top to bottom. The top row builds a factory; the middle row makes it
survivable, shareable and trustworthy; the bottom row spends the instruments, and two of those
three answers are *no*.

### 1 · What does a run cost, and in what unit?

**Big number:** `62,500 steps` / `= 1.024B tokens` · **Figure:** `21-hardware.svg`

> Every course measures training in epochs, and an epoch stops meaning anything the moment the
> dataset is the variable — which is the first thing a scaling study does. At 4M tokens the model
> sees the corpus sixty times; at 1,024M, a quarter of it once. **Tokens of updates** survives
> both. Pick the unit that survives what you intend to vary.

*56 words · `runs/scaling_law.json`*

### 2 · Why optimize before anything needs it?

**Big number:** `2.07×` / `1.32× real` · **Figure:** `14-where-the-speedup-came-from.svg`

> Nothing needed to be fast yet, which is the argument against doing it. 25.2 → 12.2 minutes —
> but only **1.32× is efficiency**; the rest is a second card. The hours saved are not the point.
> **1,089 models across 17 languages, and only one is Yoruba.** At 25 minutes a cell that control
> set is a week's work and gets cut. At 12 it runs.

*58 words · `mlm_api.results()` · report 03*

**Second block, if the cell can take two:**

> **Why train Mandarin, French and Indonesian for a Yoruba study?** Because "the penalty tracks
> coverage" cannot be shown on one language — Yoruba is Latin-script, African and under-served all
> at once, and you cannot tell which one is doing the work. Ten covered languages against seven
> uncovered, four scripts, three continents. **Mandarin is the row that kills the script
> explanation.**

*58 words · `runs/gradient_table.json`*

### 3 · What belongs in a notebook, and what belongs in a queue?

**Big number:** `53 s vs 85 min` / `96×` · **Figure:** `15-what-a-run-is-made-of.svg`

> Do not choose on principle — measure the ratio and let it choose. Preparing the corpus is **53
> seconds**; one pretraining run is **85 minutes**. A gap of **96×** is not a judgment call.
> Everything cheap stays interactive, everything expensive goes in a queue you can walk away from,
> and the part that makes it work rather than merely tidy: **both paths write the same record.**

*55 words · `runs/pipeline_bench.json`*

**Second block — the rush order:**

> A queue that only runs the plan it was given is a batch job. On 9 August there were **44.7
> GPU-hours across 69 runs** committed to both cards when Patrick needed four cells now. Pin the
> urgent fleet to one card; `reuse=True` means restarting costs the one cell in flight, not ten
> hours; and `estimate()` measures twenty real steps, so the answer was **"fifty minutes"** rather
> than "sometime tonight."

*62 words · `mlm_api.estimate()` · `mlm_fleet.py --gpu-base`*

### 4 · What makes a record survive you?

**Big number:** `fingerprint` / `15abd33de5af` · **Figure:** `07-dashboard.png`

> Two runs silently overwrote each other. Two people prepared "the same" corpus and got different
> vocabularies, and nothing in either record said so. A filename is an identity, and an identity
> has to include **everything that changes the answer** — so the vocabulary gets a hash rather
> than a label. 197 pretraining runs and 892 fine-tuning runs later, no collision has survived.

*54 words · `mlm_api.results()` · `ft_api.results()`*

### 5 · What does someone else have to be able to call?

**Big number:** `9` / `functions` · **Figure:** `19-the-interface.svg`

> **Nine functions**, nothing else to import, on fourteen thousand lines they never open. Three
> things it had to be — callable, shareable, runnable on one card — and a fourth we got wrong.
> Leon read the documentation and asked whether there was an interface he should be using. There
> was. **A tool nobody can find does not exist**, and nobody files a bug to tell you.

*57 words · counted at render time from `mlm_api.py`*

**Second block — is it a masked-LM factory, or a factory?**

> **35 of 1,114 lines touch masking — about 3%.** The corpus prep, the resident token store, the
> records, the scheduler and the estimator have no opinion about what the model predicts. Proof
> rather than claim: a **second study on a different objective** — next-token prediction, LSTM vs
> GPT — already shares the token store, the scheduler and the dashboard. A general study needs a
> different `pretrain()`, not a different API.

*63 words · counted from `mlm_data.py`, `mlm_train.py`*

### 6 · Is this difference real?

**Big number:** `2.27×` / `not 1.0×` · **Figure:** `13-how-many-seeds.svg`

> Our rule was "bigger than the cell's own seed spread," and that is half a rule — sound at
> rejecting noise, silent just above it. At three seeds a difference must clear **2.27×** the
> spread, and the exact test **cannot return a p below 0.10** at three a side however far apart
> the arms land. Two of our own claims were sitting in that gap. Both were retired.

*58 words · `runs/claims_audit.json`*

### 7 · Which of your units are not units?

**Big number:** `5.1×` / `at "matched" steps` · **Figure:** *none — set the table as type*

> Two vocabularies give two losses that are not on one scale. Convert to **bits per character**.
> Then count what "matched steps" actually bought: a 250k output head is 5.1× the compute per
> step, so twelve thousand steps each handed one arm five times the budget.

| 12,000 steps each | 16k vocab | 250k vocab |
|---|---|---|
| **scored** — bits/char | 1.131 | **0.989** *(looks better)* |
| **cost** — min/seed | 8 | **42** |

> Same runs, two readings, opposite answers. Fairness is a property of the unit.

*61 words + table · `runs/tokenizer_seeds.json`*

### 8 · Does a tuned setting transfer?

**Big number:** `7e-4` / `fatal to a fourth` · **Figure:** `16-lr-transfer.svg`

> "Adding a language is one function call" — true only if the settings come too. Five languages,
> six rates, sixty runs. Hausa, Nyanja and Swahili all peak at **7e-4**; **Igbo collapses** there
> and at every rate above. And the winner is not identified in *any* of the five: the gap to the
> runner-up is smaller than the gap between two seeds at the same rate.

*57 words · `runs/lr_transfer.json` · `runs/budget.json`*

### 9 · Detect the failure, or prevent it?

**Big number:** `0 of 11` / `checkpoints` · **Figure:** `10-early-signal.svg`

> 35 of 195 runs never learned — 17.9%, wasting 25.5 GPU-hours, so catch them early. Scored
> against their own untrained baselines at eleven checkpoints, the two outcomes **overlap at all
> eleven**: the best doomed run always looks better than the worst healthy one. Do not build the
> detector. And tighter clipping does not prevent divergence either.

*54 words · `runs/early_signal.json` · `runs/clip_prevention.json`*

---

## The strip — three blocks across the foot

### What it cost

**Figure:** `06-what-it-cost.svg`

> **148 GPU-hours. 71 kWh. $7 of electricity.** Rent the same work on Colab and it is **$104 on an
> L4 or $99 on an A100** — the A100 costs 4.4× per hour and returns 4.5× the throughput, so the
> tier changes the wait (28 days against 6) and not the bill. Free T4: 41 days, $0. **You buy
> latency, not access.** Team: Jeffrey Stall (factory), Patrick Kwok and Leon (Yoruba science).

*63 words*

### What we do not claim

> **Nobody on this team reads Yoruba.** Every quality judgement here is a benchmark number, not a
> judgement about whether the output is *good Yoruba* — so every claim is relative. The corpus is
> scraped web text nobody consented to. And the gate reports **9 claims: 6 supported, 2 not, 1
> underpowered** — two of the three failures ours, printed rather than dropped.

*57 words · `runs/claims_audit.json` · report 09 §Ethics*

### Next · sources · AI

> **Next:** the hardware figure, an audit of which sites dominate 69M tokens of Yoruba, and a
> Yoruba-speaking evaluation this project could not do. **Sources:** 19 references, shortened.
> **AI:** an assistant wrote much of this code — **120 of 144 commits carry a `Co-Authored-By`
> trailer**, which is `git log` rather than a promise. It made the building 3–5× faster and the
> *verifying* no faster at all.

*62 words · report 09 §Panel 10*

---

## Does the board answer the assignment?

Checked against the rubric rather than assumed, because a panel that satisfies nothing required is
expensive wall space. The pair of boards is what gets marked; this column is only the bottom one.

| required | where it is on this board |
|---|---|
| Team member names | header band + strip 1 |
| Problem / motivation | title block + cell 1 |
| Goals | title block, and report 09 §Goals in full |
| Tool stack | cells 2, 3, 4, 5 |
| Pre-existing vs from scratch | cell 5, and the top board's panels 3 and 5 |
| How performance was explored | cells 6, 8, 9 |
| Dataset EDA | cell 4's inventory; the top board's panel 2 carries the corpus EDA |
| Results / summary statistics | cells 1–9, every one a measured number |
| Discussion / limitations | cells 6–9 and strip 2 |
| **Ethical impact** | **strip 2**, with the full account in report 09 |
| Next steps | strip 3 |
| Citations + AI statement | strip 3, with all 19 references in report 09 |

---

## Gaps

**Cell 1's figure is drawn — `21-hardware.svg`. Nothing on this board is blocked any more.**

Four machines: the workstation, a free Colab T4, a paid Colab L4, and the 8 GB Surface Studio
Laptop — that last one measured twice, on its GPU and with `--cpu` on its own processor. The T4
is the *floor* of what a student with no budget gets — **5.9× slower, 4.4 hours for one run, 36
days of card time for the whole project, and the 98M model fits too** — the L4 is the paid tier
at **4.3× and 27 days**, and the laptop is the machine a student most likely already owns:
**3.8 h / 8.8 h per run, 64× faster than its own CPU, and quicker than the free T4 on both model
sizes.** More rows drop into `runs/hardware.json` without touching the figure; the laptop on
battery, a MacBook and an A100 are still wanted and would each add a bar.

**The first T4 reading said 33× and would have killed the claim.** `bench_portable.py` chose its
precision with `torch.cuda.is_bf16_supported()`, whose signature is `(including_emulation=True)` —
so a card whose tensor cores have no bf16 answered yes and ran it in software. 11,566 tok/s against
64,644 once it was measured in fp16. It also reported the 98M model as not fitting in 15 GB, which
was the same cause: in fp16 it fits at batch 128 with 9.85 GB. Nothing errored and nothing warned,
and both readings looked equally plausible on the page. **Every row on the figure carries its dtype
for that reason.**

**The laptop's first reading failed the same way, in the opposite direction.** The 98M model wants
~10 GB and the card has 8. Windows does not refuse: the driver spills the overflow into system RAM
and the benchmark "worked" at 5,075 tok/s — the PCIe bus wearing a GPU costume, six times under the
card's honest rate. On Linux the same run is an OutOfMemoryError. The fix was already in the
factory: gradient accumulation (`mlm_train.pretrain(accum=)`, now mirrored by `bench_portable.py`)
folds the same 16,384-token step into micro-batches — identical update math, the batch never stops
being 128 — and the model trains in **5.98 GB at 32,267 tok/s**. The script now treats an
allocation past free memory exactly like an OOM, walks the fallback ladder, and **the row records
the configuration it took**, because a rate without one is not reproducible. Report 09 carries the
full story under *"The 8 GB story: we needed 10 GB and had 8."*

**`11-metric-validity.svg`, `08-why-not-shorter.svg` and `09-scaling-with-cards.svg` are drawn and
carried by neither board.** Available if a cell wants one.

**Figure 03 went to the top board.** It was claimed by both, which neither document could see from
the inside; `check_boards.py` exists because of it and exits 0 now. Cell 7 sets its table as type
instead, which is the better panel anyway.

**Do not reuse the retracted floor sentence.** An earlier draft claimed the difference between the
two tasks' floors explained the divergence, argued from shares of 57% and 52%. Sweeping the
MasakhaNER floor moved it to **0.6261** and the shares to 61% and 78% — so the sentence was wrong
*and* its supporting numbers have evaporated.

---

## Order of work

1. ~~Figures for cells 3, 5 and 8~~ — **done**.
2. ~~Settle figure 03~~ — **done**, it is the top board's.
3. **Set the nine cells and three strip blocks from the words above.** They are written to the
   measure; do not re-expand them.
4. **Check every big number against the 18-character measure** before setting it.
5. ~~Cell 1's figure~~ — **done**, `21-hardware.svg`, three machines on it and the CPU baseline
   in its side panel.
6. **The print gate, last, once nothing is still writing:** `check_links.py`, `check_boards.py`,
   `check_provenance.py`, regenerate every figure, run the staleness pass. Do it once.

---

## What the board must not do

Quote a count typed by hand. Every number comes from `poster_bottom.ipynb`, which recomputes from
the records and flags any sentence in report 09 the data no longer supports.

The counts moved three times while this was being written: 105 pretraining runs became 156, then
**197**. Fine-tuning records stand at **278**, of which 161 are test-scored and 117 dev-scored —
the dev rows pick a learning rate and are chosen on the items they are scored on, so they are not
reportable numbers.

**Recompute, do not copy from this sheet.** Anything quoted here was true on the day it was
written.

---

# Appendix — measuring what a run costs on your machine

This is the missing cell-1 figure. `bench_portable.py` needs no corpus, no tokenizer and no
repository data, because its token stream is random integers — worthless for learning and
identical for timing, which means it pastes straight into a fresh Colab cell.

**What we are collecting.** For each machine: tokens per second on both model shapes, whether the
model fits in memory, and the extrapolated wall-clock for one full 62,500-step run. The reference
row exists — the workstation sustains **381,817 tok/s** on the 33.8M `poc` preset and **184,329
tok/s** on the 86M `afriberta` preset, medians over 96 and 55 real runs.

**Before you start, on every machine:** close other GPU work. A number taken while something else
holds the card is simply wrong, and it is the most common way these tables mislead.

## A. The workstation — re-run to confirm

```bash
cd /o/Sources/GitHub/TrueRottweiler/WashingtonCsed504/src/a2-nlp
CUDA_VISIBLE_DEVICES=0 bash py.sh bench_portable.py --out runs/hardware.json --note "Toothless, RTX PRO 6000 Blackwell Max-Q, 1 card"
```

Two minutes. One card deliberately — the figure compares *a card*, and a two-card number is not
comparable to a MacBook.

## B. Surface Studio Laptop (RTX 2000 Ada Mobile)

The interesting row, because it is the machine a student is most likely to own — **measured
12 August, plugged in**: 75,583 tok/s on the 33.8M preset; 32,267 tok/s on the 98M via gradient
accumulation (micro-batch 64 × 2 — it does not fit in 8 GB whole); 1,179 and 516 tok/s with
`--cpu`. Still wanted from this machine: **the battery run**, same command with
`--note "... on battery"`. Mobile GPUs throttle hard, and "can a student do this overnight" is a
different question unplugged; if it differs by more than ~20%, the board says so (our a1-cv
notes measured a 17% swing on this chassis).

For anyone repeating it on their own laptop:

1. Copy `src/a2-nlp/bench_portable.py` — it needs `torch` and `transformers`, nothing else.
2. `pip install torch --index-url https://download.pytorch.org/whl/cu124`, then
   `pip install transformers`
3. `python bench_portable.py --out hardware.json --note "<your machine>, plugged in"`

Two traps we hit so the next person does not. **Run it with the Python that actually has
torch** — a bare `python` on Windows is often the Store alias or a different environment, and
the resulting error names `transformers` even when the real problem is the interpreter. And
**trust the spill note**: on Windows a model too big for VRAM does not error — it swaps over
PCIe and reports a plausible number. The script detects the overflow, folds the batch until the
measurement is honest, and records how. An out-of-memory that survives every fallback is still
a result rather than a failure — record it.

## C. MacBook Pro (M4 Pro, MPS)

```bash
python bench_portable.py --out hardware.json --note "MacBook Pro M4 Pro, MPS"
```

MPS is detected automatically. Two things to expect: MPS does not support the same
mixed-precision path, so the number is honest but not comparable in *kind*; and unified memory
means the larger preset may fit where a discrete GPU of similar size would not. Both are footnotes
on the figure.

## D. Google Colab — the important set

This decides whether the "$120 and you can do this" claim survives.

```python
!wget -q https://raw.githubusercontent.com/TrueRottweiler/WashingtonCsed504/main/src/a2-nlp/bench_portable.py
!python bench_portable.py --note "Colab T4"
```

> **Do not `pip install torch` on Colab.** An earlier version of this appendix said
> `!pip -q install torch --upgrade` and it is wrong in a way that silently ruins the measurement.
> Colab ships torch already, pinned against the CUDA libraries in that image; upgrading it pulls a
> newer torch over them and leaves `torchvision`, `cuda-python` and the RAPIDS stack mismatched.
> The run may still complete, and its throughput number is then wrong in an unpredictable
> direction — which is the worst failure mode a benchmark has, because it looks like data.
> `bench_portable.py` imports nothing Colab does not already ship (`torch` and `transformers`).
> If you have already run the upgrade: **Runtime → Disconnect and delete runtime**, reconnect,
> and run the two lines above.*

| runtime | why it is on the list |
|---|---|
| **T4** (free) | the honest floor — what somebody with no budget gets |
| **L4** | the cheap paid tier, probably the best-value row |
| **A100 40GB** | the fast tier, for the "under two hours" claim |
| **TPU** | *skip* — the script refuses; it needs a different training loop and would not be comparable |

**Copy the printed JSON out of each session before it disconnects.** Also record the actual cost
per tier, so the strip's "$120" is measured rather than remembered.

## E. What happens then

Drop the rows into `runs/hardware.json` and the cell-1 figure generates: machines on one axis,
wall-clock for one full run on the other, a memory-fit marker, and the three cost routes
annotated. Then nothing on this board is blocked.

**The sentence the figure has to earn:** *you do not need the workstation.* It is only true if
the numbers say so. Four machines now say it — the free tier's card, the paid L4 at 4.3×, and
an 8 GB laptop that runs both models overnight-sized and beats its own CPU by 64× — with the
MacBook, the A100 and the battery run still to come.
