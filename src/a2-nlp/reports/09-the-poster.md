# Building a model factory: what we made, what it found, and what went wrong

*A2-NLP · CSED 504 · the plain-language version*

Written for someone who has taken an introductory machine learning course and has not built a
training pipeline before. Every technical term is explained where it first appears.

## Two posters, stacked

The group is presenting **two posters, hung one above the other**, because the project genuinely
has two halves and squeezing both onto one board would serve neither.

**The top poster is Patrick and Leon's: the experiment.** When is it better to train a small
language model from scratch on one under-served language than to reuse a big multilingual model?
Yoruba as the test case, the downstream tasks, the comparison against XLM-R and mmBERT, and what
the answer turned out to be.

**The bottom poster is mine: the factory.** The machinery that produced the models the top poster
compares — corpus preparation, the training fleet, the interfaces the other two called, the
dashboards, and what building it taught us about when a number is a result and when it is an
artifact.

They are meant to be read in that order, and the join between them is the point: the top poster
asks a question, the bottom poster is why it was answerable a hundred and five times instead of
twice. **The fifteen sections below are the fifteen panels of the bottom poster.** Where a
finding belongs upstairs it is stated here only as far as needed to explain what the factory was
for; the top poster carries it properly.

---

## The course this would be

The clearest way to say what the bottom poster is about: **it is the syllabus for a class that
does not exist yet.**

The sequence already teaches you to build a model, one layer of abstraction at a time.

| | what it gave you |
|---|---|
| **501** | classical ML with `sklearn` — `model.fit()` **is** the training. No loop, no device, no batch |
| **502** | everything by hand, in NumPy: kNN and softmax, backprop, ConvNets, BatchNorm and Dropout, the LSTM cell, and attention — first built here, over image captions |
| **503** | the language stack: n-gram models and perplexity, GloVe, attention again and checked against PyTorch to 1e-7, minGPT on Shakespeare, decoding and distillation |
| **504** | scale, twice — images then text — on hardware you have to schedule |

One thread runs the whole way through and it is worth naming, because it is the reason this poster
exists. **In 503 assignment 1 you wrote `hyper_sweep.py`** — a cross-platform hyperparameter sweep
that uploads its features to the GPU once and reuses them. That file is the direct ancestor of
`train_fleet.py`. The first version of the factory was a homework submission.

Every one of those ends when a model finishes training. **None of them covers what happens when
you need a hundred models and have to believe the differences between them.** That is a different
skill, it is most of what this term actually consisted of, and it is what the fifteen panels below
would be if they were a ten-week course.

### CSED 505: Building a Model Factory

Nine weeks as a three-by-three grid, and the tenth as the strip along the bottom, because week 10
*is* the conclusion.

Every week is a **question we could not look up**. Not one of these was answered by reproducing
somebody's result — each needed a measurement that did not exist, and three of them were run on
the two cards over the weekend this poster was written.

| | | |
|---|---|---|
| **1 · What does a run cost, and in what unit?**<br>Epochs stop being a unit the moment the dataset is a variable. Had to learn: 62,500 steps is not a choice, it is 1.024B tokens ÷ a batch — and what the same run costs on a laptop, a Colab session and this box.<br>*panel 5* | **2 · Why optimize before anything needs it?**<br>2.07×, measured before it was necessary. Had to learn: the hours saved are not the point. Speed changes which experiments you are willing to **start**, and four of our five corrections came from a cheap re-run somebody did on a hunch.<br>*panel 4* | **3 · Infrastructure that survives you**<br>Two runs silently overwrote each other; two people prepared "the same" corpus and got different vocabularies. Had to learn: a filename is an identity, a vocabulary needs a fingerprint, and a dashboard that shows an empty page while both cards sit at 85% is worse than none.<br>*panels 6–8 · figure 07* |
| **4 · Is this difference real?**<br>Two lessons, the second learned this weekend. Spread is a property of the **cell**, not a constant — it ranges 0.003 to 2.156 and we used one number everywhere. And our rule, *"bigger than the spread"*, is half a rule: at three seeds a difference must be **2.27×** the spread, not 1.0×. Our own headline number was sitting in that gap.<br>*panel 10 · figures 04, 13* | **5 · What does a model that learned nothing score?**<br>0.403 and 0.414. Had to learn that you cannot read any result before that number exists: one of our "findings" was a baseline that had never trained, and the floor is what exposed it.<br>*panels 9–10 · figure 01* | **6 · Units that silently are not units**<br>Two vocabularies produce two losses that are not on one scale. Had to learn to convert to bits per character — and that matched *steps* handed one arm 5.1× the compute and reversed the conclusion.<br>*panel 10 · figure 03* |
| **7 · Is your metric the right metric?**<br>**Run this weekend.** We minimized validation loss for a term and never checked it. 19 checkpoints, both tasks: it predicts topic classification (r = −0.888) and entity recognition **not at all** (r = +0.303). The aggregate −0.935 is three broken models holding up a line.<br>*panel 14 · figure 11* | **8 · Does a tuned setting transfer?**<br>**Run this weekend.** The board claims adding a language is one function call — true only if the settings come too. Five languages, four rates: the best rate is **not** the same, and one language diverges exactly where three others improve. The sweep then caught itself hitting its own boundary.<br>*panel 14* | **9 · Detect the failure, or prevent it?**<br>13 runs never learned — 16% of all compute. Had to learn that every abandonment rule **loses money**: best case −24 GPU-hours, because a 12% base rate against an asymmetric penalty cannot close. **Running now:** whether clipping prevents them instead.<br>*panel 14 · figure 10* |

**10 · Writing it down so it stays true** — *the strip along the bottom.* What it cost, what we
got wrong, and the machinery that keeps a report honest after the run that produced it is
forgotten: numbers generated from records rather than typed, a staleness check that names any
sentence the data no longer supports, and the five-constants table.
*panels 11, 13, 15 · figure 06*

**Prerequisite: 504. Assessment: build the factory, then find the five places it lied to you.**

The joke in that last line is that it is the actual assessment. Every panel that reports a finding
also reports the mistake that nearly buried it, and the mistakes are the transferable part — the
models are not.

### Why this framing, and not "here is our experiment"

Two reasons, and the second is the honest one.

A poster that says *we got five things wrong* reads as a confession. A poster that says *here are
the ten weeks it would take not to get them wrong* reads as expertise. The content is identical.

And it is what you would actually tell the next student. Nobody needs our Yoruba checkpoints. What
transfers is the order in which the traps arrive, which is what a syllabus is.

---

## 1. Summary

Our group asked a question about language: **when is it better to train a small language model
from scratch on one under-served language than to reuse a big multilingual model trained on a
hundred languages at once?** Yoruba — spoken by around 45 million people, and badly served by
most language technology — was the test case.

That question is the **top poster**. My half was **building the machinery that could answer it**,
and then finding out what building it teaches you — and that is this poster, the one below it.

Six things we can now say, none of which we believed at the start:

**A tiny model came out ahead of a much larger one at the task that needs meaning.** A
33.8-million-parameter model trained on 64 million words of Yoruba scored **0.666** at sorting
Yoruba text into topics. mmBERT — 246 million parameters, and reported by its authors as trained
on roughly three trillion words across 1,800 languages — scored **0.595**. Ours is about a seventh
the size and saw something like one part in fifty thousand of the text.

**Say "ahead", not "beats".** The margin is 0.071 on a 204-item test set, and the confidence
intervals overlap: [0.603, 0.711] against [0.520, 0.652]. Both numbers are the best of a
learning-rate sweep, and both sweeps picked their winner on the same 204 items they are then
reported on — which inflates both. The honest statement is *the small model is at least
competitive with a model seven times its size*, and that is remarkable enough without
overclaiming. Patrick is re-running both arms with the winner chosen on a held-out development
split; that number, not this one, is the one to trust.

![Topic classification and entity recognition, three models each, with the untrained floor drawn
as a dashed rule](figures/01-headline.png)

(The parameter counts we measured ourselves from the published configurations. The training-data
figure is the authors' claim, not something we can verify.)

**Yoruba is not a hard language to model. It is an under-served one.** Given the same amount of
text and the same amount of computer time, a model learns as much structure from Yoruba as from
English. Nothing about the language resists being learned; what is missing is the text and the
tooling.

**The disadvantage a multilingual model carries is in its vocabulary, not in the language.** We
measured this two independent ways and they agree.

![Seventeen languages ranked by how many XLM-R tokens they need per purpose-built
token](figures/02-tokenizer-gradient.png)

**More Yoruba text would not have helped much.** Past roughly 64 million words, adding sixteen
times more text produced no measurable improvement. That is a surprising and slightly deflating
result: the field's usual complaint about low-resource languages — *there isn't enough data* —
was not our binding constraint.

![Validation loss against corpus size, flattening after 64 million words, with the run-to-run
noise band drawn behind it](figures/05-data-saturation.png)

**The bigger model was not too big. It was misconfigured.** One setting, changed from 1.0 to 0.5,
moved its score by a full unit and made it thirty-eight times more reproducible.

![Thirteen identical runs of the 98M model landing in two separate populations with a gap between
them](figures/04-two-outcomes.png)

Thirteen runs of that model, every setting identical. They do not scatter around an average —
they land in two groups with nothing in between. Reporting the mean of those thirteen numbers
would describe a run that never happened.

**And the most useful finding is not on that list.** Five separate times, a number that looked
like a scientific result turned out to be an artifact of a setting nobody had questioned. Section
9 is about those five, because they are the part a student can actually use.

---

## 2. The problem, in two halves

Think of the project as a building with two floors — which is literally how the two posters are
hung, one above the other.

**Upstairs, Patrick and Leon** are asking the research question. Does a small, language-specific
model beat a large multilingual one for Yoruba? To answer that they need trained models to
compare, evaluation harnesses to score them, and enough repetitions to know whether a difference
is real or luck.

**Why Yoruba, and what it is like.** Yoruba is spoken by about 45 million people in Nigeria, Benin
and Togo. It is written in the Latin alphabet with three tone marks, which matters more than it
sounds — *ọkọ* can mean husband, vehicle, or hoe depending on the marks, so software that strips
accents destroys meaning rather than merely looking untidy. It is a **low-resource** language, and
that phrase means something concrete you can measure:

| | English | Yoruba |
|---|---|---|
| Text we could collect | 4.7 billion characters | 260 million characters |
| Usable training tokens | 1.1 billion | 69 million |
| Wikipedia articles | ~7,000,000 | ~34,000 |
| Characters per token, our vocabulary | 4.25 | 3.73 |

**Sixteen times less text.** That is the whole difficulty in one number, and it is not a property
of the language — it is a property of what has been written down and digitized. This is why "is
Yoruba harder to model, or just less served?" was worth asking, and why the answer mattered.
**Downstairs — my half — is the machinery that produces those models.** Collecting text.
Converting it into a form a GPU can train on. Running dozens of training jobs across two graphics
cards without them colliding. Recording what happened so a result can be traced back to the exact
settings that produced it. Noticing when a run has gone wrong.

The split matters because the two floors fail differently. Upstairs, a wrong answer looks like a
wrong answer — you argue about it. Downstairs, a wrong answer looks like **a perfectly reasonable
number**, and nobody argues with it at all.

Almost everything this project got wrong, it got wrong downstairs. That asymmetry is why the
downstairs half is worth a poster of its own rather than a footnote on Patrick and Leon's: the
failures that cost this project the most time were never wrong *answers*, they were reasonable
numbers produced by machinery nobody was looking at.

**So why did we train models on English, Mandarin, French, Indonesian, and twelve more?** Not
drift. Three of those languages are doing specific jobs no Yoruba run could do:

- **English is the ruler.** To say "Yoruba is not harder to model, it is just under-served" you
  need a language where data is *not* the constraint, run at the same sizes with the same tooling.
  The 1.1-billion-token English ladder exists so the Yoruba result has something to be measured
  against. It is also what showed that more data stops helping past 64 million words — a fact we
  could never have established on Yoruba, because Yoruba does not have more.
- **Mandarin and French are the control group.** Our central claim is that XLM-R's vocabulary is
  expensive *for languages it was not trained on*. The obvious objection is that it might just be
  expensive for African languages, or for languages with tone marks, or for anything unfamiliar.
  The only way to kill that objection is to run the same measurement on languages XLM-R **was**
  trained on. French and Indonesian come out at 1.04 and 1.01 — essentially free — and Mandarin at
  0.95, actually cheaper. Seventeen corpora, and the split falls exactly along coverage.
- **The twelve African languages are the gradient.** One language gives you an anecdote. Seventeen
  gives you a slope, and a visible exception (Wolof, at 1.31, is uncovered but cheap) that an
  anecdote would have hidden.

And there is a fourth reason that is honestly about the tooling rather than the science: **adding
a language is one function call**. Twelve of them trained in 48 minutes. A factory that can only
make one thing is not a factory, and the cheapest way to prove the layering worked was to point it
somewhere it had never been pointed.

I joined the group partway through the term, after they had a working proof of concept in a
notebook. The immediate question was not "what should we discover" but "what would let these two
people run ten times as many experiments without ten times the effort".

---

## 3. The hypothesis

I had already built something similar for the computer-vision half of this course: a set of tools
for training many image classifiers across a grid of settings and comparing them fairly.

**The hypothesis was that the machinery would transfer even though the subject would not.**

An image classifier and a language model have almost nothing in common at the level of the maths.
One looks at pixels and predicts a category; the other looks at text and predicts missing words.
But the *scaffolding* around them is nearly identical: you still need to prepare data once and
reuse it, still need to run many jobs across the hardware you have, still need to record results
so they can be compared, still need to notice when something has gone wrong at 3am.

If that guess was right, most of the work would be adaptation rather than invention, and the
group would get their tooling in days instead of weeks.

**It was mostly right, and the exceptions were the interesting part.** We measured how right
later: of the code involved, about 1,700 lines are specific to language modeling and everything
underneath — the scheduler, the dashboards, the record-keeping — is not. The two studies share
that layer.

---

## 4. Preparation: from coursework to real hardware

In coursework you are usually handed a dataset that fits in memory and a model that trains in
minutes. Neither is true here, and the differences are where the engineering lives.

**The data does not fit.** Our largest corpus is 4.7 billion characters. Loading it as ordinary
text would need more memory than the machine has. So text is converted once into a compact
numeric form and stored on disk in a way that lets the program read pieces of it without loading
the whole thing.

**The numbers can be made smaller.** Each word-piece gets an ID number. If your vocabulary has
16,000 entries, every ID fits in two bytes; if it has 250,000, you need four. Choosing the
smaller one where possible halves the memory the data occupies on the graphics card. This sounds
like a triviality. It is the difference between a dataset fitting on the card and not.

**The hardware is genuinely fast, and that changes what you measure.** Two RTX PRO 6000 cards,
96 GB of memory each. Early on, the training was leaving the cards mostly idle — not because they
were slow, but because the program was feeding them badly. Fixing that made the same work run
**twice as fast**, and none of the fix had anything to do with machine learning. It was about
keeping data on the card instead of shuttling it back and forth.

**Everything is measured, not assumed.** Before committing a night of computer time to an
experiment, we measure how fast it actually runs and predict how long it will take. Several
times that prediction changed the plan — once it showed a comparison we were about to run would
have taken ten hours to answer a question we had already answered by accident.

### Why do this first, when nothing needs it yet?

This is the panel we would argue hardest for, because the instinct is exactly backwards. Making
things fast feels like the thing you do at the end, once the science is settled and you want to
scale it up. On this project it was the thing that **made the science possible at all**, and the
argument is not sentimental — it is arithmetic.

**What the optimization was.** Two changes, neither of them clever: raise the batch from 64 to 128,
and actually use the second card. Measured on the same work: **25.2 minutes → 12.2 minutes, a
2.07× speedup**, with utilization going to 91% and 93%. The batch change alone was 1.31× measured
against 1.33× predicted.

**What that bought in hours.** The project spent **83.3 GPU-hours**. Without the 2.07× it would
have been about 172, which on two cards and realistic evenings is ten or eleven nights instead of
five. Real, but not the interesting part.

**What it bought in experiments.** This is the part worth putting on a wall. Speed does not just
finish the same work sooner — it changes *which work you are willing to start*:

- **We ran 20 cells at more than one seed.** Nobody replicates a cell three times when a cell costs
  an evening. Replication is what turned "run-to-run spread is 0.049" into "it ranges from 0.003 to
  2.156", and that correction invalidated every earlier claim we had judged against the constant.
- **Most of the corrections came from impulsive re-checks.** A nine-minute run gets re-run on a
  hunch. A forty-minute run does not. Four of the five entries in the constants table were found
  by somebody going "hang on" and re-running something cheap.
- **A live example from this weekend.** The study asking whether validation loss predicts
  downstream score fine-tunes at 1.2 minutes a seed, so 19 checkpoints × 2 tasks × 3 seeds is 114
  runs in 2.3 hours. At ten times that cost it would have been a 23-hour job, and we would have
  picked five checkpoints instead of nineteen — which is not enough points to see that the
  correlation holds on one task and vanishes on the other. **The finding needs the speed to exist.**

**The general form**, which is the transferable bit: the value of making a run faster is not the
time saved on the runs you were already going to do. It is the runs you would otherwise have
talked yourself out of. Below some cost per experiment, checking a suspicion becomes cheaper than
arguing about it — and that is the threshold where a project starts finding its own mistakes
instead of shipping them.

There is a documented instance of this in [report 03](03-efficiency.md) §6b: the faster
configuration did not merely finish sooner, it produced a cleaner scientific reading, because the
budget that had been spent on one slow run could be spent on several fast ones.

### What changed going from images to text

The same factory had already trained image classifiers for the previous assignment. Roughly two
thirds of it carried over untouched; the third that did not is instructive about where the real
differences between the two fields sit.

| | images (A1) | text (A2) |
|---|---|---|
| One example is | a fixed 32×32 grid of pixels | a variable-length sequence of word-pieces |
| Preparation | resize and normalize | **build a vocabulary first**, then encode |
| What the model predicts | one label out of 100 | a word-piece at every masked position |
| Output layer size | 100 | **16,000 — or 250,000** |
| Data on the card | 0.6 GB of pixels | 0.13 GB of `uint16` tokens |
| Scores comparable across datasets? | yes, accuracy is accuracy | **no** — loss depends on the vocabulary |

**The vocabulary is the whole difference.** An image model's input is a number a camera produced;
a text model's input is a number *you invented* when you built the vocabulary, and it appears
twice — once at the input and once at the output layer. Everything awkward about A2 traces back to
that. Two people can prepare "the same" corpus and get incompatible models. The final layer can be
five times the cost of the entire rest of the network. Two models' losses cannot be compared at
all unless you convert to bits per character first. None of these problems exist for images.

What carried over unchanged: the scheduler, the run supervisor, the checkpoint and resume logic,
the stall detectors, the dashboard's run discovery, and the habit of storing every setting in the
filename. Those never needed to know whether they were moving pixels or tokens — which is exactly
the payoff of having drawn the layer boundary in the right place. `text_data.py` and
`text_prepare.py`, the modules that feed the GPU, contain **zero** references to masking; the
masked-language study was built on top of them without changing a line.

---

## 5. Budgeting a training run: the unit you were taught, and where it stops working

Every visitor will eventually ask why one run takes ninety minutes. The honest answer is a short
lesson in why the unit you learned first quietly stops being a unit — and you have already written
every piece of it by hand in an earlier course.

### The habit, and why it was correct

By the end of 503 you had built a language model from the bottom: n-gram counts with add-k
smoothing in A2, attention from scratch in A4, plugged into minGPT on Shakespeare. In 502 you
wrote `lstm_step_forward` and its backward pass by hand. And in every one of those, and in the
image half of 504, work was measured in **epochs**.

That was correct, because of a property so dependable nobody names it: **the dataset was a fixed
object.** In 503 A4, a batch was a batch of *lines* — every Shakespeare line truncated or padded
to `MAX_LEN = 100`. A fixed set of lines has a well-defined "once through":

```
N lines  ÷  batch size  =  steps per epoch          <- a real, countable thing
```

Same for CIFAR-100's 50,000 images. Because the denominator never moves, "40 epochs" and "15,600
steps" and "2 million examples seen" are three names for one quantity. Pick whichever reads best.

### The exact moment it breaks

The factory replaced A4's padded lines with **windows over a continuous stream** — `text_data.py`
holds the whole corpus as one flat array and each batch is a gather of random `seq_len`-token
windows. No padding, no `<PAD>`, no wasted positions.

That change is worth about a third more useful compute per batch, and it costs you the epoch.
Once you are sampling windows from a stream, **"one pass over the data" is a choice of stride, not
an object you can count.** There is no natural moment at which the corpus has been consumed.

And then the second break, which is the one that actually matters here. Our English study trains
on 4M, 16M, 64M, 256M and 1024M tokens — **the dataset size is the experiment.** Suppose we had
asked for forty epochs at every rung:

| corpus | 40 epochs means | compute used |
|---|---|---|
| 4M tokens | 160M tokens processed | 1× |
| 1024M tokens | 40,960M tokens processed | **256×** |

The big-data arm gets 256 times the compute. When it wins, we have learned nothing — "more data
helped" and "more compute helped" are indistinguishable. **The experiment would be confounded by
its own unit.**

### Where 62,500 steps actually comes from

It was never chosen. It is derived, and the direction is the whole point:

```
1. choose the COMPUTE BUDGET     1,024,000,000 tokens of training
2. choose the batch shape        128 sequences × 128 tokens = 16,384 tokens per step
3. divide                        1,024,000,000 ÷ 16,384    = 62,500 steps
```

Nobody picked 62,500. Somebody picked **1.024 billion tokens**, and 62,500 is what that becomes
once the batch is fixed. Run the same study at batch 256 tomorrow and it is 31,250 steps — same
experiment, different number, which is the tell that steps were never the real unit.

And 1.024 billion is the **top rung of the data ladder**, chosen so the largest corpus gets
exactly one pass: every token seen once, no repetition. That is the clean reference point, and
every smaller rung becomes a measured amount of repetition against it:

| corpus | passes, at a fixed 1.024B-token budget |
|---|---|
| 4M | 256 |
| 16M | 64 |
| 64M | **16** |
| 256M | 4 |
| 1024M | **1** |

Identical compute in every row. Only the data moves. *That* is an experiment about data.

### So why not train for 30,000 steps and save an hour?

Three reasons, worst last.

**It asks a different question.** 30,000 steps is 491M tokens — a different budget, so the answer
cannot be set beside the 105 runs already recorded.

**It breaks the design.** At the 1024M rung, 30,000 steps is 0.48 passes: the model never sees
half its corpus, and the "largest corpus gets exactly one clean pass" property that makes the
ladder readable is gone.

**The run is genuinely unfinished.** The learning-rate schedule anneals to zero at the *planned*
end. Stopping at 30,000 of 62,500 does not give a slightly worse model — it gives a model caught
mid-schedule with a large learning rate still applied, which is the top half of the figure in
panel 11a.

### Where the epochs went

They stopped being what we set and became what we measure, recorded in every run as `passes`:

| | 503 A4 / CIFAR-100 | this study |
|---|---|---|
| one example | a padded 100-token line / a 32×32 image | a 128-token window of a stream |
| the dataset | fixed | **the independent variable** |
| one epoch | N ÷ batch, countable | undefined — depends on stride |
| what we FIX | epochs (40) | tokens (1.024B) |
| what FALLS OUT | steps (~15,600) | passes (**14.8** on Yoruba) |

### The bug this left in our own records

Open any loss curve in this project. There is a field called `epoch`. At the end of the Yoruba run
it reads **125**. The true number of passes is **16**.

```
step 62,500     epoch field = 125     <- step ÷ 500, the logging interval
                passes      = 16.0    <- the truth
```

The field came across with the image code, where it genuinely *was* an epoch because the
denominator never moved. Ported to a token stream it kept the name and lost the meaning: it counts
log lines now. For weeks the dashboard showed "125 epochs" to people who reasonably read it as 125
passes — an eightfold error — and nobody caught it, because the number looked plausible.

### Ninety minutes on what?

"A run is ninety minutes" is only useful if you know what it is ninety minutes *on*. Almost nobody
reading this board owns two Blackwell cards, and the number a student needs is the one for the
machine in front of them. `bench_portable.py` measures it anywhere — no corpus, no tokenizer, no
repository data, because a transformer's cost per step does not depend on which token ids arrive,
so a stream of random integers times identically to real text and the file pastes into a fresh
Colab cell.

| machine | 33.8M model | 98M model | one 62,500-step run (98M) |
|---|---|---|---|
| **Toothless** — 1 × RTX PRO 6000 Blackwell Max-Q, 96 GB | 382k tok/s | 184k tok/s | **93 min** |
| Surface Studio Laptop — RTX 2000 Ada Mobile | *to measure* | *to measure* | |
| Google Colab — free tier | *to measure* | *to measure* | |
| Google Colab — paid tier | *to measure* | *to measure* | |
| MacBook Pro — Apple Silicon, MPS | *to measure* | *to measure* | |

The Toothless figures are medians across 96 and 55 completed runs, not a short probe — sustained
rates after the card is hot, which is the only kind worth quoting. A twenty-step benchmark on a
cold card reads about 10% high, and a benchmark run while another job holds the same card reads
about **half**, which is a mistake we made and the script now warns about.

### How to tell these cards apart

Colab offers you a menu — T4, L4, A100, two kinds of TPU — with no explanation, and most students
have no reason to know that "T4" is a 2018 part or that a TPU is not a fast GPU. Three properties
decide everything, and none of them is the headline number on a spec sheet.

| | generation | memory | bf16? | what that means here |
|---|---|---|---|---|
| **T4** | Turing, 2018 | 16 GB GDDR6 | **no** | Colab's free card. Predates bf16, so our stack silently falls back to fp16 — which works, but is the precision we chose *not* to use |
| **L4** | Ada, 2023 | 24 GB GDDR6 | yes | the value option: modern instruction set, modest bandwidth |
| **A100** | Ampere, 2020 | 40 or 80 GB **HBM2e** | yes | older architecture than the L4 and far faster anyway, because HBM has roughly 5× the memory bandwidth |
| **RTX 2000 Ada Mobile** | Ada, 2023 | 8 GB GDDR6 | yes | a laptop card. Modern, but 8 GB is the binding constraint |
| **RTX PRO 6000 Max-Q** | Blackwell, 2025 | 96 GB GDDR7 | yes | ours. Compute capability 12.0, 188 SMs, both measured rather than quoted |
| **Apple M-series** | — | *unified* | not in PyTorch | a different world; see below |
| **TPU v5e / v6e** | Google | — | — | **not a GPU.** Different programming model entirely — `torch_xla`, a different training loop. Our code exits rather than pretend |

*Vendor figures except where noted as measured. Treat them as ordering, not arithmetic.*

**The three things that actually decide it**

1. **Does the model fit?** This is binary and it beats everything else. The 98M model at our batch
   needs about 10 GB; on an 8 GB laptop card the answer is simply no, and no amount of patience
   changes it.
2. **Memory bandwidth, not FLOPS.** Our models are small and our batches are small, so the card
   spends its time moving parameters, not multiplying them. That is why an A100 — a *older*
   architecture than an L4 — beats it comfortably: HBM2e against GDDR6.
3. **Generation, because of bf16.** Anything before Ampere lacks bfloat16. On a T4 our stack falls
   back to fp16, which needs loss scaling and is the failure mode we deliberately avoided.
   Compute capability is the honest way to read this: our card reports 12.0, a T4 reports 7.5.

**Apple Silicon, which is the comparison people get most wrong**

A MacBook Pro with 48 GB of unified memory can *hold* a model that will not fit on an A100-40GB.
It will still train it many times slower, and the reason is worth understanding because it
generalizes: **holding and computing are different constraints.**

- **Unified memory is not VRAM.** It is shared with the operating system and everything else
  running, and its bandwidth is on the order of a few hundred GB/s against an A100's ~1,500–2,000.
  For a workload that is bandwidth-bound, that ratio *is* the performance ratio.
- **The Neural Engine does not participate.** It accelerates inference through CoreML; PyTorch
  training runs on the GPU cores through MPS and never touches it.
- **The software stack is thinner.** Many fused kernels simply do not exist on MPS, and our
  benchmark runs fp32 there deliberately rather than pretend autocast is equivalent.

So the MacBook row in the table above is **not comparable to the CUDA rows**, and we label it
rather than quietly listing them side by side. It is there to answer "can I try this on my
laptop?" — for which the answer is yes for the small model, and the honest cost is a number, not
a shrug.

### You do not need the workstation

This is the part we would put in the largest type on the board, because the $24,000 of graphics
cards is the number that makes a reader decide this work is not available to them.

**It is available to them.** The entire project — 105 pretrained models, 172 fine-tuning runs,
83.3 GPU-hours — is reproducible on a Colab subscription. Working from measured throughput and
Colab's published compute-unit rates:

| where | the whole project | inside a $500 budget? |
|---|---|---|
| Colab A100 | ~100 GPU-hours, ~$120 | **yes, at a quarter of it** |
| Colab L4 | ~185 GPU-hours, ~$220 | **yes** |
| Colab T4 (free tier's card) | ~460 GPU-hours, ~$550 | marginally not — but every experiment that matters fits |

Those rows are estimates until `colab_reproduce.ipynb` returns real numbers; that notebook
retrains the exact headline model on Colab and computes the table from what it measures rather
than from a guess.

**Against $24,000 of cards, $120 is two hundred times cheaper.** The workstation bought us
*latency* — an answer in ninety minutes rather than tomorrow — and panel 4 argues that latency is
what let the project find its own mistakes. It did not buy access to the science. A student with a
subscription and patience can run every experiment on this board.

### And the third route, which costs nothing at all

There is an option between "buy a workstation" and "pay for cloud" that almost nobody proposes to
students: **plug your laptop in and let it train while you sleep.**

Eight hours a night, five nights a week, is **40 GPU-hours a month for free** — on hardware you
already own and are not using between midnight and eight. Our entire project was 83.3 GPU-hours.
Even at four or five times a laptop's disadvantage, the small-model half of this study is a
month of nights. Drop the 98M model, which will not fit in 8 GB anyway, and it is comfortably
less.

| route | money | elapsed | what you need |
|---|---|---|---|
| buy the workstation | ~$24,000 | ~5 nights | the money |
| rent Colab | ~$120–220 | a few weeks of sessions | a subscription and patience with queues |
| **your own laptop, overnight** | **$0** | **~a month of nights** | a power cable and a queue that survives being left alone |

That last column is the point, and it is where this whole board turns out to be about something
other than two expensive cards.

**The factory is worth more the less hardware you have, not less.** Everything on this poster that
looks like infrastructure for a big machine is exactly what makes the laptop route possible:

- a **queue** you describe once and walk away from — because you are asleep, not supervising;
- **`reuse=True` on everything**, so a night that ends early costs nothing and the next one
  resumes rather than restarting;
- **records that survive the run**, because on a thirty-night study you will not remember what
  night eleven was for;
- **a dashboard**, because the failure you cannot see at midnight is the one that wastes the
  whole month.

Someone with two Blackwell cards can get away without any of that. They will notice a wasted night
the next morning. On a laptop over a month, a silent failure on night three is discovered on night
thirty — which is the difference between a study and nothing.

**Three practical notes for the overnight route.** Keep it plugged in: our own a1-cv measurements
found a 17% swing from thermal boost behavior, and a battery-throttled night is not comparable to
a mains one. Disable sleep, not just the screen. And expect sustained throughput to sit below any
short benchmark, because a laptop chassis cannot hold peak clocks for eight hours — which is
another reason the honest number is the one measured over a real run rather than over forty steps.

Three caveats, because this is the number people will quote:

- A subscription buys a **queue, not a machine.** Sessions end. The 34-hour studies here would
  have to be cut into resumable pieces — which our `reuse=True` convention already supports, and
  which is a good argument for having built it.
- You get **whichever GPU is free**, so a study split across an A100 and an L4 has hardware as an
  uncontrolled variable. Week 3's fingerprint discipline is exactly the tool for noticing that.
- **Owning wins eventually.** The crossover against rental is ~9,300 GPU-hours. This project used
  83 — 0.9% of the way there.

Two things this table is really for. The first is that "it does not fit" is a legitimate entry: a
mobile card with 8 GB cannot hold the 98M model at this batch, and knowing that before you plan a
term is worth more than any throughput number. The second is the ratio — if the same study is four
days on a laptop and thirty-four hours here, that difference is not convenience, it is the
difference between a study you run and a study you abandon. Which is the whole argument of panel 4.

### You have met this exact trap before

In 503 A2 you learned that perplexity depends on the `<unk>` threshold: change which rare words
collapse to a single token and you change the vocabulary, and two n-gram models with different
thresholds cannot be compared by perplexity at all.

That is *precisely* the problem this project hit again with tokenizers. A 16k vocabulary and a
250k vocabulary produce losses that are not on the same scale, which is why nothing here is
comparable until it is converted to **bits per character** — the unit that divides out the
vocabulary, exactly as A2's fixed-vocabulary rule did by hand.

Same lesson, five years of abstraction apart, and we rediscovered it the expensive way. That is
the argument for this whole poster: the traps do not get more sophisticated as the models do, they
just get harder to see.

---

## 6. Setting up a factory: workloads, interfaces, and adjustments

A "model factory" is not a metaphor for anything clever. It is the boring observation that if you
are going to train a hundred models, you should build a production line rather than assemble each
one by hand.

The production line has four stations:

**Prepare a corpus, once.** Download text in a language, build a vocabulary from it, convert
everything to numbers, save it under a name. Done once per language and reused forever. Preparing
1.1 billion words of English takes twenty minutes; every experiment afterwards costs nothing to
set up.

**Train one model.** Given a corpus name, a size, and a budget, train a model and record
everything about it.

**Run many models.** Given a list of configurations, work through them across both graphics
cards, starting a new one the moment a card frees up.

**Read the results.** Every finished run leaves a record. Ask for all of them and get a table.

The interface Patrick and Leon actually call is nine functions. That number is deliberate. A tool
with a hundred options is a tool nobody else can use, and the point was for two other people to be
able to run experiments without reading my code.

Here is the whole surface. Nine functions to pretrain, and six more to fine-tune and score:

```python
import mlm_api as factory

# --- once per language --------------------------------------------------------------
factory.prepare_corpus('yor', lang='yor_Latn')   # text -> vocabulary -> numbers on disk
factory.corpus_info('yor')                       # how big, what vocabulary, what fingerprint

# --- before committing a night to it ------------------------------------------------
factory.estimate('yor', [(64_000_000, 62_500)])  # measures THIS box, predicts the hours

# --- the experiment -----------------------------------------------------------------
factory.pretrain('yor', tokens=64_000_000, steps=62_500, seed=0)
factory.random_init('yor')                       # the untrained control, in seconds

# --- reading it back ------------------------------------------------------------------
factory.results('yor_*')                         # every matching run, as a list of records
factory.curve('yor_64M_62.5k_s0')                # the loss over time for one run
factory.bits_per_char('yor_64M_62.5k_s0')        # the only unit comparable across vocabularies
```

And the downstream half, which answers "is the model any good at a real task":

```python
import ft_api as ft

ft.finetune_once('runs/yor_64M_62.5k_s0', task='sib200', lang='yor_Latn', seed=0)
ft.evaluate('runs/yor_64M_62.5k_s0', task='sib200', seeds=(0, 1, 2))   # mean, sd, CI
ft.table()                                       # every downstream result, one table
```

Three design decisions in there are worth a student's attention, because each one was bought with
a mistake:

- **`estimate()` measures rather than looks up.** It runs twenty real steps on the actual card and
  extrapolates. A table of expected throughput would have been wrong the first time something else
  was using the GPU.
- **`random_init()` is one line.** Making the untrained control trivially cheap to produce is why
  it eventually got produced. When a control costs an afternoon, it does not get run.
- **The tag encodes the settings.** `yor_64M_62.5k_s0` is corpus, data, steps, seed. Two runs that
  differ in anything that matters cannot collide, and a filename is enough to know what you are
  looking at.

**The adjustments were the education.** Every one came from something going wrong:

- Two runs quietly overwrote each other's results because their names collided. Names now include
  every setting that changes the answer.
- Two people who both downloaded "the same" text got different vocabularies, and their scores
  stopped being comparable. Every vocabulary now carries a fingerprint, and runs only compare
  across matching fingerprints.
- A run that finished did not exit, and sat holding a graphics card for thirty hours.
- Two jobs writing the same status file at the same moment crashed one of them mid-experiment.

None of these is a machine learning problem. All of them cost real time.

---

## 7. Notebooks, for going fast

A notebook is an interactive document where you write a bit of code, run it, see the answer, and
write the next bit. It is the right tool for figuring out what you want to do.

The group's original proof of concept was a notebook, and it worked. The trouble is that a
notebook is a bad place to *keep* work. Close the laptop and the state is gone. Run the cells in
a different order and get a different answer. Train for three hours and lose it when the kernel
restarts.

So the factory is arranged so that notebooks stay useful for the part they are good at:

**Exploration stays in the notebook.** Look at the data. Try one small model. Plot something.

**Long work moves to files that run outside it.** A three-hour training job runs as a program you
can start and walk away from.

**Both write to the same place.** A result from the notebook and a result from an overnight job
are the same kind of record, in the same folder, comparable without translation.

We also rewrote the group's original notebook to call the factory instead of doing everything
itself, with each change marked and the code it replaced left visible underneath — so they could
see exactly what had changed rather than being handed something unfamiliar.

---

## 8. Offline processing: queues, dashboards, and catching problems early

Once experiments take hours, three things become necessary that a notebook never needs.

**A queue.** You describe a night's work as a list, and a scheduler works through it across both
cards. The order matters more than you would think: putting the longest jobs first keeps both
cards busy, because a short job started last finishes early and leaves a card idle, while a long
job started last sets the finish time on its own.

**A dashboard.** A web page showing what is training, how far along, how fast, and when it will
finish. This sounds like a nicety. It is not: you cannot fix at 8am a problem you could not see at
midnight.

![The training dashboard: a queue panel, live GPU cards, and a comparison view of five languages](figures/07-dashboard.png)

Reading that screen top to bottom is roughly how the project was run:

- **The queue panel** names every job in tonight's list before any of them start — done, running,
  and *not started yet*. That last category is the one an earlier version hid, which made a
  twelve-job night look like a three-job night and made it impossible to tell "nearly finished"
  from "barely begun".
- **The two cards** show memory, watts, and temperature. Watts is the useful one: a card drawing 4
  W is not training, whatever the rest of the page claims.
- **The comparison view** is where the results get read. Pick an experiment, pick what to compare
  by, and it draws where each run landed against how it got there. The dashed line across both
  charts is what a model that learned nothing scores — so a curve still hugging that line has not
  started, and you can see it at a glance rather than deducing it from a number.
- **The sentence under the title is generated, not written.** "Best so far: French at 2.54, 1.92
  nats of loss ahead of Mandarin." It restates the chart in words, and it comes from the same
  records the chart does, so it cannot say something the data does not. Ours went through several rounds of being wrong in instructive ways — it showed runs as
"training" that had actually finished, it hid everything not yet started so a twelve-job night
looked like a three-job night, and it labeled a counter "epoch" that was not an epoch, which
confused everyone who read it including the people who wrote it.

**Detectors.** Automatic checks that shout when a run has gone wrong:

- *Has this model learned anything at all?* Some runs never get going. Warn halfway rather than
  waste the second half.
- *Has this model learned something and then lost it?* This one we added late, after four runs had
  each burned an hour. A model can train normally for twenty thousand steps and then fall apart,
  and the first detector could not see it — it compared against the *start*, so a run that
  improved and then collapsed still looked like it had improved.
- *Is this score better than a model that never trained?* Patrick added this one, and it caught
  the single biggest error in the project.

---

## 9. How the factory got used, and how it changed

It was not built and then used. It was built *by* being used, and every capability traces to a
question somebody could not otherwise answer.

**"Their corpus is tiny — how tiny?"** Prepare it and count. All the Yoruba in our source is 69
million words. It trains in seconds. That single measurement reframed the project: the group had
been planning around data scarcity, and the scarcity was more extreme than anyone had assumed.

**"Is this result real or luck?"** Train the same thing several times with different random
starting points. If the difference between two setups is smaller than the difference between two
runs of the *same* setup, you have not measured anything. This is the single most valuable habit
the project developed, and section 10 explains what it cost us to learn it.

**"Can we compare languages?"** Not directly — a model's score depends on the vocabulary it uses,
so scores from different languages are not on the same scale. Subtract a baseline that captures
how predictable each language is on its own, and what remains is comparable. That is what let us
say Yoruba is not harder to model than English.

**"Does the tokenizer actually matter?"** This needed a genuinely new capability: measuring
quality in *bits per character* rather than per word-piece, which is the only unit that survives
changing vocabularies. Without it the central experiment could not have been read.

By the end the factory had trained **105 models across 22 corpora and 17 languages, using about
79 hours of graphics-card time** — and the number that matters more is that any of those runs can
be traced to the exact settings that produced it.

---

## 10. What we computed, and what made it hard

**105 models sounds like a lot. Why so many?**

Not because of a grid search. A grid search asks "which settings give the best score?" — you try
combinations and keep the winner. We were asking something different: **"is this difference real?"**

Those need different amounts of compute. To find the best setting you need one run per setting. To
know whether two settings genuinely differ, you need several runs of *each*, because the same
setting run twice gives different answers, and until you know how different, you cannot say
whether anything you measured means anything.

**What the 105 runs were actually for.** They are not one grid. They are seven separate
questions, and the run counts follow from the questions rather than from a product of axis
lengths:

| what it was asking | runs |
|---|---|
| How much text does a model need before more stops helping? (English, 4M → 1024M words) | 46 |
| Does the answer change at a larger model size? (33.8M vs 98M, same ladder) | included above |
| What does a purpose-built vocabulary buy? (Yoruba, 16k vs XLM-R's 250k) | 9 |
| Does the vocabulary penalty track XLM-R's coverage, or something else? (17 languages) | 17 |
| Does gradient clipping fix the unstable model? (clip 1.0 vs 0.5) | 10 |
| Is this difference real, or is it seed noise? (repeats of cells already run) | 20 cells run more than once |
| Baselines, controls, and the untrained floor | the rest |

**And the hyperparameters we did search — and what each one actually did.** This is a genuine
grid search, and it is a *small* part of the total. Every "what happens" column below is measured
on our own runs, not quoted from a textbook.

| knob | what it is | values tried | turn it **down** | turn it **up** |
|---|---|---|---|---|
| **learning rate**, pretraining | how large a correction the model makes at each step | 1e-4 … 1e-3 | slower to converge, but safe — 0 failures below 3e-4 | it stops converging and starts thrashing. 19% of runs at 3e-4 never learned anything; the one run at 1e-3 failed outright |
| **learning rate**, fine-tuning | the same knob, on the downstream task | 5e-6 … 2e-4 | badly under-fits — our model scores **0.275** at 5e-6 | improves, peaks, then turns over. SIB-200 peaks at 3e-5 (**0.666**); NER climbs all the way to 1e-4 (**0.837**) and turns over at 2e-4 |
| **gradient clipping** | a hard ceiling on any single correction | 1.0, 0.5 | *tighter is better here.* At 256M tokens, clip 0.5 gives spread **0.052** and best 2.454 | the default 1.0 gives spread **0.224** and best 2.672 at the same cell — worse *and* four times less reproducible |
| **training steps** | how many corrections the model makes | 2,930 … 62,500 | the run stops before the cliff — see the curve above; the first 11,500 steps look like failure | past the point where the schedule has annealed, nothing more happens; the last third of our run is worth 0.10 |
| **training data** | how much text the model sees | 2M … 1,024M tokens | below ~16M the model is data-starved (6.74 at 4M, 4.35 at 16M) | above ~64M it stops mattering: 16× more text moves the loss **−0.080**, against a spread of 0.185 |
| **model size** | parameters | 33.8M, 98M, 154M | the small model is faster *and* not worse — 419k vs 201k tokens/s | bigger did **not** win. At 256M tokens the two sizes are indistinguishable, and at clipping 1.0 the larger one failed 31% of the time |
| **batch size** | sequences per step | 64, 128, 256, 512 | more, smaller steps; noisier gradients | fewer, larger steps. Fixed at **128** for 99 of 105 runs so nothing in the study depends on it — the other four were throughput probes, not a controlled sweep |
| **warmup fraction** | how long the rate ramps before annealing | none, 0.06 | — | — *we cannot say.* No two runs differ in warmup alone, so we have no controlled comparison. Listing it as "searched" would be overclaiming |
| **sequence length** | tokens per example | 128 | — | — fixed for every single run, deliberately, so no result can depend on it |

Two of those rows are worth a student's attention beyond the numbers:

**The fine-tuning learning rate moves the score more than the model does.** Our model scores 0.275
at one rate and 0.666 at another — a range of 0.391 — while the entire gap to mmBERT is 0.071. A
comparison where one side was tuned and the other was not is not measuring the models at all. That
is exactly the mistake report 08 §2b had to unwind three times.

**"Turn it up" is not the same lever as "make it better."** Clipping is the clean example: the
tighter setting was *better on the metric and more reproducible*, which is the opposite of the
usual speed/quality tradeoff. We only found it because a run's spread looked wrong, not because we
were searching for it.

**Why multiple seeds, when the settings are identical?** Because a neural network starts from
random numbers and shuffles its data randomly, so the same recipe run twice gives two different
models with two different scores. Until you know how far apart *those* are, you cannot interpret
the gap between any two experiments.

We measured that spread on 20 cells that were run more than once. **It is not a constant.** The
median is 0.112, the smallest 0.003, the largest **2.156** — a range of nearly a thousandfold
across cells of the same study. For most of the term we used a single number, 0.049, measured once
on one model on one language, and applied it everywhere. That was the first of the five constants
in the table below, and it is why every early claim was judged too generously.

**The rule we worked to for most of the term:** run the same cell at three seeds *before*
comparing it to anything, and treat a difference smaller than that cell's own spread as no
difference at all.

### The rule was half right, and we read it as if it were whole

That rule is sound in one direction and **silent in the other**, and we had been using it in both.
It correctly rejects a difference smaller than the noise. It says nothing whatever about a
difference slightly *larger* than the noise — and we had been treating "clears the spread" as
"is real."

Here is the bar a two-sample test actually sets:

![The multiple of the seed spread a difference must reach, against the number of seeds, with our
own claims plotted on it](figures/13-how-many-seeds.png)

| seeds per arm | the difference must be | |
|---|---|---|
| 2 | 4.30× the spread | |
| **3** | **2.27×** | ← where we worked |
| 5 | 1.46× | |
| 6 | 1.29× | |
| 10 | 0.94× | the rule finally becomes true here |

At three seeds a difference has to be **more than twice** the spread before it is distinguishable
from nothing. Our threshold was 1.0. Everything between those two lines is a difference we would
have called real and could not have.

**And one of ours was sitting in that gap.** The tokenizer penalty — 0.144 bits per character at
matched compute, the headline of report 08 and a cell on this board — is 1.4× the spread. It
passed our rule and fails the test: Welch t = 1.68, **p = 0.22** at three seeds against three. The
direction is consistent across every seed and the number is not wrong; it is simply not
established.

The audit that found it is [`claims_audit.py`](../claims_audit.py), and it exists because the
check we already had verified that *numbers* in the prose still matched the records while having
nothing to say about whether the claim wrapped around a number was supported. Every mistake this
project shipped lived in that gap.

**What we did about it is the part worth copying.** Not a hedge — three more seeds per arm, which
is four GPU-hours. At the observed effect size n = 6 reaches p = 0.039 and n = 5 only reaches
0.061, so six was fixed in advance and written into the script before the runs started. If the
penalty still fails at six seeds, report 08 will say it is not established. Choosing the sample
size after seeing the result is the failure this whole board is about, and the only thing
separating buying power from p-hacking is which of those two you decided first.

That is why the count is high, and it is where the project's hardest lessons came from.

**How much do two identical runs differ?** We spent most of the term using one number for this —
0.049 — measured once, on one model, on one language. It turned out to be wrong nearly everywhere
it was applied: the true figure was three times larger on some experiments and **twenty-seven times
larger** on others. Every claim we had judged against it was judged too generously.

**Sometimes an average is the wrong summary entirely.** The larger model's results, sorted, looked
like this:

```
2.57  2.67  2.70  2.82  2.90  3.05  3.20  3.28  3.76   |   5.38  5.66  6.72  7.47
```

Nothing between 3.8 and 5.3. That is not a spread around an average — it is two different
outcomes. Some runs learned and some did not, at about a 31% failure rate, and reporting the mean
of those thirteen numbers would have described neither group.

**And five times, a constant decided a result.** This is the finding I would put at the center of
the poster:

| the setting | chosen for a good reason | what it silently decided |
|---|---|---|
| run-to-run spread of 0.049 | measured honestly, once | whether every later difference counted as real |
| a field named "epoch" | compatibility with an older tool | that nobody could tell two different things apart |
| "use the first 400 documents" | a default nobody revisited | a clean-looking result that reverses at 800 |
| "train for 352 steps" | reproducing an earlier notebook exactly | the project's central conclusion about a baseline |
| "give both models the same number of steps" | the obvious way to be fair | that the tokenizer made no difference |

**None of these was a bug.** Every one was a sensible choice where it was made. Each stopped being
sensible somewhere else, and nothing announced the change. That last one is the sharpest: giving
two models the same number of training steps sounds like the definition of a fair comparison. But
one of them was five times more expensive per step, so "the same number of steps" quietly handed
it five times the resources. Read one way the tokenizer made no difference; read correctly it
costs **0.144 bits per character** — 1.6 times the run-to-run spread. **Same three experiments,
opposite conclusions.**

![The same three experiments with two brackets over them, one comparing at matched steps and one
at matched GPU time](figures/03-matched-steps-vs-compute.png)

The left two bars are the comparison as it was first run: same number of steps, and the scores
land on top of each other. The bracket above them is the conclusion that reading supports. The
number printed inside each bar is the one nobody looked at — the middle model used five times the
GPU time to reach it. The third bar is the first model given that same time, and the second
bracket is the conclusion *that* reading supports. Neither bracket is wrong about the data. They
are answers to two different questions, and only one of them was the question we meant to ask.

That is what made this hard, and it is not a machine learning skill. It is the habit of asking, of
every number in a result, *what did I hold fixed, and did I mean to?*

---

## 11. What it cost to build and run

Cost is usually left out of student projects, which is a shame, because it is the thing that
decides what you are allowed to attempt. Here are real figures.

### What the work actually consumed

Everything below is measured from the run records, not estimated.

| | |
|---|---|
| Models pretrained | 105 |
| Fine-tuning experiments | 46, made of 172 individual runs |
| Total graphics-card time | **83.3 GPU-hours** |
| Electricity for that (cards only) | ~27 kWh |
| Electricity at the wall, with cooling and the rest of the machine | ~43 kWh |
| Storage: prepared text | 5.5 GB |
| Storage: model checkpoints and logs | 95.7 GB |

Forty-three kilowatt-hours is about what a domestic fridge uses in a month. **The entire
computational output of this project cost roughly four dollars of electricity.** That is worth
sitting with, because it contradicts the impression most people have of what machine learning
costs — and the reason is that these are small models. The same tooling pointed at a
billion-parameter model would multiply everything below by a hundred or more.

### Time to train one model

| what | how long |
|---|---|
| 33.8M parameters, 196.6M tokens of training | **8.8 minutes** |
| 33.8M parameters, 1.024B tokens of training | 39.8 minutes |
| 98M parameters, 1.024B tokens of training | 92.1 minutes |
| 33.8M parameters but a 250,000-word vocabulary | 40.9 minutes |

That last row is the one to notice. It is the *same model size* as the first two and takes five
times as long as it should, because a bigger vocabulary makes the final layer enormous. Section 10
explains how that nearly produced a wrong scientific conclusion; here it is simply a reminder that
"model size" and "cost to train" are not the same number.

Measured throughput: **419,000 tokens per second** for the small model, **185,000** for the larger
one, taken as the median across all 105 runs.

### Why so many steps, and why we could not just go faster

This is the question an AI/ML student should ask, and the answer is the most transferable thing in
this section.

**Where the number 62,500 comes from.** Each step shows the model 128 sequences of 128 tokens —
16,384 tokens. 62,500 steps is therefore **1.024 billion tokens of training**. The Yoruba corpus
holds 69 million tokens, so the model sees the whole corpus **14.8 times over**. It is not that we
wanted a long run; it is that a language model has to see text repeatedly before the statistics
stick, and 14.8 passes was the smallest number that got us to a stable answer.

**Why not just turn the learning rate up?** We tried. The learning rate is the size of the step
the model takes each time it corrects itself, and turning it up is exactly the "go faster" lever.
Here is what the 105 runs say about it:

| peak learning rate | runs | runs that never learned anything |
|---|---|---|
| 0.0001 – 0.00015 | 4 | 0 |
| 0.0003 | 37 | 7 (19%) |
| 0.0005 | 63 | 5 (8%) |
| 0.001 | 1 | 1 (100%) |

Push the step size up and past some point the model stops converging and starts thrashing. It does
not fail loudly — it produces a finished run, a saved checkpoint, and a number. The number is just
worthless. So the practical limit on "train faster" is not patience. It is that the faster
settings **silently produce garbage**, and you only find out by running the slow version too.

**Why not just stop the run early?** Look at what a run actually does:

![The learning-rate schedule above the loss curve of a single run, with the first 11,500 steps
shaded as a flat plateau](figures/08-why-not-shorter.png)

For the first **11,500 steps — seven minutes, a sixth of the run** — the loss barely moves. Then it
falls off a cliff. A run that is going to fail looks *identical* to a run that is about to succeed,
right up until the cliff, and 31% of our large-model runs never reached it. There is no early stop
you can apply that keeps the good runs and kills the bad ones, because for the first sixth of their
lives they are indistinguishable.

The tail is the honest counterweight: the last third of the run is worth about **0.10** of loss,
which is inside the run-to-run noise. We could have cut the tail. We could not have cut the head,
and the head is where the uncertainty lives.

### Ten more cards would not have made this ten times faster

"Training is slow, buy more GPUs" is the reflex answer, and at this scale it is only half right.

![Wall-clock to finish a twenty-job night against the number of graphics cards, flattening onto
the length of the longest single job](figures/09-scaling-with-cards.png)

More cards finish a *queue* sooner. They never make a single *run* shorter. Twenty jobs on two
cards take 16.1 hours; on ten cards, 3.7; on twenty, 2.2 — and there it stops, because at twenty
cards every job is running already and the night is exactly as long as its longest job. Card
twenty-one changes nothing at all.

That matters more than it sounds, because **the thing we were actually waiting for was usually one
run**. "Does clipping fix the large model?" is one experiment, three seeds, and no amount of
hardware returns the answer in less than ninety minutes. Splitting a single run of this size across
cards is possible but does not pay — the model is small enough that the cards spend the saved time
talking to each other.

So the bottleneck on this project was never money and never card count. It was **the latency of a
single question**, and the only lever that moved it was asking better questions before starting.

### The parts of the pipeline, and what each one costs

Training is the part people picture, but it is not the whole job. Measured end to end on the
Yoruba corpus — 80,000 documents, 260 million characters — by `pipeline_bench.py`:

| stage | what happens | time | where |
|---|---|---|---|
| 1. Collect documents | pull text, filter, normalize to NFC | seconds | CPU |
| 2. Train the tokenizer | learn 16,000 sub-word pieces from a 12.7M-character sample | **21 s** | CPU |
| 3. Encode the corpus | text → token numbers, at 3.25M tokens/s | **21 s** | CPU |
| 4. Store it | 69.1M tokens as `uint16` | — | 0.13 GB on disk |
| 5. Load onto the card | the entire corpus goes GPU-resident, once | 11 s | 0.13 GB of 96 GB |
| 6. Pretrain, 33.8M model | 62,500 steps at 470k tokens/s | **37 min** | 5.8 GB peak |
| 6. Pretrain, 98M model | the same 62,500 steps at 201k tokens/s | **85 min** | 10.0 GB peak |
| 7. Fine-tune and score | 1,056 steps on 701 labeled examples, ×3 seeds | ~4 min | GPU |

**Preparation is about forty seconds. Training is an hour and a half.** That ratio is the whole
reason the factory is shaped the way it is: everything cheap happens interactively in a notebook,
and everything expensive goes into a queue that runs while nobody is watching.

Two details worth pulling out for a student:

- **The corpus lives on the graphics card.** 69 million tokens as 16-bit integers is 138 MB, and
  the card has 96 GB. So there is no data loader, no worker processes, no disk in the training
  loop at all — the batch is a slice of an array already in GPU memory. This is why throughput is
  419,000 tokens per second rather than something bottlenecked by Windows file I/O.
- **`uint16` is a decision, not a default.** It holds numbers up to 65,535, which covers a 16,000
  vocabulary. XLM-R's 250,002-word vocabulary does not fit, so that corpus has to be stored as
  `uint32` and **doubles on disk** — 0.5 GB against 0.13 GB for the same text. The vocabulary
  choice reaches all the way down to the storage layer.

### Buying the hardware versus renting it

![Electricity, cloud rental, and the workstation purchase price on a logarithmic dollar
axis](figures/06-what-it-cost.png)

**What we own (capital expenditure).** Two RTX PRO 6000 Blackwell cards with 96 GB of memory each,
in a workstation. These cards are about **$12,000 each** at the time of writing, so **$24,000** in
cards and call it **$28,000** for the whole machine. The hardware prices are current market
figures; every number in the table above is measured from our own records.

**What renting would have cost (operating expenditure).** Cloud providers charge by the
graphics-card-hour. A card in this class rents for roughly $2–4 per hour depending on provider and
commitment. At 83.3 GPU-hours:

| | cost |
|---|---|
| Cloud, at ~$3/GPU-hour | **~$250** |
| Owning, electricity only | **~$4** |
| Owning, hardware amortized over three years | ~$1.07/hour of *wall-clock ownership*, whether or not you use it |

**So renting would have been cheaper for this project, by a lot.** $250 against $28,000. If you
are doing one term's work, rent.

The arithmetic flips when the machine is busy. $28,000 of hardware pays for itself against
$3/hour at about **9,300 GPU-hours** — roughly six and a half months of both cards running
continuously. A research group that keeps its cards busy is better off owning; a student doing one
project is not. We used 83 hours over about three weeks. Two cards running continuously for three
weeks would be 1,008 card-hours, so that is **8% utilization** — and 8% is generous, since most of
those hours were overnight batches rather than steady work.

**But three things do not show up in that comparison**, and they mattered more than the money:

- **Iteration speed.** A three-minute experiment you can run on impulse is a different activity
  from one that needs a cloud instance provisioned, data uploaded, and a card waited for. Most of
  the corrections in section 10 came from cheap, impulsive re-checks. I do not think I would have
  run them against a meter.
- **Data transfer.** 5.5 GB of prepared text and 96 GB of checkpoints. Moving that in and out of a
  cloud provider costs money and time, and the checkpoints are what let us go back and re-measure
  things weeks later.
- **The overnight pattern.** The most productive thing in the project was queuing eight hours of
  work at midnight. That is free on hardware you own and metered on hardware you rent, and the
  metering changes what you are willing to try speculatively.

### Queue depth, and why it matters

The scheduler works through a list across both cards. The deepest queue we ran was **twenty
jobs**; a typical overnight run was ten to eighteen.

Queue depth is what turns a fast machine into a productive one. Two cards can only run two things
at once, so the useful question is not "how fast is a job" but "how much work can I describe
before going to bed". Some numbers from real nights:

- Twelve languages trained in **48 minutes** running two at a time — but **96 minutes** the first
  time, because a scheduling flaw left one card idle throughout. Same work, same hardware, twice
  the wall-clock.
- The longest single job was 92 minutes; the longest night was about ten hours.
- **Ordering matters.** Longest jobs first, always. A short job started last finishes early and
  leaves a card idle; a long job started last sets the finish time by itself. On one grid, the
  natural reading order produced 12.2 minutes of wall-clock for 10.1 minutes of work.

### The cost nobody counts

The largest expense on this project was not hardware or electricity. It was **the compute spent on
experiments that turned out to answer the wrong question** — and the human time spent finding out.

- Ten hours of queued work canceled after discovering the comparison it would run was confounded.
- Four separate runs that trained normally and then fell apart, an hour each, before we built a
  detector that noticed.
- An entire tokenizer comparison run at the wrong control, then re-run.
- Three learning-rate sweeps whose best value sat at the edge of the range, needing extension.

Call it fifteen to twenty hours of the eighty-three — **around 20% of all compute spent on work
that had to be redone.** For a student project that is fine. It is a useful ratio to know before
you scale it up, because at a hundred times the model size that 20% is the whole budget.

---

## 12. Enabling the upstairs half

The measure of the factory is not what it computed. It is whether two other people could use it —
and the honest test of that is the poster hanging above this one. Every trained model, every
score, and every comparison on Patrick and Leon's board came out of this machinery. If their
poster stands up, that is the result this panel is reporting.

**Something they could call.** Nine functions, documented where they are defined, and nothing else
to import.

**Something that survives being shared.** Corpora, vocabularies and every result are committed to
the repository, so Patrick and Leon can plot our findings without re-running anything.

**Something that works on their machines.** Everything runs on a single graphics card, including
a free cloud notebook. Nothing requires the two-card workstation.

**Something they could find.** This one we got wrong. Leon cloned the repository, read the
documentation, and asked whether there was an interface he was supposed to be using. There was —
he could not find it, because the folder's front page described a *different* study and the
relevant material started 130 lines in. That is a failure of my half, not his, and it is worth
putting on the poster: **a tool nobody can find does not exist.**

**And crucially, something they could argue with.** The most valuable thing the factory produced
was not a model. It was Patrick being able to check a number I was confident about and show it was
wrong — twice. He found that a baseline everyone had quoted for weeks had never actually trained,
and that an evaluation dataset used a different text encoding from our vocabulary, which had
silently reversed a comparison. Both came from him re-running things rather than trusting them.

The collaboration worked because results were reproducible enough to be challenged. That is a
property of the tooling, and it is the property I would defend hardest.

---

## 13. How we used AI, honestly

I used an AI coding assistant throughout, and the useful thing to report is not that it helped —
it is *where* it helped and where it actively made things worse.

**First, what it did not do.** No AI ran overnight, and no AI analyzed a result unsupervised. The
pattern was: compose the night's queue together in the evening, hand it to `mlm_fleet.py`, and let
**plain Python** work through it for six to twelve hours with nothing intelligent in the loop. In
the morning we read the records together. This matters because the alternative — an agent running
unattended for eight hours making judgment calls about experiments — is what people imagine, and
it is not what happened. The scheduler is four hundred lines and has no model in it.

**Where it was genuinely strong.** Writing the scaffolding: schedulers, dashboards, record-keeping,
the hundred small pieces that are tedious rather than hard. Adapting the existing image-model
tooling to language models. Producing careful written explanations of what had been measured.

**The unexpected one: it made a three-person group work.** We used it to write our coordination
emails, and this worked far better than anyone predicted. Three things happened at once:

- **The sender wrote more, not less.** An AI-drafted handoff runs longer than a human would type
  and stays dense — it includes the numbers, the file paths, the thing that was tried and failed —
  where a hand-typed message would have said "pushed, take a look".
- **The receiver got help reading it.** The other end could ask their own assistant what a message
  meant, what a term referred to, what was actually being requested. That is a genuinely different
  experience from re-reading a dense paragraph and guessing.
- **It flattened the language barrier.** Coordinating precisely across first languages is hard,
  and the friction usually shows up as under-communication rather than misunderstanding. This
  removed most of it.

The result was that problems got fixed in **hours instead of at the next meeting**, and that
handoffs arrived with enough context to act on. The group never met in person. Of everything on
this poster, this was the least expected and, in terms of hours saved, possibly the largest single
effect.

**Where it was wrong, and confidently so.** Several times it produced a clean, well-reasoned,
completely incorrect analysis:

- It diagnosed a broken baseline as a real scientific finding, argued itself out of the correct
  first instinct with a plausible-sounding story, and told me the risk was overstated. Patrick
  checked the individual runs and found the model had never trained at all.
- It designed the central tokenizer experiment with a flaw that produced the *opposite* of the
  right answer, and only caught it by noticing a timing column looked odd.
- It reported a "perfect separation" between two groups of languages, with a significance test to
  match, that vanished when the sample size was doubled.

**What actually made it work.** Every claim was checked against the stored results rather than
against what anyone remembered, including the assistant's own written reports. A surprising number
of errors were caught that way — including errors in text the assistant had itself just written.

**The honest summary for a student:** it made me perhaps three to five times faster at building
things, and no faster at all at knowing whether a result was true. Those are different skills, and
only one of them was automated. If you take one thing from this panel, take that the verification
habit is not optional overhead — on this project it was the difference between eight findings and
about three.

---

## 14. What the factory could answer about itself

Everything above is the factory answering questions about *language*. This panel is the factory
answering a question about **training**, using nothing but its own history — 105 stored loss
curves, no new compute, no GPU touched.

**The question.** Thirteen of our 105 runs never learned anything. They did not crash: they
trained for up to ninety minutes, saved a checkpoint, and produced a number that was worthless.
Between them they consumed **13.1 GPU-hours, 16% of everything we ever spent**. The obvious fix is
to notice early and kill them. Should we build that?

![Two panels: the early signal fails to separate the two outcomes, and every abandonment rule
costs more than it saves](figures/10-early-signal.png)

**The obvious detector does not work, and not by a little.** Every model drops about three nats
almost immediately — that is the unigram distribution, which anything learns in a few hundred
steps — and then the doomed ones simply stop. At step 8,000 a run that never learned has gained
**more** (3.31 nats) than the weakest run that went on to succeed (3.22). The populations overlap
at every checkpoint we tested, in both directions. No threshold on "how far has the loss fallen"
can work at any step.

**The rule that does discriminate still loses money.** What actually separates the two groups is
whether the run has *left its plateau* — the cliff from panel 11a. A rule that waits and then
abandons anything still flat catches **all thirteen** doomed runs. It also kills between 35 and 92
good ones, because the cliff arrives anywhere from 15% to 90% of the way through a run. The best
operating point available nets **−24.2 GPU-hours**.

**Why it is structural rather than a tuning failure.** Only 12% of runs are doomed. A doomed run
wastes at most its *remaining* time; a false kill wastes a *whole* run and you have to do it
again. With that base rate and that asymmetry, the arithmetic cannot close no matter how good the
classifier is. That is worth knowing before building the feature, and it took no compute to find
out.

**And the deadline cannot be a constant, because the cliff moves.** Median 16,000 steps, range
2,200 to 48,500 — a 22× spread. By configuration: 7,200 steps for the 33.8M model against 30,000
for the 98M. (Learning rate and model size are perfectly confounded in our runs, so we cannot say
which causes it. That is a real gap, and a cheap experiment would close it.)

**So the honest recommendation is: do not build the detector, and go and test prevention
instead.** We had been saying that tighter gradient clipping "fixes" the large model. Pulled
apart, that turns out to be two claims we had merged: clipping is *established* to tighten the
spread of runs that succeed (0.052 against 0.224 at the matched cell), and it is **entirely
untested** as to whether it prevents failures. On our data the failure rate is 20% against 17% at
n=10 versus n=36, which is no difference at all. Naming that as unmeasured is worth more than
asserting it either way.

---

### Week 7 — is your metric the right metric?

Every pretraining decision this term was made on **validation loss**. Not one was ever checked
against whether the model is any *good* at anything, because of an accident of history: 107
checkpoints on disk and exactly one had ever been fine-tuned.

Checking it cost 2.3 GPU-hours — the pretraining was already paid for, and a fine-tuning seed is
1.2 minutes. Nineteen checkpoints spanning validation loss 2.25 to 5.67, both downstream tasks,
three seeds each.

![Pretraining loss against downstream score on both tasks, with the three under-trained models
shaded](figures/11-metric-validity.png)

| | all 19 models | the 16 that actually trained |
|---|---|---|
| Topic classification | r = −0.616 | **r = −0.888** |
| Entity recognition | r = −0.935 | **r = +0.303 — i.e. none** (t = 1.19, p ≈ 0.25) |

Read the second row twice. Across all nineteen the correlation looks like the tightest number in
the study. Restrict it to models that trained, and **it disappears**.

Not *inverts* — disappears. The +0.303 is tempting to read as a reversal and it is not one: at
n = 16 that is t = 1.19, p ≈ 0.25, indistinguishable from zero. Asserting an inversion would claim
something the data cannot carry, which is exactly the failure the panel beside this one is about.
The supportable statement is that the relationship is **absent**, and that is enough.

That −0.935 is manufactured entirely by three under-trained models sitting far to the right.
Among the sixteen working ones, 0.79 nats of pretraining loss buys a score range of 0.044 with a
standard deviation of 0.013 — against a seed noise of 0.004. There is real variation between those
models; it simply is not tracking the thing we spent a term minimizing.

**So validation loss tells you a model is broken. On entity recognition it does not tell you which
working model is better.** On topic classification it does, strongly. Two tasks, opposite answers,
the same checkpoints.

### Why the two tasks differ — and the explanation we got wrong first

The first version of this panel blamed the floor: entity recognition starts closer to its ceiling,
so less is left for pretraining to supply. That is wrong, and it survived into a figure and an
email before anyone checked it. **The floors are near enough identical** — 0.403 of 0.705 on topic
classification is 57%, 0.414 of 0.798 on entity recognition is 52%. A five-point difference
explains nothing.

![Floor, the gain every trained model receives, and the band the sixteen actually
span](figures/12-floors.png)

What separates them is not the *size* of the benefit but its **variability**:

| | the gain every model gets | the band 16 trained models span |
|---|---|---|
| Topic classification | +0.159 | **0.143** wide |
| Entity recognition | +0.340 | **0.044** wide |

Entity recognition hands every working model a large and **nearly constant** benefit — the spread
between the best and worst of sixteen is 0.044, about 13% of the smallest gain. Topic
classification spreads them over 0.143, about 90% of the smallest gain.

**A benefit that everyone receives equally cannot be predicted by anything**, because there is
nothing left to predict. That is the whole reason validation loss tracks one task and not the
other, and it has nothing to do with where the floor sits.

The uncomfortable implication is for the data ladder, the step budgets and the seed replication —
all chosen to move a number that, on one of our two tasks, moves nothing downstream.

### Week 8 — does a tuned setting transfer to a new language?

This board claims adding a language is one function call. That is only true if the *settings* come
with it, and we had never checked: Yoruba and English were tuned separately and the two best rates
were never compared.

Five languages at a fixed 16M-token budget, four learning rates, two seeds — 4.6 GPU-hours.

| | 1.5e-4 | 3e-4 | 5e-4 | 7e-4 | best |
|---|---|---|---|---|---|
| Swahili | 6.297 | 3.400 | 2.943 | **2.853** | 7e-4 |
| Hausa | 6.204 | 3.518 | 3.207 | **3.139** | 7e-4 |
| Yoruba | 5.346 | 3.226 | **3.006** | 3.026 | 5e-4 |
| Igbo | 4.472 | 2.956 | **2.889** | 5.638 | 5e-4 |
| Chichewa | 6.316 | 3.433 | 2.856 | **2.734** | 7e-4 |

**The best rate is not the same.** And look at Igbo: it *diverges* at 7e-4 — 5.638 — exactly where
three other languages are still improving. Different languages have different ceilings, so "the
rate that worked last time" is not a safe default for a new one.

Then the sweep caught itself. Swahili, Hausa and Chichewa all peaked at **7e-4, the top of the
range**, which means their true optima are outside it and those three "best" numbers are not
quotable. Three of the five sweeps in this project have now done that. The script prints the
warning itself rather than leaving it to be noticed, and the range has been extended to 1e-3 —
for all five languages, not only the three that ran out of room, because extending just the arms
that hit the edge is the asymmetry report 08 spent three passes removing.

### The mistake inside this panel

The first version of this analysis called a run dead if it finished within four nats of an
untrained model. That labels every *data-starved* run a failure — at 4M tokens the best anyone
achieved is 6.72, so the threshold condemned the whole rung — and it produced a conclusion that
was **backwards**: it made tighter clipping look like the *cause* of failures, when clipping had
simply been run on the small corpora on purpose.

The fix was to stop asking "did it end high" and start asking "did it end much worse than this
configuration is capable of", using the best result any run achieved at the same corpus and model
size as the reference.

That is the sixth entry for the table in section 10, and it is the first one we caught in the same
afternoon we made it — which is the only reason it is a footnote here rather than a result.

---

## 15. Conclusions and recommendations

**About the science.** A small model trained on one under-served language can beat a large
multilingual one at tasks needing meaning, and comes close at tasks needing surface patterns. The
multilingual model's disadvantage is mostly in its vocabulary: it chops Yoruba into 1.76 times as
many pieces as a purpose-built one, and at equal computer time that costs real quality. Yoruba
itself is not hard to model.

**About the engineering.** The scaffolding transfers between problems far better than the models
do. Roughly 1,700 lines of our code know they are doing language modeling; everything underneath
would work as well for images. If you are going to run more than a handful of experiments, build
the production line first.

**About measurement — the part I would put on the poster in the largest type.**

> **Run everything twice before you believe it once.** Almost every claim this project had to
> withdraw was a single run reported as a measurement.
>
> **Write down what you held fixed.** Five times, a setting chosen for one good reason silently
> decided an answer somewhere else. None was a bug. All were invisible until someone asked.
>
> **Build the thing that tells you a number is wrong.** The detectors caught more real problems
> than any analysis did. A number that looks reasonable and is wrong is far more dangerous than
> one that crashes.
>
> **Measure against a floor.** Always know what an untrained model scores. One of our baselines
> was beating an untrained model by 0.004 and had been quoted for weeks as a real result.

**What we would do differently.** Seed everything from the start — it is cheaper than re-running
studies later. Measure the run-to-run spread for each configuration rather than borrowing one.
Check whether a sweep's best value sits at the edge of the range you swept, because three of our
five did. And write the front page of a repository for the person who has never seen it.

**What is still open.** Whether the vocabulary penalty costs anything on a real downstream task is
the one experiment we did not finish. We know it exists, we know it tracks which languages a model
was trained on, and we know it costs quality in raw language modeling. Whether it costs *accuracy
on a task someone cares about* is unmeasured, and it is the obvious next thing.
