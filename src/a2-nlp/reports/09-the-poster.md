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
twice. **The thirteen sections below are the thirteen panels of the bottom poster.** Where a
finding belongs upstairs it is stated here only as far as needed to explain what the factory was
for; the top poster carries it properly.

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

## 5. Setting up a factory: workloads, interfaces, and adjustments

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

## 6. Notebooks, for going fast

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

## 7. Offline processing: queues, dashboards, and catching problems early

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

## 8. How the factory got used, and how it changed

It was not built and then used. It was built *by* being used, and every capability traces to a
question somebody could not otherwise answer.

**"Their corpus is tiny — how tiny?"** Prepare it and count. All the Yoruba in our source is 69
million words. It trains in seconds. That single measurement reframed the project: the group had
been planning around data scarcity, and the scarcity was more extreme than anyone had assumed.

**"Is this result real or luck?"** Train the same thing several times with different random
starting points. If the difference between two setups is smaller than the difference between two
runs of the *same* setup, you have not measured anything. This is the single most valuable habit
the project developed, and section 9 explains what it cost us to learn it.

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

## 9. What we computed, and what made it hard

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

**And the hyperparameters we did search, with the values.** This is a genuine grid search, and it
is a *small* part of the total:

| knob | values tried | what it is |
|---|---|---|
| learning rate (pretraining) | 1e-4, 1.5e-4, 3e-4, 5e-4, 1e-3 | how big a correction the model makes each step |
| learning rate (fine-tuning) | 5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4 | the same knob, on the downstream task |
| batch size | 64, 128, 256, 512 | how many sequences per step |
| gradient clipping | 1.0, 0.5 | a ceiling on how large one correction may be |
| warmup fraction | none, 0.06 | how long the learning rate ramps before annealing |
| sequence length | 128 | fixed — every run, so nothing depends on it |
| model size | 33.8M, 98M, 154M total parameters | we called the middle one "the 86M model" all term — that is its count *excluding* the embedding table |
| training data | 2M → 1,024M tokens | |
| random seed | 0, 1, 2 | |

**Why multiple seeds, when the settings are identical?** Because a neural network starts from
random numbers and shuffles its data randomly, so the same recipe run twice gives two different
models with two different scores. Until you know how far apart *those* are, you cannot interpret
the gap between any two experiments.

We measured that spread on 20 cells that were run more than once. **It is not a constant.** The
median is 0.112, the smallest 0.003, the largest **2.156** — a range of nearly a thousandfold
across cells of the same study. For most of the term we used a single number, 0.049, measured once
on one model on one language, and applied it everywhere. That was the first of the five constants
in the table below, and it is why every early claim was judged too generously.

**The rule we ended up with, which is the one worth putting on a poster:** run the same cell at
three seeds *before* comparing it to anything, and treat a difference smaller than that cell's own
spread as no difference at all.

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
it five times the resources. Read one way the tokenizer made no difference; read correctly it made
a large one. **Same three experiments, opposite conclusions.**

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

## 10. What it cost to build and run

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
times as long as it should, because a bigger vocabulary makes the final layer enormous. Section 9
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
  the corrections in section 9 came from cheap, impulsive re-checks. I do not think I would have
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

## 11. Enabling the upstairs half

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

## 12. How we used AI, honestly

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

## 13. Conclusions and recommendations

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
