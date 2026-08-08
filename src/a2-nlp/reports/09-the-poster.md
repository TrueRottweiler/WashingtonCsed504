# Building a model factory: what we made, what it found, and what went wrong

*A2-NLP · CSED 504 · the plain-language version*

Written for someone who has taken an introductory machine learning course and has not built a
training pipeline before. Every technical term is explained where it first appears. The thirteen
sections below are the thirteen poster panels.

---

## 1. Summary

Our group asked a question about language: **when is it better to train a small language model
from scratch on one under-served language than to reuse a big multilingual model trained on a
hundred languages at once?** Yoruba — spoken by around 45 million people, and badly served by
most language technology — was the test case.

My half of the project was not that question. My half was **building the machinery that could
answer it**, and then finding out what building it teaches you.

Six things we can now say, none of which we believed at the start:

**A tiny model beat a much larger one at the task that needs meaning.** A 33.8-million-parameter
model trained on 64 million words of Yoruba scored **0.666** at sorting Yoruba text into topics.
mmBERT — 246 million parameters, and reported by its authors as trained on roughly three trillion
words across 1,800 languages — scored **0.595**. Ours is about a seventh the size and saw
something like one part in fifty thousand of the text.

(The parameter counts we measured ourselves from the published configurations. The training-data
figure is the authors' claim, not something we can verify.)

**Yoruba is not a hard language to model. It is an under-served one.** Given the same amount of
text and the same amount of computer time, a model learns as much structure from Yoruba as from
English. Nothing about the language resists being learned; what is missing is the text and the
tooling.

**The disadvantage a multilingual model carries is in its vocabulary, not in the language.** We
measured this two independent ways and they agree.

**More Yoruba text would not have helped much.** Past roughly 64 million words, adding sixteen
times more text produced no measurable improvement. That is a surprising and slightly deflating
result: the field's usual complaint about low-resource languages — *there isn't enough data* —
was not our binding constraint.

**The bigger model was not too big. It was misconfigured.** One setting, changed from 1.0 to 0.5,
moved its score by a full unit and made it thirty-eight times more reproducible.

**And the most useful finding is not on that list.** Five separate times, a number that looked
like a scientific result turned out to be an artefact of a setting nobody had questioned. Section
9 is about those five, because they are the part a student can actually use.

---

## 2. The problem, in two halves

Think of the project as a building with two floors.

**Upstairs, Patrick and Leon** are asking the research question. Does a small, language-specific
model beat a large multilingual one for Yoruba? To answer that they need trained models to
compare, evaluation harnesses to score them, and enough repetitions to know whether a difference
is real or luck.

**Downstairs — my half — is the machinery that produces those models.** Collecting text.
Converting it into a form a GPU can train on. Running dozens of training jobs across two graphics
cards without them colliding. Recording what happened so a result can be traced back to the exact
settings that produced it. Noticing when a run has gone wrong.

The split matters because the two floors fail differently. Upstairs, a wrong answer looks like a
wrong answer — you argue about it. Downstairs, a wrong answer looks like **a perfectly reasonable
number**, and nobody argues with it at all.

Almost everything this project got wrong, it got wrong downstairs.

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
later: of the code involved, about 1,700 lines are specific to language modelling and everything
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
midnight. Ours went through several rounds of being wrong in instructive ways — it showed runs as
"training" that had actually finished, it hid everything not yet started so a twelve-job night
looked like a three-job night, and it labelled a counter "epoch" that was not an epoch, which
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

**And five times, a constant decided a result.** This is the finding I would put at the centre of
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
| 86M parameters, 1.024B tokens of training | 92.1 minutes |
| 33.8M parameters but a 250,000-word vocabulary | 40.9 minutes |

That last row is the one to notice. It is the *same model size* as the first two and takes five
times as long as it should, because a bigger vocabulary makes the final layer enormous. Section 9
explains how that nearly produced a wrong scientific conclusion; here it is simply a reminder that
"model size" and "cost to train" are not the same number.

Measured throughput: **349,000 tokens per second** for the small model, **180,000** for the larger
one.

### Buying the hardware versus renting it

**What we own (capital expenditure).** Two RTX PRO 6000 Blackwell cards with 96 GB of memory each,
in a workstation. At the time of writing these cards are roughly $8,000–9,000 each, so call the
pair **$17,000**, and perhaps **$20,000** for the whole machine. That is an estimate — the measured
figures above are not.

**What renting would have cost (operating expenditure).** Cloud providers charge by the
graphics-card-hour. A card in this class rents for roughly $2–4 per hour depending on provider and
commitment. At 83.3 GPU-hours:

| | cost |
|---|---|
| Cloud, at ~$3/GPU-hour | **~$250** |
| Owning, electricity only | **~$4** |
| Owning, hardware amortised over three years | ~$0.76/hour of *wall-clock ownership*, whether or not you use it |

**So renting would have been cheaper for this project, by a lot.** $250 against $20,000. If you
are doing one term's work, rent.

The arithmetic flips when the machine is busy. $20,000 of hardware pays for itself against
$3/hour at about **6,700 GPU-hours** — roughly five months of both cards running continuously.
A research group that keeps its cards busy is better off owning; a student doing one project is
not. We used 83 hours over about three weeks. Two cards running continuously for three weeks would be
1,008 card-hours, so that is **8% utilisation** — and 8% is generous, since most of those hours
were overnight batches rather than steady work.

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

- Ten hours of queued work cancelled after discovering the comparison it would run was confounded.
- Four separate runs that trained normally and then fell apart, an hour each, before we built a
  detector that noticed.
- An entire tokenizer comparison run at the wrong control, then re-run.
- Three learning-rate sweeps whose best value sat at the edge of the range, needing extension.

Call it fifteen to twenty hours of the eighty-three — **around 20% of all compute spent on work
that had to be redone.** For a student project that is fine. It is a useful ratio to know before
you scale it up, because at a hundred times the model size that 20% is the whole budget.

---

## 11. Enabling the upstairs half

The measure of the factory is not what it computed. It is whether two other people could use it.

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

**Where it was genuinely strong.** Writing the scaffolding: schedulers, dashboards, record-keeping,
the hundred small pieces that are tedious rather than hard. Adapting the existing image-model
tooling to language models. Producing careful written explanations of what had been measured.
Working overnight on queued experiments while I slept.

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
do. Roughly 1,700 lines of our code know they are doing language modelling; everything underneath
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
was trained on, and we know it costs quality in raw language modelling. Whether it costs *accuracy
on a task someone cares about* is unmeasured, and it is the obvious next thing.
