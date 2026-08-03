# What is the model actually learning?

*A2-NLP · August 2026 · everything below is output from the trained checkpoint, not description*

We trained a 33.8M-parameter model on 32 million tokens of Yoruba and it reached a validation
loss of 2.886. That number means nothing on its own. This note turns it into something you can
see.

---

## 1. The task it was trained on

The model plays one game, hundreds of thousands of times: **15% of the words in a passage are
hidden, and it has to guess them from the words around them.** That's it. No labels, no human
annotation, no translation pairs — just Yoruba text with holes punched in it.

The bet behind that game is that you cannot fill the holes well without incidentally learning
the language. To guess the missing word in *"Àwọn ọmọ ___ lọ sí ilé ìwé"* you need to know
Yoruba word order, which words go together, and what kind of word belongs in that slot. Nobody
teaches the model grammar; grammar is what falls out of getting good at the guessing game.

"Training on all of the text" means: sample a random 128-token window from the corpus, hide 15%
of it, ask the model to fill the blanks, nudge its weights toward the right answer, repeat
12,000 times. Each step sees 128 windows at once, so the best model saw about **197 million
tokens of practice** drawn from a 32-million-token pool — roughly six passes over the same text.

---

## 2. What it predicts, next to what it predicted before training

Held-out Yoruba the model has never seen. One word hidden. Top five guesses from the trained
model, and from the identical architecture with random weights.

### Example A

> `... ti o nfunni to` **`___`** `akoko 3,5 lori idiyele kan ...`
> *hidden word:* `iye`

| | top five guesses | where the true answer ranked |
|---|---|---|
| **trained** | `awọn` 71%, `ni` 7%, `to` 3%, `fun` 3%, `akoko` 2% | **#7** of 16,000 |
| untrained | `Aisha` 0%, `raw` 0%, `KUN` 0%, `Cross` 0%, `stery` 0% | #3,496 of 16,000 |

### Example B

> `... ati pe o fihan` **`___`** `olupese yii dagbasoke lati pese ...`
> *hidden word:* `pe`

| | top five guesses | where the true answer ranked |
|---|---|---|
| **trained** | `pe` **96%**, `boya` 1%, `bi` 1%, `ibiti` 0%, `eyiti` 0% | **#1** of 16,000 |
| untrained | `Aisha` 0%, `raw` 0%, `KUN` 0%, `Cross` 0%, `bel` 0% | #12,684 of 16,000 |

### Example C

> `... La Asopọmọra` **`___`** `ojiji lori ohun elo ti o fanimọra ...`
> *hidden word:* `ṣe`

| | top five guesses | where the true answer ranked |
|---|---|---|
| **trained** | `ati` 17%, `awọn` 6%, `tabi` 6%, `pẹlu` 5%, `ti` 5% | #120 of 16,000 |
| untrained | `PVDF` 0%, `atako` 0%, `didanu` 0%, `bel` 0%, `Mà` 0% | #12,996 of 16,000 |

**What to notice.** The untrained model produces noise — `Aisha`, `PVDF`, `KUN` — words with no
relationship to the sentence, and it buries the correct answer around position 12,000 out of
16,000, which is what "no knowledge" looks like.

The trained model's guesses are all real Yoruba grammatical words: *awọn* (plural marker), *ni*
(is/at), *ti* (of/that), *ati* (and), *pẹlu* (with), *pe* (that), *jẹ* (is). It is not always
right — in Example C it misses — but **it is always wrong in a Yoruba-shaped way.** It has
learned what *kind* of word belongs in a slot even when it can't recover the exact one. In
Example B, where the context genuinely determines the answer, it gets it at 96% confidence.

---

## 3. What the loss number means

Cross-entropy loss converts into something concrete: **the effective number of words the model
is still choosing between.** A model with no knowledge is choosing between all 16,000. Take
`exp(loss)`:

| model | loss | effectively choosing between |
|---|---|---|
| untrained | 9.680 | **16,000 words** |
| 2M text, short training | 5.701 | 299 words |
| 32M text, short training | 5.626 | 278 words |
| 2M text, long training | 3.494 | 33 words |
| **32M text, long training** | **2.886** | **18 words** |

That is the whole result in one line. Training narrowed the field from **16,000 candidates to
about 18**. The model has not learned Yoruba the way a speaker knows it, but at every position in
a sentence it has ruled out 99.9% of the vocabulary.

It also shows why the grid looks the way it does: short training barely moves the number
(16,000 → ~290 regardless of how much text you give it), while long training moves it an order
of magnitude further. Compute is the lever; text only matters once you have enough compute to
use it.

---

## 4. What it does *not* learn, and why

Two limits are visible in the corpus itself.

### Tone marks are dropped; letter marks are kept

Yoruba uses two different kinds of diacritic, and the corpus treats them very differently.
Counting every combining mark across 1.97 million words (`audit_corpus.py`):

| mark | what it does | occurrences |
|---|---|---|
| dot below — `ọ ẹ ṣ` | distinguishes **letters** (`ọ` is a different letter from `o`) | **701,331** |
| acute — `á` | marks **high tone** | 140,712 |
| grave — `à` | marks **low tone** | 112,236 |

So the letter-distinguishing marks are largely written, and the **tone marks are largely not** —
they appear about 2.8× less often. Yoruba is tonal, and tone carries meaning (`ọkọ̀` vehicle vs
`ọkọ` husband), so this is a real loss of information in the source text.

The effect on the model is that one word arrives as several tokens. Words seen in more than one
spelling, from the same sample:

| bare form | total uses | written bare | written with marks | spellings seen |
|---|---|---|---|---|
| `ti` | 105,743 | 98,618 | 7,125 | ti, tí, tì |
| `awon` | 94,445 | 3,698 | 90,747 | awọn, àwọn, awon |
| `ni` | 75,583 | 70,569 | 5,014 | ni, ní, nì |
| `se` | 29,940 | 2,804 | 27,136 | ṣe, se, ṣẹ |

Read `ti` and `awon` together and the pattern is clear: `awọn` keeps its dot (90,747 of 94,445
uses) while `ti` loses its tone (98,618 of 105,743 uses are unmarked). Each spelling is a
separate token, so the model learns them separately. Nothing in the pipeline is broken — the
source text is written that way — but it caps what any model trained here can know about tone.

*(An earlier draft of this note reported "6% of words carry their marks", from a hand-picked list
of tone-marked pairs. That figure is right for tone marks specifically and wrong as a statement
about diacritics in general: measured across all marks, 35.4% of word occurrences carry one. The
table above is the version to quote.)*

### The language-specific vocabulary is measurably a better fit

Tokens per word on the same text — lower means the tokenizer fragments the language less, and
fragmentation wastes the model's context window:

| tokenizer | tokens per word |
|---|---|
| **this corpus' own 16k BPE** | **1.38** |
| mmBERT (256k, multilingual) | 2.12 |
| XLM-R (250k, multilingual) | 2.28 |

XLM-R needs **65% more tokens** to write the same Yoruba. That is a direct, quantified argument
for the group's thesis: a vocabulary built for one language buys real efficiency over one shared
across a hundred, before any question of model quality.

### The corpus is a mixture, and part of it is machine-translated

Random document openings from the sample:

```
Sudan Protest: Àwọn òṣìṣẹ́ aláàbò kọlu àwọn tó n ṣe ìwọ́de ní Sudan ...
Florence Babaṣọla, Oṣogbo Gomina ipinlẹ Ọṣun, Alhaji Adegboyega Oyetọla ...
Kenya ti ṣeto lati gbalejo Apejọ Isuna Digital Digital Digital lododun keji ...
Ọja didara ni awọn aye ti awọn kekeke. Focusway attaches nla pataki lori didara ...
```

The first two are genuine Yoruba journalism, fully tone-marked. The third and fourth show
machine-translation artifacts — `Digital Digital Digital`, and English left untranslated
mid-sentence (`Focusway attaches nla pataki`). GlotLID confirms 97.3% of paragraphs are Yoruba,
and that is true; it does not certify that the Yoruba is *well-formed*.

This is a plausible partial explanation for the gap we measured between the two downstream tasks
(topic classification 0.527, close to mmBERT; entity recognition 0.698, well short of it).
Topic classification can be solved from word-frequency cues that survive sloppy translation.
Entity recognition needs precise word identity, which inconsistent spelling and MT noise erode.

---

## 5. So what did we actually build?

A model that, shown Yoruba text with a word removed, proposes Yoruba function words that fit the
grammatical slot, and narrows 16,000 possibilities to roughly 18. Ten minutes of training on one
GPU, from 32 million tokens of web text.

That representation is then reused: the fine-tuning stage keeps the model's learned innards and
bolts a small classifier on top, which is why 701 labelled examples are enough to reach 0.527 on
topic classification. The pretraining did the expensive part.

**The number that proves it was worth doing** is the untrained control. Same architecture, same
fine-tuning, random weights: 0.100 on topic classification and 0.346 on entity recognition. Every
point above those lines is what the guessing game bought.

---

## How to reproduce anything here

```bash
# fill-in-the-blank predictions, best checkpoint vs the untrained control
python explain_model.py --corpus yor

# try your own sentence
python explain_model.py --corpus yor --text "Àwọn ọmọ ilé ìwé lọ sí ọjà ní àárọ̀"

# a specific checkpoint instead of the best one
python explain_model.py --corpus yor --model runs/yor_2M_3k_s0
```

`explain_model.py` picks the lowest-loss completed run by default and prints the
effective-choices table for every run on disk. Checkpoints live in `runs/yor_32M_12k_s0/` (best)
and `runs/yor_random_init/` (control).

The corpus statistics in §4 come from `factory.sample_docs('yor', 4000)` — the raw documents kept
aside at prepare time, which are the same documents the tokenizer was trained on.
