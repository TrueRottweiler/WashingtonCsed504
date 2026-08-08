# Beyond one language: a resource and script gradient

*A2-NLP · August 2026 · what the factory looks like when it stops being about Yoruba*

The group's study is one point: a low-resource language, compared against multilingual models
that barely cover it. That design cannot separate *"from-scratch works"* from *"from-scratch
works when the alternative is bad at your language"* — because it never sees a language the
alternative is good at.

This note adds the missing axis. Five languages spanning three orders of magnitude of web
presence and two writing systems, all prepared identically, so the question becomes a curve
rather than a verdict.

| corpus | language | script | web presence | who can read the output |
|---|---|---|---|---|
| `eng` | English | Latin | vast | Jeffrey |
| `fra` | French | Latin | large | Jeffrey, and a fluent second reader |
| `ind` | Indonesian | Latin | mid | a fluent reader |
| `cmn` | Mandarin | Han | large | fluent readers, possibly Patrick and Leon |
| `yor` | Yoruba | Latin + tone | tiny — 69M tokens is all of it | GlotLID only |

The human-validation column is not a footnote. Every claim in the earlier reports about Yoruba
rests on a loss number and a language-ID score, because nobody on this project reads Yoruba. On
four of these five, someone can look at the model's output and say whether it is nonsense.

---

## 1. Preparing corpora that do not fit in memory

The old `collect_docs` held every document as a Python string. That is fine for the 260 MB of
Yoruba that exists and impossible for the billions of tokens a corpus large enough to occupy this
hardware would need — 81 billion tokens of updates is what a 48-hour run consumes, and the old
path would have died long before reaching a corpus that size.

The streaming path never holds the corpus:

- `stream_docs` is a generator, so one document is resident at a time.
- `train_tokenizer_streaming` reads a **bounded prefix** — 200M characters by default. A 16k
  byte-level BPE cannot use more than that; the merge table stops changing long before.
- `encode_stream_to_disk` encodes in batches of 2,000 documents and appends raw ids to a file.
- `bin_to_npy` converts that to the memory-mapped `.npy` the trainer reads, copying in 8M-element
  chunks. The `.npy` header has to state the final length, which is not known until the stream
  ends — hence the two-step.

Peak memory is now the tokenizer sample plus one batch, regardless of corpus size.

**Verified equivalent, not just plausible.** Run against the same local text, the streaming path
produced 445,186 train tokens and 2.48 chars/token — identical to what the collect-then-tokenize
path produced, with the same vocabulary fingerprint.

---

## 2. Equal characters is not equal tokens

The five corpora were capped at the same character count, because characters are the only budget
that means the same thing to a human across scripts. That does **not** equalise what the model
sees:

| corpus | chars | train tokens | chars/token |
|---|---|---|---|
| Mandarin | 260M | **180.8M** | 1.43 |
| Yoruba | 260M | 69.1M | 3.73 |
| French | 260M | 65.2M | 3.96 |
| English | 260M | 60.6M | 4.26 |
| Indonesian | 260M | 55.2M | 4.67 |

Mandarin yields **three times** the training tokens from the same characters, because a Han
character carries roughly a morpheme where a Latin character carries a phoneme.

This is a methodological trap worth stating plainly: *a cross-language study that matches corpora
by size in characters has not matched them by how much the model trains on, and one that matches
by tokens has not matched them by how much text a person would say it read.* Neither is wrong;
choosing silently is. The training runs below match on **tokens**, because tokens are what the
model actually consumes.

---

## 3. The tooling had Latin-script assumptions baked in

Predicted before running it, and true: `audit_corpus` measured tokenizer fit as *tokens per
word*, and Mandarin has no whitespace word boundaries. A whole clause counts as one "word":

```
深度學習模型需要大量的訓練資料才能達到良好的效果。
  word-regex finds: 1 word
  whitespace split: 1 token
```

Left alone it reported **15.57 tokens per word** for Chinese and a type/token ratio of 0.654 —
both meaningless, both plausible-looking enough to end up in a table.

The fix does not hard-code a language list. `uses_word_boundaries` asks the text: if the average
run between spaces is far longer than any alphabetic language's average word, there are no word
boundaries to count, and the audit switches to **tokens per character**, which compares across
every script. Variety switches from type/token ratio to distinct characters.

A first attempt at this passed a synthetic test and still failed on the real corpus — FineWeb
Chinese carries enough embedded whitespace (numbers, URLs, English fragments) that the run-length
threshold needed checking against actual data rather than a hand-written sentence.

---

## 4. The tokenizer-fit penalty tracks resource level exactly

Each corpus tokenized by its own 16k BPE and by XLM-R's 250k multilingual vocabulary:

| language | unit | own BPE | XLM-R | XLM-R costs |
|---|---|---|---|---|
| English | /word | 1.400 | 1.457 | 1.04× |
| French | /word | 1.522 | 1.580 | 1.04× |
| Indonesian | /word | 1.503 | 1.497 | 1.00× |
| Mandarin | /char | 0.675 | 0.643 | **0.95×** |
| **Yoruba** | /word | 1.384 | 2.282 | **1.65×** |

**A multilingual vocabulary is not inherently worse.** On four of the five it is a wash, and on
Mandarin it is *better* than a dedicated 16k BPE — unsurprising, since 250k slots buy a lot of
room for Han characters and 16k does not.

The penalty appears only where the language is under-represented. Yoruba pays **65% more tokens
for the same text**, which is 65% of its context window spent on fragments.

That is the group's thesis with a control arm attached. Their claim — that a language-specific
vocabulary is worth building — is true for Yoruba and would be false if they had run the same
experiment on French. The interesting version of their question is not *whether* from-scratch
wins but *at what resource level it starts to*.

---

## 5. The matched training comparison

Five runs, identical in everything except the language: the 33.8M-parameter model, 50M tokens of
training data, 12,000 steps at batch 128 and sequence length 128 — 196.6M tokens of updates each.

| language | val loss | train loss | unigram entropy | context gained |
|---|---|---|---|---|
| French | 2.536 | 2.731 | 7.401 | **4.865** |
| Indonesian | 3.273 | 3.311 | 7.828 | 4.555 |
| **Yoruba** | 2.920 | 3.114 | 7.034 | **4.114** |
| English | 3.585 | 3.586 | 7.491 | 3.906 |
| Mandarin | 4.456 | 4.150 | 8.149 | 3.693 |

**Validation loss does not compare across these rows.** Each corpus has its own 16k BPE with its
own frequency distribution, so Yoruba's 2.920 and English's 3.585 are not measurements of the same
quantity — most of that gap is the vocabulary, not the language. Any cross-language table of raw
loss is measuring its own tokenizers.

The last column is the one that compares. **Context gained** is the corpus's unigram entropy minus
the model's final loss: the loss a model would score by predicting token frequencies and ignoring
context entirely, less what the model actually scored. What remains is what it learned *from
context*, with each vocabulary's own entropy canceled out.

On that measure, **Yoruba (4.114) sits mid-pack and ahead of English (3.906)**. Given the same
amount of data and the same amount of compute, a from-scratch Yoruba model extracts as much
contextual structure as a from-scratch English one. Yoruba's apparent advantage on raw loss was an
artifact; on the normalized measure the advantage is smaller but it survives, and it now means
something.

That is a stronger statement than the single-language study can make. It separates *"Yoruba is
hard"* from *"Yoruba is under-served"* — and the answer is the second one. Nothing about the
language resists modeling. What is scarce is text.

How much that scarcity actually costs is measured in [report 05](05-when-data-stops-mattering.md)
§3, and the answer is: less than it sounds. At these compute budgets the text Yoruba does not
have would have bought almost nothing.

### The units caveat, which is not small

Matching on tokens means **not** matching on text. Applying §2's chars/token to the shared 50M
token budget:

| language | tokens | characters actually seen |
|---|---|---|
| Indonesian | 50M | 233M |
| English | 50M | 213M |
| French | 50M | 198M |
| Yoruba | 50M | 187M |
| Mandarin | 50M | **72M** |

Mandarin came last while reading **a third as much text as Indonesian**. Its position in this table
is substantially a units artifact and should not be read as "Mandarin is harder to model." Matching
on characters instead would invert the comparison and introduce the mirror-image problem — the
model would see three times the tokens for Mandarin. There is no budget that is neutral across
scripts; there is only the choice, stated.

One further asymmetry worth naming: 50M tokens is 72% of every Yoruba token in FineWeb-2, and a
rounding error against available English. Both models are equally trained here, but only one of
them is anywhere near the ceiling of what its language could supply. This table says Yoruba does
well at a budget English has barely started to use — not that the two would stay level if both were
allowed to run to the end of their data. That is the experiment this note does not contain, and it
is the next one to run.

Every figure here is one seed, and the spread this originally cited (0.049) turned out to be the
wrong yardstick — see [report 05](05-when-data-stops-mattering.md) §2. Re-seeding at a larger
budget gives **0.149** for 33.8M English and **0.103** for 33.8M Yoruba, so the honest reference
for these runs is somewhere near 0.1 rather than 0.05, and it has not been measured at *this*
budget at all.

Against ~0.1 the Yoruba-to-English gap of 0.208 is roughly twice the noise: still pointing the
same way, with less room than the original figure suggested. The gaps between adjacent languages
in the middle of the table are inside it and should not be read as an ordering.

---

## What this does not settle

Corpus size here is capped at 260M characters per language for comparability, which for English
and French is a rounding error against what exists. This measures *the same amount of text in
different languages*, not *what each language can actually muster* — the latter is the group's
real question, and answering it means letting English run to billions of tokens while Yoruba
stays at 69M. The streaming path exists so that experiment is now possible; it has not been run.

Nor does anything here test the model-size axis honestly. The 86M preset lost to the 33.8M one at
every rung of the Yoruba ladder, which reads as undertraining rather than a verdict on capacity —
the crossover, if there is one, is somewhere past the budgets used in this note.
