# Shared tokenizers

Small enough to commit, and the one thing that has to be identical across machines.

| tokenizer | vocab | fingerprint | trained on |
|---|---|---|---|
| `eng-bpe16k/` | 16,000 | `7820319faa75` | 260M chars of FineWeb-Edu (`sample-10BT`) |
| `fra-bpe16k/` | 16,000 | `8d19fea36313` | 260M chars of FineWeb-2 French (`fra_Latn`) |
| `ind-bpe16k/` | 16,000 | `d30303ef89aa` | 260M chars of FineWeb-2 Indonesian (`ind_Latn`) |
| `cmn-bpe16k/` | 16,000 | `a40a6e85972c` | 260M chars of FineWeb-2 Mandarin (`cmn_Hani`) |
| `yor-bpe16k/` | 16,000 | `15abd33de5af` | 259.9M chars of FineWeb-2 Yoruba (`yor_Latn`) — all of it |

All five were trained on the same number of **characters**, which is deliberate: it is the only
budget that means the same thing across scripts. It does not equalise tokens — see the note at
the bottom.

## Why this exists

Everyone downloads their own text. Two people streaming FineWeb-2 do **not** receive identical
documents — the order varies, caps land in different places, and the shard changes over time. If
each of them then trains a BPE on what they happened to collect, they end up with different
vocabularies, and **their losses stop meaning the same thing**: a cross-entropy of 2.9 over one
set of 16,000 tokens is not the same measurement as 2.9 over a different set.

Sharing the vocabulary fixes that without sharing gigabytes of token arrays. Different text, same
prediction task, comparable numbers.

## Using one

```bash
python mlm_data.py --name fra --lang fra_Latn --tokenizer tokenizers/fra-bpe16k
```

```python
factory.prepare_corpus("fra", lang="fra_Latn", tokenizer="tokenizers/fra-bpe16k")
```

`--tokenizer` also accepts a bare `tokenizer.json` or a Hugging Face hub id.

## Checking you have the right one

Every prepared corpus records a fingerprint in `stats.json`, and every training run records the
vocabulary it scored against:

```python
factory.corpus_info("fra")["tokenizer_fingerprint"]     # '8d19fea36313'
```

The fingerprint hashes the token-to-id mapping rather than the file, so it is equal exactly when
the prediction task is. **If two sets of numbers disagree and the fingerprints differ, that is
the reason.**

## Adding one

```bash
python mlm_data.py --name swh --lang swh_Latn --vocab-size 16000     # trains a fresh BPE
cp -r data/swh/tokenizer tokenizers/swh-bpe16k
python -c "import mlm_api as f; print(f.tokenizer_fingerprint(f.load_tokenizer('swh')))"
```

Then record the fingerprint in the table above, and everyone else prepares `swh` with
`--tokenizer tokenizers/swh-bpe16k`.

## Equal characters is not equal tokens

Worth knowing before designing a cross-language comparison. The same 260M characters yield very
different token counts, because scripts differ in how much meaning a character carries:

| corpus | chars | train tokens | chars/token |
|---|---|---|---|
| Mandarin | 260M | **180.8M** | 1.43 |
| Yoruba | 260M | 69.1M | 3.73 |
| French | 260M | 65.2M | 3.96 |
| English | 260M | 60.6M | 4.26 |
| Indonesian | 260M | 55.2M | 4.67 |

Mandarin gets **three times** the training tokens from the same characters. A study that matches
corpora by character count has not matched them by how much the model sees, and one that matches
by tokens has not matched them by how much text a human would call it. Pick deliberately and say
which — the runs in `runs/multi_*` match on tokens (50M each), because tokens are what the model
is actually trained on.
