# Shared tokenizers

Small enough to commit, and the one thing that has to be identical across machines.

| tokenizer | vocab | fingerprint | trained on |
|---|---|---|---|
| `yor-bpe16k/` | 16,000 | `15abd33de5af` | 259.9M chars of FineWeb-2 Yoruba (`yor_Latn`), byte-level BPE |

## Why this exists

Everyone downloads their own text. Two people streaming FineWeb-2 do **not** receive identical
documents — the order varies, caps land in different places, and the shard changes over time. If
each of them then trains a BPE on what they happened to collect, they end up with different
vocabularies, and **their losses stop meaning the same thing**: a cross-entropy of 2.9 over one
set of 16,000 tokens is not the same measurement as 2.9 over a different set.

Sharing the vocabulary fixes that without sharing 130 MB of token arrays. Different text, same
prediction task, comparable numbers. It is the same reasoning `text_prepare.py` uses to make one
BPE serve both WikiText rungs.

## Using one

```bash
python mlm_data.py --name yor --lang yor_Latn --wiki yo \
    --tokenizer tokenizers/yor-bpe16k
```

```python
factory.prepare_corpus("yor", lang="yor_Latn", wiki="yo",
                       tokenizer="tokenizers/yor-bpe16k")
```

`--tokenizer` also accepts a bare `tokenizer.json` or a Hugging Face hub id.

## Checking you have the right one

Every prepared corpus records a fingerprint in `stats.json`, and every training run records the
vocabulary it scored against. Compare them:

```python
import mlm_api as factory
factory.corpus_info("yor")["tokenizer_fingerprint"]     # '15abd33de5af'
```

The fingerprint hashes the token-to-id mapping, not the file, so it is equal exactly when the
prediction task is the same — incidental metadata differences do not move it. **If two sets of
numbers disagree and the fingerprints differ, that is the reason.**

## Adding one

Prepare a corpus without `--tokenizer` (which trains a fresh BPE), then copy it here and record
its fingerprint in the table above:

```bash
python mlm_data.py --name ibo --lang ibo_Latn --wiki ig --vocab-size 16000
cp -r data/ibo/tokenizer tokenizers/ibo-bpe16k
python -c "import mlm_api as f; print(f.tokenizer_fingerprint(f.load_tokenizer('ibo')))"
```

Then everyone else prepares `ibo` with `--tokenizer tokenizers/ibo-bpe16k` and gets the same
prediction task on whatever text they collected.
