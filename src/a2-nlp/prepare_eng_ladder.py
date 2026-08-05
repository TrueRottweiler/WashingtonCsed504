"""Prepare the large English corpus the data ladder needs.

The Yoruba ladder found that data barely moved validation loss while compute moved it forty-fold.
That is a real result about that ladder, but it cannot be the general answer, because every rung
got roughly the same tokens of updates -- the data axis was never actually tested. It also cannot
be tested in Yoruba: all of FineWeb-2 Yoruba is 69.1M tokens, and the 64M rung already consumes
93% of the language.

English has the headroom. This builds one corpus large enough that the ladder's rungs are prefixes
of it, exactly as the Yoruba rungs are prefixes of `yor` -- so a single preparation serves every
rung and no rung sees text a smaller rung did not.

It reuses the committed `tokenizers/eng-bpe16k` rather than training a fresh vocabulary. Two BPEs
trained on different samples of the same source are different vocabularies, and losses across two
vocabularies are not comparable. Sharing it means these runs can be read against the existing
`multi_eng` result directly.

Measured before committing to it: FineWeb-Edu streams at 49.5M chars/sec here and the BPE encodes
at 9.1M chars/sec, so encoding dominates and a billion tokens costs about eight minutes.

    bash src/a2-nlp/py.sh prepare_eng_ladder.py
"""
import mlm_data as D

# Room above the top rung. The ladder's largest step is 1024M tokens and the validation split is
# carved out of the same stream, so aiming exactly at 1024M would leave the rung short.
TARGET_TOKENS = 1_100_000_000
CHARS_PER_TOKEN = 4.25          # measured on this source with this vocabulary

stats = D.prepare_corpus(
    name='eng_1b',
    lang='sample-10BT',
    source='fineweb_edu',
    tokenizer='tokenizers/eng-bpe16k',
    max_chars=int(TARGET_TOKENS * CHARS_PER_TOKEN),
    # A ceiling, not a target. The default 300s would truncate this to a fraction of the corpus
    # and the ladder would silently run on less data than its rungs claim.
    max_seconds=7200,
    val_tokens=2_000_000,       # larger than the 500k default; the top rungs deserve a val set
    #                             big enough that its noise is not the thing being measured
)

print()
for k in ('lang', 'chars', 'chars_per_token', 'tokenizer_fingerprint', 'tokenizer_source'):
    if k in stats:
        print(f'  {k:22s} {stats[k]}')
print(f'  {"train tokens":22s} {stats["n_tokens"]["train"]:,}')
print(f'  {"val tokens":22s} {stats["n_tokens"]["val"]:,}')

top = 1_024_000_000
have = stats['n_tokens']['train']
print()
print(f'top rung needs {top:,} train tokens; corpus has {have:,} '
      f'({"enough" if have >= top else "SHORT -- raise TARGET_TOKENS"})')
