"""The same Yoruba text, tokenized with XLM-R's vocabulary instead of our own.

This is the other arm of the tokenizer swap. Everything about it is held identical to `yor` --
the same source, the same character budget, the same validation size -- so the only difference
between the two corpora is which vocabulary turned the characters into tokens.

It also exercises the widest path in the store: 250,002 tokens does not fit in uint16, so this
corpus is written as uint32. That capability was built at the start of the project and has never
been used by a real experiment until now.
"""
import mlm_data as D

stats = D.prepare_corpus(
    name='yor_xlmr',
    lang='yor_Latn',
    source='fineweb2',
    tokenizer='FacebookAI/xlm-roberta-base',
    max_chars=260_000_000,        # identical to yor
    max_seconds=1800,
    val_tokens=500_000,
)
for k in ('lang', 'vocab_size', 'chars', 'chars_per_token', 'store_bytes',
          'tokenizer_fingerprint', 'tokenizer_source'):
    if k in stats:
        print(f'  {k:22s} {stats[k]}')
print(f'  {"train tokens":22s} {stats["n_tokens"]["train"]:,}')

own = D.T.load_stats('yor')
print(f'\n  yor       {own["chars_per_token"]:.3f} chars/token, vocab {own["vocab_size"]:,}')
print(f'  yor_xlmr  {stats["chars_per_token"]:.3f} chars/token, vocab {stats["vocab_size"]:,}')
print(f'  XLM-R needs {own["chars_per_token"]/stats["chars_per_token"]:.2f}x the tokens '
      f'for the same text')
