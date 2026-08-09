"""
audit_corpus.py -- what is actually in the text you are about to train on.

Language ID tells you the corpus IS the language you asked for. It does not tell you whether the
language is written well, consistently, or with enough variety to learn from. Those are separate
questions, and each of them has bitten this project once:

    is there enough?        FineWeb-2's entire Yoruba shard is 69M tokens. A 64M-token rung
                            consumes 93% of it. Worth knowing before designing a ladder.

    is it consistently      Yoruba is tonal, and the web mostly drops the tone marks. `àwọn` and
    written?                `awọn` are the same word and two different tokens, so the model
                            splits its statistics across spellings. This check finds that class
                            of problem in any language with diacritics, without being told which
                            marks to look for.

    is it varied, or        A corpus of near-duplicate boilerplate has a high token count and
    the same page twice?    little information. Type/token ratio and repeated-line share catch it.

    does the vocabulary     Tokens per word ("fertility") says how well the tokenizer fits the
    fit the language?       language. A high number means the model spends its context window on
                            fragments -- the usual complaint about multilingual vocabularies on
                            under-represented languages.

Usage:
    python audit_corpus.py --corpus yor
    python audit_corpus.py --corpus yor --rungs 4000000 16000000 64000000
    python audit_corpus.py --corpus yor --compare-tokenizer FacebookAI/xlm-roberta-base
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata

WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def strip_marks(word: str) -> str:
    """The word with every combining mark removed -- its bare spelling.

    Decomposing and dropping the combining characters turns `àwọn` into `awon` and `ti`/`tí`
    into the same key, which is what lets two spellings of one word be recognized as such
    without hard-coding any language's diacritics.
    """
    return ''.join(c for c in unicodedata.normalize('NFD', word.lower())
                   if not unicodedata.combining(c))


def orthographic_consistency(words: list[str], top: int = 8) -> dict:
    """How often the same word appears with and without its marks."""
    by_bare = collections.defaultdict(collections.Counter)
    for w in words:
        by_bare[strip_marks(w)][w.lower()] += 1

    marked = unmarked = 0
    split_words = []
    for bare, forms in by_bare.items():
        # One arithmetic for every case: anything spelled exactly like its mark-stripped form is
        # unmarked, everything else is marked. A word with a single spelling falls out of this
        # without a branch -- either bare_count is zero or it is the whole total.
        total = sum(forms.values())
        bare_count = forms.get(bare, 0)
        marked += total - bare_count
        unmarked += bare_count

        # Report only words genuinely written both ways, and only if common enough to matter.
        if total > 50 and 0 < bare_count < total:
            split_words.append((bare, total, bare_count, forms.most_common(3)))

    split_words.sort(key=lambda r: -r[1])
    total_all = marked + unmarked
    return {'marked': marked, 'unmarked': unmarked,
            'marked_share': marked / total_all if total_all else 0.0,
            'split_words': split_words[:top]}


def repetition(docs: list[str]) -> dict:
    """Type/token ratio and how much of the corpus is repeated lines."""
    lines = [ln.strip() for d in docs for ln in d.split('\n') if len(ln.strip()) > 40]
    counts = collections.Counter(lines)
    repeated = sum(c for c in counts.values() if c > 1)
    return {'lines': len(lines),
            'unique_lines': len(counts),
            'repeated_share': repeated / len(lines) if lines else 0.0,
            'worst': counts.most_common(3)}


def uses_word_boundaries(text: str, sample: int = 200_000) -> bool:
    """Does this writing system separate words with whitespace?

    Chinese, Japanese and Thai do not, and that quietly breaks the usual "tokens per word"
    fertility metric: a whitespace split of a Chinese clause returns the whole clause, so the
    measure reports something like thirty tokens per word and means nothing. Rather than
    hard-coding a list of languages, ask the text -- if the average run between spaces is far
    longer than any alphabetic language's average word, there are no word boundaries to count.
    """
    t = text[:sample]
    runs = [len(w) for w in t.split() if w]
    if not runs:
        return False
    return (sum(runs) / len(runs)) < 12


def fertility_per_char(tokenizer, texts: list[str]) -> float:
    """Tokens per CHARACTER -- comparable across every writing system.

    This is the measure to quote when scripts differ. Tokens per word cannot be compared between
    a language that marks word boundaries and one that does not; tokens per character can.
    """
    n_tok = n_chr = 0
    for t in texts:
        for piece in t.split(chr(10)):
            piece = piece.strip()
            if not piece:
                continue
            n_tok += len(tokenizer(piece, add_special_tokens=False,
                                   truncation=False, verbose=False)['input_ids'])
            n_chr += len(piece)
    return n_tok / max(n_chr, 1)


def fertility(tokenizer, texts: list[str]) -> float:
    """Tokens per whitespace word -- how much the tokenizer fragments this language.

    Measured on short pieces rather than whole documents. Encoding a 2,000-character document
    with a tokenizer whose model_max_length is 128 works fine for counting, but transformers
    prints a length warning for every call, and the warning is noise here: nothing is being fed
    to a model, only counted.
    """
    n_tok = n_word = 0
    for t in texts:
        for piece in t.split('\n'):
            piece = piece.strip()
            if not piece:
                continue
            n_tok += len(tokenizer(piece, add_special_tokens=False,
                                   truncation=False, verbose=False)['input_ids'])
            n_word += len(piece.split())
    return n_tok / max(n_word, 1)


def main():
    p = argparse.ArgumentParser(description='Audit a prepared corpus before training on it.')
    p.add_argument('--corpus', required=True)
    p.add_argument('--docs', type=int, default=4000, help='documents to sample')
    p.add_argument('--rungs', type=int, nargs='+',
                   default=[4_000_000, 16_000_000, 64_000_000])
    p.add_argument('--compare-tokenizer', nargs='*', default=[],
                   help='other tokenizers to measure fertility against, e.g. a multilingual one')
    p.add_argument('--json', default=None)
    args = p.parse_args()

    import mlm_api as factory

    info = factory.corpus_info(args.corpus)
    docs = factory.sample_docs(args.corpus, args.docs)
    text = '\n'.join(docs)
    words = WORD.findall(text)
    # Decided once from the text itself: Chinese, Japanese and Thai do not delimit
    # words with spaces, and every word-based statistic below is meaningless there.
    spaced = uses_word_boundaries(text)

    print('=' * 74)
    print(f'  CORPUS AUDIT: {args.corpus}')
    print('=' * 74)

    # -- how much is there ------------------------------------------------------------------
    n = info['n_tokens']['train']
    print(f'\n  SIZE')
    print(f'    {n:,} train tokens | vocab {info["vocab_size"]:,}'
          + (f' | {info["chars_per_token"]:.2f} chars/token' if 'chars_per_token' in info else ''))
    print(f'\n    {"planned rung":>16}{"share of corpus":>18}')
    for r in args.rungs:
        share = r / n
        note = '  DOES NOT FIT' if share > 1 else ('  little headroom' if share > 0.8 else '')
        print(f'    {r:>16,}{share:>17.0%}{note}')

    # -- is it consistently written --------------------------------------------------------
    orth = orthographic_consistency(words)
    print(f'\n  ORTHOGRAPHY  ({len(words):,} words sampled)')
    if orth['marked_share'] < 0.001:
        print('    no diacritics in this text -- nothing to be inconsistent about')
    else:
        print(f'    {orth["marked_share"]:.1%} of word occurrences carry diacritics')
        if orth['split_words']:
            print(f'\n    words that appear both with and without their marks:')
            print(f'      {"bare form":<14}{"uses":>9}{"bare":>9}{"marked":>9}   spellings seen')
            for bare, total, bare_count, forms in orth['split_words']:
                spell = ', '.join(f'{w}' for w, _ in forms)
                print(f'      {bare:<14}{total:>9,}{bare_count:>9,}{total-bare_count:>9,}'
                      f'   {spell}')
            print('\n    Each spelling is a separate token, so the model learns them separately.'
                  '\n    That is a property of the source text, not a bug -- but it caps what a'
                  '\n    model trained here can know about the marks.')

    # -- is it varied ----------------------------------------------------------------------
    rep = repetition(docs)
    print(f'\n  VARIETY')
    if spaced:
        ttr = len(set(w.lower() for w in words)) / max(len(words), 1)
        print(f'    type/token ratio {ttr:.3f}  '
              f'({len(set(w.lower() for w in words)):,} distinct words in {len(words):,})')
    else:
        chars = [c for c in text[:2_000_000] if not c.isspace()]
        print(f'    distinct characters {len(set(chars)):,} in {len(chars):,}')
        print('    (word-level ratios do not apply to a script without word boundaries)')
    print(f'    repeated lines   {rep["repeated_share"]:.1%} of {rep["lines"]:,} '
          f'substantial lines')
    if rep['repeated_share'] > 0.15:
        print('    a large share of this corpus is boilerplate repeated across documents.')
        for line, c in rep['worst']:
            print(f'      {c:>5}x  {line[:70]!r}')

    # -- does the vocabulary fit ------------------------------------------------------------
    sample = [d[:2000] for d in docs[:200]]
    unit = 'tokens per word' if spaced else 'tokens per character'
    print(f'\n  TOKENIZER FIT  ({unit} -- lower means a better fit)')
    if not spaced:
        print('    this script has no whitespace word boundaries, so tokens-per-word would be')
        print('    meaningless here; tokens-per-character is what compares across scripts.')

    measure = fertility if spaced else fertility_per_char
    own = factory.load_tokenizer(args.corpus)
    base = measure(own, sample)
    print(f'    {"this corpus own BPE":<34}{base:>8.3f}')
    for name in args.compare_tokenizer:
        try:
            from transformers import AutoTokenizer
            other = AutoTokenizer.from_pretrained(name)
            v = measure(other, sample)
            print(f'    {name:<34}{v:>8.3f}   {v/base:.2f}x')
        except Exception as e:
            print(f'    {name:<34}  failed: {repr(e)[:40]}')

    print()
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump({'corpus': args.corpus, 'size': info,
                       'orthography': {k: v for k, v in orth.items() if k != 'split_words'},
                       'variety': {'type_token_ratio': ttr,
                                   'repeated_share': rep['repeated_share']}},
                      f, indent=2)
        print(f'  wrote {args.json}\n')


if __name__ == '__main__':
    main()
