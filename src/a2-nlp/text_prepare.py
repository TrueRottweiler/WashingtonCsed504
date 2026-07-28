"""
text_prepare.py -- one-time download + tokenization of each corpus into flat token arrays.

This is the a2 counterpart of imagenet_prepare.py, and it exists for the same reason: tokenize
exactly once, up front, so training never touches raw text, the tokenizer, or the datasets
library again -- it just indexes into an array. The arrays are tiny by Part 1 standards
(WikiText-103 tokenized is ~230 MB against ImageNet-32's 3.9 GB), so the GPU-resident trick
carries over with room to spare.

The dataset ladder, smallest to largest:

    shakespeare   ~1.1M chars    char-level     the smoke rung; continuity with CSED 503 A2/A4
    wikitext2     ~2.5M tokens   16k BPE        the small benchmark rung
    wikitext103   ~124M tokens   16k BPE        the headline rung (49x wikitext2's ladder step)

Tokenizer decisions, because a tokenizer bug silently invalidates every perplexity comparison:

  - shakespeare is CHARACTER-level: sorted unique chars, exactly like CSED 503. No merges, no
    vocab file to get wrong, and it makes this rung a tokenizer-free sanity anchor.
  - Both wikitext rungs share ONE byte-level BPE (16,384 merges) trained on the wikitext103
    train split. Sharing it is deliberate: wikitext2 -> wikitext103 is the scaling axis of the
    study, and a per-rung vocabulary would quietly change the prediction task between rungs.
  - Perplexities are only comparable within a tokenizer. Published WikiText numbers are
    word-level and ours are BPE-level; ours anchor against each other, not against the papers.

Usage:
    python text_prepare.py --dataset shakespeare
    python text_prepare.py --dataset wikitext2
    python text_prepare.py --dataset wikitext103     # downloads ~310 MB, tokenizes for a few min
    python text_prepare.py --dataset all
    python text_prepare.py --dataset wikitext2 --force

Output (into data/<dataset>/, gitignored):
    train_tokens.npy   uint16  the flat token stream for the train split
    val_tokens.npy     uint16  validation split
    test_tokens.npy    uint16  test split (held out; nothing below ever trains on it)
    stats.json         vocab size, token counts, tokenizer id -- text_data.py reads this
And for the BPE rungs, the shared tokenizer at data/bpe16k/tokenizer.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
BPE_DIR = os.path.join(DATA_DIR, 'bpe16k')
BPE_PATH = os.path.join(BPE_DIR, 'tokenizer.json')
BPE_VOCAB = 16_384

# The classic tiny-shakespeare file (Karpathy's char-rnn). We pull the raw file rather than the
# HuggingFace mirror because the HF copy is a script-loader dataset, which datasets >= 4 refuses
# to run; the file itself is 1.1 MB of plain text and this URL has been stable for a decade.
SHAKESPEARE_URL = ('https://raw.githubusercontent.com/karpathy/char-rnn/'
                   'master/data/tinyshakespeare/input.txt')

WIKITEXT = {
    'wikitext2': ('Salesforce/wikitext', 'wikitext-2-raw-v1'),
    'wikitext103': ('Salesforce/wikitext', 'wikitext-103-raw-v1'),
}


def out_dir(dataset: str) -> str:
    return os.path.join(DATA_DIR, dataset)


def already_done(dataset: str) -> bool:
    d = out_dir(dataset)
    return all(os.path.exists(os.path.join(d, f))
               for f in ('train_tokens.npy', 'val_tokens.npy', 'test_tokens.npy', 'stats.json'))


def save_split(dataset: str, split: str, ids: np.ndarray) -> None:
    # uint16 holds any id below 65,536; our largest vocab is 16,384. The assert is cheap and the
    # bug it guards against (an id silently wrapping around) would poison every downstream number.
    assert ids.max() < 65_536, f'{split}: token id {ids.max()} does not fit uint16'
    np.save(os.path.join(out_dir(dataset), f'{split}_tokens.npy'), ids.astype(np.uint16))


def save_stats(dataset: str, tokenizer: str, vocab_size: int, counts: dict,
               extra: dict | None = None) -> None:
    stats = {'dataset': dataset, 'tokenizer': tokenizer, 'vocab_size': vocab_size,
             'n_tokens': counts}
    if extra:
        stats.update(extra)
    with open(os.path.join(out_dir(dataset), 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)


def sanity_report(dataset: str, decode) -> None:
    """Decode the head of each split and print it, the a2 version of a1's post-convert checks.

    ImageNet-32 taught us the shape of this trap: the file was sorted by class and a 'quick
    subset off the front' would have been garbage, and only a post-convert check caught it. Text
    has no class order to worry about, but a tokenizer round-trip bug (wrong vocab, shifted ids,
    doubled whitespace) is just as silent -- unless a human reads a decoded sample, which is what
    this prints. If the text below is not readable prose, do not train on it.
    """
    for split in ('train', 'val', 'test'):
        ids = np.load(os.path.join(out_dir(dataset), f'{split}_tokens.npy'), mmap_mode='r')
        sample = decode(np.asarray(ids[:120]).tolist())
        print(f'  {split:5s} {len(ids):>12,} tokens | starts: {sample[:90]!r}')


# -- shakespeare: char-level ------------------------------------------------------------------

def prepare_shakespeare() -> None:
    os.makedirs(out_dir('shakespeare'), exist_ok=True)
    raw_path = os.path.join(out_dir('shakespeare'), 'input.txt')
    if not os.path.exists(raw_path):
        print(f'  downloading {SHAKESPEARE_URL}')
        urllib.request.urlretrieve(SHAKESPEARE_URL, raw_path)
    text = open(raw_path, encoding='utf-8').read()

    # Sorted unique characters, exactly the CSED 503 tokenizer. The vocab is stored in stats.json
    # so anything downstream (generation, the notebooks) can decode without re-reading input.txt.
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    # A 90/5/5 split by position. Splitting a single continuous work by position means val and
    # test are later acts the model never saw -- the same held-out-text discipline as wikitext,
    # just at 1/100th the scale.
    n = len(ids)
    a, b = int(n * 0.90), int(n * 0.95)
    splits = {'train': ids[:a], 'val': ids[a:b], 'test': ids[b:]}
    for split, arr in splits.items():
        save_split('shakespeare', split, arr)
    save_stats('shakespeare', 'char', len(vocab),
               {s: int(len(v)) for s, v in splits.items()}, extra={'vocab': vocab})

    itos = {i: c for c, i in stoi.items()}
    sanity_report('shakespeare', lambda seq: ''.join(itos[i] for i in seq))


# -- wikitext: shared 16k byte-level BPE ------------------------------------------------------

def ensure_bpe_tokenizer():
    """Train the shared BPE on wikitext103's train split, or load it if it already exists.

    One tokenizer for both wikitext rungs (see the module docstring). Training it needs the
    wikitext103 download either way, which is why preparing wikitext2 alone still pulls the
    larger dataset once -- the cost is a one-time download, and the alternative is two rungs
    whose perplexities cannot be compared.
    """
    from tokenizers import Tokenizer

    if os.path.exists(BPE_PATH):
        return Tokenizer.from_file(BPE_PATH)

    from datasets import load_dataset
    from tokenizers import decoders, models, pre_tokenizers, trainers

    print(f'  training the shared {BPE_VOCAB}-token BPE on wikitext103 train (one-time, ~min)')
    ds = load_dataset(*WIKITEXT['wikitext103'], split='train')

    tok = Tokenizer(models.BPE())
    # Byte-level pre-tokenization, the GPT-2 scheme: every string is representable, so there is
    # no <unk> and no lossy normalization to un-decode later.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=BPE_VOCAB, show_progress=True,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def line_iter(batch=10_000):
        for i in range(0, len(ds), batch):
            yield ds[i:i + batch]['text']

    tok.train_from_iterator(line_iter(), trainer=trainer, length=len(ds))
    os.makedirs(BPE_DIR, exist_ok=True)
    tok.save(BPE_PATH)
    print(f'  saved {BPE_PATH}')
    return tok


def encode_split(tok, ds_split, chunk_lines: int = 20_000) -> np.ndarray:
    """Encode one HF split into a flat id array, in chunks so the Rust encoder can parallelize.

    The rows of wikitext are lines with the trailing newline stripped, so we re-join each chunk
    with '\\n' and append one, which reconstructs the original text exactly across chunk
    boundaries. Encoding chunk-by-chunk (rather than one giant string) keeps memory flat and
    lets encode_batch fan out across cores.
    """
    parts = []
    n = len(ds_split)
    t0 = time.time()
    for i in range(0, n, chunk_lines):
        chunk = '\n'.join(ds_split[i:i + chunk_lines]['text']) + '\n'
        parts.append(np.array(tok.encode(chunk).ids, dtype=np.uint16))
        if i // chunk_lines % 10 == 0:
            done = min(i + chunk_lines, n)
            print(f'\r  encoding {done:>10,}/{n:,} lines '
                  f'({done / max(1e-9, time.time() - t0):,.0f} lines/s)', end='', flush=True)
    print()
    return np.concatenate(parts) if parts else np.array([], dtype=np.uint16)


def prepare_wikitext(dataset: str) -> None:
    from datasets import load_dataset

    os.makedirs(out_dir(dataset), exist_ok=True)
    tok = ensure_bpe_tokenizer()
    ds = load_dataset(*WIKITEXT[dataset])

    counts = {}
    for split, hf_split in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        print(f'  {dataset} {split}:')
        ids = encode_split(tok, ds[hf_split])
        save_split(dataset, split, ids)
        counts[split] = int(len(ids))

    save_stats(dataset, 'bpe16k', tok.get_vocab_size(), counts,
               extra={'tokenizer_path': os.path.relpath(BPE_PATH, HERE)})
    sanity_report(dataset, tok.decode)


# -- driver ------------------------------------------------------------------------------------

PREPARERS = {
    'shakespeare': prepare_shakespeare,
    'wikitext2': lambda: prepare_wikitext('wikitext2'),
    'wikitext103': lambda: prepare_wikitext('wikitext103'),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=[*PREPARERS, 'all'], required=True)
    p.add_argument('--force', action='store_true', help='re-do it even if outputs exist')
    args = p.parse_args()

    names = list(PREPARERS) if args.dataset == 'all' else [args.dataset]
    for name in names:
        if already_done(name) and not args.force:
            print(f'{name}: already prepared (use --force to redo)')
            continue
        print(f'{name}:')
        t0 = time.time()
        PREPARERS[name]()
        print(f'{name}: done in {time.time() - t0:.0f}s\n')


if __name__ == '__main__':
    main()
