"""
mlm_data.py -- corpora and masked batches for the from-scratch-vs-transfer study.

This is the masked-language-modeling counterpart of text_prepare.py + text_data.py, and it
exists because the group's POC needs three things the causal side already had, in a shape their
notebook can call:

    collect + tokenize once      their notebook re-streams and re-tokenizes every session, which
                                 is minutes of wall-clock before any training starts and a fresh
                                 tokenizer (so a fresh vocabulary) each time
    a GPU-resident token stream  they already gather random windows exactly the way GpuTokens
                                 does, but hold the stream as int64 -- four times the memory the
                                 factory's int16 store needs for the same 16k vocabulary
    masking as a tested function the 80/10/10 BERT recipe, on-device, seeded

The corpus format is deliberately the SAME one text_prepare.py writes -- flat token arrays plus
stats.json under data/<name>/ -- so a Yoruba corpus is a first-class citizen of the factory and
GpuTokens loads it unchanged. The width rules are imported from the causal side rather than
restated here, so there is one definition of "how wide does this vocabulary need to be".

Prepared layout (data/<name>/, gitignored):
    train_tokens.npy   uint16 or uint32   the pretraining pool
    val_tokens.npy     "                  held out from the tail of the stream
    stats.json         vocab size, token counts, chars/token, provenance
    tokenizer/         the HF tokenizer, so fine-tuning loads the same vocabulary
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch

import text_data as T
import text_prepare as P

# Their POC's specials, in their order. <mask> has to exist for the masking below, and the four
# ids before it are skipped when drawing random replacements so masking never injects a <pad>.
SPECIALS = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']

# Where a language's raw text comes from, in preference order. FineWeb-2 is the group's primary
# source; Wikipedia is the fallback for languages whose FineWeb-2 shard is thin or missing.
SOURCES = {
    'fineweb2': lambda lang: ('HuggingFaceFW/fineweb-2', dict(name=lang, split='train',
                                                              streaming=True)),
    'wikipedia': lambda wiki: ('wikimedia/wikipedia', dict(name=f'20231101.{wiki}',
                                                           split='train', streaming=True)),
    # FineWeb-2 covers 1,800+ languages but not English, which lives in its own repositories.
    # fineweb-edu is the filtered educational subset -- cleaner text than the raw crawl, which
    # matters at the small model sizes this study trains, and readable enough that a person can
    # judge the model's output rather than taking a loss number on faith.
    'fineweb_edu': lambda cfg: ('HuggingFaceFW/fineweb-edu',
                                dict(name=cfg or 'sample-10BT', split='train', streaming=True)),
    'fineweb': lambda cfg: ('HuggingFaceFW/fineweb',
                            dict(name=cfg or 'sample-10BT', split='train', streaming=True)),
}


def probe_capacity(lang: str, seconds: int = 60) -> tuple[int, bool]:
    """Stream a corpus without storing it, to find out how much text actually exists.

    Separate from collection on purpose. The POC's v2 measured the size of what it happened to
    keep and then reported a shortfall that did not exist -- the stream had hit a character cap,
    not the end of the corpus. Returns (chars_seen, exhausted); exhausted=False means the time
    limit stopped us and there is more.
    """
    from datasets import load_dataset

    repo, kw = SOURCES['fineweb2'](lang)
    ds = load_dataset(repo, **kw)
    n, docs, t0, exhausted = 0, 0, time.time(), True
    for rec in ds:
        n += len(rec.get('text') or '')
        docs += 1
        if time.time() - t0 > seconds:
            exhausted = False
            break
    print(f'  probe: {n:,} chars over {docs:,} docs in {time.time() - t0:.0f}s'
          f'{"  (corpus exhausted)" if exhausted else "  (time limit, more remains)"}')
    return n, exhausted


def collect_docs(lang: str, wiki: str | None = None, max_chars: int = 300_000_000,
                 max_seconds: int = 300) -> tuple[list[str], int]:
    """Pull documents from the first source that works, stopping at a character or time cap.

    Kept as a list of documents rather than one joined string because the tokenizer trains on
    document iterators and because each document gets its own <s> ... </s> framing below.
    """
    from datasets import load_dataset

    attempts = [('fineweb2', SOURCES['fineweb2'](lang))]
    if wiki:
        attempts.append(('wikipedia', SOURCES['wikipedia'](wiki)))

    for label, (repo, kw) in attempts:
        try:
            ds = load_dataset(repo, **kw)
            docs, n, t0 = [], 0, time.time()
            for rec in ds:
                text = rec.get('text') or ''
                if not text:
                    continue
                docs.append(text)
                n += len(text)
                if n >= max_chars or time.time() - t0 > max_seconds:
                    break
            print(f'  [{label}] kept {n:,} chars / {len(docs):,} docs '
                  f'in {time.time() - t0:.0f}s')
            return docs, n
        except Exception as e:
            print(f'  [{label}] failed: {repr(e)[:110]}')
    raise RuntimeError(f'no corpus source worked for {lang}')


def tokenizer_fingerprint(hf_tok) -> str:
    """A short hash of the vocabulary itself -- the identity of the prediction task.

    Two runs are only comparable if they scored the same task, and the task is set by the
    vocabulary: a loss of 2.9 over 16,000 possible tokens means something different from a loss
    of 2.9 over a different 16,000. Hashing the token-to-id mapping (rather than the file, which
    carries incidental metadata) gives a value that is equal exactly when the task is.

    Recorded in stats.json at prepare time and in every run record, so a mismatch between two
    sets of numbers is visible instead of being a silent puzzle.
    """
    vocab = hf_tok.get_vocab()
    blob = '\n'.join(f'{i}\t{t}' for t, i in sorted(vocab.items(), key=lambda kv: kv[1]))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


def load_shared_tokenizer(spec: str, max_len: int):
    """Load a tokenizer someone else built: a local directory, a tokenizer.json, or a hub id.

    This is how a group shares one vocabulary without sharing 130 MB of token arrays. Each
    person downloads their own text -- the streams will differ, because a web crawl does not
    hand out identical documents -- but tokenizing them with the same vocabulary keeps the
    perplexities comparable, which is the property that actually matters.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    if os.path.isfile(spec) and spec.endswith('.json'):
        return PreTrainedTokenizerFast(
            tokenizer_file=spec, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
            pad_token='<pad>', mask_token='<mask>', cls_token='<s>', sep_token='</s>',
            model_max_length=max_len)
    return AutoTokenizer.from_pretrained(spec, model_max_length=max_len)


def train_tokenizer(docs: list[str], vocab_size: int, max_len: int):
    """Train a byte-level BPE on this language's own text and wrap it for transformers.

    Byte-level means every string is representable, so there is no <unk> to lose information to
    on a language with diacritics -- which matters here, since the whole premise of the study is
    that a language-specific vocabulary beats a multilingual one that barely covers the language.
    The post-processor adds <s> ... </s> per document, so the concatenated stream carries
    document boundaries the model can see.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE(unk_token='<unk>'))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(iter(docs), trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.post_processor = processors.TemplateProcessing(
        single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>',
        special_tokens=[('<s>', tk.token_to_id('<s>')), ('</s>', tk.token_to_id('</s>'))])

    return PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>', cls_token='<s>', sep_token='</s>',
        model_max_length=max_len)


def encode_docs(hf_tok, docs: list[str], vocab_size: int, batch: int = 2000) -> np.ndarray:
    """Encode every document into one flat id array, at the width the vocabulary requires.

    The dtype comes from P.disk_dtype rather than being hardcoded, which is the fix that lets
    this same function encode a 250k-vocab multilingual tokenizer without silently truncating.
    """
    dtype = P.disk_dtype(vocab_size)
    parts, t0 = [], time.time()
    for i in range(0, len(docs), batch):
        for enc in hf_tok.backend_tokenizer.encode_batch(docs[i:i + batch]):
            parts.append(np.array(enc.ids, dtype=dtype))
        if (i // batch) % 20 == 0:
            print(f'\r  encoding {min(i + batch, len(docs)):,}/{len(docs):,} docs', end='')
    print()
    ids = np.concatenate(parts)
    print(f'  {len(ids):,} tokens ({len(ids)/1e6:.1f}M) in {time.time() - t0:.0f}s')
    return ids


def sample_chars_for_docs(n_docs: int) -> int:
    """Roughly how many characters yield n_docs documents, at web-average length."""
    return max(2_000_000, n_docs * 4_000)


def stream_docs(lang: str, wiki: str | None = None, max_chars: int | None = None,
                max_seconds: float | None = None, source: str = 'fineweb2'):
    """Yield documents one at a time from the first source that works.

    A generator rather than a list, and that is the whole point. collect_docs holds every
    document as a Python string -- fine for the 260 MB of Yoruba that exists, fatal for the
    billions of tokens a corpus large enough to occupy this hardware would need. Nothing here
    keeps more than the document in hand.
    """
    from datasets import load_dataset

    attempts = [(source, SOURCES[source](lang))]
    if wiki:
        attempts.append(('wikipedia', SOURCES['wikipedia'](wiki)))

    last_error = None
    for label, (repo, kw) in attempts:
        try:
            ds = load_dataset(repo, **kw)
            n, docs, t0 = 0, 0, time.time()
            for rec in ds:
                text = rec.get('text') or ''
                if not text:
                    continue
                yield text
                n += len(text)
                docs += 1
                if max_chars and n >= max_chars:
                    break
                if max_seconds and time.time() - t0 > max_seconds:
                    break
                if docs % 20_000 == 0:
                    rate = n / max(1e-9, time.time() - t0) / 1e6
                    print(f'\r    streamed {docs:,} docs / {n/1e6:,.0f}M chars '
                          f'({rate:.1f}M chars/s)', end='', flush=True)
            print(f'\r    [{label}] {docs:,} docs / {n/1e6:,.0f}M chars in '
                  f'{time.time()-t0:.0f}s' + ' ' * 24)
            return
        except Exception as e:                        # noqa: BLE001 -- try the next source
            last_error = e
            print(f'  [{label}] failed: {repr(e)[:110]}')
    raise RuntimeError(f'no corpus source worked for {lang}: {last_error!r}')


def train_tokenizer_streaming(lang, wiki, vocab_size, max_len, sample_chars,
                              sample_docs_keep, source='fineweb2'):
    """Train the BPE on a bounded prefix of the stream, keeping a document sample on the way.

    A 16k byte-level BPE cannot use more text than this: a couple of hundred million characters
    is already far past the point where the merge table stops changing. Reading a bounded prefix
    keeps memory flat however large the corpus is, which is the difference between preparation
    that scales and preparation that dies.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    kept, sample = [], []
    print(f'  reading up to {sample_chars/1e6:,.0f}M chars to train the vocabulary')
    for doc in stream_docs(lang, wiki, max_chars=sample_chars, source=source):
        kept.append(doc)
        if len(sample) < sample_docs_keep:
            sample.append(doc)

    tk = Tokenizer(models.BPE(unk_token='<unk>'))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(iter(kept), trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.post_processor = processors.TemplateProcessing(
        single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>',
        special_tokens=[('<s>', tk.token_to_id('<s>')), ('</s>', tk.token_to_id('</s>'))])
    del kept

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tk, bos_token='<s>', eos_token='</s>', unk_token='<unk>',
        pad_token='<pad>', mask_token='<mask>', cls_token='<s>', sep_token='</s>',
        model_max_length=max_len)
    return hf, sample


def encode_stream_to_disk(hf_tok, docs_iter, dtype, bin_path, batch: int = 2000):
    """Encode a document stream straight to a raw file, holding only one batch at a time.

    Returns (tokens_written, chars_read). It writes raw bytes rather than a .npy because the
    final length is not known until the stream ends, and a .npy header has to state it.
    """
    n_tok = n_chars = n_docs = 0
    t0 = time.time()
    buf = []

    with open(bin_path, 'wb') as out:
        def flush(chunk):
            nonlocal n_tok
            if not chunk:
                return
            encs = hf_tok.backend_tokenizer.encode_batch(chunk)
            arr = np.concatenate([np.asarray(e.ids, dtype=dtype) for e in encs])
            out.write(arr.tobytes())
            n_tok += len(arr)

        for doc in docs_iter:
            buf.append(doc)
            n_chars += len(doc)
            n_docs += 1
            if len(buf) >= batch:
                flush(buf)
                buf = []
                if n_docs % (batch * 20) == 0:
                    rate = n_tok / max(1e-9, time.time() - t0) / 1e6
                    print(f'\r    encoded {n_docs:,} docs -> {n_tok/1e6:,.1f}M tokens '
                          f'({rate:.2f}M tok/s)', end='', flush=True)
        flush(buf)

    print(f'\r    encoded {n_docs:,} docs -> {n_tok:,} tokens in {time.time()-t0:.0f}s'
          + ' ' * 24)
    return n_tok, n_chars


def bin_to_npy(bin_path: str, npy_path: str, dtype, n: int, chunk: int = 8_000_000):
    """Turn the raw stream into a .npy GpuTokens can memory-map, without loading it.

    Copied in chunks: the array can be tens of gigabytes, and the point of the streaming path is
    that no step ever needs all of it resident.
    """
    arr = np.lib.format.open_memmap(npy_path, mode='w+', dtype=dtype, shape=(n,))
    itemsize = np.dtype(dtype).itemsize
    with open(bin_path, 'rb') as f:
        pos = 0
        while pos < n:
            raw = f.read(min(chunk, n - pos) * itemsize)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=dtype)
            arr[pos:pos + len(block)] = block
            pos += len(block)
    arr.flush()
    del arr
    os.remove(bin_path)


def prepare_corpus(name: str, lang: str, wiki: str | None = None, vocab_size: int = 16_000,
                   max_len: int = 128, max_chars: int = 300_000_000, max_seconds: int = 300,
                   val_tokens: int = 500_000, sample_docs: int = 4000,
                   tokenizer: str | None = None,
                   tokenizer_sample_chars: int = 200_000_000,
                   source: str = 'fineweb2', force: bool = False) -> dict:
    """Collect, tokenize, and store one language once. Returns the stats dict.

    This is the step the POC repeats every session and the factory does exactly once. Re-running
    is a no-op unless force=True, which also matters for reproducibility: a re-trained BPE is a
    different vocabulary, and perplexities across two vocabularies are not comparable.

    Validation is taken from the TAIL of the stream, matching the POC, so the held-out text is
    documents the pretraining pool never contains.

    `tokenizer` is how a group stays comparable without sharing corpora. Left None, this trains a
    fresh BPE on whatever text it collected -- and since two people streaming the same web source
    do not receive identical documents, they get different vocabularies and their losses stop
    meaning the same thing. Point it at a shared tokenizer instead (a directory, a tokenizer.json,
    or a hub id) and everyone scores the same prediction task on their own text.
    """
    out = P.out_dir(name)
    if os.path.exists(os.path.join(out, 'stats.json')) and not force:
        print(f'{name}: already prepared (force=True to redo)')
        stats = T.load_stats(name)

        # Corpora prepared before fingerprints existed have no record of which vocabulary they
        # used, which makes them incomparable-by-inspection rather than genuinely incomparable.
        # The tokenizer is sitting right there, so fill it in rather than forcing a re-prepare.
        if 'tokenizer_fingerprint' not in stats:
            try:
                stats['tokenizer_fingerprint'] = tokenizer_fingerprint(load_tokenizer(name))
                stats.setdefault('tokenizer_source', 'unknown (prepared before this was recorded)')
                with open(os.path.join(out, 'stats.json'), 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2)
                print(f'  backfilled vocabulary fingerprint '
                      f'{stats["tokenizer_fingerprint"]}')
            except Exception as e:
                print(f'  could not fingerprint the existing tokenizer: {repr(e)[:70]}')
        return stats

    os.makedirs(out, exist_ok=True)
    print(f'{name}: preparing {lang}')

    # Pass one: the vocabulary, plus a document sample for the language-ID gate. Skipped when a
    # shared tokenizer is supplied, which also saves a pass over the network.
    sample = []
    if tokenizer:
        print(f'  using the shared tokenizer at {tokenizer!r} (not training a new one)')
        hf_tok = load_shared_tokenizer(tokenizer, max_len)
        for doc in stream_docs(lang, wiki, max_chars=sample_chars_for_docs(sample_docs),
                               source=source):
            sample.append(doc)
            if len(sample) >= sample_docs:
                break
    else:
        hf_tok, sample = train_tokenizer_streaming(
            lang, wiki, vocab_size, max_len, tokenizer_sample_chars, sample_docs, source)

    actual_vocab = hf_tok.backend_tokenizer.get_vocab_size()
    fingerprint = tokenizer_fingerprint(hf_tok)
    if not sample:
        raise ValueError(f'{name}: the stream yielded no documents')

    # Pass two: the whole corpus, encoded straight to disk. Memory holds one batch of documents
    # and never the corpus, so this is bounded whether the language has 69 million tokens or ten
    # billion.
    dtype = P.disk_dtype(actual_vocab)
    tmp_bin = os.path.join(out, '_tokens.bin')
    n_tok, n_chars = encode_stream_to_disk(
        hf_tok, stream_docs(lang, wiki, max_chars, max_seconds, source=source),
        dtype, tmp_bin)
    if n_tok <= val_tokens * 2:
        os.remove(tmp_bin)
        raise ValueError(f'{name}: only {n_tok:,} tokens, too few to hold out {val_tokens:,}')

    all_path = os.path.join(out, '_all_tokens.npy')
    bin_to_npy(tmp_bin, all_path, dtype, n_tok)

    # Split by slicing the memory-mapped array, so neither half is ever fully resident.
    allarr = np.load(all_path, mmap_mode='r')
    n_train = n_tok - val_tokens
    np.save(os.path.join(out, 'train_tokens.npy'), allarr[:n_train])
    np.save(os.path.join(out, 'val_tokens.npy'), allarr[n_train:])
    del allarr
    os.remove(all_path)

    with open(os.path.join(out, 'sample_docs.json'), 'w', encoding='utf-8') as f:
        json.dump(sample[:sample_docs], f)

    hf_tok.save_pretrained(os.path.join(out, 'tokenizer'))
    P.save_stats(name, f'bpe{actual_vocab}', actual_vocab,
                 {'train': int(n_train), 'val': int(val_tokens)},
                 extra={'lang': lang, 'chars': int(n_chars),
                        'chars_per_token': n_chars / n_tok,
                        'tokenizer_fingerprint': fingerprint,
                        'tokenizer_source': tokenizer or 'trained here',
                        'tokenizer_path': os.path.join('data', name, 'tokenizer')})

    # Read a sample back before anyone trains on it. Same discipline as the causal rungs: a
    # tokenizer round-trip bug is silent, and a human reading prose is what catches it.
    head = np.asarray(np.load(os.path.join(out, 'train_tokens.npy'), mmap_mode='r')[:60])
    print(f'\n  decoded sample: {hf_tok.decode(head.tolist())[:200]!r}')
    print(f'  {n_train:,} train + {val_tokens:,} val tokens, {n_chars/n_tok:.2f} chars/token')
    print(f'  vocabulary fingerprint {fingerprint} -- runs only compare across matching '
          f'fingerprints\n')
    return T.load_stats(name)


def load_tokenizer(name: str):
    """The tokenizer this corpus was prepared with -- fine-tuning must use the same one."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(P.out_dir(name), 'tokenizer'))


def sample_docs(name: str, n: int | None = None) -> list[str]:
    """The raw documents kept aside at prepare time, for the language-ID gate.

    These are the same documents the tokenizer was trained on, so a language-ID verdict over
    them is a verdict about the text that actually became training tokens -- not about a fresh
    sample that may have drawn from a different part of the stream.
    """
    path = os.path.join(P.out_dir(name), 'sample_docs.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- prepare this corpus with a current mlm_data.py')
    with open(path, encoding='utf-8') as f:
        docs = json.load(f)
    return docs[:n] if n else docs


class MlmTokens(T.GpuTokens):
    """A GPU-resident token stream that serves masked batches instead of shifted ones.

    Everything about storage, width, and windowing is inherited from GpuTokens -- this class
    only changes what a batch MEANS. Where the causal stream returns (window, window shifted by
    one), this returns (window with some tokens corrupted, labels that are -100 everywhere the
    model is not being asked to predict).
    """

    def __init__(self, device, dataset, split='train', seq_len=128, subset=None,
                 store_dtype='auto', mask_id=None, n_special=len(SPECIALS)):
        super().__init__(device, dataset, split, seq_len=seq_len, subset=subset,
                         store_dtype=store_dtype)
        # mask_id has to come from the tokenizer that built this corpus; guessing it would
        # corrupt the objective in a way that still trains and still reports a loss.
        if mask_id is None:
            mask_id = load_tokenizer(dataset).mask_token_id
        self.name = dataset
        self.mask_id = int(mask_id)
        self.n_special = int(n_special)

    def windows(self, batch_size: int, generator=None) -> torch.Tensor:
        """One batch of random (batch_size, seq_len) windows, int64, ready for masking."""
        starts = torch.randint(0, self.n - self.seq_len - 1, (batch_size,),
                               device=self.device, generator=generator)
        pos = starts.view(-1, 1) + torch.arange(self.seq_len, device=self.device)
        return self.t[pos].long()

    def mask(self, x: torch.Tensor, mlm_prob: float = 0.15, generator=None):
        """The BERT 80/10/10 corruption, on-device. Returns (corrupted inputs, labels).

        Of the positions selected at mlm_prob: 80% become <mask>, 10% become a random real
        token, and 10% are left alone. The last two exist because a model that only ever sees
        <mask> at prediction time learns a representation that is useless when no <mask> is
        present -- which is every downstream fine-tuning batch.

        Labels are -100 outside the selection, which is the ignore_index HF's loss expects, so
        the loss is averaged over masked positions only.
        """
        labels = x.clone()
        sel = torch.rand(x.shape, device=x.device, generator=generator) < mlm_prob
        labels[~sel] = -100

        r = torch.rand(x.shape, device=x.device, generator=generator)
        out = x.clone()
        out[sel & (r < 0.8)] = self.mask_id
        # Random replacements skip the special ids, so corruption never injects <pad> or <mask>
        # as if it were a real word.
        rnd = torch.randint(self.n_special, self.vocab_size, x.shape,
                            device=x.device, generator=generator)
        swap = sel & (r >= 0.8) & (r < 0.9)
        out[swap] = rnd[swap]
        return out, labels

    def masked_batches(self, batch_size: int, steps: int, mlm_prob: float = 0.15,
                       generator=None):
        """Yield `steps` masked training batches. The training loop's only data call."""
        for _ in range(steps):
            x = self.windows(batch_size, generator)
            yield self.mask(x, mlm_prob, generator)

    def fixed_val_batches(self, batch_size: int = 64, n_batches: int = 4,
                          mlm_prob: float = 0.15, seed: int = 1234):
        """A masked validation set built once with a fixed seed, so every checkpoint is scored
        on exactly the same corrupted text. Without this the val loss moves when the masking
        moves, and two checkpoints stop being comparable."""
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        usable = ((self.n - 1) // self.seq_len) * self.seq_len
        flat = self.t[:usable].long().view(-1, self.seq_len)[:batch_size * n_batches]
        return [self.mask(flat[i:i + batch_size], mlm_prob, g)
                for i in range(0, len(flat), batch_size)]


def corpus_report(name: str) -> dict:
    """Everything the notebook wants to print about a prepared corpus, without loading it."""
    s = T.load_stats(name)
    with_ppl = dict(s)
    with_ppl['store_gb_int16'] = s['n_tokens']['train'] * 2 / 1e9
    with_ppl['store_gb_int32'] = s['n_tokens']['train'] * 4 / 1e9
    return with_ppl


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Prepare one language into the factory format.')
    p.add_argument('--name', required=True, help='corpus name, e.g. yor')
    p.add_argument('--lang', required=True, help='FineWeb-2 code, e.g. yor_Latn')
    p.add_argument('--wiki', default=None, help='Wikipedia fallback code, e.g. yo')
    p.add_argument('--source', default='fineweb2', choices=sorted(SOURCES),
                   help='fineweb2 for most languages; fineweb_edu/fineweb for English')
    p.add_argument('--vocab-size', type=int, default=16_000)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--max-chars', type=int, default=300_000_000)
    p.add_argument('--max-seconds', type=int, default=300)
    p.add_argument('--val-tokens', type=int, default=500_000)
    p.add_argument('--sample-docs', type=int, default=4000,
                   help='raw documents kept aside for the language-ID gate')
    p.add_argument('--tokenizer', default=None,
                   help='reuse a shared tokenizer (dir, tokenizer.json, or hub id) instead of '
                        'training a new one -- required for comparability across machines')
    p.add_argument('--tokenizer-sample-chars', type=int, default=200_000_000,
                   help='characters read to train the BPE; the rest of the corpus '
                        'streams to disk and is never held in memory')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    # Keyword arguments, not positional. Passing these positionally is what silently handed
    # --force to the sample_docs parameter when a new argument was inserted ahead of it: the
    # document sample came out empty and --force stopped doing anything at all, neither of
    # which raised. Keywords make the signature free to grow.
    stats = prepare_corpus(name=a.name, lang=a.lang, wiki=a.wiki, vocab_size=a.vocab_size,
                           max_len=a.max_len, max_chars=a.max_chars,
                           max_seconds=a.max_seconds, val_tokens=a.val_tokens,
                           sample_docs=a.sample_docs, tokenizer=a.tokenizer,
                           tokenizer_sample_chars=a.tokenizer_sample_chars,
                           source=a.source, force=a.force)
    print(json.dumps({k: v for k, v in stats.items() if k != 'vocab'}, indent=2))
