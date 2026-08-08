"""Time and size every stage of the text pipeline, so the writeup can quote measurements.

The question this answers is one a student asks and the literature almost never does: when you
say "we pretrained a language model", what were you actually doing with the hours? Tokenizer
training, encoding, and evaluation are invisible in every paper, and two of the three turn out
to be non-trivial.

Everything here runs against the Yoruba corpus that the study actually used -- 80,000 documents,
260 million characters -- so the numbers are the real ones rather than a synthetic benchmark.
Stages that would take longer than a few minutes are measured on a fixed sample and reported as a
rate plus the extrapolation, which is labeled as such.

    bash src/a2-nlp/py.sh pipeline_bench.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import mlm_data as D
import mlm_api as f
import text_data as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'pipeline_bench.json')
CORPUS = 'yor'


def gb(n_bytes):
    return n_bytes / 1024 ** 3


def main():
    stats = json.load(open(os.path.join(HERE, 'data', CORPUS, 'stats.json'), encoding='utf-8'))
    n_chars = stats['chars']
    n_docs = stats['docs']
    n_tokens = stats['n_tokens']['train']
    out = {'corpus': CORPUS, 'chars': n_chars, 'docs': n_docs, 'train_tokens': n_tokens,
           'stages': []}

    def stage(name, **kw):
        out['stages'].append(dict(name=name, **kw))
        bits = '  '.join(f'{k}={v}' for k, v in kw.items())
        print(f'{name:<34} {bits}')

    print(f'{CORPUS}: {n_docs:,} documents, {n_chars:,} characters, {n_tokens:,} tokens\n')

    # ---- 1. Reading the corpus off disk -----------------------------------------------------
    # The documents are no longer on disk as text -- only the token array is kept -- so this
    # stage is timed on the tokenizer's own sample cache, which is the same I/O shape.
    t0 = time.perf_counter()
    docs = D.sample_docs(CORPUS, 20_000)
    dt = time.perf_counter() - t0
    sample_chars = sum(len(d) for d in docs)
    stage('1. read documents', seconds=round(dt, 2), docs=len(docs),
          chars_per_s=round(sample_chars / dt) if dt else None,
          extrapolated_full_s=round(n_chars / (sample_chars / dt)) if dt else None)

    # ---- 2. Training a 16k BPE tokenizer ----------------------------------------------------
    # Trained on the same sample the real preparation uses. This is CPU-only and single-pass.
    t0 = time.perf_counter()
    tok = D.train_tokenizer(docs, 16_000, 128)
    dt_tok = time.perf_counter() - t0
    stage('2. train 16k BPE tokenizer', seconds=round(dt_tok, 1), trained_on_chars=sample_chars,
          device='CPU')

    # ---- 3. Encoding text to token ids ------------------------------------------------------
    t0 = time.perf_counter()
    ids = D.encode_docs(tok, docs, 16_000)
    dt_enc = time.perf_counter() - t0
    stage('3. encode to token ids', seconds=round(dt_enc, 1), tokens=len(ids),
          tokens_per_s=round(len(ids) / dt_enc) if dt_enc else None,
          extrapolated_full_s=round(n_tokens / (len(ids) / dt_enc)) if dt_enc else None,
          dtype=str(ids.dtype), device='CPU')

    # ---- 4. What the token store costs on disk and in memory --------------------------------
    npy = os.path.join(HERE, 'data', CORPUS, 'train.npy')
    disk = os.path.getsize(npy) if os.path.exists(npy) else n_tokens * stats['store_bytes']
    stage('4. token store', disk_gb=round(gb(disk), 3), bytes_per_token=stats['store_bytes'],
          note='uint16 holds a 16k vocabulary; a 250k vocabulary needs uint32 and doubles this')

    # ---- 5. Moving the corpus onto the card -------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ds = f.stream(CORPUS, seq_len=128, gpu=0)
        torch.cuda.synchronize()
        dt_up = time.perf_counter() - t0
        stage('5. corpus onto the GPU', seconds=round(dt_up, 2),
              resident_gb=round(gb(torch.cuda.memory_allocated()), 3),
              note='the whole corpus lives on the card; there is no DataLoader in the loop')

        # ---- 6. Peak memory and throughput for one training step, both model sizes ----------
        for preset, label in (('poc', '33.8M model'), ('afriberta', '98M model')):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            est = f.estimate(CORPUS, [(n_tokens, 62_500)], preset=preset, batch=128)
            stage(f'6. train step, {label}',
                  tokens_per_s=round(est['tok_s_measured']),
                  peak_gb=round(gb(torch.cuda.max_memory_allocated()), 2),
                  hours_for_62500_steps=round(est['cells'][0]['hours'], 2))
    else:
        print('  (no CUDA visible -- GPU stages skipped)')

    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2)
    print(f'\nwrote runs/pipeline_bench.json')


if __name__ == '__main__':
    main()
