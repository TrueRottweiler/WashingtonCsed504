"""
mlm_api.py -- the factory's published interface for the from-scratch-vs-transfer study.

This is the surface Patrick and Leon call. Everything here is meant to stay stable while the
modules underneath change: if a call in this file works today it should work next week, and the
internals are free to move. Import it and nothing else:

    import mlm_api as factory

Five things it does, in the order a study uses them:

    factory.prepare_corpus(...)   collect and tokenize a language ONCE into token arrays
    factory.stream(...)           that corpus, resident on a GPU, serving masked batches
    factory.estimate(...)         measure throughput and predict hours BEFORE committing a night
    factory.pretrain(...)         train one grid cell, checkpoint it, record it
    factory.results(...)          every completed run, as rows ready for a table or a plot

Two design notes worth knowing, because they explain why the calls look the way they do.

Everything is addressed by a corpus NAME, not by a path or an in-memory object. A name means the
same tokens and the same vocabulary in the notebook, in the console runner, and on the other
card -- which is what makes a number from a notebook comparable with a number from an overnight
fleet run. Corpora live under data/<name>/ and survive kernel restarts.

Nothing here fine-tunes. The fine-tuning half of the POC -- SIB-200, MasakhaNER, the seeded
harness with pooled bootstrap CIs -- stays exactly as it is. It runs on a few hundred labeled
examples in seconds, so the factory has nothing to offer it, and the checkpoints produced below
are ordinary save_pretrained directories that AutoModelFor*.from_pretrained already loads.
"""
from __future__ import annotations

import glob
import json
import os

import torch

import mlm_data as _data
import mlm_train as _train
import text_data as _text

# Bumped when a call here changes shape. Pin against it if a script must not silently change
# behavior: assert mlm_api.API_VERSION == (1, 0)
API_VERSION = (1, 0)

RUNS = _train.RUNS
SIZE_PRESETS = _train.SIZE_PRESETS

# The count formatter the run names use. The plots label their axes with it too, so a point on a
# chart and the run that produced it are named the same way.
compact = _train.compact

# Re-exported so the notebook can reach the pieces without importing the private modules.
prepare_corpus = _data.prepare_corpus
probe_capacity = _data.probe_capacity
load_tokenizer = _data.load_tokenizer
load_shared_tokenizer = _data.load_shared_tokenizer
tokenizer_fingerprint = _data.tokenizer_fingerprint
sample_docs = _data.sample_docs
MlmTokens = _data.MlmTokens


def bits_per_char(tag: str = None, *, val_loss: float = None, corpus: str = None) -> float:
    """Validation loss in bits per CHARACTER -- the only unit that compares two vocabularies.

    Loss in nats per token is meaningless across tokenizers, and this project has already been
    caught by that twice. Report 04 says so and then has to caveat every cross-language table;
    report 05 had to invent "context gained" to work around it. Neither is a substitute for the
    standard unit.

    A token is worth `chars_per_token` characters, so:

        bits/char = (nats/token) / ln 2 / (chars/token)

    Both factors matter and they pull in opposite directions. A vocabulary that fits a language
    badly produces MORE tokens per character, which lowers chars_per_token and therefore RAISES
    bits per character even if the per-token loss looks better. That is exactly the trap in
    comparing a 16k language-specific BPE against XLM-R's 250k: the second has an easier per-token
    job and does more of them.

        bits_per_char('multi_yor')                     -> from the run's own record
        bits_per_char(val_loss=2.92, corpus='yor')     -> for a loss you have in hand

    Returns None when the corpus was prepared before chars_per_token was recorded, rather than
    guessing at it.
    """
    import math

    if tag is not None:
        rows = results(tag)
        if not rows:
            raise ValueError(f'no completed run named {tag!r}')
        val_loss = rows[0]['val_loss']
        corpus = corpus or rows[0].get('corpus')
    if val_loss is None or not corpus:
        raise ValueError('need either a tag, or both val_loss and corpus')

    cpt = _text.load_stats(corpus).get('chars_per_token')
    if not cpt:
        return None
    return val_loss / math.log(2) / cpt


def corpus_info(name: str) -> dict:
    """What is in a prepared corpus: vocabulary, token counts, chars/token, and both widths.

    Cheap -- reads stats.json, never the arrays -- so it is safe to call while deciding whether
    a rung fits before loading anything onto a card.
    """
    return _data.corpus_report(name)


def stream(name: str, tokens: int | None = None, split: str = 'train', seq_len: int = 128,
           gpu: int = 0, store_dtype: str = 'auto') -> _data.MlmTokens:
    """A corpus resident on one GPU, serving masked batches.

    `tokens` is the DATA axis of the grid: it takes that many tokens off the front of the
    stream, so the 2M rung is a prefix of the 32M rung and the two differ only in how much
    unique text the model sees.

    store_dtype is 'auto' by default, which picks the narrowest integer width the vocabulary
    allows -- int16 for a 16k BPE, meaning a quarter of the memory an int64 stream costs for the
    identical batches. Pass 'int32' to force the wide store; measured on this hardware it costs
    exactly twice the stream's memory and no measurable throughput (see store_bench.py), so
    forcing it is cheap insurance if a vocabulary might grow.
    """
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    return _data.MlmTokens(device, name, split, seq_len=seq_len, subset=tokens,
                           store_dtype=store_dtype)


def estimate(name: str, cells: list[tuple[int, int]], preset: str = 'poc', batch: int = 128,
             seq_len: int = 128, gpu: int = 0, scale_to: str | None = None,
             probe_steps: int = 20) -> dict:
    """Measure this box's throughput, then predict wall-clock for a grid.

    `cells` is a list of (tokens, steps) pairs -- the same grid you would hand to pretrain().
    `scale_to` rescales the prediction to a bigger preset using the POC's hidden^2 x depth rule,
    so a few seconds at 'poc' can answer "does the afriberta-scale study fit a weekend".

    The throughput comes from a real short run rather than a table, because it is the one number
    that changes with hardware, batch shape, and whatever else is on the card at the time.
    """
    ds = stream(name, tokens=min(4_000_000, corpus_info(name)['n_tokens']['train']),
                seq_len=seq_len, gpu=gpu)
    tok = load_tokenizer(name)
    tok_s = _train.measure_throughput(ds, tok, preset, batch, steps=probe_steps)
    est = _train.estimate_hours(tok_s, cells, batch, seq_len, preset, scale_to)

    print(f'measured {tok_s/1e3:.0f}k tok/s at preset {preset!r}'
          + (f' -> {est["tok_s_effective"]/1e3:.0f}k at {scale_to!r} '
             f'({est["scale"]:.1f}x cost/token)' if scale_to else ''))
    print(f'\n{"tokens":>12}{"steps":>9}{"passes":>9}{"hours":>8}')
    for r in est['cells']:
        print(f'{r["n_tokens"]:>12,}{r["steps"]:>9,}{r["passes"]:>9.1f}{r["hours"]:>8.2f}')
    print(f'{"total":>12}{"":>9}{"":>9}{est["total_hours"]:>8.2f}\n')
    return est


def pretrain(name: str, tokens: int, steps: int, seed: int = 0, preset: str = 'poc',
             batch: int = 128, seq_len: int = 128, lr: float | None = None,
             mlm_prob: float = 0.15,
             gpu: int = 0, tag: str | None = None, store_dtype: str = 'auto',
             reuse: bool = True, clip: float = 1.0, warmup: float | None = None) -> dict:
    """Pretrain one grid cell and return its record.

    The two axes are `tokens` (how much unique text) and `steps` (how much compute). Note that
    steps are only comparable at a FIXED batch: what the study actually varies is tokens of
    updates, steps x batch x seq_len, and mlm_fleet's queues are written in those units so the
    batch size can change for throughput reasons without moving the experiment.

    reuse=True returns the existing record if this cell has already been trained, instead of
    retraining it. That is what makes the notebook cheap to re-run: the fine-tuning gates below
    the grid are the ones being iterated on, and re-executing the notebook should not spend
    twenty minutes reproducing checkpoints that are already on disk. Pass reuse=False to force
    a retrain.

    `clip` and `warmup` reach mlm_train unchanged. They were exposed on the fleet before they
    were exposed here, which meant the one setting known to change the 98M model's behaviour
    could not be varied from a script -- and so the question of whether it PREVENTS failures,
    as opposed to tightening the spread of the runs that succeed, went unasked for a term.

    Writes a save_pretrained checkpoint under runs/<tag>/, a JSONL curve the dashboard reads
    live, and runs/<tag>_result.json for the results notebook. Returns the same record.

    For anything longer than a few minutes, prefer the console runners -- a notebook kernel that
    dies at hour three takes the run with it:
        python mlm_run.py --corpus <name> --tokens <n> --steps <n> --gpu 0
        python mlm_fleet.py --corpus <name> --queue poc
    """
    ds = stream(name, tokens=tokens, seq_len=seq_len, gpu=gpu, store_dtype=store_dtype)
    if tag is None:
        tag = _train.cell_tag(name, ds.n, steps, seed, preset)

    if reuse:
        done = os.path.join(RUNS, f'{tag}_result.json')
        ckpt = os.path.join(RUNS, tag, 'config.json')
        if os.path.exists(done) and os.path.exists(ckpt):
            with open(done, encoding='utf-8') as f:
                rec = json.load(f)
            print(f'[{tag}] reusing the completed run '
                  f'(val {rec["val_loss"]:.3f}, {rec["seconds"]/60:.1f} min) -- '
                  f'pass reuse=False to retrain')
            return rec

    val = stream(name, split='val', seq_len=seq_len, gpu=gpu, store_dtype=store_dtype)
    tok = load_tokenizer(name)
    return _train.pretrain(ds, tok, tag, steps, preset=preset, batch=batch, lr=lr,
                           mlm_prob=mlm_prob, seed=seed, clip=clip, warmup=warmup,
                           val_batches=val.fixed_val_batches(mlm_prob=mlm_prob))


def random_init(name: str, preset: str = 'poc', seq_len: int = 128) -> str:
    """Write the untrained control checkpoint and return its path.

    This is the row the whole study turns on: identical architecture, identical tokenizer,
    identical fine-tuning, no pretraining. Whatever the pretrained cells score above it is what
    pretraining bought, and a gain inside the seed spread is not a gain.
    """
    out = os.path.join(RUNS, f'{name}_random_init')
    if os.path.exists(os.path.join(out, 'config.json')):
        return out
    return _train.save_random_init(_text.load_stats(name)['vocab_size'], load_tokenizer(name),
                                   preset, seq_len, out)


def random_init_like(model_id: str, tag: str | None = None) -> str:
    """An untrained model with SOMEONE ELSE'S architecture and vocabulary. No pretrained weights.

    `random_init` above builds our architecture from a corpus we prepared, which cannot express
    "XLM-R, minus the pretraining". That distinction decides an argument. The existing control
    differs from XLM-R in three ways at once -- size, tokenizer and pretraining -- so a gap
    between them cannot separate "XLM-R's Yoruba is weak" from "XLM-R's vocabulary caps what it
    can do", and the second is the study's thesis.

    This loads the published CONFIG and instantiates from it, which is the one line that matters:
    `from_config` initialises fresh weights where `from_pretrained` would download trained ones.
    Same layer count, same hidden size, same 250k vocabulary, same tokenizer -- and no knowledge
    of any language.

        random_init_like('FacebookAI/xlm-roberta-base')   -> runs/xlm-roberta-base_random_init

    Returns the checkpoint directory, ready for ft_api.evaluate() like any other model path.
    """
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    out = os.path.join(RUNS, f'{tag or model_id.split("/")[-1]}_random_init')
    if os.path.exists(os.path.join(out, 'config.json')):
        return out

    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_config(cfg)
    tok = AutoTokenizer.from_pretrained(model_id)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def results(pattern: str = '*', include_smoke: bool = False) -> list[dict]:
    """Every completed pretraining record, newest last, with the history dropped.

    The history stays in the JSON on disk; this returns the one-row-per-run summary that a table
    or a scatter wants. Pass a glob to narrow it, e.g. results('yor_*').

    Each row merges `<tag>_meta.json` under `<tag>_result.json`. The result file records what the
    run achieved and the meta file records what it WAS -- which corpus, which preset, which seed
    -- and without the merge no row said what language it trained on or what size the model was.
    Every consumer then had to re-derive that from the tag, and the results notebook drew the
    86M and 33.8M runs at one rung as a single series because it could not tell them apart.

    The result wins on conflict: meta is written when the run starts, the result when it ends.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, f'{pattern}_result.json'))):
        base = os.path.basename(path)
        if not include_smoke and base.startswith('smoke-'):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if 'steps' not in rec:      # a causal run from the LSTM-vs-GPT study; not this grid
            continue

        meta = {}
        meta_path = path[:-len('_result.json')] + '_meta.json'
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass                    # a run from before meta files existed; the row is thinner

        row = {k: v for k, v in meta.items() if k != 'history'}
        row.update({k: v for k, v in rec.items() if k != 'history'})
        row.setdefault('tag', base[:-len('_result.json')])
        if not row.get('corpus'):
            row['corpus'] = _corpus_from_tag(row['tag'])
        rows.append(row)
    return rows


def _corpus_from_tag(tag: str) -> str | None:
    """Recover which corpus a run used from its name, for runs written before meta files existed.

    Matched against the corpora actually prepared on this machine, longest name first, rather
    than by splitting on underscores -- a corpus name can contain one (`eng_1b`), so splitting
    would report the corpus as `eng` and silently merge two different studies.
    """
    prepared = sorted(
        (os.path.basename(os.path.dirname(p))
         for p in glob.glob(os.path.join(_data.P.out_dir('*'), 'stats.json'))),
        key=len, reverse=True)
    for name in prepared:
        if tag == name or tag.startswith(name + '_'):
            return name
    return None


def curve(tag: str) -> list[dict]:
    """One run's logged points, for plotting a training curve."""
    path = os.path.join(RUNS, f'{tag}_result.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)['history']
