"""
ft_api.py -- the fine-tuning half of the study, behind one stable interface.

The pretraining half already lives behind mlm_api.py, which is why a notebook cell that trains a
grid cell is three lines long. This is the same idea for the other half: SIB-200, MasakhaNER, the
seeded harness with pooled bootstrap CIs. Import it and nothing else:

    import ft_api as ft

The reason it exists is not tidiness. The open questions are all COMPARISONS -- 86M against
33.8M, XLM-R against mmBERT, 701 labels against 6,876 -- and each one wants its own small
notebook. Copy the harness into three notebooks and there are three harnesses within a week: a
bug fixed in one, a seed count changed in another, and the numbers stop being comparable in a way
nothing announces. One code path, one set of numbers.

Five things it does, in the order a study uses them:

    ft.load_sib200(...)        topic classification data, cached
    ft.load_masakhaner(...)    NER data, read from CoNLL (see "MasakhaNER" below)
    ft.token_lengths(...)      what a tokenizer does to those items -- no GPU, no training
    ft.evaluate(...)           fine-tune over seeds, score, bootstrap, write the record
    ft.results(...)            every completed record, as rows ready for a table

Four design notes, because they explain why the calls look the way they do.

FIXED STEP BUDGET, NOT EPOCHS. The notebook trained `for _ in range(FT_EPOCHS)`. On 701 SIB-200
examples at batch 16 that is 352 updates; on 6,876 MasakhaNER sentences it is 3,440. So the
notebook's comparison between the two tasks varied the update budget by 10x at the same time as
the task and the label count, and subsampling MasakhaNER to 701 sentences under that harness
would have cut its budget by 10x as well -- measuring the budget cut and calling it a label-count
effect. Here `steps` is an explicit budget that does not move when the training set shrinks. The
defaults reproduce the notebook on the FULL splits (see FT_STEPS), so the existing numbers still
come out; what changes is only the case the notebook got wrong.

RECORDS GO IN runs/, ONE PER CELL. Keyed by (model, task, lang, n_train, lr, steps, seed-set),
written as runs/ft_*.json. Two consequences, both deliberate. session.save_results() copies
runs/*.json off a Colab runtime's disposable disk, so downstream results travel by the same route
the pretraining ones already do -- a separate results directory would be silently lost with the
session. And the suffix is _ft.json, not _result.json, so mlm_api.results() does not pick these
up: the pretraining table stays a pretraining table.

POOLED BOOTSTRAP, NOT THE LAST SEED. `evaluate` resamples test ITEMS and averages across seeds
inside each resample, so the interval covers both sources of variation. A per-seed sd is reported
alongside it because the two answer different questions: the sd says whether a rerun would land
somewhere else, the CI says whether the test set is large enough to tell two models apart. On 204
SIB-200 items it usually is not, which is the point.

MASAKHANER IS READ FROM CoNLL, NOT load_dataset. The HuggingFace copy ships a custom loading
script, and that path is no longer executed after the July 2026 incident. The files come from the
masakhane-ner GitHub repo instead, and the loader records and prints the split sizes -- comparing
against the published Table 4 baseline is only valid if they match.

TEXT IS NFC-NORMALISED ON THE WAY IN, and that is not cosmetic. MasakhaNER ships in DECOMPOSED
form: 17.1% of its characters vanish under NFC, because Yoruba tone marks are stored as separate
combining codepoints. The committed 16k BPEs have no normalizer (they are byte-level, trained on
FineWeb-2, which is precomposed), so on the raw files they see `náà` as five codepoints and cut
it into four tokens instead of one. XLM-R does not care -- SentencePiece carries a precompiled
charsmap that folds the difference away.

The result was a measurement that reversed the study's central claim. Raw, yor-bpe16k spends 3.15
tokens/word on MasakhaNER against XLM-R's 2.83, and 18.9% of sentences overflow a 128-token
window against XLM-R's 11.8% -- the project's own tokenizer looking WORSE than the multilingual
one, on the one dataset where the thesis says it should look best. Under NFC it is 1.67 against
2.91, and 0.2% against 13.3%. Same data, same tokenizers, opposite conclusion.

So `normalize` defaults to 'NFC', it is recorded on every result and encoded in the record tag,
and normalize=None reproduces the pre-fix numbers rather than silently overwriting them.
"""
from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import random
import time
import unicodedata
import urllib.request

import numpy as np
import torch
import torch.nn as nn

import mlm_api as factory

# Bumped when a call here changes shape or a default moves. Pin against it if a script must not
# silently change behaviour: assert ft_api.API_VERSION == (1, 1)
#   (1, 0)  extraction from POC_v4_factory.ipynb
#   (1, 1)  NFC normalisation on by default; `normalize` added to the record and the tag
API_VERSION = (1, 1)

RUNS = factory.RUNS
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LEN = 128
FT_BATCH = 16

# The notebook's fine-tuning defaults, kept so the recorded numbers reproduce.
FT_LR = 2e-5            # pretrained baselines (XLM-R, mmBERT)
FT_LR_SCRATCH = 5e-5    # from-scratch and random-init checkpoints
NER_LR = 3e-5

# Step budgets, chosen to equal what the notebook's epoch loop spent on the FULL split at batch
# 16 -- SIB-200 701/16 = 44 batches x 8 epochs, MasakhaNER 6,876/16 = 430 batches x 5 epochs.
# That makes the extraction behaviour-preserving where the notebook was already right, and fixed
# where it was not: a 701-sentence NER subsample now gets 2,150 updates like the full split does,
# instead of 215.
FT_STEPS = 352
NER_STEPS = 2150

SEEDS = (0, 1, 2)
N_BOOT = 500

# The subsample is drawn with its own seed, NOT the training seed, so every model and every
# training seed sees the IDENTICAL 701 sentences. Varying the subsample with the seed would fold
# "which examples were drawn" into the seed spread and make the label-count comparison noisier
# than it needs to be.
SUBSAMPLE_SEED = 12345

# Unicode normalisation applied to every item before it reaches a tokenizer. See the module
# docstring: on MasakhaNER this is the difference between the study's central claim holding and
# reversing. None disables it and reproduces the pre-fix numbers.
NORMALIZE = 'NFC'

SIB_TEXT, SIB_LABEL = 'text', 'category'
MASAKHANER_URL = ('https://raw.githubusercontent.com/masakhane-io/masakhane-ner/main'
                  '/MasakhaNER2.0/data/{lang}/{split}.txt')

_SIB_CACHE: dict = {}
_NER_CACHE: dict = {}


def set_seed(s: int) -> None:
    """Seed every generator the fine-tuning path touches."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def gpu_name() -> str:
    """What card this is running on. Recorded with every result -- a GPU-hour on a Blackwell and
    a GPU-hour on a T4 differ by an order of magnitude, so an unqualified figure is not a
    measurement."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def _norm(s: str, form: str | None) -> str:
    """Unicode-normalise one string. Word counts are unaffected -- normalisation composes
    combining marks into precomposed characters, it never splits or joins words -- so a
    normalised NER sentence still lines up with its tag sequence."""
    return unicodedata.normalize(form, s) if form else s


def decomposition_report(texts) -> dict:
    """How much of this text is in decomposed form. A quick way to check a new corpus, or to
    confirm the prepared pretraining corpus matches what the tokenizer was trained on.

    `shrink` is the fraction of characters that disappear under NFC. Near 0 means precomposed
    (FineWeb-2, SIB-200); MasakhaNER is 0.171.
    """
    s = ''.join(texts)
    nfc = unicodedata.normalize('NFC', s)
    return {'chars': len(s), 'chars_nfc': len(nfc), 'is_nfc': s == nfc,
            'shrink': (len(s) - len(nfc)) / max(len(s), 1)}


def load_sib200(lang: str = 'yor_Latn', normalize: str | None = NORMALIZE) -> dict:
    """SIB-200 topic classification for one language.

    Returns {'train'/'validation'/'test': {'text': [...], 'label': [...]}, 'labels': [...]}.
    Label ids are assigned from the sorted category names, so they are stable across sessions and
    across languages -- a model fine-tuned in one notebook and scored in another agrees about
    what class 3 is.
    """
    key = (lang, normalize)
    if key in _SIB_CACHE:
        return _SIB_CACHE[key]

    from datasets import load_dataset

    ds = load_dataset('Davlan/sib200', lang)
    labels = sorted(set(ds['train'][SIB_LABEL]))
    l2i = {l: i for i, l in enumerate(labels)}

    out = {'labels': labels, 'lang': lang, 'task': 'sib200', 'normalize': normalize}
    for split in ('train', 'validation', 'test'):
        if split in ds:
            out[split] = {'text': [_norm(t, normalize) for t in ds[split][SIB_TEXT]],
                          'label': [l2i[l] for l in ds[split][SIB_LABEL]]}
    sizes = ' '.join(f'{s} {len(out[s]["text"])}' for s in ('train', 'validation', 'test')
                     if s in out)
    dec = decomposition_report(ds['train'][SIB_TEXT])
    print(f'sib200/{lang}: {sizes} | {len(labels)} classes (chance {1/len(labels):.3f}) '
          f'| {dec["shrink"]:.1%} decomposed -> {normalize}')
    _SIB_CACHE[key] = out
    return out


def _parse_conll(text: str) -> tuple[list[list[str]], list[list[str]]]:
    """CoNLL -> parallel lists of word sequences and tag sequences."""
    tokens, tags, cur_t, cur_g = [], [], [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('-DOCSTART-'):
            if cur_t:
                tokens.append(cur_t)
                tags.append(cur_g)
                cur_t, cur_g = [], []
            continue
        parts = line.split()
        if len(parts) >= 2:
            cur_t.append(parts[0])
            cur_g.append(parts[-1])
    if cur_t:
        tokens.append(cur_t)
        tags.append(cur_g)
    return tokens, tags


def load_masakhaner(lang: str = 'yor', normalize: str | None = NORMALIZE) -> dict:
    """MasakhaNER 2.0 for one language, read from the project's CoNLL files.

    NOT via load_dataset: that copy ships a custom loading script and HuggingFace no longer
    executes those. The files are cached under data/masakhaner/<lang>/ after the first call, so a
    warm runtime does not re-download and an offline kernel still works.

    The split sizes are printed and stored on the record. The comparison against MasakhaNER 2.0
    Table 4 is only valid if they match the published ones -- for Yoruba, 6,876 / 983 / 1,964.
    """
    key = (lang, normalize)
    if key in _NER_CACHE:
        return _NER_CACHE[key]

    cache = os.path.join(DATA, 'masakhaner', lang)
    os.makedirs(cache, exist_ok=True)

    raw = {}
    for split, fname in (('train', 'train'), ('validation', 'dev'), ('test', 'test')):
        path = os.path.join(cache, f'{fname}.txt')
        if not os.path.exists(path):
            url = MASAKHANER_URL.format(lang=lang, split=fname)
            with urllib.request.urlopen(url) as r:
                body = r.read().decode('utf-8')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(body)
        with open(path, encoding='utf-8') as f:
            raw[split] = _parse_conll(f.read())

    seen = sorted({t for toks, tags in raw.values() for row in tags for t in row})
    names = ['O'] + [t for t in seen if t != 'O']    # 'O' is id 0 by construction
    t2i = {t: i for i, t in enumerate(names)}

    out = {'tags': names, 'lang': lang, 'task': 'masakhaner', 'normalize': normalize}
    for split, (toks, tags) in raw.items():
        out[split] = {'tokens': [[_norm(w, normalize) for w in s] for s in toks],
                      'ner_tags': [[t2i[t] for t in row] for row in tags]}

    sizes = ' '.join(f'{s} {len(out[s]["tokens"])}' for s in ('train', 'validation', 'test'))
    dec = decomposition_report(' '.join(s) for s in raw['train'][0])
    print(f'masakhaner2/{lang}: {sizes} | {len(names)} tags {names}')
    action = f'normalising to {normalize}' if normalize else 'LEFT AS-IS (normalize=None)'
    print(f'  {dec["shrink"]:.1%} of characters are decomposed -> {action}'
          + ('  (see the module docstring -- this one matters)' if dec['shrink'] > 0.02 else ''))
    _NER_CACHE[key] = out
    return out


def subsample(split: dict, n: int | None, seed: int = SUBSAMPLE_SEED) -> dict:
    """Take n items from a split, deterministically and identically for every caller.

    n=None returns the split unchanged. Drawn with SUBSAMPLE_SEED rather than the training seed
    so that all models and all seeds see the same items -- see the note on that constant.

    The draws are NESTED: taking a prefix of one permutation means the 701-item subsample is
    contained in the 2,000-item one, which is contained in the full split. So a label-quantity
    ladder varies only what was ADDED at each rung, never which items were swapped -- the same
    property the pretraining data ladder has, where the 2M rung is a prefix of the 32M rung. Two
    rungs that differed in membership as well as size would confound "more labels" with "these
    particular labels".
    """
    keys = list(split)
    total = len(split[keys[0]])
    if n is None or n >= total:
        return split
    idx = np.random.default_rng(seed).permutation(total)[:n]
    return {k: [split[k][i] for i in idx] for k in keys}


# ----------------------------------------------------------------------------------------------
# tokenizer diagnostics -- no GPU, no training
# ----------------------------------------------------------------------------------------------

def load_ft_tokenizer(spec: str, max_len: int = MAX_LEN):
    """A tokenizer by hub id ('FacebookAI/xlm-roberta-base'), local checkpoint, or shared
    vocabulary directory ('tokenizers/yor-bpe16k'). Goes through the factory's loader so the
    committed 16k vocabularies work here exactly as they do in pretraining."""
    return factory.load_shared_tokenizer(spec, max_len)


def token_lengths(spec: str, texts: list[str] | None = None,
                  words: list[list[str]] | None = None, max_length: int = MAX_LEN) -> dict:
    """How long these items are under this tokenizer, and how many would be truncated.

    Pass `texts` for sentence items (SIB-200) or `words` for pre-tokenized ones (MasakhaNER).
    Lengths include special tokens, because those occupy the window too.

    This is the measurement that decides whether the tokenizer penalty is a CONTEXT WINDOW
    argument or a REPRESENTATION argument. "65% of the window spent on fragments" is a claim
    about truncation; if frac_over is ~0 at max_length=128, nothing is being truncated and the
    penalty has to hurt some other way.
    """
    if (texts is None) == (words is None):
        raise ValueError('pass exactly one of texts= (sentences) or words= (pre-tokenized)')

    tok = load_ft_tokenizer(spec, max_len=max_length)
    if words is not None:
        # truncation=False is the whole point: the question is how long these items WOULD be,
        # which a truncating tokenizer cannot answer -- it would report max_length and nothing
        # over it.
        enc = tok(list(words), is_split_into_words=True, truncation=False,
                  add_special_tokens=True)
        denom = sum(len(w) for w in words)
    else:
        enc = tok(list(texts), truncation=False, add_special_tokens=True)
        denom = sum(len(t.split()) for t in texts)
    n = np.array([len(ids) for ids in enc['input_ids']])

    return {
        'tokenizer': spec,
        'n_items': int(n.size),
        'mean': float(n.mean()),
        'median': float(np.median(n)),
        'p90': float(np.percentile(n, 90)),
        'p95': float(np.percentile(n, 95)),
        'p99': float(np.percentile(n, 99)),
        'max': int(n.max()),
        'max_length': max_length,
        'n_over': int((n > max_length).sum()),
        'frac_over': float((n > max_length).mean()),
        'tokens_per_word': float(n.sum() / denom) if denom else None,
        'lengths': n,
    }


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------

def macro_f1(y_true, y_pred, k: int) -> float:
    """Unweighted mean per-class F1. Written out rather than imported so the number does not move
    if sklearn changes a default -- and so the zero-division convention is visible: a class the
    model never predicts and that never appears scores 0, not nan."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(out))


def entity_f1(gold: list[list[str]], pred: list[list[str]]) -> float:
    """Span-level entity F1 (seqeval). Imported lazily so the classification path and the
    tokenizer diagnostics do not require seqeval to be installed."""
    from seqeval.metrics import f1_score
    return float(f1_score(gold, pred))


def _take(x, idx):
    return x[idx] if isinstance(x, np.ndarray) else [x[i] for i in idx]


def pooled_ci(score_fn, gold, preds: list, n_boot: int = N_BOOT, seed: int = 0
              ) -> tuple[float, float]:
    """Bootstrap over test ITEMS, averaging across seeds within each resample.

    Not the last seed's interval: resampling one seed's predictions measures test-set size only,
    and reporting it as the model's interval understates the uncertainty by leaving out the run-
    to-run variation entirely. Each resample here scores every seed on the same resampled items
    and averages, so the interval carries both.
    """
    rng = np.random.default_rng(seed)
    n = len(gold)
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        g = _take(gold, b)
        vals.append(float(np.mean([score_fn(g, _take(p, b)) for p in preds])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------------------------------------
# fine-tuning
# ----------------------------------------------------------------------------------------------

def _load_model(cls, path: str, n_labels: int):
    """from_pretrained, retrying with the sdpa attention kernel. Some checkpoint/transformers
    pairs refuse the default kernel; the retry is what the notebook did and it is load-time only,
    so it does not change what gets trained."""
    try:
        return cls.from_pretrained(path, num_labels=n_labels)
    except Exception as e:                                  # noqa: BLE001 - reported, then retried
        print(f'    retrying with sdpa ({repr(e)[:60]})')
        return cls.from_pretrained(path, num_labels=n_labels, attn_implementation='sdpa')


def _train_steps(model, loader, opt, sch, steps: int, forward) -> None:
    """Train for exactly `steps` updates, cycling the loader as many times as that takes.

    The fixed budget is the point: it is what lets a 701-example condition and a 6,876-example
    one be compared without the comparison secretly being about how many gradient updates each
    got. It does mean the small condition makes many more passes over its data -- that is a real
    tradeoff, not a bug. Fixed steps holds compute equal and lets passes vary; fixed epochs holds
    passes equal and lets compute vary. Neither is neutral, and this one is the choice the
    project already made (no early stopping either: the smallest conditions have no dev set to
    stop on).
    """
    model.train()
    done = 0
    while done < steps:
        for batch in loader:
            loss = forward(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            done += 1
            if done >= steps:
                break


def _finetune_clf(model_path: str, data: dict, lr: float, seed: int, steps: int,
                  n_train: int | None, batch: int, max_length: int):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    n_labels = len(data['labels'])
    tok = AutoTokenizer.from_pretrained(model_path)
    model = _load_model(AutoModelForSequenceClassification, model_path, n_labels).to(DEVICE)

    def pack(split):
        enc = tok(split['text'], truncation=True, max_length=max_length,
                  padding='max_length', return_tensors='pt')
        return torch.utils.data.TensorDataset(enc['input_ids'], enc['attention_mask'],
                                              torch.tensor(split['label']))

    train = subsample(data['train'], n_train)
    tr = torch.utils.data.DataLoader(pack(train), batch_size=batch, shuffle=True, drop_last=False)
    te = torch.utils.data.DataLoader(pack(data['test']), batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps),
                                              pct_start=0.1)

    def forward(m, batch):
        ids, am, y = batch
        out = m(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE), labels=y.to(DEVICE))
        return out.loss

    t0 = time.time()
    _train_steps(model, tr, opt, sch, steps, forward)

    model.eval()
    pred, gold = [], []
    with torch.no_grad():
        for ids, am, y in te:
            logits = model(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE)).logits
            pred.append(logits.argmax(-1).cpu())
            gold.append(y)
    del model
    torch.cuda.empty_cache()
    return torch.cat(pred).numpy(), torch.cat(gold).numpy(), time.time() - t0, len(train['text'])


def _finetune_ner(model_path: str, data: dict, lr: float, seed: int, steps: int,
                  n_train: int | None, batch: int, max_length: int):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    set_seed(seed)
    names = data['tags']
    tok = AutoTokenizer.from_pretrained(model_path)
    model = _load_model(AutoModelForTokenClassification, model_path, len(names)).to(DEVICE)

    def pack(split):
        enc = tok(split['tokens'], is_split_into_words=True, truncation=True,
                  max_length=max_length, padding='max_length', return_tensors='pt')
        labels = []
        for i, tags in enumerate(split['ner_tags']):
            wid, prev, row = enc.word_ids(i), None, []
            for w in wid:
                # -100 on specials, padding, and every subword after the first: the metric is
                # over words, so only the first piece of each word carries its label.
                row.append(-100 if w is None or w == prev else tags[w])
                prev = w
            labels.append(row)
        return torch.utils.data.TensorDataset(enc['input_ids'], enc['attention_mask'],
                                              torch.tensor(labels))

    train = subsample(data['train'], n_train)
    tr = torch.utils.data.DataLoader(pack(train), batch_size=batch, shuffle=True, drop_last=False)
    te = torch.utils.data.DataLoader(pack(data['test']), batch_size=64)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps),
                                              pct_start=0.1)

    def forward(m, batch):
        ids, am, y = batch
        return m(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE), labels=y.to(DEVICE)).loss

    t0 = time.time()
    _train_steps(model, tr, opt, sch, steps, forward)

    model.eval()
    pred, gold = [], []
    with torch.no_grad():
        for ids, am, y in te:
            pr = model(input_ids=ids.to(DEVICE), attention_mask=am.to(DEVICE)).logits.argmax(-1)
            for pi, yi in zip(pr.cpu(), y):
                keep = yi != -100
                pred.append([names[t] for t in pi[keep].tolist()])
                gold.append([names[t] for t in yi[keep].tolist()])
    del model
    torch.cuda.empty_cache()
    return pred, gold, time.time() - t0, len(train['tokens'])


def finetune_once(model_path: str, task: str = 'sib200', lang: str | None = None,
                  lr: float | None = None, seed: int = 0, steps: int | None = None,
                  n_train: int | None = None, batch: int = FT_BATCH,
                  max_length: int = MAX_LEN, data: dict | None = None):
    """One fine-tuning run. Returns (predictions, gold, seconds, n_train_used).

    Mostly you want evaluate(), which does this over seeds and writes the record. This is here
    for the same reason mlm_api exposes the pieces: a notebook debugging one cell should not have
    to reimplement it.
    """
    if task == 'sib200':
        data = data or load_sib200(lang or 'yor_Latn')
        return _finetune_clf(model_path, data, lr if lr is not None else FT_LR, seed,
                             steps if steps is not None else FT_STEPS, n_train, batch, max_length)
    if task == 'masakhaner':
        data = data or load_masakhaner(lang or 'yor')
        return _finetune_ner(model_path, data, lr if lr is not None else NER_LR, seed,
                             steps if steps is not None else NER_STEPS, n_train, batch,
                             max_length)
    raise ValueError(f'unknown task {task!r}; known: sib200, masakhaner')


# ----------------------------------------------------------------------------------------------
# the study-level call
# ----------------------------------------------------------------------------------------------

def _slug(model_path: str) -> str:
    """A filesystem-safe short name. The full path stays on the record; this is only the key."""
    s = model_path.rstrip('/\\').replace('\\', '/').split('/')[-1]
    return ''.join(c if c.isalnum() or c in '-.' else '-' for c in s)


def record_tag(model_path: str, task: str, lang: str, n_train: int | None, lr: float,
               steps: int, normalize: str | None = NORMALIZE) -> str:
    """The record's name. Everything that changes the number is in it, so two conditions cannot
    overwrite each other and a sweep does not silently keep only its last cell.

    Normalisation is in the tag because it moves MasakhaNER fertility by 47% -- an NFC run and a
    raw run are different measurements and must not land on the same file.
    """
    n = 'full' if n_train is None else str(n_train)
    return (f'ft_{task}_{lang}_{_slug(model_path)}_n{n}_lr{lr:g}_st{steps}'
            f'_{(normalize or "raw").lower()}')


def evaluate(model_path: str, task: str = 'sib200', lang: str | None = None,
             lr: float | None = None, seeds=SEEDS, steps: int | None = None,
             n_train: int | None = None, batch: int = FT_BATCH, max_length: int = MAX_LEN,
             label: str = '', n_boot: int = N_BOOT, reuse: bool = True,
             data: dict | None = None, normalize: str | None = NORMALIZE) -> dict:
    """Fine-tune over seeds, score, bootstrap, write runs/<tag>_ft.json, return the record.

    This is the call every notebook makes. reuse=True returns the existing record rather than
    retraining, which is what makes a notebook cheap to re-run -- pass reuse=False to force it.

    Never claim a difference below ~0.06 macro-F1 on SIB-200: 204 test items do not resolve it,
    and the CI on the record is there to be quoted rather than worked around.
    """
    lang = lang or ('yor_Latn' if task == 'sib200' else 'yor')
    if lr is None:
        lr = FT_LR if task == 'sib200' else NER_LR
    if steps is None:
        steps = FT_STEPS if task == 'sib200' else NER_STEPS
    seeds = list(seeds)
    # If a caller passed pre-loaded data, ITS normalisation is the one that applies -- otherwise
    # the record would claim NFC while the tensors came from raw text.
    if data is not None:
        normalize = data.get('normalize', normalize)

    tag = record_tag(model_path, task, lang, n_train, lr, steps, normalize)
    path = os.path.join(RUNS, f'{tag}_ft.json')
    if reuse and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            rec = json.load(f)
        if rec.get('seeds') == seeds:
            print(f'  {label or _slug(model_path):<26} {rec["mean"]:.3f} (reusing record; '
                  f'reuse=False to rerun)')
            return rec

    if task == 'sib200':
        data = data or load_sib200(lang, normalize=normalize)
        k = len(data['labels'])
        score_fn = lambda g, p: macro_f1(g, p, k)       # noqa: E731 - closes over k
        metric = 'macro_f1'
    else:
        data = data or load_masakhaner(lang, normalize=normalize)
        score_fn = entity_f1
        metric = 'entity_f1'

    preds, gold, secs, used = [], None, [], None
    for s in seeds:
        p, g, t, used = finetune_once(model_path, task=task, lang=lang, lr=lr, seed=s,
                                      steps=steps, n_train=n_train, batch=batch,
                                      max_length=max_length, data=data)
        preds.append(p)
        gold = g
        secs.append(t)

    scores = [score_fn(gold, p) for p in preds]
    lo, hi = pooled_ci(score_fn, gold, preds, n_boot=n_boot)
    mean, sd = float(np.mean(scores)), float(np.std(scores))

    rec = {
        'tag': tag, 'label': label or _slug(model_path),
        'task': task, 'lang': lang, 'metric': metric,
        'model': model_path, 'model_slug': _slug(model_path),
        'n_train': used, 'n_train_requested': n_train, 'n_test': len(gold),
        'steps': steps, 'batch': batch, 'lr': lr, 'max_length': max_length,
        'normalize': normalize,
        'seeds': seeds, 'scores': scores,
        'mean': mean, 'sd': sd, 'ci': [lo, hi],
        'seconds_per_seed': float(np.mean(secs)), 'n_boot': n_boot,
        'gpu': gpu_name(), 'ft_api_version': list(API_VERSION),
        'created': _dt.datetime.now().isoformat(timespec='seconds'),
    }

    os.makedirs(RUNS, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rec, f, indent=2)

    print(f'  {rec["label"]:<26} {metric} {mean:.3f} +/-{sd:.3f} (seed sd) '
          f'| 95% CI [{lo:.3f}, {hi:.3f}] | n_train {used} | {np.mean(secs):.0f}s/seed')
    return rec


# ----------------------------------------------------------------------------------------------
# reading results back
# ----------------------------------------------------------------------------------------------

def results(pattern: str = '*', task: str | None = None, lang: str | None = None) -> list[dict]:
    """Every completed fine-tuning record, as rows ready for a table.

    This is what makes the canonical downstream table a query rather than a reconciliation: the
    numbers on the poster come from here, not from whichever notebook last printed them.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, f'ft_{pattern}_ft.json'))):
        try:
            with open(p, encoding='utf-8') as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if task and rec.get('task') != task:
            continue
        if lang and rec.get('lang') != lang:
            continue
        rows.append(rec)
    return sorted(rows, key=lambda r: (r['task'], r['lang'], -r['mean']))


def table(pattern: str = '*', task: str | None = None, lang: str | None = None) -> list[dict]:
    """results(), printed. Shows the CI next to every mean because that is the number that
    decides whether two rows differ."""
    rows = results(pattern, task=task, lang=lang)
    if not rows:
        print('no fine-tuning records yet')
        return rows
    print(f'{"model":<26}{"task":<13}{"n_train":>8}{"lr":>8}{"steps":>7}{"norm":>6}'
          f'{"score":>8}{"sd":>7}   95% CI')
    for r in rows:
        print(f'{r["label"]:<26}{r["task"]:<13}{r["n_train"]:>8}{r["lr"]:>8.0e}{r["steps"]:>7}'
              f'{str(r.get("normalize") or "raw"):>6}'
              f'{r["mean"]:>8.3f}{r["sd"]:>7.3f}   [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]')
    return rows
