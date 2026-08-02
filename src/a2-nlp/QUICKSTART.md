# Quickstart — for Patrick and Leon

Everything here runs on **one GPU**, including a free Colab session. Nothing requires the
two-card workstation; the scheduler just uses whatever cards it finds.

If you only read one section, read [§2](#2-the-thing-that-will-save-you-the-most-time).

---

## 1. Setup (one cell)

```python
!git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
%pip install -q -U datasets transformers tokenizers seqeval

import os, sys
FACTORY_DIR = "/content/WashingtonCsed504/src/a2-nlp"
os.environ["FACTORY_DIR"] = FACTORY_DIR      # POC_v4_factory.ipynb reads this
sys.path.insert(0, FACTORY_DIR)
os.chdir(FACTORY_DIR)                        # data/ and runs/ are resolved relative to here

import mlm_api as factory
print("factory API", factory.API_VERSION)
```

`fasttext` (for the GlotLID language-ID gate) installs normally on Colab — it only needs special
handling on Windows.

---

## 2. The thing that will save you the most time

**Prepare the corpus once, and keep it.** A Colab session throws away its disk when it ends, so
without this you re-stream and re-tokenize every session — several minutes each time, and, more
importantly, **a re-trained BPE is a different vocabulary.** Two sessions' losses are then not
comparable, and neither are two checkpoints.

Point the factory's data directory at Drive and the problem disappears:

```python
from google.colab import drive
drive.mount('/content/drive')

import text_prepare, text_data
CACHE = '/content/drive/MyDrive/csed504-corpora'
os.makedirs(CACHE, exist_ok=True)
text_prepare.DATA_DIR = text_data.DATA_DIR = CACHE     # both modules resolve paths off these
```

Then prepare — the first time it downloads and tokenizes, every time after it's instant:

```python
stats = factory.prepare_corpus("yor", lang="yor_Latn", wiki="yo", vocab_size=16_000)
```

It prints a decoded sample at the end. **Read it.** If it isn't recognisable Yoruba, stop there —
that check has already caught one real problem for us.

---

## 3. Know the cost before you spend it

```python
factory.estimate("yor", cells=[(2_000_000, 3_000), (32_000_000, 12_000)], preset="poc")
```

Runs for a few seconds, measures this machine's actual throughput, prints predicted hours per
cell. Scale it to the size the real study needs with `scale_to="afriberta"`.

---

## 4. Train

```python
rec = factory.pretrain("yor", tokens=32_000_000, steps=12_000, seed=0)
print(rec["val_loss"], rec["path"])
```

- Already trained that exact cell? It returns the saved record instantly instead of retraining.
  Pass `reuse=False` to force it.
- The checkpoint is an ordinary `save_pretrained` directory, so
  `AutoModelForSequenceClassification.from_pretrained(rec["path"])` works — **your fine-tuning
  code does not need to change at all.**
- Don't forget the control: `factory.random_init("yor")`.

For anything longer than a few minutes use the console runner instead of a notebook cell — a
Colab kernel that dies at hour two takes the run with it:

```bash
python mlm_run.py --corpus yor --tokens 32000000 --steps 12000 --gpu 0
python mlm_fleet.py --corpus yor --queue poc          # the whole 2x2 grid
```

---

## 5. Two defaults we changed, and why you should care

**Batch size is 128, not 64.** Measured 1.33× faster on identical work, and it saturates there —
past 256 it gets *worse*. If you have a reason to change it, use `batch=`.

**Compute budgets are counted in tokens of updates, not steps.** This one matters for your
science. `steps × batch × seq_len` is the work actually done, so 6,000 steps at batch 64 and
6,000 at batch 128 are *different experiments*. If you compare runs at different batch sizes by
step count, you will attribute a doubling of compute to whatever else changed. `mlm_fleet.py`
takes `--update-tokens` for exactly this reason; your old 6k/24k steps at batch 64 are 49.2M and
196.6M tokens of updates.

---

## 6. See what a checkpoint learned

```bash
python explain_model.py --corpus yor
python explain_model.py --corpus yor --text "Àwọn ọmọ ilé ìwé lọ sí ọjà ní àárọ̀"
```

Hides a word in held-out Yoruba, prints the model's top guesses beside an untrained model's, and
converts each run's loss into "how many words is it still choosing between" (16,000 → about 18
for the best run so far).

---

## 7. Reading results

```python
factory.results()          # one row per completed run
factory.curve("yor_32M_12k_s0")   # the training curve
```

Or open `results_factory_mlm.ipynb`, which reads the same records and plots the data axis against
the compute axis.

---

## What we are *not* taking over

Your fine-tuning half — SIB-200, MasakhaNER, the seeded harness with pooled bootstrap CIs — is
untouched and should stay that way. It runs on a few hundred labelled examples in seconds; a
GPU-resident token stream has nothing to offer it. `POC_v4_factory.ipynb` is your v3 notebook
with only the pretraining plumbing swapped out, each change marked
`# BEGIN: factory` / `# END: factory` with the old code left commented underneath. **v3 is
untouched.**

---

## Findings you should know before you write anything up

Detail in [`reports/`](reports/). The three that change decisions:

1. **All the Yoruba in FineWeb-2 is 69.1M tokens** — it exhausts in 7 seconds of streaming. Your
   64M rung uses 93% of it and the 128M rung is not reachable from that source.
2. **The study is compute-bound.** More training moves validation loss by 2.2–2.7; 16× more text
   moves it by 0.08–0.61. Spend the budget on updates, not on scraping.
3. **XLM-R's 0.127 on SIB-200 is probably a fine-tuning failure, not a coverage result** — the
   same model scores 0.843 on MasakhaNER, which it could not do without usable Yoruba
   representations. Your headline contrast leans on that number; please re-run it with more seeds
   and a higher learning rate before it goes on the poster.

---

## If something breaks

- `no CUDA device found` — Colab runtime is set to CPU. Runtime → Change runtime type → GPU.
- `--gpu 1 was requested but this machine has 1` — use `--gpu 0`, or let `mlm_fleet.py` pick.
- Cells showing **"Could not render content"** after a headless run — `python nb_clean.py <nb>`.
- A corpus that won't re-prepare — pass `force=True`; without it, preparation is a deliberate
  no-op.

Ask Jeffrey. The interface in `mlm_api.py` is meant to stay stable — if you write against it
today it should still work next week, and if it doesn't, that's a bug worth reporting.
