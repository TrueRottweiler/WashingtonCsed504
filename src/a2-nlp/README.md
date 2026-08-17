# A2 · NLP — when is it worth training your own model?

**The question.** For a language with little text and few labels, the received answer is
*transfer*: fine-tune a multilingual encoder someone else pretrained. The alternative is to
pretrain your own small encoder on whatever in-language text exists, with a vocabulary fitted to
it. This project asks when transfer stops being the right answer, using **Yoruba** as the test
case — about 47 million speakers, and all of FineWeb-2 Yoruba is 69.1M tokens.

**What we found.** A 33.8M-parameter model trained from scratch on 64M Yoruba tokens reaches
**0.688 macro-F1 on SIB-200 topic classification, against mmBERT's 0.582** — a model that saw
about 3 trillion tokens across 1,800 languages. It loses entity recognition by 0.026. The reason
is the **vocabulary, not the data**: the data axis saturates at or before 64M tokens, so Yoruba
is not text-starved at these budgets, while XLM-R's 250k vocabulary costs 1.76 tokens per word
against a fitted 16k BPE. Swapping *only* the vocabulary, holding architecture, text and compute
fixed, moves downstream scores by +0.144 on topic and +0.061 on entities, with every seed of the
fitted vocabulary beating every seed of the multilingual one.

**Read the conclusions properly in [`reports/13-the-top-board.md`](reports/13-the-top-board.md)**,
which is the argument in fourteen panels with every number traced to its record. The limits are
stated there too, and they are real: the causal evidence is one language with four seeds a side,
SIB-200 has only 204 test items, and nobody on the team reads Yoruba, so every quality judgement
is a benchmark number rather than a judgement about the output.

---

**Two studies live in this folder.** They share the token store, the scheduler, and the
dashboard, which is why they are together — but they are separate work and it is easy to open
the wrong one.

| you are here for | start at |
|---|---|
| **The Yoruba study** — masked LM, from-scratch vs multilingual transfer | **[QUICKSTART.md](QUICKSTART.md)**, then [`mlm_api.py`](mlm_api.py) and [`ft_api.py`](ft_api.py) |
| The causal LM study — LSTM vs GPT on WikiText, held-out perplexity | this README, below |

## Running the Yoruba study

The pretraining interface is **[`mlm_api.py`](mlm_api.py)**. That is the whole surface: nine
functions, each documented in place, and nothing else in this folder needs to be imported to use
the factory.

```python
import mlm_api as factory                       # from inside src/a2-nlp/

factory.prepare_corpus('yor', lang='yor_Latn')   # once per language, cached on disk
factory.pretrain('yor', tokens=16_000_000, steps=12_000)
factory.results()                                # every finished run, as a list of dicts
factory.curve('yor_16M_12k_s0')                  # one run's loss history
```

Two arguments worth knowing about early:

- `pretrain(..., reuse=True)` is the default and returns the existing record if that exact cell
  has already been trained. Re-running a notebook top to bottom costs nothing.
- `prepare_corpus(..., tokenizer=...)` points a corpus at a shared vocabulary instead of training
  a fresh one. Two people streaming the same web source do not receive identical documents, so
  they get different BPEs and their losses stop being comparable. Pass a directory, a
  `tokenizer.json`, or a hub id. `tokenizers/` has five committed vocabularies with fingerprints.

The downstream half is **[`ft_api.py`](ft_api.py)** — `load_sib200`, `load_masakhaner`,
`evaluate`, `results`, `table`. It is separate on purpose: it runs on a few hundred labelled
examples in seconds and has nothing to gain from a GPU-resident stream.

[QUICKSTART.md](QUICKSTART.md) is the ten-minute version, written for a single GPU including a
free Colab session — nothing here needs the two-card workstation. `POC_v4_factory.ipynb` is the
original proof-of-concept notebook with its plumbing replaced by these calls, every change
bracketed by `# BEGIN: factory` / `# END: factory` with the replaced code left commented
underneath, so the diff is readable.

### Reproducing the study end to end

The whole project measured **148 GPU-hours** across every run it kept; the smoke path below is
minutes, and `mlm_run.py --estimate` projects a step's cost from what it measures on your machine
before you commit a night to it. Nothing needs credentials — every dataset is public and
downloads on first use.

```bash
cd src/a2-nlp

# 1. Corpus. Collects, tokenizes and caches one language; prints a decoded sample -- read it.
python mlm_data.py --name yor --lang yor_Latn --wiki yo --vocab-size 16000

# 2. Check the machine before spending a night on it.
python diagnose.py --corpus yor --scale-to afriberta

# 3. Pretrain. --smoke proves the wiring in about a minute first.
python mlm_fleet.py --corpus yor --queue poc --smoke
python mlm_fleet.py --corpus yor --queue poc
python mlm_run.py --corpus yor --random-init        # the untrained control every gap is measured against

# 4. Fine-tune and evaluate (see QUICKSTART for the dev-split selection rule).
python sweep_fromscratch.py                          # dev-selected LR sweep, then test
python sweep_ner_baselines.py

# 5. The studies behind individual panels, each writing runs/<name>.json
python study_label_quantity.py --report              # says what this machine can contribute first
python study_tokenizer_seeds.py
python study_swap_downstream.py

# 6. Regenerate every figure and check nothing drifted from its records
python poster_figures.py
git status --porcelain reports/figures/               # empty means no figure drifted
```

**Verification gates**, all runnable without a GPU:

```bash
python -m unittest discover -p "test_*.py"   # 53 tests; board numbers pinned to runs/
python claims_audit.py                       # states the null for each comparative claim
python check_links.py                        # every relative link in the markdown resolves
python check_boards.py                       # no figure claimed by both posters
python poster/board_content.py               # every poster panel still fits its column
```

Findings, with the methodology written out, are in **[reports/](reports/)** — start with
[reports/README.md](reports/README.md), which indexes all thirteen and says which are superseded.

### Citations

Full list with links in [reports/09-the-bottom-report.md](reports/09-the-bottom-report.md)
§References. The load-bearing ones:

| | |
|---|---|
| **XLM-R** | Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale* — [arXiv:1911.02116](https://arxiv.org/abs/1911.02116) |
| **mmBERT** | [`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base) |
| **RoBERTa** | Liu et al. (2019) — [arXiv:1907.11692](https://arxiv.org/abs/1907.11692); the architecture the from-scratch models use |
| **AfriBERTa** | Ogueji, Zhu and Lin (2021), *Small Data? No Problem!* — [ACL Anthology](https://aclanthology.org/2021.mrl-1.11/); the 86M preset is shaped after it |
| **SIB-200** | Adelani et al. (2024) — [arXiv:2309.07445](https://arxiv.org/abs/2309.07445) |
| **MasakhaNER 2.0** | Adelani et al. (2022) — [arXiv:2210.12391](https://arxiv.org/abs/2210.12391) |
| **FineWeb-2** | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| **Statistics** | Welch (1947) for the unequal-variance *t*; Pitman (1937) for the permutation tests |

**AI disclosure.** An assistant (Claude Code) was used throughout to turn ideas and action points
into working code. It made the building much faster and the verifying no faster at all — every
number here still had to be checked against the records.

---

## The causal LM study — When Does Attention Beat Recurrence?

Parameter-matched LSTM and GPT language models raced across a ladder of data scales, measured by
held-out perplexity. This part duplicates the a1-cv factory files and adapts them from
images/top-1 to tokens/perplexity; a1-cv itself is untouched. The proposal has the full study
design; what follows is *how to run it*.

## The pieces

| file | job (a1-cv counterpart) |
|---|---|
| `text_prepare.py` | tokenize each corpus once into flat uint16 arrays (`imagenet_prepare.py`) |
| `text_data.py` | the GPU-resident token stream serving random windows (`imagenet_data.py`) |
| `models.py` | the four builders: `lstm`, `gpt`, `lstm_large`, `gpt_medium` (`models.py`) |
| `train_loop.py` | epoch loop, perplexity metrics, checkpoints, JSONL (`train_loop.py`) |
| `train_run.py` | one process, one GPU, one model (`train_run.py`) |
| `train_fleet.py` | the two-card scheduler with preset queues (`train_fleet.py`) |
| `dashboard.py` | the live read-only dashboard (`dashboard.py`) |
| `store_bench.py` | what the wide (int32) resident token store costs, measured |

And the masked-LM half, which is what the Yoruba from-scratch-vs-transfer study runs on:

| file | job |
|---|---|
| `mlm_api.py` | **the published pretraining interface** — nine functions, the only module a notebook needs |
| `mlm_data.py` | collect + tokenize a language once; the GPU-resident stream and BERT masking |
| `mlm_train.py` | the HF model builders, the step-based pretraining loop, the cost estimator |
| `mlm_run.py` | one process, one GPU, one cell of the (data × compute) grid |
| `mlm_fleet.py` | that grid across both cards |
| `diagnose.py` | **measure this machine**: GPUs, batch sweep, named bottleneck, projected cost |
| `audit_corpus.py` | **measure this corpus**: enough text? consistent spelling? does the vocab fit? |
| `explain_model.py` | ask a checkpoint to fill in blanks — what the model actually learned |
| `nb_clean.py` | make an executed notebook render (see the note below) |
| `py.sh` | run any of the above with a working interpreter, from this folder, with UTF-8 output |
| [`tokenizers/`](tokenizers/) | **shared vocabularies** — commit-sized, and what keeps everyone's numbers comparable |

And the downstream half, plus the analysis and reporting layer:

| file | job |
|---|---|
| `ft_api.py` | **the published fine-tuning interface** — load SIB-200 / MasakhaNER, evaluate, read the canonical table |
| `sweep_fromscratch.py`, `sweep_ner_baselines.py` | the dev-selected learning-rate sweeps behind the reported rows |
| `study_*.py` | one console study per question, each writing `runs/<name>.json` — label quantity, tokenizer seeds, the vocabulary swap, the NER control sweep, clipping, LR transfer |
| `poster_figures.py` | all figures, generated from `mlm_api.results()` / `ft_api.results()` so no chart can drift from its records |
| [`poster/`](poster/) | turns a board markdown file into printable PowerPoint + PDF ([README](poster/README.md)) |
| `claims_audit.py` | states the null for each comparative claim and computes what would refute it |
| `check_links.py`, `check_boards.py` | the markdown gates — dangling links, figures claimed twice |
| `test_*.py` | 53 tests, no GPU needed; `test_board_numbers.py` pins every published number to its record |

**Written up in [`reports/`](reports/):** what the study asks and where the limits are, what
the model actually learned, and the throughput investigation. Start with
[reports/01-what-were-building.md](reports/01-what-were-building.md).

Extra packages beyond the a1 stack (already in the workstation's `uw-csed504` env):
`datasets`, `tokenizers`, `transformers`, `seqeval`, `fasttext-numpy2`.

`fasttext` (the GlotLID language-ID runtime) is source-only and needs two MSVC flags its own
setup.py does not pass — `/std:c++17` (it uses `std::string_view`; MSVC still defaults to C++14)
and `/Dssize_t=Py_ssize_t` (it uses the POSIX `ssize_t`, which MSVC does not define). Supply them
through `CL` and it builds against stock pybind11:

```bash
CL="/std:c++17 /Dssize_t=Py_ssize_t" pip install fasttext-numpy2
```

`setup_windows.ps1` does this for you.

## The token store is as wide as the vocabulary needs

`text_data.resolve_store_dtype` picks the narrowest signed type that holds the ids: `int16` for
everything in this study (char-level, and the 16k BPE), `int32` once a vocabulary passes 32,768
types — which is every multilingual checkpoint (mBERT ≈ 120k, XLM-R ≈ 250k). `--store-dtype`
forces either. An explicit `int16` on a vocabulary too large for it raises rather than wrapping,
because a truncated id is not a crash: it is a different, perfectly valid-looking token that
would quietly poison every perplexity downstream.

Measured with `store_bench.py` on wikitext103/gpt, 5 interleaved repeats:

| width | store | peak | step tok/s |
|---|---|---|---|
| int16 | 0.248 GB | 9.07 GB | 773 k |
| int32 | 0.495 GB | 9.32 GB | 774 k |

So the wide store costs **exactly 2× the token stream and no measurable throughput**. Quote the
step column, not the gather column: gather timing on this box is bimodal (≈470–820 M tok/s) at
*both* widths, and a single unrepeated measurement of it will show a 1.5× difference that is
pure noise. That is what `--repeat` and the printed spread are for.

## One-time data preparation

```bash
python text_prepare.py --dataset all
```

Downloads and tokenizes all three rungs (shakespeare is instant; wikitext103 downloads ~310 MB
and tokenizes for a few minutes, and also trains the shared 16k BPE both wikitext rungs use).
Outputs land in `data/` (gitignored). The Hugging Face cache keeps both the parquet download and
a decompressed Arrow copy, so budget ~850 MB there on top of the ~245 MB under `data/`.
Each rung prints a decoded sample at the end — **read it**; if it is not readable prose, do not
train on it.

| rung | tokens (train) | tokenizer | role |
|---|---|---|---|
| `shakespeare` | ~1.0M chars | char-level | smoke rung; continuity with CSED 503 |
| `wikitext2` | ~2.5M | shared 16k BPE | small benchmark rung |
| `wikitext103` | ~124M | shared 16k BPE | the headline rung |

## Training

```bash
# prove the wiring first (~1 min):
python train_fleet.py --queue wikitext2 --smoke

# the quick rungs:
python train_fleet.py --queue shakespeare      # both baselines, minutes
python train_fleet.py --queue wikitext2        # all four models on the small rung

# the overnight run: wikitext103, both headliners x3 seeds + both capacity controls:
python train_fleet.py --queue overnight
```

Watch from another terminal with `python dashboard.py`. Single runs:
`python train_run.py --model gpt --dataset wikitext103 --gpu 0`.

Same discipline as Part 1: notebooks (when they arrive) are fast pipeline checks; every quoted
number comes from these console runs, whose history lands in `runs/*.jsonl` and
`runs/*_result.json`.

## Things to know before quoting numbers

- **Perplexity is only comparable within a tokenizer.** Both wikitext rungs share one BPE
  precisely so they can be compared to each other; none of our numbers compare to published
  word-level WikiText perplexities.
- **Lower is better.** `best_ppl` in the result files is a minimum; the dashboard flips all its
  comparisons accordingly.
- **The two families are matched on backbone parameters** (~10.6M each), not total — with a 16k
  vocab both models carry an identical ~6M tied embedding on top. `train_run.py` prints both
  counts at startup; `gpt_medium` and `lstm_large` are the capacity controls.
- **wikitext2 runs will overfit hard late in training.** That is the small rung being small, not
  a bug: best-so-far val perplexity is what gets recorded, and *when* each family peaks is part
  of the finding.
- **Grad clipping defaults ON for both families here** (unlike Part 1's per-family clip). It is
  standard practice in both families' canonical LM recipes, so it cannot tilt the race.

## The masked-LM side (the Yoruba study)

One-time corpus preparation, then the grid. A corpus is addressed by *name* everywhere — in the
notebook, in the console runner, and on the other card — so a number from a notebook and a number
from an overnight run are comparable by construction.

```bash
# collect, tokenize, and cache one language (reads a decoded sample back -- check it)
python mlm_data.py --name yor --lang yor_Latn --wiki yo --vocab-size 16000

# prove the wiring (~1 min), then the POC's 2x2 grid across both cards
python mlm_fleet.py --corpus yor --queue poc --smoke
python mlm_fleet.py --corpus yor --queue poc
python mlm_run.py --corpus yor --random-init      # the control everything is measured against

# predict before committing a night
python mlm_run.py --corpus yor --steps 24000 --estimate
```

Watch with `python dashboard.py` — the MLM runs write the same JSONL schema as the causal ones,
so the existing dashboard displays them unchanged.

After executing any notebook headlessly, run `python nb_clean.py <notebook>`. Hugging Face's
progress bars leave `widget-view` outputs whose backing state does not survive a headless run,
and those cells render as **"Could not render content"**; the cleaner drops the dead views (their
`text/plain` fallback is kept), strips the ANSI codes transformers prints, and restores cell ids.
It took `POC_v4_factory.ipynb` from 787 KB to 244 KB.

`factory_diagnostics.ipynb` runs `diagnose.py` and `audit_corpus.py` with charts — the same
analysis the reports contain, but computed for whatever machine and corpus you point it at.
Run it first on a new box.

Notebooks: `POC_v4_factory.ipynb` is the v3 proof-of-concept with its plumbing replaced
by these calls (every change bracketed by `# BEGIN: factory` / `# END: factory`, with the code it
replaced left commented underneath); `results_factory_mlm.ipynb` reads `runs/*_result.json` and
plots the data axis against the compute axis. `POC_v3_...ipynb` is the original and is frozen.

Two things the factory deliberately does **not** take over: the fine-tuning harness (SIB-200,
MasakhaNER, the seeded bootstrap CIs) runs on a few hundred labeled examples in seconds and has
nothing to gain from a GPU-resident stream, and checkpoints are ordinary `save_pretrained`
directories so `AutoModelFor*.from_pretrained` keeps working.
