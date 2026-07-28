# A2 · NLP — When Does Attention Beat Recurrence?

Part 2's model factory: parameter-matched LSTM and GPT language models raced across a ladder of
data scales, measured by held-out perplexity. This folder duplicates the a1-cv factory files and
adapts them from images/top-1 to tokens/perplexity; a1-cv itself is untouched. The proposal has
the full study design; this README is *how to run it*.

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

Extra packages beyond the a1 stack (already in the workstation's `uw-csed504` env):
`datasets`, `tokenizers`.

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
