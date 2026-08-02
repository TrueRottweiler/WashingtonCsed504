# Reports

Written investigations behind the A2-NLP factory. Each one is self-contained and every number in
them was measured on the CSED 504 workstation (2 × RTX PRO 6000 Blackwell) unless labelled as a
projection.

| | report | what it answers |
|---|---|---|
| 01 | [What we're building](01-what-were-building.md) | What is the group's research question, what does the factory provide, and where are the data and hardware limits? Start here. |
| 02 | [What the model actually learned](02-what-the-model-learned.md) | We trained a model — what did it *do*? Fill-in-the-blank predictions from the real checkpoint, what a loss of 2.886 means, and what the corpus does and doesn't contain. |
| 03 | [The throughput investigation](03-efficiency.md) | Why GPU utilisation was poor, what we measured, what we changed, and which of our guesses were wrong. The engineering log. |

## The short version

- **Yoruba is scarce.** All of it on FineWeb-2 is 69.1M tokens — less than one English benchmark
  dataset. The group's top rung uses 93% of everything available.
- **The small from-scratch model works.** 33.8M parameters, ten minutes on one card, and it
  matches mmBERT on topic classification (0.527 vs 0.537). It does *not* match on entity
  recognition (0.698 vs 0.848), so the easy task was flattering it.
- **The study is compute-bound, not data-bound.** More training moves validation loss by 2.2–2.7;
  16× more text moves it by 0.08–0.61. Convenient, since text is what they've run out of.
- **The factory made the same work 2.07× faster** — 1.33× from batch size, the rest from using the
  second card at all — and the full study fits in 7.6 GPU-hours against a ~20 hour budget.

## Two things not to quote yet

- **XLM-R's 0.127 on topic classification** is almost certainly a fine-tuning failure, not a
  coverage result — the same model scores 0.843 on entity recognition. The group's headline
  contrast depends on this number and it needs re-running with more seeds.
- **Everything is one seed per cell.** The differences between the four pretrained models are
  currently smaller than the run-to-run wobble. `python mlm_fleet.py --corpus yor --queue seeds`
  exists for this.

## Reproducing

All three reports are generated from artifacts in `runs/` and `data/`, which the factory writes.
See the parent [README](../README.md) for how to run the pipeline, and `explain_model.py`,
`store_bench.py`, and `results_factory_mlm.ipynb` for the tools that produced the figures.
