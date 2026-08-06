# Reports

Written investigations behind the A2-NLP factory. Each one is self-contained and every number in
them was measured on the CSED 504 workstation (2 × RTX PRO 6000 Blackwell) unless labelled as a
projection.

| | report | what it answers |
|---|---|---|
| 01 | [What we're building](01-what-were-building.md) | What is the group's research question, what does the factory provide, and where are the data and hardware limits? Start here. |
| 02 | [What the model actually learned](02-what-the-model-learned.md) | We trained a model — what did it *do*? Fill-in-the-blank predictions from the real checkpoint, what a loss of 2.886 means, and what the corpus does and doesn't contain. |
| 03 | [The throughput investigation](03-efficiency.md) | Why GPU utilisation was poor, what we measured, what we changed, and which of our guesses were wrong. The engineering log. |
| 04 | [The language gradient](04-the-language-gradient.md) | Five languages, prepared identically. Is Yoruba hard, or under-served? What a multilingual vocabulary actually costs, and where the tooling had Latin-script assumptions in it. |
| 05 | [When more data stops helping](05-when-data-stops-mattering.md) | The English ladder — 256× of data at fixed compute — plus the Yoruba rungs that check whether its threshold transfers. Where the data axis saturates and whether the bigger model ever catches up. |

## The short version

- **Yoruba is scarce.** All of it on FineWeb-2 is 69.1M tokens — less than one English benchmark
  dataset. The group's top rung uses 93% of everything available.
- **The small from-scratch model works.** 33.8M parameters, ten minutes on one card, and it
  matches mmBERT on topic classification (0.527 vs 0.537). It does *not* match on entity
  recognition (0.698 vs 0.848), so the easy task was flattering it.
- **The study is compute-bound, not data-bound.** More training moves validation loss by 2.2–2.7
  against a measured seed spread of 0.049 — 45–56× the noise. More *text* does nothing measurable
  until the training budget is large enough to use it (0.075 at low compute, 1.5× the spread).
  Convenient, since text is what they've run out of.
- **The factory made the same work 2.07× faster** — 1.33× from batch size, the rest from using the
  second card at all — and the full study fits in 7.6 GPU-hours against a ~20 hour budget.
- **Yoruba is under-served, not hard.** At matched data and compute, a from-scratch Yoruba model
  gains 4.114 nats of context against English's 3.906 — measured after subtracting each corpus's
  unigram entropy, since raw loss across two vocabularies compares nothing. Nothing about the
  language resists modelling.
- **A multilingual vocabulary is not inherently worse.** XLM-R costs English 1.04×, Indonesian
  1.00×, and Mandarin 0.95× — better than a dedicated 16k BPE. It costs Yoruba **1.65×**. The
  penalty appears only where the language is under-represented, which is the group's thesis with
  a control arm attached.
- **The data axis saturates between 64M and 256M tokens** — and all of Yoruba is 69.1M. Past 64M,
  16× more English text buys 0.024 nats, half the seed spread. Checked in Yoruba rather than
  assumed: the two curves agree to within 0.025 nats at the one step both languages can take, so
  the text Yoruba lacks is probably worth about as little. The study cannot rest on "there is too
  little Yoruba text"; it has to rest on tokenizer fit.

## The one that changes the study plan

**Scaling the model up made it worse, at every rung.** The proposal sizes the real study at
AfriBERTa scale (86M) on the assumption that bigger is better once the pipeline works. Run at
matched budgets against the 33.8M model, it lost every time — 2.612 vs 5.315 on the 64M rung.

The curves say why. Both start at 5.65. The smaller model *breaks through* the unigram plateau
at ~40% of training and converges at 2.61; the larger one grinds along the plateau and converges
at 5.32, its last three checkpoints moving 0.003. It is finished, not merely slow. So the larger
model costs 2.5× the compute per token **and** never reaches the regime where a language model
becomes useful.

Getting to that answer took fixing a separate bug first: at 86M the old recipe *collapsed*
outright (loss flat at 6.73 for 69 minutes), because the peak learning rate was too high for
that size and the percentage-based warmup gave short runs far too few steps. Both are fixed and
verified across three seeds. Had the study run on the old settings, every cell would have
produced a dead model and "from-scratch pretraining doesn't work for Yoruba" would have looked
like the answer. Detail in [03-efficiency.md §6d–6e](03-efficiency.md).

**More compute does not rescue it.** The 16M rung re-run at 3× the step budget reached 5.385
against 5.494 at 1× — 0.109 for triple the compute — and converged on the plateau again. The
smaller model reaches 3.008 on that rung with a third of the budget.

**Batch shape was the promising lead, and it is not the answer either.** Quadrupling the batch
to 65,536 tokens per step moved the 16M rung from 5.494 to 5.342 — 0.15 against a gap of 2.3 —
and converged on the plateau just the same. Larger batches help in the expected direction and
by nowhere near enough.

## Two things not to quote yet

- **XLM-R's 0.127 on topic classification** is almost certainly a fine-tuning failure, not a
  coverage result — the same model scores 0.843 on entity recognition. The group's headline
  contrast depends on this number and it needs re-running with more seeds.
- **The AfriBERTa ladder rows in `runs/` are collapsed runs**, kept as evidence for the finding
  above. They are not results and the ladder needs re-running at the corrected settings.
- **The downstream numbers are still unresolved.** Seeds have now been run on the pretraining
  side (sd 0.049), and the pretraining differences clear it easily. But the four models score
  0.448–0.527 on topic classification with overlapping CIs, so *which* grid cell fine-tunes best
  is not yet answerable.

## Reproducing

All three reports are generated from artifacts in `runs/` and `data/`, which the factory writes.
See the parent [README](../README.md) for how to run the pipeline, and `explain_model.py`,
`store_bench.py`, and `results_factory_mlm.ipynb` for the tools that produced the figures.
