# What's generic, what's specific

*A map of the factory across a1-cv and a2-nlp. Every claim here was derived from the import graph
and line counts, not from memory — the measurements are at the bottom.*

The natural way to sketch this is by project: images over here, text over there, shared tools in
the middle. That turns out to be the wrong axis. Two files in different projects can be near
copies of each other, and two files in the *same* folder can share nothing.

The axis that actually predicts what can be reused is **what a component has to know**. A
scheduler that hands work to GPUs doesn't need to know whether the work is images or text. A
masking function can't exist without knowing it's doing masked language modeling. Sorting by
that gives four layers, and a component belongs to the highest layer whose vocabulary it uses.

---

## The tree

```
0. KNOWS ONLY ABOUT THE MACHINE
   └── src/common/gpu_check.py .................. device detection, multi-GPU helpers  [636 lines]
       The only genuinely shared module. Consumed mostly by NOTEBOOKS
       (hello_text, csed503_to_pytorch, cifar100_train, report_factory_performance)
       rather than by the training scripts -- of the .py files, only
       a1-cv/perf/collect.py imports it. The factory scripts do their own
       torch.cuda setup instead, which is a small, real duplication.

1. KNOWS ABOUT TRAINING, NOT ABOUT WHAT IS BEING TRAINED
   ├── the fleet scheduler ...................... queue of specs, one process per card, refill on exit
   │     a1-cv/train_fleet.py · a2-nlp/train_fleet.py · a2-nlp/mlm_fleet.py
   ├── the run supervisor ....................... one process, one GPU, one cell; owns its log
   │     a1-cv/train_run.py · a2-nlp/train_run.py · a2-nlp/mlm_run.py
   ├── the dashboards ........................... read runs/ and logs/, render progress
   │     a1-cv/dashboard.py · a2-nlp/dashboard.py · a2-nlp/webboard.py
   └── checkpoint / resume / stall detection ..... inside each train loop
       ── Nothing at this layer needs to know images from tokens. All of it is
          duplicated three times anyway. See "The honest part" below.

2. KNOWS ABOUT THE MODALITY (pixels vs tokens), NOT THE OBJECTIVE
   ├── IMAGES
   │   ├── a1-cv/imagenet_prepare.py ............ raw images  -> flat arrays
   │   ├── a1-cv/imagenet_data.py ............... GPU-resident image stream        [428]
   │   ├── a1-cv/cifar_data.py .................. the same, for CIFAR              [343]
   │   └── a1-cv/subsets.py ..................... which examples are in a data rung
   └── TOKENS
       ├── a2-nlp/text_prepare.py ............... corpus -> tokenizer -> flat uint16/int32  [275]
       └── a2-nlp/text_data.py .................. GPU-resident token stream, int16/int32    [162]
           ── Zero references to masking in either file. This is the layer the
              masked-LM study was built ON, not a part of it.

3. KNOWS THE OBJECTIVE (what the loss is)
   ├── CLASSIFICATION (a1) ...................... a1-cv/models.py, a1-cv/train_loop.py
   │     resnet18 / vit / resnet50 / vit_base, top-1 accuracy
   ├── CAUSAL LM (a2, the LSTM-vs-GPT study) .... a2-nlp/models.py, a2-nlp/train_loop.py
   │     lstm / gpt / lstm_large / gpt_medium, held-out perplexity
   └── MASKED LM (a2, the group's Yoruba study)
       ├── a2-nlp/mlm_data.py ................... corpus prep + the 80/10/10 masking   [595]
       ├── a2-nlp/mlm_train.py .................. the loop, LR presets, checkpointing  [477]
       └── a2-nlp/models.py is NOT used here — MLM builds RoBERTa through
           HuggingFace AutoModelForMaskedLM instead of the hand-written builders.

4. KNOWS THE STUDY (this experiment, these questions)
   ├── a2-nlp/mlm_api.py ........................ the nine-function surface Patrick and Leon call
   ├── a2-nlp/mlm_fleet.py QUEUES ............... poc / full / seeds / engladder
   ├── a2-nlp/prepare_eng_ladder.py ............. builds one specific 1.1B-token corpus
   ├── a2-nlp/audit_corpus.py, diagnose.py, explain_model.py, store_bench.py
   └── a2-nlp/reports/ .......................... the findings
```

---

## Answering the question directly

**How much of the factory is masked-LM only?** Measured by the import graph: the files carrying
masked-LM vocabulary are `mlm_data.py`, `mlm_train.py`, `mlm_run.py`, `mlm_fleet.py`,
`mlm_api.py` — about **1,720 lines**. Everything they stand on is not MLM-specific.

The clearest evidence is `text_data.py` and `text_prepare.py`: **zero** occurrences of "mask"
between them. They were written for the causal LM study and the masked-LM study imports them
unchanged. Swapping the objective did not require touching the modality layer, which is what a
layer boundary is supposed to buy you.

The second piece of evidence is the dashboards. `webboard.py` is 1,279 lines and imports exactly
**one** name from `mlm_train`: `cell_tag`, which turns a run's parameters into a filename. That's
a naming convention, not an objective. The dashboard is generic infrastructure wearing a thin
MLM-shaped hat.

---

## The honest part: layer 1 is duplicated, not shared

Layer 1 needs to know nothing about images or tokens. It is nonetheless copied three times, and
the copies have drifted:

| file | a1-cv | a2-nlp | lines identical |
|---|---|---|---|
| `dashboard.py` | 513 | 492 | **283** (55%) |
| `train_fleet.py` | 327 | 186 | 114 |
| `train_loop.py` | 312 | 180 | 112 |
| `train_run.py` | 366 | 194 | 88 |
| `models.py` | 219 | 225 | 18 (genuinely different) |

`src/common/` contains `gpu_check.py` and its test, and nothing else. Everything that *could* live
there doesn't.

This was a deliberate call at the time and the a2-nlp README says so — a1-cv was finished and
working, and editing it to extract shared code risked breaking a graded deliverable to serve a
different assignment. That trade was probably right then. It is worth naming now because the cost
is real and compounding:

- Three fleet schedulers means a scheduling fix lands in one and not the others. The
  longest-first ordering was learned in a1-cv and had to be re-applied to `mlm_fleet.py`.
- The queue-plan panel, the stale-page banner, and the per-cell grouping fixes all went into
  `webboard.py` only. `a2-nlp/dashboard.py` and `a1-cv/dashboard.py` still have the old behavior.
- `models.py` sharing 18 lines is *correct* — a ResNet builder and an LSTM builder have nothing
  in common. That row is not duplication and should not be "fixed".

**What extraction would actually be worth it?** Only two things, both at layer 1:

1. **The fleet scheduler.** The queue, the per-card slots, the longest-first ordering, and the
   plan file are identical in concept across all three. The only thing that varies is the command
   line it builds for a child process, which is one function.
2. **The dashboard's run discovery.** Reading `runs/*_result.json` and `logs/*.log` into a list of
   run records is the same problem regardless of what trained. Rendering differs; discovery does
   not.

Everything else is either genuinely modality-specific or small enough that sharing it would cost
more in indirection than it saves.

---

## What this means for adding a fifth thing

The layering predicts the work. To add:

- **a new language** — layer 4 only. One call to `prepare_corpus`. This is why five languages took
  an afternoon.
- **a new objective on text** (say, next-sentence prediction, or a seq2seq task) — a new layer-3
  pair alongside `mlm_data`/`mlm_train`, reusing `text_data`/`text_prepare` and the whole of layer
  1. This is the path the masked-LM study itself took.
- **a new modality** (audio, say) — a new layer-2 pair, and layer 1 would need to be shared for
  real rather than copied a fourth time.

---

## How this was derived

`ast.parse` over every `.py` in `src/common`, `src/a1-cv`, `src/a2-nlp` to extract the local
import edges, plus regex counts of objective-specific vocabulary (`mlm|mask_id|mlm_prob`,
`tokenizer|bpe|vocab_size`, `image|cifar|imagenet|pixel|top-1`) per file. Duplication measured
with `diff` on comment-stripped files. Re-run it against a checkout if these numbers drift.

Two things the method cannot see, so treat them as unmeasured rather than absent: notebooks
(`*.ipynb`) are excluded, and a file that merely *mentions* a marker word in a comment counts the
same as one that depends on it. The `text_data.py` result — zero masking references — is strong
precisely because it is a zero.
