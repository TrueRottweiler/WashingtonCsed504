# The next night: three questions about the factory, not about Yoruba

*Written 2026-08-08. The Yoruba science belongs on the top poster — Patrick and Leon's. These are
questions about **how to run experiments**, which is what the bottom poster is about, and all
three are answerable with hardware we already own and checkpoints we already trained.*

> **This is a plan, and the plan has since been executed. Read it as history, not as status.**
> Leon flagged on 12 August that it still reads like a to-do list; it is not one. Study 1 ran and is
> `runs/downstream_correlation.json` — the result is in [report 09](09-the-bottom-report.md), Panel 6
> and the note on the two cut findings. Study 2 ran as the sixty-run grid and is
> `runs/lr_transfer.json`, written up in Panel 8 along with the luck-versus-skill analysis it turned
> into. Study 3 was not run. **For what is true now, go to [report 09](09-the-bottom-report.md),
> [11](11-selecting-on-the-dev-split.md), [12](12-the-bottom-board.md) and
> [13](13-the-top-board.md)** — the four current documents. The counts below (107 checkpoints, 90 GB)
> were right on the 8th and are 197 and larger now.

The starting observation is embarrassing and useful: **we have 107 trained checkpoints on disk,
90 GB of them, and exactly one has ever been fine-tuned.** The scarce resource on this project was
never GPU time. It was that we trained a hundred models and only ever asked one of them whether it
was any good.

---

## Study 1 — Does the number we optimize predict the number we care about?

**Status: built and dry-run clean. `study_downstream_correlation.py`.**

Every pretraining decision this term was made on validation loss. None was ever checked against
downstream usefulness. The field assumes that correlation is tight; we have never looked, and
looking is cheap because the expensive half is already paid for.

| | |
|---|---|
| Checkpoints | **19** Yoruba models at a verified matching vocabulary fingerprint |
| Loss range | 2.253 → 5.670 — nearly the whole span the project produced |
| Cells | 19 models × 2 tasks × 3 seeds = **114 fine-tuning runs** |
| Cost | **~2.3 GPU-hours** at the measured 1.2 min/seed |

```bash
bash src/a2-nlp/py.sh study_downstream_correlation.py --dry-run    # plan and cost, runs nothing
bash src/a2-nlp/py.sh study_downstream_correlation.py --gpu 0      # ~2.3 GPU-hours, resumable
```

**Three design choices worth defending.** Every arm uses the *default* learning rate, because the
question is whether pretraining loss induces the right **rank order**, not what the best
achievable score is — a per-model sweep answers a different question at twelve times the cost.
Both tasks, because report 06 showed they diverge and a correlation that holds on one and not the
other is the more interesting outcome. `reuse=True`, so an interrupted night resumes rather than
re-spending.

**Three checkpoints are excluded and the script says so.** Each carries two result records with
different losses pointing at the *same* directory — the run was repeated and the checkpoint
overwritten while both result files survived. The weights belong to one of those losses and
nothing on disk records which. Pairing a checkpoint with the wrong loss would put a wrong point on
the very curve we are drawing, so an ambiguous tag is dropped rather than guessed at.

**What we expect:** a tight negative correlation. **What would be more interesting:** it breaking
down among the runs that never learned, where every model is bad in the same way while their
losses still differ by nats. Panel 13 gives a reason to suspect exactly that.

---

## Study 2 — Does a tuned hyperparameter transfer to a new language?

**Status: not built. Needs a driver over `mlm_fleet.py`, which already takes `--lr`, `--clip`,
`--tag-prefix` and `--gpu-base`.**

The bottom poster claims *adding a language is one function call*. That is only true if the
settings transfer. We tuned Yoruba and English separately and never compared the two argmins —
so the claim on our own board is untested.

| | |
|---|---|
| Grid | 4 learning rates × 5 languages × 2 seeds, 12,000 steps |
| Runs | 40 |
| Cost | **~4.6 GPU-hours** |
| Languages | pick across the coverage gradient: swh (covered, cheap), hau, yor, ibo, wol (the exception at 1.31) |

**The outcome is binary and both branches are printable.** If the best rate is the same everywhere,
adding a language really is one call and the poster can say so with evidence. If it moves, every
new language costs a sweep and we should put *that* on the board instead. Right now we are
asserting the first without having checked.

**Confound to fix while we are here.** Panel 13 found the plateau ends at 7,200 steps for the
33.8M model and 30,000 for the 98M — but learning rate and model size are perfectly confounded
across all 105 of our runs, so we cannot say which moves it. This grid breaks that confound for
free, since it varies the rate at a fixed size.

---

## Study 3 — Ten more languages, and an honest failure count

**Status: not built. One `prepare_corpus` call per language plus one fleet queue.**

| | |
|---|---|
| Runs | 10 languages × 1 seed at 62,500 steps |
| Cost | **~6 GPU-hours**, plus preparation |

The deliverable is not the models. It is the **count of how many needed manual intervention**, and
what each one needed. A measured failure rate is more persuasive than the claim it tests, and if
the answer is "three of ten needed a code change" then the layering claim needs qualifying and we
should qualify it.

---

## The whole thing fits in one night

| | GPU-hours |
|---|---|
| Study 1 — downstream correlation | 2.3 |
| Study 2 — hyperparameter transfer | 4.6 |
| Study 3 — ten languages | 6.0 |
| Slack for re-runs, which we will need | ~6 |
| **Total** | **~19** ≈ 10 hours of wall-clock on two cards |

Study 1 alone is 2.3 hours and could run this evening.

## What is deliberately not here

**Nothing about Yoruba specifically.** The repetition-ceiling work in `scaling_law.py` — how far
re-reading a small corpus substitutes for having more of it — is a genuinely open question and our
runs already gesture at an answer, but it is a question about a *language's* data constraints and
it belongs upstairs. It is written up in panel 13's sibling analysis and left for Patrick and Leon
to take or leave.

## Order, and why it changed

Study 1 moved to the front after panel 13. That analysis found we cannot tell a doomed run from a
good one early, which means the pretraining loss curve carries less information than we assumed —
and Study 1 asks whether the *final* loss carries as much as we assumed either. Those are the same
doubt at two points in the pipeline, and the second one is cheaper to resolve.
