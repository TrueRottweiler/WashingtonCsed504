"""Tests for study_label_quantity -- the decision rule, the band arithmetic, and the guards.

WHY THIS FILE EXISTS. The study's output is a VERDICT, not a number: `reading()` maps a measured
band onto one of three conclusions about the poster. A study that prints a conclusion needs its
conclusion tested, because the failure mode is not a crash or an obviously silly figure -- it is a
defensible-looking paragraph that says the opposite of what the data supports. That has now happened
twice in this file's short life:

  * The first version of `reading()` had the branches in the wrong order, so a synthetic set where
    nothing moved read as LABEL QUANTITY. Caught before the experiment ran.
  * The set the band is measured over grew from 16 models to 21 when a downstream sweep elsewhere
    in the project gave four tokenizer-swap checkpoints their missing full-split rows. With no
    change to the study at all, the band went 0.069 -> 0.182 and the verdict went BETWEEN THE TWO
    -> LABEL QUANTITY. Caught by re-running the script against merged records rather than by
    reading its output from an email.

Both are pinned below. The second is the one to keep: a set that is *computed* from a shared
directory rather than hardcoded still is not pinned, because it is a query against everyone else's
work.

NO GPU AND NO TRAINING. Everything here is arithmetic over records already on disk. On a machine
with torch this drives the real `ft_api`/`mlm_api`; on a laptop without it, faithful stubs read the
same `runs/*.json` by the same rules. Both paths are exercised in CI terms by whoever runs it --
`python -m unittest test_label_quantity -v` works in either.

The real-data assertions are deliberately pinned to values recomputed on 11 August, and the numbers
they pin are NOT the ones the script's docstring originally claimed. It said the band reproduces
"0.0441 over sixteen models"; over the seventeen that now carry a full-split row on the band's
vocabulary it is 0.0441 over seventeen, and the published sixteen remain a subset. Updating that
deliberately, with the old value recorded here, is the point of the exercise.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


# --------------------------------------------------------------------------------------------
# Import the study, with ft_api/mlm_api stubbed only if the real ones cannot load.
#
# Preferring the real modules matters: a stub that drifts from ft_api.results() would let this file
# keep passing while the study read something else. The stubs replicate exactly two behaviours --
# the test/dev split filter and the meta-under-result merge -- because those are the two that
# decide which rows the band sees.
# --------------------------------------------------------------------------------------------
def _stub_ft_results(pattern='*', task=None, lang=None, eval_split='test'):
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, f'ft_{pattern}_ft.json'))):
        try:
            with open(p, encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if task and rec.get('task') != task:
            continue
        if lang and rec.get('lang') != lang:
            continue
        # Dev-scored cells are not reportable numbers; ft_api hides them by default and so must
        # anything standing in for it.
        if eval_split and rec.get('eval_split', 'test') != eval_split:
            continue
        rows.append(rec)
    return sorted(rows, key=lambda r: (r['task'], r['lang'], -r['mean']))


def _stub_mlm_results(pattern='*', include_smoke=False):
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, f'{pattern}_result.json'))):
        if not include_smoke and os.path.basename(p).startswith('smoke-'):
            continue
        try:
            with open(p, encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if 'steps' not in rec:          # a causal run from the Part 1 study; not this grid
            continue
        meta_path = p[:-len('_result.json')] + '_meta.json'
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as fh:
                meta = json.load(fh)
        merged = {**meta, **rec}        # the result wins on conflict, as in mlm_api
        merged.pop('history', None)
        rows.append(merged)
    return rows


try:
    import torch                                            # noqa: F401
    STUBBED = False
except ImportError:
    STUBBED = True
    _ft = types.ModuleType('ft_api')
    _ft.results = _stub_ft_results
    _ft.record_tag = lambda *a, **k: 'tag'

    def _no_training(*a, **k):
        raise AssertionError('ft_api.evaluate must never be called from a test')
    _ft.evaluate = _no_training
    sys.modules.setdefault('ft_api', _ft)

    _mlm = types.ModuleType('mlm_api')
    _mlm.results = _stub_mlm_results
    sys.modules.setdefault('mlm_api', _mlm)

sys.path.insert(0, HERE)
import study_label_quantity as S                            # noqa: E402


def cell(mean, sd=0.005, slug=None, n=701):
    """One band row, in the shape `spread()` and the report consume."""
    return {'kind': 'band', 'model_slug': slug or f'm{mean:.4f}', 'n_train_requested': n,
            'mean': mean, 'sd': sd, 'seeds': [0, 1, 2]}


class DecisionRule(unittest.TestCase):
    """`reading()` is the study's output. These are its three branches and its two known bugs."""

    def test_unchanged_band_reads_task_type(self):
        """The regression that started this file.

        A set whose baseline spread is already wide satisfies the "as wide as SIB-200" test while
        showing no change at all. The first version tested that branch first and called this LABEL
        QUANTITY. "Did it move" must be asked before "is it big".
        """
        base = {'range': 0.14, 'between_sd': 0.045, 'n_models': 16}
        at = {'range': 0.14, 'between_sd': 0.045, 'within_sd': 0.005, 'n_models': 16}
        self.assertIn('TASK TYPE', ' '.join(S.reading(at, base)))

    def test_band_matching_sib200_reads_label_quantity(self):
        base = {'range': 0.044, 'between_sd': 0.013, 'n_models': 16}
        at = {'range': 0.140, 'between_sd': 0.045, 'within_sd': 0.005, 'n_models': 16}
        self.assertIn('LABEL QUANTITY', ' '.join(S.reading(at, base)))

    def test_partial_widening_reads_between_the_two(self):
        base = {'range': 0.044, 'between_sd': 0.0129, 'n_models': 16}
        at = {'range': 0.069, 'between_sd': 0.0195, 'within_sd': 0.005, 'n_models': 16}
        self.assertIn('BETWEEN THE TWO', ' '.join(S.reading(at, base)))

    def test_verdict_does_not_depend_on_how_many_models_are_in_the_set(self):
        """Trap 4, as a rule about the rule.

        A range grows with the number of models in it, so a decision rule that compares a range
        against SIB-200's fixed 0.1426 changes its answer when the set size changes -- which is the
        trap the study's own docstring opens with, committed by its own decision rule. Doubling the
        set inflates the range while leaving the between-model sd alone; the verdict must not move.
        """
        base = {'range': 0.044, 'between_sd': 0.0129, 'n_models': 16}
        small = {'range': 0.069, 'between_sd': 0.0195, 'within_sd': 0.005, 'n_models': 16}
        # Same population, more draws: the range stretches, the sd does not.
        large = {'range': 0.115, 'between_sd': 0.0195, 'within_sd': 0.005, 'n_models': 32}
        self.assertEqual(S.reading(small, base)[0], S.reading(large, base)[0])
        self.assertIn('BETWEEN THE TWO', S.reading(large, base)[0])


class Spread(unittest.TestCase):
    """The band arithmetic, including the sample-vs-population correction."""

    def test_range_and_between_sd(self):
        s = S.spread([cell(0.70), cell(0.75), cell(0.80)])
        self.assertAlmostEqual(s['range'], 0.10, places=9)
        self.assertAlmostEqual(s['between_sd'], 0.05, places=9)
        self.assertEqual(s['n_models'], 3)
        self.assertAlmostEqual(s['lo'], 0.70, places=9)
        self.assertAlmostEqual(s['hi'], 0.80, places=9)

    def test_within_sd_is_converted_to_a_sample_sd(self):
        """The sd stored on a record is a population sd over three seeds, while every "N times the
        spread" rule in this project was derived for a sample sd. At three seeds that is a 22% gap,
        and it is applied in the direction that makes the rule harder to pass, not easier."""
        s = S.spread([cell(0.70, sd=0.01), cell(0.80, sd=0.01)])
        self.assertAlmostEqual(s['within_sd'], 0.01 * (3 / 2) ** 0.5, places=9)
        self.assertGreater(s['within_sd'], 0.01)

    def test_a_single_model_has_no_band(self):
        s = S.spread([cell(0.70)])
        self.assertEqual(s['range'], 0.0)
        self.assertEqual(s['between_sd'], 0.0)


class VocabularyGuard(unittest.TestCase):
    """Trap 2b/4: the band is a spread over models that differ in PRETRAINING, so a model scored
    over a different vocabulary is a different experiment, not another point on the axis."""

    def setUp(self):
        self.band, self.dropped = S.band_models()
        self.slugs = {s for s, _, _ in self.band}

    def test_the_xlmr_vocabulary_swap_arm_is_excluded_by_name(self):
        dropped = {s for s, _ in self.dropped}
        self.assertEqual(dropped, {f'swap-yor-xlmr-121.3M-12k-s{i}' for i in range(4)},
                         'the 250k-vocabulary arm must be dropped, and only it')
        self.assertFalse(self.slugs & dropped)

    def test_the_dropped_models_pass_the_val_loss_cut_they_are_not_being_caught_by(self):
        """The guard is not redundant with the trap-2 cut, and this is why: those four models sail
        through `val_loss < 3.1` at 1.24-1.70. Fewer nats per token because there are more tokens
        -- not better training. If someone ever deletes the vocabulary guard believing the loss cut
        already covers it, this fails."""
        index = S.pretrain_index()
        for slug, _ in self.dropped:
            tag = slug.replace('-', '_')
            rec = index.get(tag)
            self.assertIsNotNone(rec, f'no pretraining record for {tag}')
            self.assertLess(rec['val_loss'], S.TRAINED_MAX_LOSS,
                            f'{tag} was expected to pass the loss cut and be caught by vocabulary')

    def test_every_band_model_shares_one_vocabulary_fingerprint(self):
        index = S.pretrain_index()
        fps = {(index.get(os.path.basename(p)) or {}).get('vocab_fingerprint')
               for _, p, _ in self.band}
        self.assertEqual(len(fps - {None}), 1, f'band spans several vocabularies: {fps}')

    def test_the_guard_reports_rather_than_silently_dropping(self):
        for slug, why in self.dropped:
            self.assertIn('vocabulary', why, 'the reason must name the vocabulary')


class RealRecords(unittest.TestCase):
    """The bands, against the records actually in the repository.

    These are the numbers the poster panel quotes. Pinning them here means that if the records
    change under the study -- which is exactly what happened between the run and the merge -- the
    test says so, instead of the report quietly measuring something else.
    """

    @classmethod
    def setUpClass(cls):
        path = os.path.join(RUNS, 'label_quantity.json')
        if not os.path.exists(path):
            raise unittest.SkipTest('runs/label_quantity.json not in this checkout')
        with open(path, encoding='utf-8') as fh:
            cls.rows = [r for r in json.load(fh) if 'mean' in r]
        band, _ = S.band_models()
        cls.trained = {s for s, _, t in band if t}

    def level(self, n):
        return S.spread([r for r in self.rows if r.get('kind') == 'band'
                         and r['n_train_requested'] == n and r['model_slug'] in self.trained])

    def test_full_split_band_reproduces(self):
        """Was documented as 0.0441 over SIXTEEN. It is 0.0441 over the seventeen that now carry a
        full-split row on the band's vocabulary -- the seventeenth is a fourth swap seed that lands
        inside the existing range, so the range is unchanged and the sd moves in the fourth decimal.
        Both are recorded because "the number did not move when the set grew" is only reassuring if
        somebody checked."""
        base = S.full_data_band(set(self.level(701)['models']))
        self.assertEqual(base['n_models'], 17)
        self.assertAlmostEqual(base['range'], 0.0441, places=4)
        self.assertAlmostEqual(base['between_sd'], 0.0126, places=4)

    def test_701_band_and_verdict(self):
        at = self.level(701)
        base = S.full_data_band(set(at['models']))
        self.assertEqual(at['n_models'], base['n_models'])
        self.assertAlmostEqual(at['range'], 0.0691, places=4)
        self.assertAlmostEqual(at['between_sd'], 0.0190, places=4)
        self.assertAlmostEqual(at['between_sd'] / base['between_sd'], 1.51, places=2)
        self.assertIn('BETWEEN THE TWO', S.reading(at, base)[0])

    def test_2000_band_shows_no_widening_at_all(self):
        """The dose-response, and the sharper half of the result: cutting the labels threefold does
        nothing (x0.99 on between-model sd), while cutting them tenfold moves it x1.51. Whatever
        happens between 2,000 and 701 does not happen between 6,876 and 2,000, which rules out the
        smooth version of the label-quantity story."""
        at = self.level(2000)
        base = S.full_data_band(set(at['models']))
        self.assertEqual(at['n_models'], 16)
        self.assertAlmostEqual(at['range'], 0.0559, places=4)
        self.assertAlmostEqual(at['between_sd'], 0.0127, places=4)
        self.assertAlmostEqual(at['between_sd'] / base['between_sd'], 0.99, places=2)
        self.assertIn('TASK TYPE', S.reading(at, base)[0])

    def test_including_the_wrong_vocabulary_would_reverse_the_verdict(self):
        """The regression, stated as the damage it does rather than as a set membership.

        This is what the script printed on `main` before the guard: a band of 0.182 and a verdict of
        LABEL QUANTITY, from records landing rather than from anything being measured.
        """
        rows = [r for r in self.rows if r.get('kind') == 'band' and r['n_train_requested'] == 701]
        at = S.spread(rows)
        self.assertEqual(at['n_models'], 21)
        self.assertAlmostEqual(at['range'], 0.1821, places=4)
        base = S.full_data_band(set(at['models']))
        self.assertIn('LABEL QUANTITY', S.reading(at, base)[0])

    def test_the_baseline_is_recomputed_per_level_not_over_the_union(self):
        """Trap 1's own remedy, applied where it was originally missed. The 701 and 2,000 levels
        cover different models, so a baseline computed once over their union is comparable with
        neither -- and comparing against the union is what suppressed the 2,000-label verdict."""
        s701, s2000 = set(self.level(701)['models']), set(self.level(2000)['models'])
        self.assertNotEqual(s701, s2000, 'levels expected to differ; the trap is moot if they do not')
        union = S.full_data_band(s701 | s2000)
        self.assertNotEqual(S.full_data_band(s2000)['n_models'], union['n_models'])

    def test_the_context_arms_carry_their_own_rates(self):
        """Trap 3: the floor runs at its swept best of 3e-4, not the band's 3e-5. Quoting it at the
        band's rate reproduces the 0.414 error the sweep had just corrected."""
        ctx = {r['label']: r for r in self.rows if r.get('kind') == 'context'}
        floors = [r for k, r in ctx.items() if 'floor' in k]
        self.assertTrue(floors, 'no floor arm in the results')
        for r in floors:
            self.assertEqual(r['lr'], S.FLOOR_LR)
            self.assertNotEqual(r['lr'], S.BAND_LR)


if __name__ == '__main__':
    print(f'ft_api/mlm_api: {"stubbed (no torch)" if STUBBED else "real"}')
    unittest.main(verbosity=2)
