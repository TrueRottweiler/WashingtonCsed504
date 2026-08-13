"""Every number the top board prints, pinned to the records it came from.

WHAT THIS IS FOR, and why it is not claims_audit.py. That gate states the null for each comparative
CLAIM and computes what would refute it -- whether the finding survives. This file asks a smaller
and more boring question: does the number written in the report still come out of `runs/`? Those
fail differently. A claim can stay true while the figure beside it goes stale, and this project has
watched exactly that happen: 0.4140 sat in a summary table for a fortnight after the sweep that
replaced it, and 42% was carried from a seventeen-model set into a panel whose own table is sixteen.

It is deliberately a TEST rather than a script, so it runs in the same breath as the others and
fails loudly. No GPU, no training, no network, no matplotlib, no scipy: real `ft_api`/`mlm_api`
where torch exists, and the faithful record-reading stubs from `test_label_quantity` where it does
not, so this runs on a laptop.

WHAT IT DOES NOT COVER. Only numbers derived from records. Whether the PROSE around a number is
still accurate is not checkable here -- the second stale sentence found under the 0.414 fix said
"the NER control is still a single cell", which no numeric assertion would have caught. That is the
staleness pass's job, and a person's.

    bash src/a2-nlp/py.sh -m unittest test_board_numbers -v
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
sys.path.insert(0, HERE)

# Side effect on purpose: installs the ft_api/mlm_api stubs, and ONLY if the real ones cannot load.
# Importing them from there rather than copying them is the point -- two copies of a stub is how a
# test keeps passing while the code under it reads something else.
import test_label_quantity as _records                      # noqa: F401,E402

import ft_api                                               # noqa: E402  (stub or real)
import mlm_api                                              # noqa: E402  (stub or real)
import study_label_quantity as S                            # noqa: E402

TASK_NER, TASK_SIB, LANG_NER, LANG_SIB = 'masakhaner', 'sib200', 'yor', 'yor_Latn'
NER_STEPS, SIB_STEPS = 2150, 1056

# Geography is not on the records -- gradient_table.json has no region field -- so the African-only
# control of report 07 section 4 needs its set written down. poster_figures.fig_gradient holds the
# same two sets for its caption, and this is a DELIBERATE duplication: that module imports
# matplotlib, which is not installable on every machine that should be able to run this test. Both
# copies assert they partition the table, so a language added to the study fails in both places
# rather than being silently counted as African in one of them.
AFRICAN = {'afr', 'swh', 'som', 'hau', 'amh', 'xho', 'wol', 'lug', 'nya', 'sna', 'kin',
           'yor', 'ibo'}
ELSEWHERE = {'cmn', 'ind', 'eng', 'fra'}


def load(name):
    with open(os.path.join(RUNS, name), encoding='utf-8') as fh:
        return json.load(fh)


def dev_selected(slug, task, lang, steps):
    """The cell the board reports: rate chosen on the dev split, score read off test.

    The same rule fig_headline uses, and matched on model_slug rather than a path substring, for
    the reasons written there.
    """
    test = [r for r in ft_api.results('*', task=task, lang=lang)
            if r['model_slug'] == slug and r.get('steps') == steps]
    dev = [r for r in ft_api.results('*', task=task, lang=lang, eval_split='validation')
           if r['model_slug'] == slug and r.get('steps') == steps]
    if not test:
        return None
    if dev:
        lr = max(dev, key=lambda r: r['mean'])['lr']
        hit = [r for r in test if r['lr'] == lr]
        if hit:
            return hit[0]
    return max(test, key=lambda r: r['mean'])


# ------------------------------------------------------------------------------------------
class TokenizerGradient(unittest.TestCase):
    """Panel 3, and the thesis block quote. Figure 02 draws this file."""

    def setUp(self):
        # eng_1b is English at a second corpus size, NOT a second language. Left in, the covered
        # mean is 1.140 over "11 covered languages" and anybody recomputing the board's lead
        # number concludes the report is wrong.
        self.rows = [r for r in load('gradient_table.json') if r['corpus'] != 'eng_1b']
        self.covered = [r for r in self.rows if r['in_xlmr'] is True]
        self.uncovered = [r for r in self.rows if r['in_xlmr'] is not True]

    def mean(self, group):
        return st.mean(r['penalty'] for r in group)

    def test_the_language_set_is_classified(self):
        self.assertEqual({r['corpus'] for r in self.rows}, AFRICAN | ELSEWHERE)

    def test_eng_1b_is_a_corpus_not_a_language(self):
        self.assertEqual(len(self.rows), 17)
        self.assertEqual((len(self.covered), len(self.uncovered)), (10, 7))

    def test_penalty_separates_by_coverage(self):
        self.assertAlmostEqual(self.mean(self.covered), 1.150, places=3)
        self.assertAlmostEqual(self.mean(self.uncovered), 1.593, places=3)

    def test_african_only_control(self):
        """Rules out script and region as the explanation."""
        self.assertAlmostEqual(self.mean([r for r in self.covered
                                          if r['corpus'] in AFRICAN]), 1.244, places=3)
        self.assertAlmostEqual(self.mean([r for r in self.uncovered
                                          if r['corpus'] in AFRICAN]), 1.593, places=3)

    def test_the_exception_is_wolof_and_it_is_not_smoothed(self):
        """A gradient with one exception, not a clean split. The figure title asserted a clean
        split until 12 August; this pins the shape the reports actually describe."""
        top_covered = max(self.covered, key=lambda r: r['penalty'])
        self.assertEqual(top_covered['corpus'], 'xho')
        below = [r for r in self.uncovered if r['penalty'] < top_covered['penalty']]
        self.assertEqual([r['corpus'] for r in below], ['wol'])
        self.assertEqual(len(self.uncovered) - len(below), 6)

    def test_yoruba_is_second_highest(self):
        order = sorted(self.rows, key=lambda r: -r['penalty'])
        self.assertEqual([r['corpus'] for r in order[:2]], ['ibo', 'yor'])
        self.assertAlmostEqual(next(r['penalty'] for r in self.rows
                                    if r['corpus'] == 'yor'), 1.76, places=2)


# ------------------------------------------------------------------------------------------
class Sib200Table(unittest.TestCase):
    """Panel 5 and figure 01. Every arm dev-selected, scored on 204 test items."""

    EXPECTED = {'yor-64M-62.5k-s0': 0.688, 'mmBERT-base': 0.582,
                'yor-random-init': 0.429, 'xlm-roberta-base-random-init': 0.382,
                'xlm-roberta-base': 0.358}

    def test_every_arm_reproduces(self):
        for slug, expected in self.EXPECTED.items():
            with self.subTest(slug=slug):
                rec = dev_selected(slug, TASK_SIB, LANG_SIB, SIB_STEPS)
                self.assertIsNotNone(rec, f'no cells for {slug}')
                self.assertAlmostEqual(rec['mean'], expected, places=3)

    def test_our_margin_over_mmbert(self):
        ours = dev_selected('yor-64M-62.5k-s0', TASK_SIB, LANG_SIB, SIB_STEPS)['mean']
        theirs = dev_selected('mmBERT-base', TASK_SIB, LANG_SIB, SIB_STEPS)['mean']
        self.assertAlmostEqual(ours - theirs, 0.106, places=3)

    def test_xlmr_lands_below_its_own_untrained_control(self):
        """The board's sharpest negative result, and it reversed sign once when both arms were
        selected the same way. Pinned so it cannot reverse back unnoticed."""
        xlmr = dev_selected('xlm-roberta-base', TASK_SIB, LANG_SIB, SIB_STEPS)['mean']
        control = dev_selected('xlm-roberta-base-random-init', TASK_SIB, LANG_SIB,
                               SIB_STEPS)['mean']
        self.assertLess(xlmr, control)
        self.assertAlmostEqual(xlmr - control, -0.024, places=3)

    def test_xlmr_is_a_mixture_not_a_mean(self):
        """Four seeds trained, one collapsed below chance. A cell whose mean describes no run."""
        rec = dev_selected('xlm-roberta-base', TASK_SIB, LANG_SIB, SIB_STEPS)
        scores, chance = rec['scores'], rec.get('chance') or 1 / 7
        self.assertEqual(len(scores), 5)
        self.assertEqual(sum(1 for s in scores if s < chance), 1)


# ------------------------------------------------------------------------------------------
class MasakhanerTable(unittest.TestCase):
    """Panels 7 and 8: three ceilings and the floor that has to be drawn beside them."""

    def full_split(self, slug):
        return [r for r in ft_api.results('*', task=TASK_NER, lang=LANG_NER)
                if r['model_slug'] == slug and r.get('steps') == NER_STEPS
                and r.get('n_train_requested') is None]

    def test_ceilings(self):
        for slug, expected in (('mmBERT-base', 0.8628), ('xlm-roberta-base', 0.8513),
                               ('yor-64M-62.5k-s0', 0.8373)):
            with self.subTest(slug=slug):
                cells = self.full_split(slug)
                self.assertAlmostEqual(max(r['mean'] for r in cells), expected, places=4)

    def test_the_floor_is_the_swept_best_not_the_3e5_cell(self):
        """0.4140 was this sweep's 3e-5 cell -- a rate one tenth of its best -- and it sat in the
        reports for a fortnight. The n_train_requested filter is load-bearing: the labelled-data
        study runs the same control at 701 and 2,000 labels."""
        cells = self.full_split('yor-random-init')
        best = max(cells, key=lambda r: r['mean'])
        self.assertAlmostEqual(best['mean'], 0.6261, places=4)
        self.assertAlmostEqual(best['lr'], 3e-4)
        self.assertGreaterEqual(len(cells), 12)

    def test_the_two_baselines_do_not_separate_on_ner(self):
        mm = max(r['mean'] for r in self.full_split('mmBERT-base'))
        xl = max(r['mean'] for r in self.full_split('xlm-roberta-base'))
        self.assertLess(abs(mm - xl), 0.06)          # the project's own resolution floor


# ------------------------------------------------------------------------------------------
class RawBands(unittest.TestCase):
    """Panel 8. The raw bands replaced a normalised version that the swept floor broke twice."""

    def band(self, task):
        y = [r['mean'] for r in load('downstream_correlation.json')
             if r['task'] == task and r['val_loss'] < 3.1]
        return max(y) - min(y), len(y)

    def test_ner_band_over_sixteen(self):
        band, n = self.band(TASK_NER)
        self.assertEqual(n, 16)
        self.assertAlmostEqual(band, 0.0441, places=4)

    def test_sib_band_over_the_same_sixteen(self):
        band, n = self.band(TASK_SIB)
        self.assertEqual(n, 16)
        self.assertAlmostEqual(band, 0.1426, places=4)

    def test_topic_varies_about_three_times_as_much(self):
        self.assertAlmostEqual(self.band(TASK_SIB)[0] / self.band(TASK_NER)[0], 3.2, places=1)


# ------------------------------------------------------------------------------------------
class TokenizerSwap(unittest.TestCase):
    """Panel 10. The project's only causal downstream evidence for the tokenizer argument."""

    def arms(self, stage):
        rows = load('swap_downstream.json')
        return ([r['mean'] for r in rows if r.get('stage') == stage
                 and r['arm'] == 'our vocabulary'],
                [r['mean'] for r in rows if r.get('stage') == stage
                 and r['arm'] == "XLM-R's vocabulary"])

    def test_gaps(self):
        for stage, expected in (('test', 0.1439), ('ner', 0.0606)):
            with self.subTest(stage=stage):
                ours, theirs = self.arms(stage)
                self.assertEqual((len(ours), len(theirs)), (4, 4))
                self.assertAlmostEqual(st.mean(ours) - st.mean(theirs), expected, places=4)

    def test_every_seed_of_ours_beats_every_seed_of_theirs(self):
        for stage in ('test', 'ner'):
            with self.subTest(stage=stage):
                ours, theirs = self.arms(stage)
                self.assertGreater(min(ours), max(theirs))

    def test_four_seeds_a_side_because_three_cannot_reach_p_005(self):
        """2/C(8,4) = 0.0286 is the floor of the exact test, which is where both tasks sit. At
        three a side the floor is 0.10 and no separation can clear it."""
        ours, theirs = self.arms('test')
        self.assertEqual(len(ours) + len(theirs), 8)
        self.assertAlmostEqual(2 / math.comb(8, 4), 0.0286, places=4)


# ------------------------------------------------------------------------------------------
class LabelQuantity(unittest.TestCase):
    """Panel 9 and figure 18. test_label_quantity pins the VERDICTS; this pins the dose-response
    the panel's table prints, which is over the matched sixteen rather than the seventeen."""

    def setUp(self):
        band_all, _dropped = S.band_models()
        trained = {slug for slug, _, ok in band_all if ok}
        self.levels = {}
        for r in load('label_quantity.json'):
            if r.get('kind') == 'band' and 'mean' in r and r['model_slug'] in trained:
                self.levels.setdefault(r['n_train_requested'], {})[r['model_slug']] = r
        self.matched = set.intersection(*(set(v) for v in self.levels.values()))

    def test_the_matched_set_is_sixteen(self):
        self.assertEqual(len(self.matched), 16)
        self.assertEqual(sorted(self.levels), [701, 2000])

    def test_dose_response_over_the_matched_sixteen(self):
        base = S.full_data_band(self.matched)
        self.assertAlmostEqual(base['between_sd'], 0.0129, places=4)
        for n, expected in ((2000, 0.0127), (701, 0.0195)):
            with self.subTest(labels=n):
                at = S.spread([self.levels[n][slug] for slug in self.matched])
                self.assertAlmostEqual(at['between_sd'], expected, places=4)

    def test_threefold_cut_does_nothing_and_tenfold_moves_it_half_way(self):
        base = S.full_data_band(self.matched)['between_sd']
        at2000 = S.spread([self.levels[2000][s] for s in self.matched])['between_sd']
        at701 = S.spread([self.levels[701][s] for s in self.matched])['between_sd']
        self.assertAlmostEqual(at2000 / base, 0.99, places=2)
        self.assertAlmostEqual(at701 / base, 1.51, places=2)

    def test_43_percent_and_42_percent_are_both_right(self):
        """The panel says 43% (matched sixteen) and the study's console output says 42%
        (seventeen). Whichever a future reader meets first, they must not "fix" it into the other:
        a percentage carried between two set sizes is the mistake panel 9 is about."""
        matched = S.spread([self.levels[701][s] for s in self.matched])['between_sd']
        seventeen = S.spread(list(self.levels[701].values()))['between_sd']
        self.assertEqual(len(self.levels[701]), 17)
        self.assertEqual(round(100 * matched / S.SIB_BETWEEN_SD), 43)
        self.assertEqual(round(100 * seventeen / S.SIB_BETWEEN_SD), 42)


class ResidualPermutation(unittest.TestCase):
    """Panel 11's bottom row, which lived in email for two days and got two different wrong values.

    Pinned here because prose was the only place it existed -- the rule this project keeps is that
    the repository carries numbers and email carries reasoning, and this one broke it in the panel
    whose own subject is naming your test.

    Needs matplotlib, which `residual_permutation` sits beside but does not use; skipped rather
    than failed where it is absent, so this file still runs on a laptop.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from poster_figures import residual_permutation
        except ImportError as exc:                       # matplotlib, on a machine without it
            raise unittest.SkipTest(f'poster_figures unavailable: {exc}')
        cls.fn = staticmethod(residual_permutation)
        path = os.path.join(HERE, 'runs', 'swap_downstream.json')
        if not os.path.exists(path):
            raise unittest.SkipTest('swap_downstream.json not on disk')
        with open(path, encoding='utf-8') as fh:
            rows = json.load(fh)
        cls.arms = {}
        for task, stage in (('topic', 'test'), ('entities', 'ner')):
            cls.arms[task] = (
                [r['mean'] for r in rows
                 if r.get('stage') == stage and r['arm'] == "XLM-R's vocabulary"],
                [r['mean'] for r in rows
                 if r.get('stage') == stage and r['arm'] == 'our vocabulary'])

    def test_both_tasks_match_the_panel(self):
        for task, expected in (('topic', 0.5714), ('entities', 0.0286)):
            with self.subTest(task=task):
                p, _, total, _ = self.fn(*self.arms[task])
                self.assertEqual(total, 70)
                self.assertAlmostEqual(p, expected, places=4)

    def test_entities_sits_on_the_floor_and_the_floor_is_the_top_rows(self):
        """0.029 in both rows of the panel is one number, not two that happen to agree."""
        p, hits, total, floor = self.fn(*self.arms['entities'])
        self.assertEqual(hits, 2)
        self.assertAlmostEqual(floor, 2 / 70, places=6)
        self.assertAlmostEqual(p, floor, places=6)

    def test_the_statistic_is_symmetric_under_swapping_the_arms(self):
        """The defect in the second draft. `|ratio - 1|` is not symmetric, so it was quietly
        one-sided and returned 1/70 -- a value no two-sided test at four a side can reach."""
        for task in ('topic', 'entities'):
            a, b = self.arms[task]
            with self.subTest(task=task):
                self.assertAlmostEqual(self.fn(a, b)[0], self.fn(b, a)[0], places=12)
                self.assertGreaterEqual(self.fn(a, b)[1], 2)

    def test_centering_is_what_makes_it_agree_with_the_f_test(self):
        """Without centering, topic returns ~0.057 and reads as a variance finding conjured out of
        the 0.144 mean gap. The panel's whole warning, as an assertion."""
        a, b = self.arms['topic']
        centered = self.fn(a, b)[0]
        shifted = self.fn([x + 10.0 for x in a], b)[0]
        self.assertAlmostEqual(centered, shifted, places=12)
        self.assertGreater(centered, 0.5)


if __name__ == '__main__':
    print(f'ft_api/mlm_api: {"stubbed (no torch)" if _records.STUBBED else "real"}')
    unittest.main(verbosity=2)
