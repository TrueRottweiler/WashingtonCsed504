"""Unit tests for the masked-language-model objective.

The masking is the scientific core of every number this project has produced, and it is the
kind of code that fails silently: a wrong ratio, a label left un-ignored, or a random
replacement drawn from the special tokens does not crash, does not warn, and does not look
wrong in a log. It just trains a slightly different -- usually worse -- model, and the
comparison it was built to support quietly stops meaning what it says.

These run on CPU with a synthetic corpus, so they need no GPU and no prepared data.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

# Patrick's guard, on his suggestion: without torch this module cannot import, and unittest
# discover reported the failure as an ERROR rather than a skip -- a red suite on any clone
# without a GPU stack, for a reason that is not a defect. SkipTest at import time makes
# discover report what is actually true: skipped here, for this reason.
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest(f'torch is not installed on this machine ({exc}); '
                            'these tests exercise the torch data path') from None

import mlm_data as D
import text_data as T
import text_prepare as P

VOCAB = 500
SEQ = 16


class MaskingTests(unittest.TestCase):
    """The 80/10/10 BERT recipe, on a stream we control exactly."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls._saved = (P.DATA_DIR, T.DATA_DIR)
        P.DATA_DIR = T.DATA_DIR = cls.tmp

        os.makedirs(P.out_dir('mask_demo'), exist_ok=True)
        # Ids well away from the specials, so a stray special is obvious when it appears.
        ids = np.arange(100, 100 + 20_000, dtype=np.uint16) % VOCAB
        ids[ids < 10] = 100
        P.save_split('mask_demo', 'train', ids, VOCAB)
        P.save_stats('mask_demo', 'synthetic', VOCAB, {'train': int(len(ids))})

        cls.ds = D.MlmTokens(torch.device('cpu'), 'mask_demo', 'train',
                             seq_len=SEQ, mask_id=4)

    @classmethod
    def tearDownClass(cls):
        P.DATA_DIR, T.DATA_DIR = cls._saved
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _big_batch(self, prob=0.15, seed=0):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        x = self.ds.windows(512, generator=g)
        return x, *self.ds.mask(x, mlm_prob=prob, generator=g)

    # -- what the model is asked to predict --------------------------------------------------

    def test_labels_are_ignored_everywhere_except_the_selection(self):
        """Loss must be scored ONLY on masked positions.

        If un-selected positions kept their labels the model would be graded on copying its own
        input, which is trivially easy -- the loss would look excellent and mean nothing.
        """
        x, corrupted, labels = self._big_batch()
        selected = labels != -100
        self.assertTrue((labels[~selected] == -100).all())
        # Every scored label is the ORIGINAL token, not the corrupted one.
        self.assertTrue(torch.equal(labels[selected], x[selected]))

    def test_selection_rate_is_the_requested_probability(self):
        x, corrupted, labels = self._big_batch(prob=0.15)
        rate = (labels != -100).float().mean().item()
        self.assertAlmostEqual(rate, 0.15, delta=0.01)

    def test_selection_rate_follows_the_argument(self):
        for prob in (0.05, 0.30):
            _, _, labels = self._big_batch(prob=prob, seed=1)
            rate = (labels != -100).float().mean().item()
            self.assertAlmostEqual(rate, prob, delta=0.02)

    # -- the 80/10/10 split ------------------------------------------------------------------

    def test_eighty_ten_ten(self):
        """Of the selected positions: ~80% <mask>, ~10% random, ~10% left alone.

        The last two are not decoration. A model that only ever sees <mask> at the positions it
        must predict learns a representation that is useless when no <mask> is present -- which
        is every downstream fine-tuning batch, and therefore every number in these reports.
        """
        x, corrupted, labels = self._big_batch()
        sel = labels != -100
        n = int(sel.sum())

        masked = ((corrupted == self.ds.mask_id) & sel).sum().item() / n
        unchanged = ((corrupted == x) & sel).sum().item() / n
        replaced = (((corrupted != x) & (corrupted != self.ds.mask_id)) & sel).sum().item() / n

        self.assertAlmostEqual(masked, 0.80, delta=0.03)
        self.assertAlmostEqual(replaced, 0.10, delta=0.02)
        # "Unchanged" also catches the ~1/vocab of random draws that land on the original token.
        self.assertAlmostEqual(unchanged, 0.10, delta=0.02)

    def test_nothing_outside_the_selection_is_touched(self):
        """Positions the model is not asked about must arrive exactly as they were."""
        x, corrupted, labels = self._big_batch()
        untouched = labels == -100
        self.assertTrue(torch.equal(corrupted[untouched], x[untouched]))

    def test_random_replacements_never_inject_special_tokens(self):
        """A random replacement must be a real word, never <pad>, <mask> or <s>.

        Drawing from the whole vocabulary would sprinkle padding and mask tokens through the
        input as if they were content, teaching the model that they occur mid-sentence.
        """
        x, corrupted, labels = self._big_batch(prob=0.9, seed=3)
        sel = labels != -100
        replaced = sel & (corrupted != x) & (corrupted != self.ds.mask_id)
        self.assertGreater(int(replaced.sum()), 100, 'need replacements to test')
        self.assertGreaterEqual(int(corrupted[replaced].min()), self.ds.n_special)

    def test_ids_stay_inside_the_vocabulary(self):
        x, corrupted, labels = self._big_batch(prob=0.5, seed=4)
        self.assertGreaterEqual(int(corrupted.min()), 0)
        self.assertLess(int(corrupted.max()), self.ds.vocab_size)

    # -- reproducibility ---------------------------------------------------------------------

    def test_same_seed_gives_the_same_masking(self):
        a = self._big_batch(seed=7)
        b = self._big_batch(seed=7)
        for u, v in zip(a, b):
            self.assertTrue(torch.equal(u, v))

    def test_different_seeds_differ(self):
        _, ca, _ = self._big_batch(seed=8)
        _, cb, _ = self._big_batch(seed=9)
        self.assertFalse(torch.equal(ca, cb))

    def test_validation_batches_are_fixed(self):
        """Two checkpoints must be scored on identical corrupted text.

        If the validation masking moved between evaluations, a val loss could improve or worsen
        because the questions changed rather than because the model did -- and every comparison
        in the study is a comparison of val losses.
        """
        first = self.ds.fixed_val_batches(batch_size=8, n_batches=2)
        second = self.ds.fixed_val_batches(batch_size=8, n_batches=2)
        self.assertEqual(len(first), len(second))
        for (xa, ya), (xb, yb) in zip(first, second):
            self.assertTrue(torch.equal(xa, xb))
            self.assertTrue(torch.equal(ya, yb))

    # -- the windows the masking operates on -------------------------------------------------

    def test_windows_have_the_requested_shape_and_dtype(self):
        g = torch.Generator(device='cpu')
        g.manual_seed(0)
        w = self.ds.windows(7, generator=g)
        self.assertEqual(tuple(w.shape), (7, SEQ))
        self.assertEqual(w.dtype, torch.int64)   # embedding lookups need int64 indices

    def test_windows_stay_in_bounds(self):
        g = torch.Generator(device='cpu')
        g.manual_seed(1)
        w = self.ds.windows(256, generator=g)
        self.assertGreaterEqual(int(w.min()), 0)
        self.assertLess(int(w.max()), VOCAB)


if __name__ == '__main__':
    unittest.main()
