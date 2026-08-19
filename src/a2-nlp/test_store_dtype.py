"""Unit tests for the width of the token store, on disk and resident.

These run on CPU and need no prepared corpus: the round-trip cases build a synthetic token
stream in a temp directory. The point of the exercise is that a vocabulary too wide for a store
must either widen the store or raise -- what it must never do is wrap an id around into a
different valid-looking token, which no downstream check would catch.
"""

from __future__ import annotations

import os
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

import text_data as T
import text_prepare as P


class ResolveStoreDtypeTests(unittest.TestCase):
    """resolve_store_dtype's choices, including the boundaries."""

    def test_auto_picks_int16_for_our_own_rungs(self) -> None:
        # The char rung (65) and the shared BPE (16,384), the two vocabularies Part 2 trained on.
        self.assertEqual(T.resolve_store_dtype(65), torch.int16)
        self.assertEqual(T.resolve_store_dtype(16_384), torch.int16)

    def test_auto_widens_for_multilingual_vocabularies(self) -> None:
        # Roughly mBERT and XLM-R, the checkpoints a transfer study would fine-tune.
        self.assertEqual(T.resolve_store_dtype(119_547), torch.int32)
        self.assertEqual(T.resolve_store_dtype(250_002), torch.int32)

    def test_boundary_is_the_largest_id_not_the_count(self) -> None:
        # Ids run 0..vocab_size-1, so a vocabulary of exactly 32,768 has a largest id of 32,767
        # and still fits int16. One more type does not.
        self.assertEqual(T.resolve_store_dtype(32_768), torch.int16)
        self.assertEqual(T.resolve_store_dtype(32_769), torch.int32)

    def test_explicit_int32_is_always_allowed(self) -> None:
        # Wider than necessary costs memory but cannot truncate, so it is never refused.
        self.assertEqual(T.resolve_store_dtype(65, 'int32'), torch.int32)

    def test_explicit_int16_on_a_wide_vocab_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            T.resolve_store_dtype(250_002, 'int16')
        self.assertIn('int32', str(ctx.exception))

    def test_unknown_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            T.resolve_store_dtype(65, 'int8')


class DiskDtypeTests(unittest.TestCase):
    """disk_dtype reaches one bit further than the resident store, because it is unsigned."""

    def test_uint16_covers_more_than_int16(self) -> None:
        self.assertIs(P.disk_dtype(16_384), np.uint16)
        self.assertIs(P.disk_dtype(65_536), np.uint16)
        self.assertIs(P.disk_dtype(65_537), np.uint32)

    def test_the_two_thresholds_disagree_on_purpose(self) -> None:
        # A 40k vocabulary is the case that exercises both rules at once: two bytes on disk,
        # four bytes resident. If these ever collapse to one threshold, this test should fail.
        self.assertIs(P.disk_dtype(40_000), np.uint16)
        self.assertEqual(T.resolve_store_dtype(40_000), torch.int32)


class RoundTripTests(unittest.TestCase):
    """Save a synthetic corpus and load it back, checking no id changed on the way through."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # Both modules resolve paths off a module-level DATA_DIR, so pointing them at a temp
        # directory is enough to keep the tests away from the real prepared corpora.
        self._saved = (P.DATA_DIR, T.DATA_DIR)
        P.DATA_DIR = T.DATA_DIR = self.tmp.name

    def tearDown(self) -> None:
        P.DATA_DIR, T.DATA_DIR = self._saved
        self.tmp.cleanup()

    def _write(self, name: str, ids: np.ndarray, vocab_size: int) -> None:
        os.makedirs(P.out_dir(name), exist_ok=True)
        P.save_split(name, 'train', ids, vocab_size)
        P.save_stats(name, 'synthetic', vocab_size, {'train': int(len(ids))})

    def test_wide_vocab_survives_the_round_trip(self) -> None:
        # Ids spread across the whole range, with the largest id present explicitly -- that is
        # the value a too-narrow store would wrap.
        vocab = 250_002
        ids = np.array([0, 1, 32_767, 32_768, 65_535, 65_536, 200_000, vocab - 1] * 8,
                       dtype=np.uint32)
        self._write('synth_wide', ids, vocab)

        ds = T.GpuTokens(torch.device('cpu'), 'synth_wide', 'train', seq_len=8)
        self.assertEqual(ds.store_dtype, torch.int32)
        self.assertEqual(ds.bytes_per_token, 4)
        np.testing.assert_array_equal(ds.t.numpy().astype(np.int64), ids.astype(np.int64))

    def test_narrow_disk_wide_resident(self) -> None:
        # 40k: uint16 on disk, int32 once resident. The ids above 32,767 are the ones that would
        # come back negative if the resident store stayed int16.
        vocab = 40_000
        ids = np.array([0, 32_767, 32_768, 39_999] * 16, dtype=np.uint16)
        self._write('synth_mid', ids, vocab)

        ds = T.GpuTokens(torch.device('cpu'), 'synth_mid', 'train', seq_len=8)
        self.assertEqual(ds.store_dtype, torch.int32)
        self.assertTrue((ds.t >= 0).all(), 'an id came back negative -- the store truncated')
        np.testing.assert_array_equal(ds.t.numpy().astype(np.int64), ids.astype(np.int64))

    def test_gb_tracks_the_chosen_width(self) -> None:
        ids = np.arange(1024, dtype=np.uint16) % 16_384
        self._write('synth_small', ids, 16_384)

        narrow = T.GpuTokens(torch.device('cpu'), 'synth_small', 'train', seq_len=8)
        wide = T.GpuTokens(torch.device('cpu'), 'synth_small', 'train', seq_len=8,
                           store_dtype='int32')
        self.assertEqual(narrow.bytes_per_token, 2)
        self.assertEqual(wide.bytes_per_token, 4)
        self.assertAlmostEqual(wide.gb(), 2 * narrow.gb())

    def test_save_split_refuses_an_out_of_vocab_id(self) -> None:
        # A tokenizer that returns an id past its own declared vocabulary is a real failure mode,
        # and it should stop the prepare step rather than reach the training loop.
        with self.assertRaises(AssertionError):
            self._write('synth_bad', np.array([0, 70_000] * 16, dtype=np.uint32), 1_000)


if __name__ == '__main__':
    unittest.main()
