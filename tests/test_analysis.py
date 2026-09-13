"""Tests for the descriptive and stratified analyses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cabt import analysis as A  # noqa: E402


class StrengthStratifiedTest(unittest.TestCase):
    """The stratification must use the episode as the unit, not the row."""

    def _frames(self, per_episode_rows: int = 50):
        rng = np.random.default_rng(0)
        eps = np.arange(100)
        rows = []
        for e in eps:
            label = float(e % 2)
            # Predictions must vary by episode, otherwise every episode has an
            # identical Brier and the regression has nothing to fit.
            p_model = 0.2 + 0.6 * (((e * 7) % 11) / 10.0)
            for _ in range(per_episode_rows):
                rows.append(
                    {
                        "episode_id": int(e),
                        "label": label,
                        "p_prize_only": 0.5,
                        "p_gbdt_filtered": p_model,
                    }
                )
        preds = pd.DataFrame(rows)
        manifest = pd.DataFrame(
            {"episode_id": eps, "avg_score": 1000.0 + rng.permutation(len(eps))}
        )
        return preds, manifest

    def test_buckets_partition_the_episodes(self):
        preds, manifest = self._frames()
        table, summary = A.strength_stratified(preds, manifest)
        self.assertEqual(len(table), 4)
        self.assertEqual(int(table["episodes"].sum()), 100)
        self.assertEqual(summary["n_episodes"], 100)
        # Buckets are ordered and do not overlap.
        self.assertTrue((table["score_lo"].diff().dropna() > 0).all())
        for lo, prev_hi in zip(
            table["score_lo"].iloc[1:], table["score_hi"].iloc[:-1], strict=True
        ):
            self.assertGreater(lo, prev_hi)

    def test_regression_n_is_episodes_not_rows(self):
        """A per-row fit would shrink the standard error by ~sqrt(rows/game)."""
        few, manifest = self._frames(per_episode_rows=5)
        many, _ = self._frames(per_episode_rows=500)
        _, s_few = A.strength_stratified(few, manifest)
        _, s_many = A.strength_stratified(many, manifest)
        self.assertAlmostEqual(
            s_few["model_brier_vs_score"]["stderr"],
            s_many["model_brier_vs_score"]["stderr"],
            places=10,
        )
