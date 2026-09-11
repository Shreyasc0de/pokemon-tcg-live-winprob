"""Model, filter and metric behaviour, checked on constructed data.

These are the tests that matter most for a probability model: a metric with a
sign error or a filter that quietly peeks at the future would still produce
plausible-looking output, so each property is pinned to a case with a known
answer.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cabt.evaluate import (  # noqa: E402
    auc,
    brier,
    expected_calibration_error,
    log_loss_safe,
    martingale_test,
    path_volatility,
    reliability,
    skill_score,
)
from cabt.models import BaseRateClassifier, LogitKalmanFilter, expit, logit  # noqa: E402


def _chain(p: list[float], label: float, episode_id: int = 1, seat: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": episode_id,
            "acting_player": seat,
            "decision_index": np.arange(len(p)),
            "p_raw": p,
            "label": label,
        }
    )


class TestMetrics(unittest.TestCase):
    def test_brier_on_known_values(self):
        self.assertAlmostEqual(brier(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 0.0)
        self.assertAlmostEqual(brier(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)
        self.assertAlmostEqual(brier(np.array([1.0, 0.0]), np.array([0.5, 0.5])), 0.25)

    def test_log_loss_on_known_values(self):
        self.assertAlmostEqual(
            log_loss_safe(np.array([1.0, 0.0]), np.array([0.5, 0.5])), np.log(2), places=6
        )
        self.assertLess(log_loss_safe(np.array([1.0]), np.array([0.9])), log_loss_safe(np.array([1.0]), np.array([0.6])))

    def test_log_loss_is_finite_at_the_boundary(self):
        """A confident miss must be penalised heavily but never return inf."""
        loss = log_loss_safe(np.array([1.0]), np.array([0.0]))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 10)

    def test_auc_is_half_for_constant_and_one_for_perfect(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(auc(y, np.full(4, 0.3)), 0.5)
        self.assertAlmostEqual(auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)

    def test_auc_is_nan_with_one_class(self):
        self.assertTrue(np.isnan(auc(np.ones(4), np.array([0.1, 0.2, 0.8, 0.9]))))

    def test_skill_score_signs(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        good, bad, ref = np.array([0.9, 0.1, 0.9, 0.1]), np.array([0.1, 0.9, 0.1, 0.9]), np.full(4, 0.5)
        self.assertGreater(skill_score(y, good, ref), 0)
        self.assertLess(skill_score(y, bad, ref), 0)
        self.assertAlmostEqual(skill_score(y, ref, ref), 0.0)

    def test_reliability_bins_partition_the_sample(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=2000)
        y = (rng.uniform(size=2000) < p).astype(float)
        tab = reliability(y, p, n_bins=10)
        self.assertEqual(int(tab["n"].sum()), 2000)
        self.assertTrue((tab["obs_lo"] <= tab["observed"]).all())
        self.assertTrue((tab["observed"] <= tab["obs_hi"]).all())

    def test_ece_is_zero_for_a_perfectly_calibrated_forecast(self):
        """Deterministic construction: each bin's outcomes match its prediction."""
        p, y = [], []
        for level in np.linspace(0.05, 0.95, 10):
            n = 1000
            k = int(round(level * n))
            p.extend([level] * n)
            y.extend([1.0] * k + [0.0] * (n - k))
        self.assertLess(expected_calibration_error(np.array(y), np.array(p)), 0.01)

    def test_ece_detects_overconfidence(self):
        y = np.array([1.0] * 60 + [0.0] * 40)
        p = np.array([0.95] * 60 + [0.05] * 40)
        self.assertLess(expected_calibration_error(y, p), 0.1)
        shifted = np.clip(p + 0.04, 0, 1)
        self.assertGreater(expected_calibration_error(y, shifted), 0.02)


class TestLogitHelpers(unittest.TestCase):
    def test_logit_and_expit_round_trip(self):
        p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        np.testing.assert_allclose(expit(logit(p)), p, atol=1e-6)

    def test_logit_clips_instead_of_diverging(self):
        self.assertTrue(np.isfinite(logit(np.array([0.0, 1.0]))).all())


class TestKalmanFilter(unittest.TestCase):
    def test_large_q_reproduces_the_raw_signal(self):
        """With no smoothing prior the filter is a pass-through."""
        df = _chain([0.2, 0.8, 0.3, 0.9], 1.0)
        out = LogitKalmanFilter(q=1e9, r=1.0).transform(df)
        np.testing.assert_allclose(out, df["p_raw"].to_numpy(), atol=1e-3)

    def test_small_q_smooths_toward_the_running_level(self):
        df = _chain([0.5, 0.5, 0.5, 0.95], 1.0)
        out = LogitKalmanFilter(q=0.01, r=1.0).transform(df)
        self.assertLess(out[-1], df["p_raw"].iloc[-1], "a spike should be damped")
        self.assertGreater(out[-1], out[0])

    def test_filter_is_causal(self):
        """Changing a later observation must not move an earlier output."""
        a = _chain([0.5, 0.6, 0.7, 0.8], 1.0)
        b = a.copy()
        b.loc[3, "p_raw"] = 0.01
        out_a = LogitKalmanFilter(q=0.2).transform(a)
        out_b = LogitKalmanFilter(q=0.2).transform(b)
        np.testing.assert_allclose(out_a[:3], out_b[:3], atol=1e-12)

    def test_chains_do_not_bleed_into_each_other(self):
        """Each (episode, seat) chain is filtered independently."""
        one = _chain([0.9, 0.9, 0.9], 1.0, episode_id=1)
        two = _chain([0.1, 0.1, 0.1], 0.0, episode_id=2)
        both = pd.concat([one, two], ignore_index=True)
        kf = LogitKalmanFilter(q=0.2)
        joint = kf.transform(both)
        np.testing.assert_allclose(joint[:3], kf.transform(one), atol=1e-12)
        np.testing.assert_allclose(joint[3:], kf.transform(two), atol=1e-12)

    def test_transform_preserves_row_order(self):
        """Output is realigned to the caller's index, not to sort order."""
        df = _chain([0.2, 0.4, 0.6, 0.8], 1.0).iloc[::-1].reset_index(drop=True)
        out = LogitKalmanFilter(q=1e9).transform(df)
        np.testing.assert_allclose(out, df["p_raw"].to_numpy(), atol=1e-3)

    def test_fit_selects_q_from_the_grid(self):
        rng = np.random.default_rng(0)
        frames = []
        for eid in range(40):
            label = float(eid % 2)
            truth = 0.8 if label else 0.2
            noisy = np.clip(truth + rng.normal(0, 0.25, 12), 0.02, 0.98)
            frames.append(_chain(list(noisy), label, episode_id=eid))
        kf = LogitKalmanFilter().fit(pd.concat(frames, ignore_index=True))
        self.assertIn(kf.q, kf.grid)
        self.assertTrue(np.isfinite(kf.val_log_loss_))


class TestPathDiagnostics(unittest.TestCase):
    def test_volatility_is_zero_for_a_flat_path(self):
        df = _chain([0.5] * 6, 1.0)
        vol = path_volatility(df.rename(columns={"p_raw": "p"}), "p")
        self.assertAlmostEqual(float(vol["mean_abs_step"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(vol["max_swing"].iloc[0]), 0.0)

    def test_volatility_picks_up_the_largest_jump(self):
        df = _chain([0.5, 0.52, 0.9, 0.88], 1.0)
        vol = path_volatility(df.rename(columns={"p_raw": "p"}), "p")
        self.assertAlmostEqual(float(vol["max_swing"].iloc[0]), 0.38, places=6)

    def test_martingale_slope_is_near_zero_for_a_random_walk(self):
        """A driftless random walk in logit space satisfies the property."""
        rng = np.random.default_rng(1)
        frames = []
        for eid in range(200):
            z = np.cumsum(rng.normal(0, 0.3, 20))
            frames.append(_chain(list(expit(z)), float(z[-1] > 0), episode_id=eid))
        res = martingale_test(pd.concat(frames, ignore_index=True).rename(columns={"p_raw": "p"}), "p")
        self.assertLess(abs(res["slope"]), 4 * res["stderr"] + 0.01)

    def test_martingale_slope_is_negative_for_a_reverting_path(self):
        """An overshooting path that gets pulled back shows up as negative."""
        rng = np.random.default_rng(2)
        frames = []
        for eid in range(200):
            p = [0.5]
            for _ in range(19):
                # Strong pull back to 0.5: the hallmark of an overconfident path.
                p.append(float(np.clip(p[-1] - 0.6 * (p[-1] - 0.5) + rng.normal(0, 0.1), 0.02, 0.98)))
            frames.append(_chain(p, 1.0, episode_id=eid))
        res = martingale_test(pd.concat(frames, ignore_index=True).rename(columns={"p_raw": "p"}), "p")
        self.assertLess(res["slope"], 0)
        self.assertLess(res["p_value"], 0.01)


class TestBaseRate(unittest.TestCase):
    def test_predicts_the_training_rate_everywhere(self):
        X = pd.DataFrame({"a": [0.0] * 10})
        y = np.array([1.0] * 7 + [0.0] * 3)
        m = BaseRateClassifier().fit(X, y)
        p = m.predict_proba(pd.DataFrame({"a": [0.0] * 4}))[:, 1]
        np.testing.assert_allclose(p, 0.7)


if __name__ == "__main__":
    unittest.main()
