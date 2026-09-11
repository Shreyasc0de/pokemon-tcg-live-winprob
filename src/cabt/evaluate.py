"""Scoring, calibration and path diagnostics for win-probability estimates.

Accuracy is the wrong headline for this problem. A win-probability curve is
useful only if its numbers mean what they say: states labelled 70% should win
about 70% of the time. So the primary metrics are proper scoring rules --
Brier score and log loss -- plus explicit calibration error, and every score
is quoted as a *skill* relative to the prize-differential baseline a human
player already computes for free.

Two diagnostics are less standard and more interesting:

**Calibration by phase.** A model can be well calibrated on average while
being badly overconfident early and underconfident late. Scores are therefore
broken out by turn.

**The martingale property.** If a probability path is correctly specified,
it must be a martingale: the expected change from here is zero, whatever the
current level. Regressing the next change on the current level gives a
testable coefficient -- zero for an honest path, negative for one that
systematically overshoots and gets pulled back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-15


def log_loss_safe(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    y = np.asarray(y, dtype="float64")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype="float64") - np.asarray(y, dtype="float64")) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC via the Mann-Whitney statistic; NaN if only one class present."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def skill_score(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray) -> float:
    """Brier skill score against a reference forecast: 1 is perfect, 0 is no gain."""
    ref = brier(y, p_ref)
    return float("nan") if ref == 0 else float(1 - brier(y, p) / ref)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Binned reliability table: predicted vs observed frequency per bin."""
    y = np.asarray(y, dtype="float64")
    p = np.asarray(p, dtype="float64")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if not n:
            continue
        obs = float(y[m].mean())
        # Wilson interval, so sparse high-confidence bins are not over-read.
        lo, hi = _wilson(y[m].sum(), n)
        rows.append(
            {
                "bin": b,
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "predicted": float(p[m].mean()),
                "observed": obs,
                "obs_lo": lo,
                "obs_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def _wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Sample-weighted mean gap between predicted and observed frequency."""
    tab = reliability(y, p, n_bins)
    if tab.empty:
        return float("nan")
    w = tab["n"] / tab["n"].sum()
    return float((w * (tab["predicted"] - tab["observed"]).abs()).sum())


def score(y: np.ndarray, p: np.ndarray, p_ref: np.ndarray | None = None) -> dict:
    """All headline metrics for one set of predictions."""
    out = {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "brier": brier(y, p),
        "log_loss": log_loss_safe(y, p),
        "auc": auc(y, p),
        "ece": expected_calibration_error(y, p),
    }
    if p_ref is not None:
        out["brier_skill_vs_ref"] = skill_score(y, p, p_ref)
    return out


def score_by_turn(
    df: pd.DataFrame,
    pred_col: str,
    bins: tuple[int, ...] = (0, 2, 4, 6, 8, 11, 15, 200),
) -> pd.DataFrame:
    """Metrics within turn buckets, to expose phase-dependent miscalibration."""
    d = df.copy()
    d["turn_bucket"] = pd.cut(d["turn"], bins=list(bins), right=False)
    rows = []
    for bucket, part in d.groupby("turn_bucket", observed=True):
        y, p = part["label"].to_numpy(), part[pred_col].to_numpy()
        rows.append({"turn_bucket": str(bucket), **score(y, p)})
    return pd.DataFrame(rows)


def path_volatility(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Per-game path statistics: how much the estimate moves, and how far.

    ``mean_abs_step`` is the average absolute change between consecutive
    decision points -- a jitter measure. ``max_swing`` is the largest single
    jump, i.e. the most decisive moment the model identifies in that game.
    """
    d = df.sort_values(["episode_id", "acting_player", "decision_index"])
    g = d.groupby(["episode_id", "acting_player"], sort=False)
    step = g[pred_col].diff()
    d = d.assign(_step=step)
    agg = d.groupby(["episode_id", "acting_player"], sort=False).agg(
        n=("_step", "size"),
        mean_abs_step=("_step", lambda s: float(np.nanmean(np.abs(s)))),
        max_swing=("_step", lambda s: float(np.nanmax(np.abs(s))) if s.notna().any() else np.nan),
        start=(pred_col, "first"),
        end=(pred_col, "last"),
        label=("label", "first"),
    )
    return agg.reset_index()


def martingale_test(df: pd.DataFrame, pred_col: str) -> dict:
    """Test the martingale property of the probability path.

    Regresses the next step :math:`p_{t+1} - p_t` on the centred current level
    :math:`p_t - 0.5` within each (episode, seat) chain. Under a correctly
    specified forecast the slope is zero: knowing the current level tells you
    nothing about which way it will move next. A significantly negative slope
    means the path habitually overshoots and reverts -- the forecast is too
    confident too early.
    """
    d = df.sort_values(["episode_id", "acting_player", "decision_index"]).copy()
    g = d.groupby(["episode_id", "acting_player"], sort=False)
    d["_next"] = g[pred_col].shift(-1)
    d = d.dropna(subset=["_next"])
    x = d[pred_col].to_numpy() - 0.5
    dy = d["_next"].to_numpy() - d[pred_col].to_numpy()
    if len(x) < 10:
        return {"n": int(len(x))}
    res = stats.linregress(x, dy)
    return {
        "n": int(len(x)),
        "slope": float(res.slope),
        "stderr": float(res.stderr),
        "p_value": float(res.pvalue),
        "mean_step": float(np.mean(dy)),
        "mean_abs_step": float(np.mean(np.abs(dy))),
    }


def grouped_permutation_importance(
    model,  # noqa: ANN001
    X: pd.DataFrame,  # noqa: N803
    y: np.ndarray,
    families: dict[str, tuple[str, ...]],
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Permutation importance by feature family, measured in Brier increase.

    Families are shuffled together because the features inside one are heavily
    collinear (``my_prizes``, ``opp_prizes`` and ``prize_diff`` carry the same
    information three ways). Permuting them one at a time would let the model
    read the shuffled column's twin and report near-zero importance for all
    three.
    """
    rng = np.random.default_rng(seed)
    base = brier(y, model.predict_proba(X)[:, 1])
    rows = []
    for name, cols in families.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            deltas.append(brier(y, model.predict_proba(Xp)[:, 1]) - base)
        rows.append(
            {
                "family": name,
                "n_features": len(cols),
                "brier_increase": float(np.mean(deltas)),
                "brier_increase_sd": float(np.std(deltas)),
            }
        )
    out = pd.DataFrame(rows).sort_values("brier_increase", ascending=False)
    out.attrs["baseline_brier"] = base
    return out.reset_index(drop=True)
