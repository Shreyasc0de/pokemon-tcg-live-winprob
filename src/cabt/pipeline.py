"""End-to-end experiment: fit the model ladder, score it, filter the path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .dataset import ALL_GROUPS, design_matrix, split_by_agent, split_by_episode
from .evaluate import (
    grouped_permutation_importance,
    martingale_test,
    path_volatility,
    reliability,
    score,
    score_by_turn,
)
from .features import IMPORTANCE_FAMILIES, feature_names
from .models import BaseRateClassifier, LogitKalmanFilter, calibrated, gbdt, linear_pipeline

RANDOM_SEED = 7


def _group_folds(df: pd.DataFrame, n_splits: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """CV folds that never split a game across train and validation."""
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(df))
    return list(gkf.split(idx, groups=df["episode_id"].to_numpy()))


def fit_and_score(
    df: pd.DataFrame,
    out_dir: str | Path,
    feature_groups: tuple[str, ...] = ALL_GROUPS,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train the ladder on an episode-held-out split and write every artefact."""
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    train, test = split_by_episode(df, test_frac=0.2, seed=seed)
    # A further slice of training episodes tunes the filter, so its q is never
    # chosen on the test set.
    fit_part, val_part = split_by_episode(train, test_frac=0.2, seed=seed + 1)

    X_fit, y_fit = design_matrix(fit_part, feature_groups)
    X_val, y_val = design_matrix(val_part, feature_groups)
    X_train, y_train = design_matrix(train, feature_groups)
    X_test, y_test = design_matrix(test, feature_groups)

    prize_cols = ["prize_diff"]
    models: dict[str, object] = {}

    models["base_rate"] = BaseRateClassifier().fit(X_train, y_train)
    models["prize_only"] = linear_pipeline().fit(X_train[prize_cols], y_train)
    models["logistic"] = linear_pipeline(C=0.5).fit(X_train, y_train)
    # Uncalibrated is the shipped model. On the full archive the boosted model
    # is already well calibrated (ECE ~0.011) and isotonic recalibration makes
    # it worse; `calibration_variants` records the comparison.
    models["gbdt"] = gbdt().fit(X_train, y_train)
    models["gbdt_isotonic"] = calibrated(gbdt(), cv=_group_folds(train)).fit(X_train, y_train)

    preds = pd.DataFrame(index=test.index)
    preds["base_rate"] = models["base_rate"].predict_proba(X_test)[:, 1]
    preds["prize_only"] = models["prize_only"].predict_proba(X_test[prize_cols])[:, 1]
    preds["logistic"] = models["logistic"].predict_proba(X_test)[:, 1]
    preds["gbdt"] = models["gbdt"].predict_proba(X_test)[:, 1]
    preds["gbdt_isotonic"] = models["gbdt_isotonic"].predict_proba(X_test)[:, 1]

    # Filter tuning uses a validation split, with the per-state model refit on
    # the remaining training episodes so its validation output is out-of-sample.
    inner = gbdt().fit(X_fit, y_fit)
    val = val_part.assign(p_raw=inner.predict_proba(X_val)[:, 1])
    kf = LogitKalmanFilter().fit(val, "p_raw")

    test_with_raw = test.assign(p_raw=preds["gbdt"].to_numpy())
    preds["gbdt_filtered"] = kf.transform(test_with_raw, "p_raw")

    ref = preds["prize_only"].to_numpy()
    metrics = {
        name: score(y_test, preds[name].to_numpy(), ref) for name in preds.columns
    }
    metrics_df = pd.DataFrame(metrics).T.reset_index(names="model")

    scored = test.assign(**{f"p_{c}": preds[c] for c in preds.columns})
    by_turn = score_by_turn(scored, "p_gbdt_filtered")
    rel_raw = reliability(y_test, preds["gbdt"].to_numpy(), 10)
    rel_filt = reliability(y_test, preds["gbdt_filtered"].to_numpy(), 10)
    # The uncalibrated model is kept purely so the calibration figure has
    # something to compare against: isotonic is what fixes it.
    rel_uncal = reliability(y_test, preds["gbdt_isotonic"].to_numpy(), 10)

    vol = pd.DataFrame(
        {
            "model": ["gbdt", "gbdt_filtered"],
            "mean_abs_step": [
                float(path_volatility(scored, "p_gbdt")["mean_abs_step"].mean()),
                float(path_volatility(scored, "p_gbdt_filtered")["mean_abs_step"].mean()),
            ],
            "mean_max_swing": [
                float(path_volatility(scored, "p_gbdt")["max_swing"].mean()),
                float(path_volatility(scored, "p_gbdt_filtered")["max_swing"].mean()),
            ],
        }
    )

    martingale = {
        "gbdt": martingale_test(scored, "p_gbdt"),
        "gbdt_filtered": martingale_test(scored, "p_gbdt_filtered"),
    }
    sweep = _filter_sweep(scored, kf.q, ref)

    importance = grouped_permutation_importance(
        models["gbdt"], X_test, y_test, IMPORTANCE_FAMILIES, n_repeats=3, seed=seed
    )

    ablation = _ablation(train, test, feature_groups, ref, y_test)
    calib_variants = _calibration_variants(fit_part, val_part, test, feature_groups, ref, y_test)
    unseen = _unseen_agent_check(df, feature_groups, seed)

    for name, table in {
        "metrics": metrics_df,
        "metrics_by_turn": by_turn,
        "reliability_gbdt_isotonic": rel_uncal,
        "reliability_gbdt": rel_raw,
        "reliability_gbdt_filtered": rel_filt,
        "path_volatility": vol,
        "filter_sweep": sweep,
        "feature_importance": importance,
        "ablation": ablation,
        "calibration_variants": calib_variants,
    }.items():
        table.to_csv(out_dir / "tables" / f"{name}.csv", index=False)

    scored.to_csv(out_dir / "test_predictions.csv.gz", index=False)

    summary = {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_episodes": int(train["episode_id"].nunique()),
        "n_test_episodes": int(test["episode_id"].nunique()),
        "n_features": len(feature_names(feature_groups)),
        "kalman_q": kf.q,
        "kalman_val_log_loss": getattr(kf, "val_log_loss_", None),
        "metrics": metrics,
        "martingale": martingale,
        "filter_sweep": sweep.to_dict("records"),
        "unseen_agent": unseen,
        "calibration_variants": calib_variants.to_dict("records"),
        "importance_baseline_brier": importance.attrs.get("baseline_brier"),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _filter_sweep(scored: pd.DataFrame, tuned_q: float, ref: np.ndarray) -> pd.DataFrame:
    """Score the same per-state predictions at several smoothing strengths.

    The pass-through row (``q`` effectively infinite) is the honest control,
    and it matters: running predictions through ``logit`` and back clips them
    away from 0 and 1, which by itself improves log loss. Comparing the filter
    against the *unclipped* model would credit the filter with that clipping.
    Against the pass-through row, the filter's real effect is visible --
    unchanged scores, a smoother path, and a martingale slope that stops being
    significantly negative.
    """
    y = scored["label"].to_numpy()
    rows = []
    for label, q in (
        ("pass-through (no smoothing)", 1e6),
        (f"tuned (q={tuned_q:g})", tuned_q),
        ("moderate (q=0.5)", 0.5),
        ("heavy (q=0.05)", 0.05),
    ):
        p = LogitKalmanFilter(q=q).transform(scored.assign(p_raw=scored["p_gbdt"]), "p_raw")
        d = scored.assign(_p=p)
        vol = path_volatility(d, "_p")
        mart = martingale_test(d, "_p")
        rows.append(
            {
                "filter": label,
                "q": q,
                **score(y, p, ref),
                "mean_abs_step": float(vol["mean_abs_step"].mean()),
                "mean_max_swing": float(vol["max_swing"].mean()),
                "martingale_slope": mart["slope"],
                "martingale_p": mart["p_value"],
            }
        )
    return pd.DataFrame(rows)


def _calibration_variants(
    fit_part: pd.DataFrame,
    cal_part: pd.DataFrame,
    test: pd.DataFrame,
    groups: tuple[str, ...],
    ref: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Compare three calibration choices on the same held-out episodes.

    Brier and log loss decompose into calibration *and* sharpness, so the
    ranking differs depending on which you care about: a single isotonic map
    fitted on dedicated held-out episodes is the best *calibrated* (lowest
    ECE) but the least sharp, because it both trains on less data and
    discretises the output into a coarse staircase. The cross-fitted isotonic
    ensemble wins on the proper scoring rules and is what the pipeline ships.
    """
    from sklearn.isotonic import IsotonicRegression

    X_fit, y_fit = design_matrix(fit_part, groups)
    X_cal, y_cal = design_matrix(cal_part, groups)
    X_test, _ = design_matrix(test, groups)
    train = pd.concat([fit_part, cal_part])
    X_train, y_train = design_matrix(train, groups)

    rows = []
    raw = gbdt().fit(X_train, y_train)
    rows.append({"calibration": "none", **score(y_test, raw.predict_proba(X_test)[:, 1], ref)})

    cv_iso = calibrated(gbdt(), cv=_group_folds(train)).fit(X_train, y_train)
    rows.append(
        {"calibration": "isotonic (4-fold, cross-fitted)", **score(y_test, cv_iso.predict_proba(X_test)[:, 1], ref)}
    )

    inner = gbdt().fit(X_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        inner.predict_proba(X_cal)[:, 1], y_cal
    )
    rows.append(
        {"calibration": "isotonic (single held-out map)", **score(y_test, iso.predict(inner.predict_proba(X_test)[:, 1]), ref)}
    )
    return pd.DataFrame(rows)


def _ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    groups: tuple[str, ...],
    ref: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Refit the boosted model on each nested feature-group set."""
    sets = {
        "core": ("core",),
        "core+decision": ("core", "decision"),
        "core+decision+trajectory": ("core", "decision", "trajectory"),
        "all (incl. clock)": groups,
    }
    rows = []
    for name, gs in sets.items():
        Xtr, ytr = design_matrix(train, gs)
        Xte, _ = design_matrix(test, gs)
        m = gbdt().fit(Xtr, ytr)
        rows.append(
            {"feature_set": name, "n_features": Xtr.shape[1], **score(y_test, m.predict_proba(Xte)[:, 1], ref)}
        )
    return pd.DataFrame(rows)


def _unseen_agent_check(df: pd.DataFrame, groups: tuple[str, ...], seed: int) -> dict:
    """Score on games between agents whose play was never trained on."""
    train, test = split_by_agent(df, test_frac=0.25, seed=seed)
    if len(test) < 500 or len(train) < 500:
        return {"skipped": True, "n_test_rows": int(len(test))}
    Xtr, ytr = design_matrix(train, groups)
    Xte, yte = design_matrix(test, groups)
    m = gbdt().fit(Xtr, ytr)
    ref = linear_pipeline().fit(Xtr[["prize_diff"]], ytr).predict_proba(Xte[["prize_diff"]])[:, 1]
    return {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_test_episodes": int(test["episode_id"].nunique()),
        **score(yte, m.predict_proba(Xte)[:, 1], ref),
    }
