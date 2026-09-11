"""Models for live win probability, from trivial baselines to a filtered path.

Four rungs, each one a control on the next:

1. ``base_rate`` -- the unconditional win rate. Any model must beat this.
2. ``prize_only`` -- logistic regression on the prize differential alone. This
   is the number a human player already tracks in their head, and it is a much
   harder baseline than the base rate.
3. ``logit`` / ``gbdt`` -- the full feature set, linear and gradient-boosted.
4. ``gbdt + Kalman`` -- the per-state prediction passed through a causal
   local-level filter on the logit scale.

The filter matters because a per-state model is memoryless: fed a noisy state
it produces a noisy probability, and consecutive decision points inside one
turn can swing wildly for reasons that have nothing to do with who is winning.
A live broadcast number should not. The filter is strictly causal -- it uses
only decision points already played -- so it remains a legitimate real-time
estimate rather than hindsight smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype="float64")))


class BaseRateClassifier:
    """Predicts the training base rate for every state."""

    def fit(self, X, y):  # noqa: N803, ANN001
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):  # noqa: N803, ANN001
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def linear_pipeline(C: float = 1.0) -> Pipeline:  # noqa: N803
    """Median-imputed, standardised logistic regression.

    Imputation is needed because face-down Pokemon leave genuinely unknown HP
    as NaN; the boosted model reads those directly, a linear model cannot.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000)),
        ]
    )


def gbdt(**kwargs) -> HistGradientBoostingClassifier:
    """Histogram gradient boosting with NaN support and early stopping."""
    params = {
        "max_iter": 400,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 40,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 0,
    }
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


def calibrated(estimator, cv) -> CalibratedClassifierCV:  # noqa: ANN001
    """Wrap an estimator in isotonic calibration over supplied CV folds.

    ``cv`` must be a list of ``(train_idx, test_idx)`` pairs that respect
    episode boundaries; passing an integer would let rows from the same game
    land on both sides and produce a flattering calibration curve.
    """
    return CalibratedClassifierCV(estimator, method="isotonic", cv=cv)


@dataclass
class LogitKalmanFilter:
    """Causal local-level filter on the win-probability logit.

    State-space form, one chain per (episode, seat):

    .. math::

        x_t = x_{t-1} + w_t,\\quad w_t \\sim N(0, q)

        z_t = x_t + v_t,\\quad v_t \\sim N(0, r)

    where :math:`z_t` is the per-state model's logit output and :math:`x_t` the
    latent "true" logit. Only the forward pass is run, so the estimate at
    decision point *t* depends on nothing after *t*.

    ``q / r`` is the only thing that matters -- a high ratio tracks the raw
    model, a low ratio produces a smooth path that reacts slowly. It is chosen
    by grid search on a validation split rather than assumed.
    """

    q: float = 0.5
    r: float = 1.0
    p0_var: float = 10.0
    grid: tuple[float, ...] = field(
        default=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 1e6)
    )

    def filter_chain(self, z: np.ndarray) -> np.ndarray:
        """Run the forward pass over one ordered chain of raw logits."""
        z = np.asarray(z, dtype="float64")
        out = np.empty_like(z)
        x, p = 0.0, self.p0_var
        for i, zi in enumerate(z):
            p += self.q  # predict
            k = p / (p + self.r)  # gain
            x += k * (zi - x)  # update
            p *= 1 - k
            out[i] = x
        return out

    def transform(self, df: pd.DataFrame, raw_col: str = "p_raw") -> np.ndarray:
        """Filter every (episode, seat) chain in decision order."""
        order = df.sort_values(["episode_id", "acting_player", "decision_index"]).index
        ordered = df.loc[order]
        z = logit(ordered[raw_col].to_numpy())
        out = np.empty(len(ordered), dtype="float64")
        keys = list(zip(ordered["episode_id"], ordered["acting_player"], strict=True))
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                out[start:i] = self.filter_chain(z[start:i])
                start = i
        result = pd.Series(expit(out), index=ordered.index)
        return result.reindex(df.index).to_numpy()

    def fit(self, df: pd.DataFrame, raw_col: str = "p_raw") -> LogitKalmanFilter:
        """Pick ``q`` (with ``r`` fixed at 1) by minimising validation log loss."""
        from .evaluate import log_loss_safe

        y = df["label"].to_numpy(dtype="float64")
        best_q, best_loss = self.q, np.inf
        for q in self.grid:
            self.q = q
            loss = log_loss_safe(y, self.transform(df, raw_col))
            if loss < best_loss:
                best_q, best_loss = q, loss
        self.q = best_q
        self.val_log_loss_ = float(best_loss)
        return self
