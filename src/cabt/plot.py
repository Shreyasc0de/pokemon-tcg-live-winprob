"""Figures for the report.

Shared conventions: a light off-white surface, hairline recessive grid and
axes, thin marks, a legend whenever two series share a panel, and selective
direct labels rather than a number on every point. The three categorical hues
are used in fixed order and were validated for colour-vision deficiency
separation before use, not chosen by eye.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
REFERENCE = "#b9b7b0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
    }
)


def _tidy(ax, hide_x_grid: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if hide_x_grid:
        ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def winprob_curves(
    preds: pd.DataFrame,
    episode_ids: list[int],
    path: str | Path,
    raw_col: str = "p_gbdt",
    filtered_col: str = "p_gbdt_filtered",
) -> Path:
    """Win-probability path for up to four matches, one panel each.

    Both seats are folded onto a single "seat 0 wins" axis: at a decision
    point taken by seat 1 the model's output is that seat's win probability, so
    it is flipped. That makes each panel one continuous line rather than two
    interleaved ones.
    """
    episode_ids = episode_ids[:4]
    n = len(episode_ids)
    fig, axes = plt.subplots(
        (n + 1) // 2, 2, figsize=(10.5, 3.1 * ((n + 1) // 2)), squeeze=False, sharey=True
    )
    flat = [a for row in axes for a in row]

    for ax, eid in zip(flat, episode_ids, strict=False):
        game = preds[preds["episode_id"] == eid].sort_values("step_index")
        flip = game["acting_player"].to_numpy() == 1
        p_raw = np.where(flip, 1 - game[raw_col], game[raw_col])
        p_filt = np.where(flip, 1 - game[filtered_col], game[filtered_col])
        x = np.arange(len(game))

        ax.axhline(0.5, color=REFERENCE, lw=0.8, zorder=1)
        ax.plot(x, p_raw, color=SERIES[1], lw=1.0, alpha=0.55, zorder=2, label="per-state model")
        ax.plot(x, p_filt, color=SERIES[0], lw=1.6, zorder=3, label="filtered path")

        # Mark decision points where the prize differential moved: the events a
        # viewer would recognise as the turning points.
        moved = game["d_prize_diff"].fillna(0).to_numpy() != 0
        if moved.any():
            ax.plot(
                x[moved],
                p_filt[moved],
                "o",
                ms=4.5,
                mfc=SERIES[0],
                mec=SURFACE,
                mew=1.4,
                zorder=4,
                label="prize taken",
            )

        winner_seat = 0 if game["label"].iloc[0] == (game["acting_player"].iloc[0] == 0) else 1
        won = "seat 0 won" if winner_seat == 0 else "seat 1 won"
        ax.set_title(f"Episode {eid}: {won}", color=INK)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(1, len(game) - 1))
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("decision point")
        _tidy(ax)

    for ax in flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(seat 0 wins)")

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.995, 0.985))
    fig.suptitle(
        "Live win probability, updated at every decision point",
        x=0.005,
        ha="left",
        color=INK,
        fontsize=13,
        fontweight="600",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def filter_zoom(
    preds: pd.DataFrame,
    episode_id: int,
    path: str | Path,
    window: tuple[int, int] = (0, 70),
    raw_col: str = "p_gbdt",
    tuned_q: float = 0.5,
    heavy_q: float = 0.05,
) -> Path:
    """A short window of one match, showing what the filter chose not to do.

    Three paths: the raw per-state model, the filter at the ``q`` selected by
    validation log loss, and the same filter at a much stronger smoothing
    prior. The tuned filter tracks the raw path closely -- it removes
    intra-turn jitter and little else. The heavy filter produces the smooth
    broadcast-style line people expect, and is worse on every scoring rule,
    because most of what looks like noise at this zoom is real information
    arriving.
    """
    from .models import LogitKalmanFilter

    game = preds[preds["episode_id"] == episode_id].sort_values("step_index").copy()
    paths = {}
    for label, q in (("tuned", tuned_q), ("heavy", heavy_q)):
        paths[label] = LogitKalmanFilter(q=q).transform(game, raw_col)

    lo, hi = window
    sl = slice(lo, hi)
    flip = game["acting_player"].to_numpy()[sl] == 1

    def orient(v):
        return np.where(flip, 1 - v[sl], v[sl])

    p_raw = orient(game[raw_col].to_numpy())
    p_tuned, p_heavy = orient(paths["tuned"]), orient(paths["heavy"])
    x = np.arange(len(p_raw)) + lo

    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.axhline(0.5, color=REFERENCE, lw=0.8, zorder=1)
    ax.plot(x, p_raw, "-o", ms=3.0, color=SERIES[1], lw=1.0, mec=SURFACE, mew=0.6,
            zorder=2, label="per-state model (raw)")
    ax.plot(x, p_tuned, "-", color=SERIES[0], lw=2.0, zorder=4,
            label=f"filter, tuned (q={tuned_q:g})")
    ax.plot(x, p_heavy, "-", color=SERIES[2], lw=1.8, zorder=3,
            label=f"filter, heavy smoothing (q={heavy_q:g})")

    ax.set_xlabel("decision point")
    ax.set_ylabel("P(seat 0 wins)")
    ax.set_title(f"Episode {episode_id}, decision points {lo}\u2013{lo + len(p_raw)}")
    span = np.concatenate([p_raw, p_tuned, p_heavy])
    pad = max(0.04, 0.12 * (span.max() - span.min()))
    ax.set_ylim(max(0.0, span.min() - pad), min(1.0, span.max() + pad))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.legend(loc="upper left", ncols=3)
    _tidy(ax)
    return _save(fig, path)


def calibration(tables: dict[str, pd.DataFrame], path: str | Path) -> Path:
    """Reliability diagram with Wilson intervals on the observed frequency."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    # Named in the legend rather than annotated in place: every position near
    # the diagonal collides with one of the series.
    ax.plot([0, 1], [0, 1], color=REFERENCE, lw=0.9, zorder=1, label="perfect calibration")

    for (name, tab), colour in zip(tables.items(), SERIES, strict=False):
        ax.errorbar(
            tab["predicted"],
            tab["observed"],
            yerr=[tab["observed"] - tab["obs_lo"], tab["obs_hi"] - tab["observed"]],
            fmt="o-",
            ms=5,
            lw=1.6,
            elinewidth=0.9,
            capsize=2.5,
            color=colour,
            mec=SURFACE,
            mew=1.2,
            label=name,
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("observed win frequency")
    ax.set_title("Calibration on held-out games")
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1, len(labels))) + [0]  # series first, reference last
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left")
    _tidy(ax, hide_x_grid=False)
    return _save(fig, path)


def brier_by_turn(by_turn: pd.DataFrame, baseline: float, path: str | Path) -> Path:
    """Brier score per turn bucket against a whole-sample reference line."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(by_turn))
    ax.bar(x, by_turn["brier"], width=0.56, color=SERIES[0], zorder=3)
    ax.axhline(baseline, color=SERIES[1], lw=1.4, zorder=4)
    ax.text(
        len(by_turn) - 0.4,
        baseline,
        f"  prize-only baseline ({baseline:.3f})",
        color=SERIES[1],
        fontsize=8.5,
        va="center",
    )
    for xi, b in zip(x, by_turn["brier"], strict=True):
        ax.text(xi, b + 0.004, f"{b:.3f}", ha="center", color=INK_2, fontsize=8)

    labels = []
    for bucket, n in zip(by_turn["turn_bucket"], by_turn["n"], strict=True):
        text = bucket.replace("[", "").replace(")", "").replace(", ", "\u2013")
        first, _, last = text.partition("\u2013")
        if int(float(last)) > 60:  # open-ended final bucket
            text = f"{int(float(first))}+"
        labels.append(f"{text}\nn={int(n):,}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("turn")
    ax.set_ylabel("Brier score (lower is better)")
    ax.tick_params(axis="x", length=0)
    ax.set_title("The forecast sharpens as the game resolves")
    ax.set_ylim(0, max(by_turn["brier"].max(), baseline) * 1.18)
    _tidy(ax)
    return _save(fig, path)


def feature_importance(imp: pd.DataFrame, path: str | Path) -> Path:
    """Grouped permutation importance, in Brier increase when shuffled."""
    d = imp.sort_values("brier_increase")
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(d) + 1.6))
    y = np.arange(len(d))
    ax.barh(y, d["brier_increase"], height=0.56, color=SERIES[0], zorder=3)
    ax.errorbar(
        d["brier_increase"],
        y,
        xerr=d["brier_increase_sd"],
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=0.9,
        capsize=2.5,
        zorder=4,
    )
    span = max(d["brier_increase"].max(), 1e-6)
    for yi, v in zip(y, d["brier_increase"], strict=True):
        if v >= 0:
            ax.text(v + span * 0.02, yi, f"{v:+.4f}", va="center", ha="left", color=INK_2, fontsize=8)
        else:
            ax.text(v - span * 0.02, yi, f"{v:+.4f}", va="center", ha="right", color=INK_2, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in d["family"]])
    ax.axvline(0, color=GRID, lw=0.9)
    ax.set_xlabel("Brier score increase when the family is shuffled")
    ax.set_title("What the model is actually using")
    ax.set_xlim(min(0, d["brier_increase"].min() * 3.5), span * 1.24)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    return _save(fig, path)


def prize_lookup(lookup: pd.DataFrame, model_curve: pd.DataFrame | None, path: str | Path) -> Path:
    """Empirical win rate by prize differential, with the model's mean output."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = lookup["prize_diff_clipped"].to_numpy()
    ax.axhline(0.5, color=REFERENCE, lw=0.8, zorder=1)
    ax.errorbar(
        x,
        lookup["win_rate"],
        yerr=[lookup["win_rate"] - lookup["ci_lo"], lookup["ci_hi"] - lookup["win_rate"]],
        fmt="o-",
        ms=6,
        lw=1.8,
        elinewidth=1.0,
        capsize=3,
        color=SERIES[0],
        mec=SURFACE,
        mew=1.2,
        label="observed win rate",
        zorder=3,
    )
    if model_curve is not None:
        ax.plot(
            model_curve["prize_diff_clipped"],
            model_curve["mean_pred"],
            "s--",
            ms=5,
            lw=1.5,
            color=SERIES[1],
            mec=SURFACE,
            mew=1.2,
            label="model mean prediction",
            zorder=2,
        )
    for xi, (r, n) in enumerate(zip(lookup["win_rate"], lookup["n"], strict=True)):
        ax.annotate(
            f"n={int(n):,}",
            (x[xi], r),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color=INK_MUTED,
            fontsize=7.5,
        )
    ax.set_xticks(x)
    ax.set_xlabel("prize differential (opponent's prizes left − mine; positive = I'm ahead)")
    ax.set_ylabel("P(win)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_title("The baseline a player already computes in their head")
    ax.legend(loc="upper left")
    _tidy(ax, hide_x_grid=False)
    return _save(fig, path)


def strength_stratified(table: pd.DataFrame, path: str | Path) -> Path:
    """Brier by agent-strength quartile: the model flat, the baseline drifting."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(table))
    ax.plot(x, table["brier_prize_only"], "o-", color=REFERENCE, lw=1.6, label="prize differential only")
    ax.plot(x, table["brier_model"], "o-", color=SERIES[0], lw=2.0, label="gbdt + filter")
    for i, row in table.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{row['brier_skill']:.1%} skill",
            (i, row["brier_model"]),
            textcoords="offset points", xytext=(0, -16),
            ha="center", fontsize=8, color=SERIES[0],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r.bucket}\n{r.score_lo:.0f}-{r.score_hi:.0f}" for r in table.itertuples()],
        fontsize=8,
    )
    ax.set_xlabel("held-out episodes by mean agent rating (quartiles)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Accuracy holds across the rated band; the baseline does not")
    ax.legend(frameon=False, fontsize=9)
    _tidy(ax)
    return _save(fig, path)
