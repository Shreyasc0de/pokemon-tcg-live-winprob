#!/usr/bin/env python3
"""Fit the win-probability model ladder and write metrics, tables and figures.

    python scripts/train.py --data data/processed --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cabt import plot  # noqa: E402
from cabt.dataset import load_decisions  # noqa: E402
from cabt.pipeline import fit_and_score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    df = load_decisions(args.data)
    print(f"{len(df):,} decision points from {df['episode_id'].nunique():,} episodes")

    summary = fit_and_score(df, out, seed=args.seed)
    print(json.dumps(summary["metrics"], indent=2))
    print(f"\nwrote {out / 'results.json'} and {out / 'tables'}")

    if args.no_figures:
        return 0

    tables = out / "tables"
    figures = out / "figures"
    preds = pd.read_csv(out / "test_predictions.csv.gz")

    # Pick illustrative games: the longest paths make the clearest curves.
    lengths = preds.groupby("episode_id").size().sort_values(ascending=False)
    chosen = [int(e) for e in lengths.index[:2]] + [int(e) for e in lengths.index[len(lengths) // 2 : len(lengths) // 2 + 2]]
    plot.winprob_curves(preds, chosen, figures / "winprob_curves.png")
    plot.filter_zoom(
        preds,
        chosen[0],
        figures / "filter_zoom.png",
        window=(0, 70),
        tuned_q=float(summary["kalman_q"]),
        raw_col="p_gbdt_isotonic",
    )

    plot.calibration(
        {
            "uncalibrated": pd.read_csv(tables / "reliability_gbdt.csv"),
            "shipped (isotonic + filter)": pd.read_csv(tables / "reliability_gbdt_filtered.csv"),
        },
        figures / "calibration.png",
    )

    metrics = pd.read_csv(tables / "metrics.csv").set_index("model")
    plot.brier_by_turn(
        pd.read_csv(tables / "metrics_by_turn.csv"),
        float(metrics.loc["prize_only", "brier"]),
        figures / "brier_by_turn.png",
    )
    plot.feature_importance(pd.read_csv(tables / "feature_importance.csv"), figures / "feature_importance.png")
    print(f"wrote figures to {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
