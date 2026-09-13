#!/usr/bin/env python3
"""Descriptive analysis of the archive: seating, agents, cards, game length.

    python scripts/analyse.py --data data/processed --out reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cabt import analysis as A  # noqa: E402
from cabt import plot  # noqa: E402
from cabt.dataset import load_decisions, load_table  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    decisions = load_decisions(data)
    episodes = load_table(data, "episodes")
    names = {int(k): v for k, v in json.loads((data / "card_names.json").read_text()).items()}

    lookup = A.turn_state_win_rates(decisions)
    tables = {
        "descriptive_game_length": A.game_length(episodes),
        "descriptive_winning_margin": A.winning_margin(episodes),
        "descriptive_win_route": A.win_route(episodes, decisions),
        "descriptive_agent_records": A.agent_records(episodes),
        "descriptive_elo": A.elo(episodes),
        "descriptive_prize_diff_lookup": lookup,
        "descriptive_outcome_statuses": A.outcome_mix(episodes),
    }
    try:
        decks = load_table(data, "decks")
        tables["descriptive_card_win_rates"] = A.card_win_rates(decks, names)
    except FileNotFoundError:
        print("no decks table found; skipping card win rates")

    for name, table in tables.items():
        table.to_csv(out / "tables" / f"{name}.csv", index=False)

    summary = {
        "n_episodes": int(len(episodes)),
        "n_decision_points": int(len(decisions)),
        "n_agents": int(pd.concat([episodes["agent_0"], episodes["agent_1"]]).nunique()),
        "n_distinct_cards": len(names),
        "first_player_advantage": A.first_player_advantage(episodes),
        "median_turns": float(episodes["max_turn"].median()),
        "median_decision_points_per_game": float(decisions.groupby("episode_id").size().median()),
    }
    (out / "descriptive.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.no_figures:
        model_curve = None
        pred_path = out / "test_predictions.csv.gz"
        if pred_path.exists():
            preds = pd.read_csv(pred_path)
            preds["prize_diff_clipped"] = preds["prize_diff"].clip(-4, 4)
            model_curve = (
                preds.groupby("prize_diff_clipped")["p_gbdt_filtered"]
                .mean()
                .rename("mean_pred")
                .reset_index()
            )
        plot.prize_lookup(lookup, model_curve, out / "figures" / "prize_lookup.png")
        print(f"wrote {out / 'figures' / 'prize_lookup.png'}")

    # Strength stratification needs both the archive manifest and held-out
    # predictions, so it is skipped on the sample pipeline where neither the
    # manifest nor a trained model is present.
    manifest_path = Path("data/manifest.csv")
    pred_path = out / "test_predictions.csv.gz"
    if manifest_path.exists() and pred_path.exists():
        preds = pd.read_csv(
            pred_path,
            usecols=["episode_id", "label", "p_prize_only", "p_gbdt_filtered"],
        )
        try:
            strat, strat_summary = A.strength_stratified(preds, pd.read_csv(manifest_path))
        except ValueError as exc:  # too few matched episodes, e.g. the sample run
            print(f"[analyse] skipping strength stratification: {exc}")
            return 0
        strat.to_csv(out / "tables" / "strength_stratified.csv", index=False)
        (out / "strength.json").write_text(
            json.dumps(strat_summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(strat_summary, indent=2))
        if not args.no_figures:
            plot.strength_stratified(strat, out / "figures" / "strength_stratified.png")
            print(f"wrote {out / 'figures' / 'strength_stratified.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
