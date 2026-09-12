#!/usr/bin/env python3
"""Check every number quoted in README.md against the generated report tables.

A README is the part of a project people actually read, and it is the part that
silently goes stale when a model is retrained. This script re-derives each
quoted figure from `reports/` and fails loudly on any drift, so "the numbers in
the README are the numbers the pipeline produced" is a checked claim rather than
an assurance.

Tolerances are half a unit of the last quoted digit -- a README that rounds
0.18019 to 0.1802 is correct, and the check should say so.

    python scripts/verify_readme.py            # exits non-zero on any mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _tol(quoted: float) -> float:
    """Half a unit in the last decimal place of the quoted value."""
    text = f"{quoted!r}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** (-decimals) + 1e-12


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checked = 0

    def __call__(self, label: str, got: float, quoted: float, tol: float | None = None) -> None:
        self.checked += 1
        tol = _tol(quoted) if tol is None else tol
        if abs(float(got) - float(quoted)) > tol:
            self.failures.append(f"{label}: pipeline {got!r}, README {quoted!r} (tol {tol:g})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", default=str(ROOT / "reports"))
    args = ap.parse_args()
    rp = Path(args.reports)

    t = lambda n: pd.read_csv(rp / "tables" / f"{n}.csv")  # noqa: E731
    rs = t("repeated_splits").set_index("model")
    lg = t("repeated_splits_long")
    ab, imp = t("ablation"), t("feature_importance").set_index("family")
    bt, lk = t("metrics_by_turn"), t("descriptive_prize_diff_lookup").set_index("prize_diff_clipped")
    wr = t("descriptive_win_route").set_index("route")
    cards = t("descriptive_card_win_rates")
    glen = t("descriptive_game_length").set_index("stat")
    res = json.loads((rp / "results.json").read_text())
    des = json.loads((rp / "descriptive.json").read_text())

    chk = Checker()

    # Headline table: mean and spread over five episode splits.
    for name, (brier, sd, ll, auc, ece, skill) in {
        "prize_only": (0.2167, 0.0044, 0.6209, 0.695, 0.027, 0.0),
        "gbdt": (0.1819, 0.0044, 0.5336, 0.798, 0.022, 0.161),
        "gbdt_filtered": (0.1815, 0.0044, 0.5325, 0.799, 0.019, 0.162),
        "gbdt_isotonic": (0.1802, 0.0036, 0.5297, 0.801, 0.016, 0.169),
        "gbdt_isotonic_filtered": (0.1802, 0.0036, 0.5295, 0.801, 0.018, 0.169),
    }.items():
        r = rs.loc[name]
        chk(f"{name} brier", r.brier_mean, brier)
        chk(f"{name} brier sd", r.brier_sd, sd)
        chk(f"{name} log_loss", r.log_loss_mean, ll)
        chk(f"{name} auc", r.auc_mean, auc)
        chk(f"{name} ece", r.ece_mean, ece)
        chk(f"{name} skill", r.skill_mean, skill)

    # Martingale slopes across the same splits.
    for name, (slope, sd) in {
        "gbdt": (-0.0157, 0.0014),
        "gbdt_filtered": (0.0008, 0.0004),
        "gbdt_isotonic": (-0.0093, 0.0013),
        "gbdt_isotonic_filtered": (0.0043, 0.0006),
    }.items():
        chk(f"{name} martingale slope", rs.loc[name, "martingale_slope_mean"], slope)
        chk(f"{name} martingale sd", rs.loc[name, "martingale_slope_sd"], sd)
    chk("filtered martingale p min", rs.loc["gbdt_filtered", "martingale_p_min"], 0.02)
    chk("filtered martingale p max", rs.loc["gbdt_filtered", "martingale_p_max"], 0.67)

    # The paired comparison the calibration claim rests on.
    brier = lg.pivot(index="split", columns="model", values="brier")
    ece = lg.pivot(index="split", columns="model", values="ece")
    mart_p = lg.pivot(index="split", columns="model", values="martingale_p")
    d_brier = brier["gbdt_isotonic"] - brier["gbdt"]
    d_ece = ece["gbdt_isotonic"] - ece["gbdt"]
    for i, v in enumerate([-0.00022, -0.00067, -0.00238, -0.00275, -0.00269]):
        chk(f"paired dBrier split {i}", d_brier.iloc[i], v)
    for i, v in enumerate([-0.0024, -0.0042, -0.0013, -0.0113, -0.0102]):
        chk(f"paired dECE split {i}", d_ece.iloc[i], v)
    chk("isotonic wins Brier on 5 of 5", int((d_brier < 0).sum()), 5, 0)
    chk("isotonic wins ECE on 5 of 5", int((d_ece < 0).sum()), 5, 0)
    chk("raw+filter not significant in 4 of 5", int((mart_p["gbdt_filtered"] >= 0.05).sum()), 4, 0)
    chk("shipped significant in 5 of 5", int((mart_p["gbdt_isotonic_filtered"] < 0.05).sum()), 5, 0)

    # Ablation and grouped importance.
    for i, v in enumerate([0.1827, 0.1811, 0.1811, 0.1798]):
        chk(f"ablation row {i}", ab.iloc[i].brier, v)
    for fam, v in {
        "prizes": 0.0423,
        "hit_points": 0.0330,
        "energy_and_evolution": 0.0180,
        "card_economy": 0.0174,
    }.items():
        chk(f"importance {fam}", imp.loc[fam, "brier_increase"], v)
    chk("importance status_conditions", imp.loc["status_conditions", "brier_increase"], 0.000)

    # Phase dependence, the baseline lookup, and the unseen-agent check.
    chk("turn 0-2 brier", bt.iloc[0].brier, 0.239)
    chk("turn 0-2 auc", bt.iloc[0].auc, 0.620)
    chk("lookup 2 behind", lk.loc[-2, "win_rate"], 0.27)
    chk("lookup 2 ahead", lk.loc[2, "win_rate"], 0.85)
    u = res["unseen_agent"]
    chk("unseen brier", u["brier"], 0.1433)
    chk("unseen auc", u["auc"], 0.880)
    chk("unseen episodes", u["n_test_episodes"], 99, 0)

    # Archive descriptives.
    fpa = des["first_player_advantage"]
    chk("first-player wins", fpa["first_player_wins"], 1067, 0)
    chk("first-player games", fpa["n_games"], 2000, 0)
    chk("first-player rate", fpa["win_rate"], 0.534)
    chk("first-player ci lo", fpa["ci_lo"], 0.512)
    chk("first-player ci hi", fpa["ci_hi"], 0.555)
    chk("first-player p", fpa["p_value_vs_half"], 0.0029)
    chk("episodes", des["n_episodes"], 2000, 0)
    chk("decision points", des["n_decision_points"], 332814, 0)
    chk("agents", des["n_agents"], 151, 0)
    chk("cards", des["n_distinct_cards"], 179, 0)
    chk("median decision points", des["median_decision_points_per_game"], 167, 0)
    chk("test episodes", res["n_test_episodes"], 400, 0)
    for route, v in {
        "prizes_taken": 0.868,
        "no_pokemon_left": 0.055,
        "deck_out": 0.024,
        "unclear": 0.054,
    }.items():
        chk(f"win route {route}", wr.loc[route, "share"], v)
    chk("top card decks", cards.iloc[0].decks, 455, 0)
    chk("top card win rate", cards.iloc[0].win_rate, 0.613)
    chk("median turns", glen.loc["50%", "max_turn"], 12, 0)
    chk("median steps", glen.loc["50%", "n_steps"], 172, 0)
    chk("longest game steps", glen.loc["max", "n_steps"], 1085, 0)

    if chk.failures:
        print(f"{len(chk.failures)} of {chk.checked} README figures do not match:\n", file=sys.stderr)
        for f in chk.failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"all {chk.checked} README figures match the generated reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
