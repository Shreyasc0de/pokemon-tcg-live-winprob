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
    strat = t("strength_stratified")
    res = json.loads((rp / "results.json").read_text())
    des = json.loads((rp / "descriptive.json").read_text())
    stg = json.loads((rp / "strength.json").read_text())

    chk = Checker()

    # Headline table: mean and spread over five episode splits.
    for name, (brier, sd, ll, auc, ece, skill) in {
        "prize_only": (0.2118, 0.0043, 0.6098, 0.711, 0.022, 0.0),
        "gbdt": (0.1752, 0.0050, 0.5158, 0.812, 0.011, 0.173),
        "gbdt_filtered": (0.1750, 0.0050, 0.5155, 0.812, 0.012, 0.174),
        "gbdt_isotonic": (0.1750, 0.0049, 0.5149, 0.812, 0.011, 0.174),
        "gbdt_isotonic_filtered": (0.1750, 0.0049, 0.5151, 0.812, 0.013, 0.174),
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
        "gbdt": (-0.0126, 0.0013),
        "gbdt_filtered": (0.0028, 0.0005),
        "gbdt_isotonic": (-0.0087, 0.0011),
        "gbdt_isotonic_filtered": (0.0046, 0.0004),
    }.items():
        chk(f"{name} martingale slope", rs.loc[name, "martingale_slope_mean"], slope)
        chk(f"{name} martingale sd", rs.loc[name, "martingale_slope_sd"], sd)
    # The filtered path's residual slope is now detectable on every split: the
    # test gained power with n, the forecast did not get worse. README quotes
    # the range, so check the order of magnitude at both ends.
    chk("filtered martingale p min", rs.loc["gbdt_filtered", "martingale_p_min"], 6e-17, 1e-17)
    chk("filtered martingale p max", rs.loc["gbdt_filtered", "martingale_p_max"], 3e-07, 1e-07)

    # The paired comparison the calibration claim rests on.
    brier = lg.pivot(index="split", columns="model", values="brier")
    ece = lg.pivot(index="split", columns="model", values="ece")
    mart_p = lg.pivot(index="split", columns="model", values="martingale_p")
    d_brier = brier["gbdt_isotonic"] - brier["gbdt"]
    d_ece = ece["gbdt_isotonic"] - ece["gbdt"]
    for i, v in enumerate([0.00011, -0.00013, -0.00068, -0.00010, -0.00034]):
        chk(f"paired dBrier split {i}", d_brier.iloc[i], v)
    for i, v in enumerate([0.0003, 0.0036, -0.0035, 0.0013, -0.0016]):
        chk(f"paired dECE split {i}", d_ece.iloc[i], v)
    chk("isotonic wins Brier on 4 of 5", int((d_brier < 0).sum()), 4, 0)
    chk("isotonic wins ECE on 2 of 5", int((d_ece < 0).sum()), 2, 0)
    chk("mean dECE is ~zero", float(d_ece.mean()), 0.000005, 5e-06)
    chk("raw+filter significant in 5 of 5", int((mart_p["gbdt_filtered"] < 0.05).sum()), 5, 0)
    chk("shipped significant in 5 of 5", int((mart_p["gbdt_isotonic_filtered"] < 0.05).sum()), 5, 0)

    # Ablation and grouped importance.
    for i, v in enumerate([0.1778, 0.1762, 0.1766, 0.1758]):
        chk(f"ablation row {i}", ab.iloc[i].brier, v)
    for fam, v in {
        "prizes": 0.0447,
        "hit_points": 0.0265,
        "energy_and_evolution": 0.0167,
        "card_economy": 0.0144,
    }.items():
        chk(f"importance {fam}", imp.loc[fam, "brier_increase"], v)
    chk("importance status_conditions", imp.loc["status_conditions", "brier_increase"], 0.000)

    # Phase dependence, the baseline lookup, and the unseen-agent check.
    chk("turn 0-2 brier", bt.iloc[0].brier, 0.245)
    chk("turn 0-2 auc", bt.iloc[0].auc, 0.561)
    chk("lookup 2 behind", lk.loc[-2, "win_rate"], 0.27)
    chk("lookup 2 ahead", lk.loc[2, "win_rate"], 0.86)
    u = res["unseen_agent"]
    chk("unseen brier", u["brier"], 0.1833)
    chk("unseen skill", u["brier_skill_vs_ref"], 0.129)
    chk("unseen episodes", u["n_test_episodes"], 237, 0)

    # Archive descriptives.
    fpa = des["first_player_advantage"]
    chk("first-player wins", fpa["first_player_wins"], 2424, 0)
    chk("first-player games", fpa["n_games"], 4516, 0)
    chk("first-player rate", fpa["win_rate"], 0.537)
    chk("first-player ci lo", fpa["ci_lo"], 0.522)
    chk("first-player ci hi", fpa["ci_hi"], 0.551)
    chk("first-player p", fpa["p_value_vs_half"], 8.3e-07, 1e-07)
    chk("episodes", des["n_episodes"], 4518, 0)
    chk("decision points", des["n_decision_points"], 751012, 0)
    chk("agents", des["n_agents"], 183, 0)
    chk("cards", des["n_distinct_cards"], 187, 0)
    chk("median decision points", des["median_decision_points_per_game"], 166, 0)
    chk("test episodes", res["n_test_episodes"], 903, 0)
    for route, v in {
        "prizes_taken": 0.878,
        "no_pokemon_left": 0.052,
        "deck_out": 0.019,
        "unclear": 0.050,
    }.items():
        chk(f"win route {route}", wr.loc[route, "share"], v)
    same = cards[cards.decks == 74]
    chk("archetype trio decks", len(same) >= 3, True, 0)
    chk("archetype trio win rate", same.iloc[0].win_rate, 0.622)
    chk("median turns", glen.loc["50%", "max_turn"], 12, 0)
    chk("median steps", glen.loc["50%", "n_steps"], 171, 0)
    chk("longest game steps", glen.loc["max", "n_steps"], 1085, 0)

    # Finding 4: accuracy against agent rating.
    for i, (lo, hi, bm, bp, sk) in enumerate([
        (1053, 1075, 0.1697, 0.2006, 0.154),
        (1075, 1100, 0.1688, 0.1987, 0.151),
        (1100, 1138, 0.1709, 0.2037, 0.161),
        (1138, 1277, 0.1732, 0.2200, 0.212),
    ]):
        r = strat.iloc[i]
        chk(f"strat Q{i + 1} score lo", r.score_lo, lo, 0.5)
        chk(f"strat Q{i + 1} score hi", r.score_hi, hi, 0.5)
        chk(f"strat Q{i + 1} brier model", r.brier_model, bm)
        chk(f"strat Q{i + 1} brier prize", r.brier_prize_only, bp)
        chk(f"strat Q{i + 1} skill", r.brier_skill, sk)
        chk(f"strat Q{i + 1} episodes", r.episodes, [226, 226, 225, 226][i], 0)
    chk("strat model slope", stg["model_brier_vs_score"]["slope"], 4.3e-05, 5e-07)
    chk("strat model stderr", stg["model_brier_vs_score"]["stderr"], 6.9e-05, 5e-07)
    chk("strat model p", stg["model_brier_vs_score"]["p_value"], 0.53)
    chk("strat model r2", stg["model_brier_vs_score"]["r_squared"], 0.0004)
    chk("strat prize slope", stg["prize_only_brier_vs_score"]["slope"], 1.3e-04, 5e-06)
    chk("strat prize p", stg["prize_only_brier_vs_score"]["p_value"], 0.044)
    chk("strat gain p", stg["brier_gain_vs_score"]["p_value"], 0.11)
    chk("strat span", stg["score_span"], 225, 0.5)
    chk("strat episodes", stg["n_episodes"], 903, 0)

    if chk.failures:
        print(f"{len(chk.failures)} of {chk.checked} README figures do not match:\n", file=sys.stderr)
        for f in chk.failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"all {chk.checked} README figures match the generated reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
