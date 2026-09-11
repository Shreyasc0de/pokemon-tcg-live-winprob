"""Feature extraction for a single decision point.

Every feature is computed from ``observation.current`` for the player who is
*about to act*, i.e. strictly from information that player can actually see.
Opponent hand contents are masked in the replay and stay masked here; only the
public count is used. The fully-revealed ``visualize`` payload is never touched.

Two conventions make the dataset usable by a symmetric model:

* **Perspective.** ``my_*`` always refers to the acting player and ``opp_*`` to
  their opponent, so one row is one (state, acting player) pair and the label
  is "did the acting player go on to win". Both seats of a match therefore
  contribute rows, and the model cannot learn "seat 0 wins more often" by
  accident -- seat is exposed explicitly as ``is_first_player``.
* **No clock leakage.** Nothing derived from the step index, the episode
  length, or the replay's own ``result`` field enters the feature set. An
  episode terminates the moment someone wins, so "how far in are we" is a
  direct function of the outcome.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .schema import STARTING_PRIZES

STATUS_FLAGS = ("asleep", "burned", "confused", "paralyzed", "poisoned")

#: Feature groups, so ablations can add or drop a whole family at once.
#: ``core`` is pure game state. ``decision`` describes the menu being answered.
#: ``clock`` is the agent's remaining thinking-time bank -- real information a
#: player has, but a competition artefact rather than a game-state variable.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "turn",
        "turn_action_count",
        "is_first_player",
        "my_prizes",
        "opp_prizes",
        "prize_diff",
        "prizes_taken_total",
        "my_bench",
        "opp_bench",
        "my_bench_space",
        "opp_bench_space",
        "my_board_pokemon",
        "opp_board_pokemon",
        "board_pokemon_diff",
        "my_active_present",
        "opp_active_present",
        "my_hidden_pokemon",
        "opp_hidden_pokemon",
        "my_active_hp",
        "opp_active_hp",
        "my_active_max_hp",
        "opp_active_max_hp",
        "my_active_hp_frac",
        "opp_active_hp_frac",
        "active_hp_frac_diff",
        "my_total_hp",
        "opp_total_hp",
        "total_hp_diff",
        "my_total_max_hp",
        "opp_total_max_hp",
        "my_board_damage_frac",
        "opp_board_damage_frac",
        "my_active_energy",
        "opp_active_energy",
        "my_board_energy",
        "opp_board_energy",
        "board_energy_diff",
        "my_active_stage",
        "opp_active_stage",
        "my_active_tools",
        "opp_active_tools",
        "my_hand",
        "opp_hand",
        "hand_diff",
        "my_deck",
        "opp_deck",
        "deck_diff",
        "my_discard",
        "opp_discard",
        "discard_diff",
        "my_status_count",
        "opp_status_count",
        "my_asleep",
        "my_burned",
        "my_confused",
        "my_paralyzed",
        "my_poisoned",
        "opp_asleep",
        "opp_burned",
        "opp_confused",
        "opp_paralyzed",
        "opp_poisoned",
        "stadium_present",
        "stadium_is_mine",
        "supporter_played",
        "energy_attached",
        "retreated",
    ),
    "decision": (
        "n_legal_options",
        "select_type",
        "select_context",
        "menu_min_count",
        "menu_max_count",
        "remain_energy_cost",
        "remain_damage_counter",
        "opt_has_attack",
        "opt_has_retreat",
        "opt_has_ability",
        "opt_has_attach_energy",
        "opt_has_play_card",
        "opt_only_pass",
    ),
    "clock": ("my_overage_time",),
}

#: Feature families for grouped permutation importance.
IMPORTANCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "prizes": ("my_prizes", "opp_prizes", "prize_diff", "prizes_taken_total"),
    "board_presence": (
        "my_bench",
        "opp_bench",
        "my_bench_space",
        "opp_bench_space",
        "my_board_pokemon",
        "opp_board_pokemon",
        "board_pokemon_diff",
        "my_active_present",
        "opp_active_present",
        "my_hidden_pokemon",
        "opp_hidden_pokemon",
    ),
    "hit_points": (
        "my_active_hp",
        "opp_active_hp",
        "my_active_max_hp",
        "opp_active_max_hp",
        "my_active_hp_frac",
        "opp_active_hp_frac",
        "active_hp_frac_diff",
        "my_total_hp",
        "opp_total_hp",
        "total_hp_diff",
        "my_total_max_hp",
        "opp_total_max_hp",
        "my_board_damage_frac",
        "opp_board_damage_frac",
    ),
    "energy_and_evolution": (
        "my_active_energy",
        "opp_active_energy",
        "my_board_energy",
        "opp_board_energy",
        "board_energy_diff",
        "my_active_stage",
        "opp_active_stage",
        "my_active_tools",
        "opp_active_tools",
    ),
    "card_economy": (
        "my_hand",
        "opp_hand",
        "hand_diff",
        "my_deck",
        "opp_deck",
        "deck_diff",
        "my_discard",
        "opp_discard",
        "discard_diff",
    ),
    "status_conditions": (
        "my_status_count",
        "opp_status_count",
        "my_asleep",
        "my_burned",
        "my_confused",
        "my_paralyzed",
        "my_poisoned",
        "opp_asleep",
        "opp_burned",
        "opp_confused",
        "opp_paralyzed",
        "opp_poisoned",
    ),
    "tempo_and_turn": (
        "turn",
        "turn_action_count",
        "is_first_player",
        "supporter_played",
        "energy_attached",
        "retreated",
        "stadium_present",
        "stadium_is_mine",
    ),
    "decision_menu": FEATURE_GROUPS["decision"],
    "clock": FEATURE_GROUPS["clock"],
}


def feature_names(groups: tuple[str, ...] = ("core", "decision", "clock")) -> list[str]:
    """Ordered feature names for the requested groups."""
    out: list[str] = []
    for g in groups:
        out.extend(FEATURE_GROUPS[g])
    return out


def _count(zone: Any) -> int:
    """Length of a replay zone, treating a masked (``None``) zone as empty."""
    return 0 if zone is None else len(zone)


class ZoneStats(NamedTuple):
    """Aggregates over one Pokemon zone (active, bench, or both).

    During setup the opponent's active Pokemon is placed face down and the
    replay represents it as a ``None`` entry: the player knows a Pokemon is
    *there* but not what it is. ``n_hidden`` counts those slots and the
    numeric totals cover only the revealed ones, so "hidden" never gets
    silently encoded as "zero HP".
    """

    hp: int
    max_hp: int
    energy: int
    tools: int
    n: int
    n_hidden: int


def _pokemon_stats(mons: Any) -> ZoneStats:
    """Aggregate a Pokemon zone, tolerating face-down (``None``) entries."""
    mons = mons or []
    known = [m for m in mons if isinstance(m, dict)]
    return ZoneStats(
        hp=sum(int(m.get("hp") or 0) for m in known),
        max_hp=sum(int(m.get("maxHp") or 0) for m in known),
        energy=sum(_count(m.get("energies")) for m in known),
        tools=sum(_count(m.get("tools")) for m in known),
        n=len(mons),
        n_hidden=len(mons) - len(known),
    )


def _safe_frac(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den else default


def _hp_frac(z: ZoneStats) -> float:
    """Fraction of HP remaining in a zone.

    ``NaN`` when the zone holds only face-down Pokemon -- the quantity is
    genuinely unknown to the acting player. The gradient-boosted model reads
    NaN natively; the linear baselines impute it in-pipeline.
    """
    if z.max_hp:
        return z.hp / z.max_hp
    return float("nan") if z.n_hidden else 0.0


def _active_stage(active: Any) -> float:
    """Evolution stage of the active Pokemon (0 = basic), NaN if face down."""
    if not active:
        return 0.0
    mon = active[0]
    if not isinstance(mon, dict):
        return float("nan")
    return float(_count(mon.get("preEvolution")))


def state_features(current: dict, select: dict | None, overage: float | None) -> dict:
    """Build the feature dict for one decision point.

    Parameters
    ----------
    current:
        ``observation.current`` as seen by the acting player.
    select:
        ``observation.select``, the legal-action menu, or ``None``.
    overage:
        ``observation.remainingOverageTime`` for the acting player.
    """
    me_idx = int(current["yourIndex"])
    me = current["players"][me_idx]
    opp = current["players"][1 - me_idx]

    my_prizes, opp_prizes = _count(me.get("prize")), _count(opp.get("prize"))
    my_active, opp_active = me.get("active") or [], opp.get("active") or []
    my_bench, opp_bench = me.get("bench") or [], opp.get("bench") or []

    mine = _pokemon_stats(my_active + my_bench)
    theirs = _pokemon_stats(opp_active + opp_bench)
    my_a = _pokemon_stats(my_active)
    opp_a = _pokemon_stats(opp_active)

    my_a_frac = _hp_frac(my_a)
    opp_a_frac = _hp_frac(opp_a)
    stadium = current.get("stadium") or []
    first_player = int(current.get("firstPlayer", -1))

    f: dict[str, float] = {
        "turn": int(current.get("turn", 0)),
        "turn_action_count": int(current.get("turnActionCount", 0)),
        # -1 means "not yet decided" in the replay; map it to 0.5 so the model
        # sees an explicit "unknown" rather than a fake third seat.
        "is_first_player": 0.5 if first_player < 0 else float(first_player == me_idx),
        "my_prizes": my_prizes,
        "opp_prizes": opp_prizes,
        # Positive = the acting player has taken more prizes than the opponent.
        "prize_diff": opp_prizes - my_prizes,
        "prizes_taken_total": (STARTING_PRIZES - my_prizes) + (STARTING_PRIZES - opp_prizes),
        "my_bench": len(my_bench),
        "opp_bench": len(opp_bench),
        "my_bench_space": int(me.get("benchMax", 5)) - len(my_bench),
        "opp_bench_space": int(opp.get("benchMax", 5)) - len(opp_bench),
        "my_board_pokemon": len(my_active) + len(my_bench),
        "opp_board_pokemon": len(opp_active) + len(opp_bench),
        "board_pokemon_diff": (len(my_active) + len(my_bench)) - (len(opp_active) + len(opp_bench)),
        "my_active_present": float(bool(my_active)),
        "opp_active_present": float(bool(opp_active)),
        "my_hidden_pokemon": mine.n_hidden,
        "opp_hidden_pokemon": theirs.n_hidden,
        "my_active_hp": my_a.hp,
        "opp_active_hp": opp_a.hp,
        "my_active_max_hp": my_a.max_hp,
        "opp_active_max_hp": opp_a.max_hp,
        "my_active_hp_frac": my_a_frac,
        "opp_active_hp_frac": opp_a_frac,
        "active_hp_frac_diff": my_a_frac - opp_a_frac,
        "my_total_hp": mine.hp,
        "opp_total_hp": theirs.hp,
        "total_hp_diff": mine.hp - theirs.hp,
        "my_total_max_hp": mine.max_hp,
        "opp_total_max_hp": theirs.max_hp,
        "my_board_damage_frac": 1.0 - _hp_frac(mine),
        "opp_board_damage_frac": 1.0 - _hp_frac(theirs),
        "my_active_energy": my_a.energy,
        "opp_active_energy": opp_a.energy,
        "my_board_energy": mine.energy,
        "opp_board_energy": theirs.energy,
        "board_energy_diff": mine.energy - theirs.energy,
        "my_active_stage": _active_stage(my_active),
        "opp_active_stage": _active_stage(opp_active),
        "my_active_tools": my_a.tools,
        "opp_active_tools": opp_a.tools,
        # Own hand is visible; the opponent's is masked, so only its size is used.
        "my_hand": int(me.get("handCount") or 0),
        "opp_hand": int(opp.get("handCount") or 0),
        "hand_diff": int(me.get("handCount") or 0) - int(opp.get("handCount") or 0),
        "my_deck": int(me.get("deckCount") or 0),
        "opp_deck": int(opp.get("deckCount") or 0),
        "deck_diff": int(me.get("deckCount") or 0) - int(opp.get("deckCount") or 0),
        "my_discard": _count(me.get("discard")),
        "opp_discard": _count(opp.get("discard")),
        "discard_diff": _count(me.get("discard")) - _count(opp.get("discard")),
        "stadium_present": float(bool(stadium)),
        "stadium_is_mine": float(bool(stadium) and stadium[0].get("playerIndex") == me_idx),
        "supporter_played": float(bool(current.get("supporterPlayed"))),
        "energy_attached": float(bool(current.get("energyAttached"))),
        "retreated": float(bool(current.get("retreated"))),
    }

    my_status = 0
    opp_status = 0
    for flag in STATUS_FLAGS:
        mv, ov = float(bool(me.get(flag))), float(bool(opp.get(flag)))
        f[f"my_{flag}"], f[f"opp_{flag}"] = mv, ov
        my_status += int(mv)
        opp_status += int(ov)
    f["my_status_count"], f["opp_status_count"] = my_status, opp_status

    options = (select or {}).get("option") or []
    opt_types = {int(o.get("type", -1)) for o in options}
    f.update(
        {
            "n_legal_options": len(options),
            "select_type": int((select or {}).get("type", -1)),
            "select_context": int((select or {}).get("context", -1)),
            "menu_min_count": int((select or {}).get("minCount", 0) or 0),
            "menu_max_count": int((select or {}).get("maxCount", 0) or 0),
            "remain_energy_cost": int((select or {}).get("remainEnergyCost", 0) or 0),
            "remain_damage_counter": int((select or {}).get("remainDamageCounter", 0) or 0),
            "opt_has_attack": float(13 in opt_types),
            "opt_has_retreat": float(9 in opt_types),
            "opt_has_ability": float(10 in opt_types),
            "opt_has_attach_energy": float(12 in opt_types),
            "opt_has_play_card": float(7 in opt_types),
            "opt_only_pass": float(bool(opt_types) and opt_types == {14}),
            "my_overage_time": float(overage) if overage is not None else -1.0,
        }
    )
    return f
