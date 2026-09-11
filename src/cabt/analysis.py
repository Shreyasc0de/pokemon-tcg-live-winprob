"""Descriptive analysis of the archive: seating, agents, cards, game length.

These are the context numbers a win-probability model needs to be read
against. If the first player wins 60% of games, a model that outputs 0.60 on
turn one is not being clever, and a card that appears in 90% of decks cannot
explain much variance in who wins.

Every rate is reported with a Wilson interval and an explicit denominator.
Card and agent effects here are **descriptive, not causal**: decks are chosen,
not assigned, so a card's win rate mixes the card's contribution with the
skill of the agents that pick it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import _wilson


def _rate_table(wins: pd.Series, n: pd.Series) -> pd.DataFrame:
    lo, hi = zip(*[_wilson(w, int(c)) for w, c in zip(wins, n, strict=True)], strict=True)
    return pd.DataFrame(
        {"n": n.astype(int), "wins": wins.astype(int), "win_rate": wins / n, "ci_lo": lo, "ci_hi": hi}
    )


def first_player_advantage(episodes: pd.DataFrame) -> dict:
    """Does moving first win more often? Two-sided binomial test."""
    played = episodes.dropna(subset=["winner"])
    played = played[played["first_player"] >= 0]
    n = len(played)
    if n == 0:
        return {"n": 0}
    wins = int((played["winner"] == played["first_player"]).sum())
    lo, hi = _wilson(wins, n)
    return {
        "n_games": n,
        "first_player_wins": wins,
        "win_rate": wins / n,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_value_vs_half": float(stats.binomtest(wins, n, 0.5).pvalue),
        "seat_0_is_always_first": bool((played["first_player"] == 0).all()),
    }


def outcome_mix(episodes: pd.DataFrame) -> pd.DataFrame:
    """How games end: clean finishes, draws, and non-DONE statuses."""
    rows = []
    for col in ("status_0", "status_1"):
        rows.append(episodes[col].value_counts().rename("count").to_frame().assign(seat=col))
    out = pd.concat(rows).reset_index(names="status")
    return out.pivot_table(index="status", columns="seat", values="count", fill_value=0).reset_index()


def game_length(episodes: pd.DataFrame) -> pd.DataFrame:
    """Distribution of game length in turns and in replay steps."""
    d = episodes[["max_turn", "n_steps"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    return d.reset_index(names="stat")


def winning_margin(episodes: pd.DataFrame) -> pd.DataFrame:
    """Winner's prize count in their **last observed state**.

    Important caveat: a replay's final observation for a seat precedes the
    terminal play, so this is the count *before* the winning move, not after.
    Observed prize counts in the archive bottom out at 1, never 0, which
    confirms it -- the game ends on the take that would have shown 0.

    Read it as: 1 left means the winner finished by taking a single last
    prize; 2 left means they finished by taking two at once (a knockout on a
    two-prize Pokemon); 3 or more means they won without exhausting prizes at
    all, which ``win_route`` characterises.
    """
    played = episodes.dropna(subset=["winner"]).copy()
    played["winner_prizes_left"] = np.where(
        played["winner"] == 0, played["final_prizes_0"], played["final_prizes_1"]
    )
    counts = played["winner_prizes_left"].value_counts().sort_index()
    return pd.DataFrame(
        {
            "winner_prizes_left": counts.index.astype(int),
            "games": counts.to_numpy(),
            "share": counts.to_numpy() / counts.sum(),
        }
    )


def win_route(episodes: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Classify how each game ended, from the losing seat's final state.

    Three routes are distinguishable: the winner took their last prize(s); the
    loser had no Pokemon left to promote (their bench was empty behind a lone
    active); or the loser was out of cards to draw. Because the final
    observation is one play stale, a residual group stays ``unclear`` rather
    than being forced into a bucket.
    """
    played = episodes.dropna(subset=["winner"]).copy()
    played["winner_prizes_left"] = np.where(
        played["winner"] == 0, played["final_prizes_0"], played["final_prizes_1"]
    )
    last = (
        decisions.sort_values("step_index")
        .groupby(["episode_id", "acting_player"], sort=False)
        .tail(1)
    )
    losers = last[last["label"] == 0].set_index("episode_id")

    routes = []
    for row in played.itertuples():
        if row.winner_prizes_left <= 2:
            routes.append("prizes_taken")
            continue
        state = losers.loc[row.episode_id] if row.episode_id in losers.index else None
        if state is None:
            routes.append("unclear")
        elif state["my_deck"] <= 1:
            routes.append("deck_out")
        elif state["my_bench"] == 0:
            routes.append("no_pokemon_left")
        else:
            routes.append("unclear")
    counts = pd.Series(routes).value_counts()
    return pd.DataFrame(
        {"route": counts.index, "games": counts.to_numpy(), "share": counts.to_numpy() / counts.sum()}
    )


def agent_records(episodes: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    """Per-agent win/loss record with Wilson intervals.

    Games per agent are wildly unequal in a leaderboard archive, so the
    interval matters more than the point estimate.
    """
    played = episodes.dropna(subset=["winner"])
    rows = []
    for seat in (0, 1):
        rows.append(
            pd.DataFrame(
                {
                    "agent": played[f"agent_{seat}"],
                    "won": (played["winner"] == seat).astype(int),
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    g = long.groupby("agent")["won"]
    tab = _rate_table(g.sum(), g.size())
    tab = tab[tab["n"] >= min_games].sort_values("win_rate", ascending=False)
    return tab.reset_index()


def elo(episodes: pd.DataFrame, k: float = 24.0, start: float = 1500.0) -> pd.DataFrame:
    """Sequential Elo over the archive, in episode-id order.

    Episode ids are monotone in time, so this is a genuine online rating
    rather than a batch fit. It is a convenience summary, not a feature: the
    win-probability model never sees agent identity.
    """
    played = episodes.dropna(subset=["winner"]).sort_values("episode_id")
    ratings: dict[str, float] = {}
    games: dict[str, int] = {}
    for row in played.itertuples():
        a, b = row.agent_0, row.agent_1
        ra, rb = ratings.get(a, start), ratings.get(b, start)
        exp_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        sa = 1.0 if row.winner == 0 else 0.0
        ratings[a] = ra + k * (sa - exp_a)
        ratings[b] = rb + k * ((1 - sa) - (1 - exp_a))
        games[a] = games.get(a, 0) + 1
        games[b] = games.get(b, 0) + 1
    out = pd.DataFrame({"agent": list(ratings), "elo": list(ratings.values())})
    out["games"] = out["agent"].map(games)
    return out.sort_values("elo", ascending=False).reset_index(drop=True)


def card_win_rates(
    decks: pd.DataFrame,
    card_names: dict[int, str] | None = None,
    min_decks: int = 30,
) -> pd.DataFrame:
    """Inclusion rate and win rate per card, one row per deck-appearance.

    Descriptive only. A high win rate here means "decks containing this card
    won more often", which conflates the card with everything else its pilots
    did.
    """
    d = decks.dropna(subset=["won"])
    g = d.groupby("card_id")["won"]
    tab = _rate_table(g.sum(), g.size())
    tab = tab.rename(columns={"n": "decks", "wins": "deck_wins"})
    tab["inclusion_rate"] = tab["decks"] / d["episode_id"].nunique() / 2
    copies = d.groupby("card_id")["copies"].mean().rename("mean_copies")
    tab = tab.join(copies)
    tab = tab[tab["decks"] >= min_decks]
    tab = tab.reset_index()
    if card_names:
        tab["card"] = tab["card_id"].map(lambda c: card_names.get(int(c), f"#{int(c)}"))
    return tab.sort_values("win_rate", ascending=False).reset_index(drop=True)


def turn_state_win_rates(decisions: pd.DataFrame) -> pd.DataFrame:
    """Empirical win rate by prize differential -- the model's main competitor.

    This is the lookup table a player builds by intuition: "I'm two prizes up,
    how often does that hold?" Any model has to beat it.
    """
    d = decisions.copy()
    d["prize_diff_clipped"] = d["prize_diff"].clip(-4, 4)
    g = d.groupby("prize_diff_clipped")["label"]
    tab = _rate_table(g.sum(), g.size())
    return tab.reset_index()
