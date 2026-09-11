"""Stream Kaggle `cabt` episode replays into flat per-decision-point records.

The archive is ~9.5 GB of JSON: 2,000 episodes averaging 4.75 MB each, of
which roughly 63% is the ``visualize`` payload. Episodes are therefore read and
reduced one file at a time and never held in memory together.

A **decision point** is one step at which a player is ``ACTIVE`` and holds a
non-empty legal-action menu. That is the natural unit for a live
win-probability model: it is exactly the moment a player would want a number.
Both seats contribute rows, always from the acting player's own point of view.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from .features import _count, state_features
from .schema import REWARD_DRAW, REWARD_WIN, STARTING_PRIZES


def load_episode(path: str | Path) -> dict:
    """Load one episode replay. Accepts ``.json`` and ``.json.gz``."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def episode_paths(root: str | Path) -> list[Path]:
    """All episode replays under ``root``, sorted by episode id."""
    root = Path(root)
    files = sorted(
        [*root.glob("*.json"), *root.glob("*.json.gz")],
        key=lambda p: p.name,
    )
    return files


def _label_for(rewards: list, player: int) -> float | None:
    """1.0 if ``player`` won, 0.0 if they lost, ``None`` for a draw."""
    r = rewards[player]
    if r == REWARD_DRAW:
        return None
    return 1.0 if r == REWARD_WIN else 0.0


def card_names(episode: dict) -> dict[int, str]:
    """Card ``id -> name`` map, read from the step-0 ``visualize`` payload.

    This is the only place names appear. Used for the descriptive metagame
    tables, never as a model input.
    """
    out: dict[int, str] = {}
    vis = (episode["steps"][0][0] or {}).get("visualize")
    if not vis:
        return out
    for player in vis[0]["current"]["players"]:
        for zone in ("deck", "hand", "active", "bench", "discard", "prize"):
            for card in player.get(zone) or []:
                if isinstance(card, dict) and card.get("name"):
                    out[int(card["id"])] = card["name"]
    return out


def decklists(episode: dict) -> list[list[int]] | None:
    """Both players' 60-card decklists as card ids, or ``None`` if absent.

    Episode metadata only. A player does not know their opponent's list, so
    this never reaches the feature matrix.
    """
    vis = (episode["steps"][0][0] or {}).get("visualize")
    if not vis or "action" not in vis[0]:
        return None
    return [[int(c) for c in side] for side in vis[0]["action"]]


def episode_meta(episode: dict) -> dict:
    """Episode-level summary: ids, agents, outcome, length, seating."""
    info = episode.get("info") or {}
    agents = [a.get("Name") for a in info.get("Agents") or [{}, {}]]
    rewards = list(episode.get("rewards") or [None, None])
    statuses = list(episode.get("statuses") or [None, None])

    first_player, max_turn = -1, 0
    final_prizes = [STARTING_PRIZES, STARTING_PRIZES]
    for step in episode["steps"]:
        for state in step:
            cur = (state.get("observation") or {}).get("current")
            if not cur:
                continue
            if first_player < 0 and int(cur.get("firstPlayer", -1)) >= 0:
                first_player = int(cur["firstPlayer"])
            max_turn = max(max_turn, int(cur.get("turn", 0)))
            me = int(cur["yourIndex"])
            prize = cur["players"][me].get("prize")
            if prize is not None:
                final_prizes[me] = len(prize)

    winner = None
    if rewards[0] == REWARD_WIN:
        winner = 0
    elif rewards[1] == REWARD_WIN:
        winner = 1

    return {
        "episode_id": int(info.get("EpisodeId") or -1),
        "agent_0": agents[0],
        "agent_1": agents[1],
        "reward_0": rewards[0],
        "reward_1": rewards[1],
        "status_0": statuses[0],
        "status_1": statuses[1],
        "winner": winner,
        "first_player": first_player,
        "n_steps": len(episode["steps"]),
        "max_turn": max_turn,
        "final_prizes_0": final_prizes[0],
        "final_prizes_1": final_prizes[1],
    }


def iter_decision_points(episode: dict) -> Iterator[dict]:
    """Yield one feature row per decision point in ``episode``.

    A row is emitted when a player is ``ACTIVE``, has a materialised
    ``current`` state and a non-empty option menu. Draws are skipped: with a
    binary win label there is nothing to predict.

    Pre-game setup decisions are also skipped. The two prize piles are dealt at
    slightly different moments, so a player's first few observations can show
    "opponent 6 prizes, me 0" -- which the prize-differential feature would
    read as a near-won game rather than an un-dealt board. A seat enters the
    game proper once it has seen both piles at full size.
    """
    meta = episode_meta(episode)
    labels = [_label_for(episode.get("rewards") or [0, 0], p) for p in (0, 1)]
    counters = [0, 0]
    setup_done = [False, False]

    for step_index, step in enumerate(episode["steps"]):
        for player, state in enumerate(step):
            if state.get("status") != "ACTIVE":
                continue
            obs = state.get("observation") or {}
            cur, select = obs.get("current"), obs.get("select")
            if not cur or not (select or {}).get("option"):
                continue
            label = labels[player]
            if label is None:
                continue

            me_idx = int(cur["yourIndex"])
            prizes = [_count(cur["players"][i].get("prize")) for i in (me_idx, 1 - me_idx)]
            if not setup_done[player]:
                if min(prizes) < STARTING_PRIZES:
                    continue
                setup_done[player] = True

            row = {
                "episode_id": meta["episode_id"],
                "step_index": step_index,
                "decision_index": counters[player],
                "n_steps": meta["n_steps"],
                "acting_player": player,
                "acting_agent": meta[f"agent_{player}"],
                "opponent_agent": meta[f"agent_{1 - player}"],
                "episode_status": meta[f"status_{player}"],
                "episode_reward": meta[f"reward_{player}"],
                "label": label,
            }
            row.update(state_features(cur, select, obs.get("remainingOverageTime")))
            counters[player] += 1
            yield row


def parse_episode_file(path: str | Path) -> tuple[list[dict], dict]:
    """``(decision_point_rows, episode_meta)`` for one replay file."""
    episode = load_episode(path)
    return list(iter_decision_points(episode)), episode_meta(episode)


def parse_many(
    paths: Iterable[str | Path],
    on_error: str = "warn",
) -> tuple[list[dict], list[dict], dict[int, str]]:
    """Parse a batch of replays.

    Returns ``(rows, metas, card_names)``. ``on_error`` is ``"warn"``,
    ``"raise"`` or ``"skip"``.
    """
    rows: list[dict] = []
    metas: list[dict] = []
    names: dict[int, str] = {}
    for path in paths:
        try:
            episode = load_episode(path)
            rows.extend(iter_decision_points(episode))
            metas.append(episode_meta(episode))
            names.update(card_names(episode))
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill a 2,000-file run
            if on_error == "raise":
                raise
            if on_error == "warn":
                print(f"[parse] skipping {Path(path).name}: {type(exc).__name__}: {exc}")
    return rows, metas, names


def deck_rows(episode: dict) -> list[dict]:
    """Long-format ``(episode_id, player, card_id, copies)`` decklist rows."""
    lists = decklists(episode)
    if lists is None:
        return []
    meta = episode_meta(episode)
    out: list[dict] = []
    for player, cards in enumerate(lists):
        counts: dict[int, int] = {}
        for cid in cards:
            counts[cid] = counts.get(cid, 0) + 1
        for cid, copies in sorted(counts.items()):
            out.append(
                {
                    "episode_id": meta["episode_id"],
                    "player": player,
                    "agent": meta[f"agent_{player}"],
                    "card_id": cid,
                    "copies": copies,
                    "won": None if meta["winner"] is None else int(meta["winner"] == player),
                }
            )
    return out
