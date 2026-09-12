"""Build the decision-point dataset from a directory of episode replays.

Reduction happens file-by-file in a worker pool: 21.5 GB of JSON becomes a
single tabular file of roughly 750k rows. Rows are flushed into columnar
batches as they arrive so peak memory stays near 2 GB rather than scaling
with the archive. Output is gzipped CSV so the
pipeline has no Arrow dependency; Parquet is written instead when ``pyarrow``
is importable and ``--parquet`` is passed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_GROUPS, feature_names
from .parse import card_names, deck_rows, episode_meta, episode_paths, iter_decision_points, load_episode

#: History-derived features. A player legitimately knows how the game got
#: here, so momentum terms are fair game -- and they are what lets a
#: memoryless per-state model approximate a sequential one.
TRAJECTORY_FEATURES = (
    "prize_diff_lag1",
    "d_prize_diff",
    "total_hp_diff_lag1",
    "d_total_hp_diff",
    "prize_diff_ewma",
    "total_hp_diff_ewma",
    "prize_diff_momentum",
    "total_hp_diff_momentum",
    "decisions_so_far",
    "decisions_since_prize_change",
    "board_pokemon_diff_lag1",
    "d_board_pokemon_diff",
)

FEATURE_GROUPS["trajectory"] = TRAJECTORY_FEATURES

ALL_GROUPS = ("core", "decision", "clock", "trajectory")


def _worker(path: str) -> tuple[list[dict], dict, list[dict], dict[int, str]]:
    episode = load_episode(path)
    return (
        list(iter_decision_points(episode)),
        episode_meta(episode),
        deck_rows(episode),
        card_names(episode),
    )


def add_trajectory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append within-game history features, grouped by (episode, seat).

    Every term is a backward-looking function of earlier decision points in the
    same game from the same seat, so no future information enters a row.
    """
    df = df.sort_values(["episode_id", "acting_player", "decision_index"]).reset_index(drop=True)
    g = df.groupby(["episode_id", "acting_player"], sort=False)

    for col in ("prize_diff", "total_hp_diff", "board_pokemon_diff"):
        lag = g[col].shift(1)
        df[f"{col}_lag1"] = lag
        df[f"d_{col}"] = df[col] - lag

    for col in ("prize_diff", "total_hp_diff"):
        ewma = g[col].transform(lambda s: s.ewm(halflife=5, adjust=False).mean())
        df[f"{col}_ewma"] = ewma
        df[f"{col}_momentum"] = df[col] - ewma

    df["decisions_so_far"] = df["decision_index"].astype("float64")

    changed = df["d_prize_diff"].fillna(0) != 0
    # Decisions elapsed since the prize differential last moved: a long quiet
    # stretch means neither side has scored.
    since = []
    counter = 0
    prev_key = None
    for key, was_change in zip(
        zip(df["episode_id"], df["acting_player"], strict=True), changed, strict=True
    ):
        if key != prev_key:
            counter = 0
            prev_key = key
        elif was_change:
            counter = 0
        else:
            counter += 1
        since.append(counter)
    df["decisions_since_prize_change"] = np.asarray(since, dtype="float64")
    return df


def build(
    replay_dir: str | Path,
    out_dir: str | Path,
    limit: int | None = None,
    workers: int | None = None,
    parquet: bool = False,
) -> dict:
    """Parse every replay under ``replay_dir`` and write the dataset.

    Writes ``decisions``, ``episodes``, ``decks`` and ``card_names.json`` into
    ``out_dir`` and returns a summary dict.
    """
    replay_dir, out_dir = Path(replay_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = episode_paths(replay_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no *.json or *.json.gz replays under {replay_dir}")

    workers = workers or max(1, (os.cpu_count() or 2))
    # Rows are flushed into DataFrames every FLUSH_ROWS rather than held as
    # one list of dicts to the end. A dict of ~90 float keys costs several KB
    # of interpreter overhead, so the whole archive as dicts needs multiple GB
    # and gets OOM-killed; the same rows as float columns are a few hundred MB.
    FLUSH_ROWS = 100_000
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    metas: list[dict] = []
    decks: list[dict] = []
    names: dict[int, str] = {}
    failures: list[str] = []

    def _flush() -> None:
        if rows:
            frames.append(pd.DataFrame(rows))
            rows.clear()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, result in zip(
            paths, pool.map(_worker, [str(p) for p in paths], chunksize=4), strict=True
        ):
            try:
                r, m, d, n = result
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{Path(path).name}: {exc}")
                continue
            rows.extend(r)
            metas.append(m)
            decks.extend(d)
            names.update(n)
            if len(rows) >= FLUSH_ROWS:
                _flush()
    _flush()

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    frames.clear()
    decisions = add_trajectory_features(raw)
    episodes = pd.DataFrame(metas)
    deck_df = pd.DataFrame(decks)

    _write(decisions, out_dir / "decisions", parquet)
    _write(episodes, out_dir / "episodes", parquet)
    if len(deck_df):
        _write(deck_df, out_dir / "decks", parquet)
    (out_dir / "card_names.json").write_text(
        json.dumps({str(k): v for k, v in sorted(names.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "n_episodes_found": len(paths),
        "n_episodes_parsed": int(len(episodes)),
        "n_decision_points": int(len(decisions)),
        "n_features": len(feature_names(ALL_GROUPS)),
        "n_agents": int(pd.concat([episodes["agent_0"], episodes["agent_1"]]).nunique()),
        "n_cards": len(names),
        "failures": failures,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write(df: pd.DataFrame, stem: Path, parquet: bool) -> Path:
    if parquet:
        try:
            import pyarrow  # noqa: F401

            path = stem.with_suffix(".parquet")
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            print(f"[dataset] pyarrow unavailable; writing {stem.name}.csv.gz instead")
    path = Path(str(stem) + ".csv.gz")
    df.to_csv(path, index=False)
    return path


def load_decisions(data_dir: str | Path) -> pd.DataFrame:
    """Read the decisions table, preferring Parquet when both exist."""
    data_dir = Path(data_dir)
    pq, csv = data_dir / "decisions.parquet", data_dir / "decisions.csv.gz"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"no decisions table in {data_dir}")


def load_table(data_dir: str | Path, name: str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    pq, csv = data_dir / f"{name}.parquet", data_dir / f"{name}.csv.gz"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(csv)


def split_by_episode(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole episodes.

    Rows from one game are strongly dependent, and both seats of a game share
    an outcome, so splitting rows at random would put near-duplicates of a test
    state in the training set and flatter every metric.
    """
    ids = np.sort(df["episode_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_frac)))
    test_ids = set(ids[:n_test].tolist())
    mask = df["episode_id"].isin(test_ids)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def split_by_agent(
    df: pd.DataFrame,
    test_frac: float = 0.25,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole agents: test games are played by agents never trained on.

    A harder generalisation test than the episode split. Any game whose two
    seats straddle the agent split is dropped so the two sides stay disjoint.
    """
    agents = np.sort(pd.concat([df["acting_agent"], df["opponent_agent"]]).dropna().unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(agents)
    n_test = max(1, int(round(len(agents) * test_frac)))
    test_agents = set(agents[:n_test].tolist())

    acting_test = df["acting_agent"].isin(test_agents)
    opp_test = df["opponent_agent"].isin(test_agents)
    test = df.loc[acting_test & opp_test].copy()
    train = df.loc[~acting_test & ~opp_test].copy()
    return train, test


def episode_groups(df: pd.DataFrame) -> np.ndarray:
    """Group labels for ``GroupKFold``: one group per episode."""
    return df["episode_id"].to_numpy()


def design_matrix(
    df: pd.DataFrame,
    groups: Sequence[str] = ALL_GROUPS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """``(X, y)`` for the requested feature groups."""
    cols = feature_names(tuple(groups))
    return df[cols].astype("float64"), df["label"].to_numpy(dtype="float64")


def check_no_leakage(df: pd.DataFrame, groups: Iterable[str] = ALL_GROUPS) -> None:
    """Assert that no bookkeeping column reached the feature set."""
    from .schema import NON_FEATURE_COLUMNS

    banned = set(NON_FEATURE_COLUMNS)
    used = set(feature_names(tuple(groups)))
    overlap = banned & used
    if overlap:
        raise AssertionError(f"leakage: bookkeeping columns used as features: {sorted(overlap)}")
