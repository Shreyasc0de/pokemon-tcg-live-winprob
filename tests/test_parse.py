"""Parser and dataset invariants, checked against the committed sample games."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cabt.dataset import (  # noqa: E402
    ALL_GROUPS,
    add_trajectory_features,
    check_no_leakage,
    design_matrix,
    split_by_agent,
    split_by_episode,
)
from cabt.features import feature_names, state_features  # noqa: E402
from cabt.parse import (  # noqa: E402
    card_names,
    deck_rows,
    episode_meta,
    episode_paths,
    iter_decision_points,
    load_episode,
)
from cabt.schema import NON_FEATURE_COLUMNS, STARTING_PRIZES  # noqa: E402

SAMPLE = ROOT / "data" / "sample"


def _sample_frame() -> pd.DataFrame:
    rows = []
    for path in episode_paths(SAMPLE):
        rows.extend(iter_decision_points(load_episode(path)))
    return add_trajectory_features(pd.DataFrame(rows))


class TestSampleData(unittest.TestCase):
    def test_sample_replays_exist(self):
        self.assertGreaterEqual(len(episode_paths(SAMPLE)), 5)


class TestParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = episode_paths(SAMPLE)
        cls.df = _sample_frame()

    def test_produces_rows(self):
        self.assertGreater(len(self.df), 500)

    def test_both_seats_present_in_every_game(self):
        seats = self.df.groupby("episode_id")["acting_player"].nunique()
        self.assertTrue((seats == 2).all(), "every game should yield rows for both seats")

    def test_label_is_constant_within_a_seat(self):
        n = self.df.groupby(["episode_id", "acting_player"])["label"].nunique()
        self.assertTrue((n == 1).all())

    def test_seats_have_opposite_labels(self):
        per_game = self.df.groupby(["episode_id", "acting_player"])["label"].first().unstack()
        self.assertTrue(((per_game[0] + per_game[1]) == 1).all(), "exactly one seat wins")

    def test_label_matches_episode_reward(self):
        expected = (self.df["episode_reward"] == 1).astype(float)
        pd.testing.assert_series_equal(self.df["label"], expected, check_names=False)

    def test_prize_counts_are_in_range_and_never_increase(self):
        self.assertTrue(self.df["my_prizes"].between(0, STARTING_PRIZES).all())
        self.assertTrue(self.df["opp_prizes"].between(0, STARTING_PRIZES).all())
        ordered = self.df.sort_values(["episode_id", "acting_player", "decision_index"])
        for col in ("my_prizes", "opp_prizes"):
            diff = ordered.groupby(["episode_id", "acting_player"], sort=False)[col].diff()
            self.assertTrue((diff.fillna(0) <= 0).all(), f"{col} should never increase")

    def test_setup_rows_are_excluded(self):
        """Both prize piles are dealt before the first recorded decision."""
        first = (
            self.df.sort_values("decision_index")
            .groupby(["episode_id", "acting_player"])
            .first()
        )
        self.assertTrue((first["my_prizes"] == STARTING_PRIZES).all())
        self.assertTrue((first["opp_prizes"] == STARTING_PRIZES).all())

    def test_decision_index_is_contiguous(self):
        for _, part in self.df.groupby(["episode_id", "acting_player"]):
            idx = np.sort(part["decision_index"].to_numpy())
            np.testing.assert_array_equal(idx, np.arange(len(idx)))

    def test_every_row_had_a_legal_menu(self):
        self.assertTrue((self.df["n_legal_options"] >= 1).all())

    def test_prize_diff_is_consistent(self):
        expected = self.df["opp_prizes"] - self.df["my_prizes"]
        pd.testing.assert_series_equal(self.df["prize_diff"], expected, check_names=False)

    def test_episode_meta_agrees_with_rows(self):
        meta = episode_meta(load_episode(self.paths[0]))
        rows = pd.DataFrame(iter_decision_points(load_episode(self.paths[0])))
        self.assertEqual(meta["episode_id"], int(rows["episode_id"].iloc[0]))
        self.assertLessEqual(int(rows["turn"].max()), meta["max_turn"])

    def test_card_names_and_decklists_line_up(self):
        episode = load_episode(self.paths[0])
        names, decks = card_names(episode), deck_rows(episode)
        self.assertGreater(len(names), 20)
        self.assertEqual(len(decks) % 1, 0)
        for player in (0, 1):
            copies = sum(r["copies"] for r in decks if r["player"] == player)
            self.assertEqual(copies, 60, "a decklist should hold 60 cards")


class TestNoLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _sample_frame()

    def test_bookkeeping_columns_are_not_features(self):
        check_no_leakage(self.df)

    def test_step_index_absent_from_feature_set(self):
        names = set(feature_names(ALL_GROUPS))
        for banned in ("step_index", "n_steps", "episode_reward", "label", "acting_agent"):
            self.assertNotIn(banned, names)
        self.assertTrue(set(NON_FEATURE_COLUMNS).isdisjoint(names))

    def test_design_matrix_is_numeric_and_finite_or_nan(self):
        X, y = design_matrix(self.df)
        self.assertEqual(X.shape[0], len(self.df))
        self.assertTrue(all(np.issubdtype(t, np.floating) for t in X.dtypes))
        self.assertFalse(np.isinf(X.to_numpy()).any())
        self.assertTrue(set(np.unique(y)) <= {0.0, 1.0})

    def test_trajectory_features_only_look_backwards(self):
        """`prize_diff_lag1` must equal the previous row of the same chain."""
        ordered = self.df.sort_values(["episode_id", "acting_player", "decision_index"])
        g = ordered.groupby(["episode_id", "acting_player"], sort=False)
        expected = g["prize_diff"].shift(1)
        pd.testing.assert_series_equal(
            ordered["prize_diff_lag1"], expected, check_names=False, check_dtype=False
        )
        firsts = ordered[ordered["decision_index"] == 0]
        self.assertTrue(firsts["prize_diff_lag1"].isna().all())


class TestSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _sample_frame()

    def test_episode_split_shares_no_games(self):
        train, test = split_by_episode(self.df, 0.3, seed=1)
        self.assertTrue(set(train["episode_id"]).isdisjoint(set(test["episode_id"])))
        self.assertEqual(len(train) + len(test), len(self.df))

    def test_agent_split_shares_no_agents(self):
        train, test = split_by_agent(self.df, 0.3, seed=1)
        if len(test) == 0:
            self.skipTest("sample too small to form an agent split")
        train_agents = set(train["acting_agent"]) | set(train["opponent_agent"])
        test_agents = set(test["acting_agent"]) | set(test["opponent_agent"])
        self.assertTrue(train_agents.isdisjoint(test_agents))


class TestFeatureConstruction(unittest.TestCase):
    """Hand-built states, so the perspective convention is pinned down."""

    @staticmethod
    def _board(seat0: tuple[int, int], seat1: tuple[int, int], your_index: int = 0) -> dict:
        """One board, viewed from ``your_index``.

        ``seat0`` / ``seat1`` are ``(prizes_left, active_hp)``. The players list
        is always in seat order, exactly as the replay stores it, so the only
        thing that changes between views is ``yourIndex``.
        """

        def player(prizes: int, hp: int, hand: int, deck: int) -> dict:
            return {
                "active": [{"hp": hp, "maxHp": 100, "energies": [1], "tools": [], "preEvolution": []}],
                "bench": [],
                "benchMax": 5,
                "deckCount": deck,
                "handCount": hand,
                "discard": [],
                "prize": [None] * prizes,
                "asleep": False,
                "burned": False,
                "confused": False,
                "paralyzed": False,
                "poisoned": False,
            }

        return {
            "yourIndex": your_index,
            "players": [player(*seat0, 4, 30), player(*seat1, 6, 40)],
            "turn": 5,
            "turnActionCount": 2,
            "firstPlayer": 0,
            "stadium": [],
            "supporterPlayed": False,
            "energyAttached": False,
            "retreated": False,
        }

    MENU = {"type": 1, "context": 0, "option": [{"type": 13}], "minCount": 1, "maxCount": 1}

    def test_perspective_is_antisymmetric(self):
        """The same board seen from either seat gives exactly negated differentials."""
        seat0, seat1 = (4, 80), (6, 40)
        a = state_features(self._board(seat0, seat1, your_index=0), self.MENU, 600)
        b = state_features(self._board(seat0, seat1, your_index=1), self.MENU, 600)
        for key in ("prize_diff", "total_hp_diff", "active_hp_frac_diff", "hand_diff", "deck_diff"):
            self.assertAlmostEqual(a[key], -b[key], msg=key)
        # And each seat's "mine" is the other seat's "theirs".
        for mine, theirs in (
            ("my_prizes", "opp_prizes"),
            ("my_active_hp", "opp_active_hp"),
            ("my_hand", "opp_hand"),
        ):
            self.assertAlmostEqual(a[mine], b[theirs], msg=mine)
            self.assertAlmostEqual(a[theirs], b[mine], msg=theirs)

    def test_is_first_player_follows_the_seat(self):
        board = self._board((6, 100), (6, 100), your_index=0)
        self.assertEqual(state_features(board, self.MENU, 600)["is_first_player"], 1.0)
        board = self._board((6, 100), (6, 100), your_index=1)
        self.assertEqual(state_features(board, self.MENU, 600)["is_first_player"], 0.0)

    def test_undecided_first_player_is_encoded_as_unknown(self):
        board = self._board((6, 100), (6, 100))
        board["firstPlayer"] = -1
        self.assertEqual(state_features(board, self.MENU, 600)["is_first_player"], 0.5)

    def test_prize_diff_sign_convention(self):
        ahead = state_features(self._board((3, 100), (6, 100)), self.MENU, 600)
        behind = state_features(self._board((6, 100), (3, 100)), self.MENU, 600)
        self.assertGreater(ahead["prize_diff"], 0, "fewer own prizes left means ahead")
        self.assertLess(behind["prize_diff"], 0)

    def test_face_down_pokemon_yield_nan_not_zero(self):
        """An unrevealed active Pokemon is unknown HP, not dead."""
        board = self._board((6, 100), (6, 100))
        board["players"][1]["active"] = [None]
        f = state_features(board, self.MENU, 600)
        self.assertTrue(np.isnan(f["opp_active_hp_frac"]))
        self.assertEqual(f["opp_hidden_pokemon"], 1)
        self.assertEqual(f["opp_active_present"], 1.0)

    def test_masked_opponent_hand_contributes_only_its_count(self):
        board = self._board((6, 100), (6, 100))
        board["players"][1]["hand"] = None
        f = state_features(board, self.MENU, 600)
        self.assertEqual(f["opp_hand"], 6)

    def test_option_flags_read_the_menu(self):
        menu = {
            "type": 1,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 13}, {"type": 12}, {"type": 7}],
        }
        f = state_features(self._board((6, 100), (6, 100)), menu, 600)
        self.assertEqual(f["opt_has_attack"], 1.0)
        self.assertEqual(f["opt_has_attach_energy"], 1.0)
        self.assertEqual(f["opt_has_play_card"], 1.0)
        self.assertEqual(f["opt_has_retreat"], 0.0)
        self.assertEqual(f["n_legal_options"], 3)

    def test_feature_names_are_unique(self):
        names = feature_names(ALL_GROUPS)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
