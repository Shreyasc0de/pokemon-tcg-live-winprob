"""Constants describing the Kaggle `cabt` episode-replay schema.

An episode JSON has the shape::

    {
      "id": str, "name": "cabt", "version": "1.0.0",
      "info": {"EpisodeId": int, "Agents": [{"Name": str}, ...], "TeamNames": [str, str]},
      "rewards": [r0, r1],          # 1 = win, -1 = loss, 0 = draw
      "statuses": [s0, s1],         # "DONE" / "TIMEOUT" / "ERROR" / ...
      "steps": [ [agent0_state, agent1_state], ... ]
    }

Each ``agent_state`` carries:

``action``
    The list of option indices the agent submitted for the *previous*
    ``observation.select`` menu.
``observation.current``
    That agent's own view of the game. Private opponent zones are masked:
    ``players[opponent]["hand"]`` is ``None``, and ``prize`` is a list of
    ``None`` whose *length* is the public remaining-prize count.
``observation.select``
    The legal-action menu the agent must answer, or ``None``.
``observation.logs``
    Public events emitted since the previous step.
``visualize``
    Present only on step 0. Holds the fully-revealed state including both
    60-card decklists -- the only place card ``id -> name`` is available.
    **Never used as a model input**: it is ground truth the players cannot see.
"""

from __future__ import annotations

#: Number of prize cards each player starts with. Reducing your own pile to
#: zero wins the game, so "my prizes remaining" is a countdown, not a score.
STARTING_PRIZES = 6

#: Reward encoding in ``episode["rewards"]``.
REWARD_WIN, REWARD_DRAW, REWARD_LOSS = 1, 0, -1

#: Episode-level statuses that mean the game was played to a normal finish.
CLEAN_STATUSES = frozenset({"DONE"})

#: ``observation.logs[i]["type"]`` codes observed in the archive. Names are
#: inferred from the accompanying payload keys and event ordering, not from
#: official documentation, so they are descriptive rather than authoritative.
LOG_TYPES = {
    0: "move_card",
    1: "check_basic_pokemon",
    2: "draw",
    3: "shuffle_or_search",
    4: "reveal",
    5: "place_pokemon",
    6: "damage_counter",
    7: "discard",
    8: "swap_active_bench",
    10: "play_stadium_or_supporter",
    11: "attach",
    12: "knockout_or_take_prize",
    14: "evolve",
    15: "attack",
    16: "state_change",
    22: "coin_flip",
}

#: ``observation.select["type"]`` codes: what kind of menu is being answered.
SELECT_TYPES = {
    0: "confirm",
    1: "main_action",
    2: "choose_number",
    4: "choose_energy",
    5: "choose_tool",
    6: "choose_card_in_play",
    7: "choose_card_in_play_alt",
    8: "choose_from_zone",
    9: "choose_from_revealed",
}

#: ``observation.select["option"][i]["type"]`` codes: what a single legal
#: option does.
OPTION_TYPES = {
    0: "number",
    1: "zone_card",
    2: "revealed_card",
    3: "card_in_hand_or_zone",
    4: "tool",
    5: "energy",
    6: "count",
    7: "play_card",
    8: "target_in_play",
    9: "retreat_or_switch",
    10: "ability",
    12: "attach_energy",
    13: "attack",
    14: "pass_or_confirm",
    15: "named_card",
}

#: Columns the dataset carries for bookkeeping / grouping but that must never
#: be handed to a model. ``step_index`` and ``n_steps`` in particular encode how
#: far through the episode we are, and an episode ends exactly when someone
#: wins -- using them would leak the outcome.
NON_FEATURE_COLUMNS = (
    "episode_id",
    "step_index",
    "decision_index",
    "n_steps",
    "acting_player",
    "acting_agent",
    "opponent_agent",
    "label",
    "episode_reward",
    "episode_status",
)
