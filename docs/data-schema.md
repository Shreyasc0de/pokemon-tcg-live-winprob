# The `cabt` episode replay format

Notes from reverse-engineering the archive. Nothing here comes from official
documentation, so treat the enum names as descriptive labels rather than
authoritative ones; the structural claims are all verified against the files by
the test suite.

## File layout

One JSON file per episode, named `<EpisodeId>.json`, averaging 4.75 MB. Roughly
63% of each file is the `visualize` payload on step 0.

```
{
  "id":              str,      # run uuid
  "name":            "cabt",
  "title":           "Card Battle",
  "description":     "Limited Card Battle.",
  "version":         "1.0.0",
  "module_version":  str,
  "schema_version":  1,
  "configuration":   {"actTimeout", "episodeSteps", "runTimeout", "seed"},
  "info": {
    "EpisodeId":     int,
    "Agents":        [{"Name": str, "ThumbnailUrl": null}, ...],
    "TeamNames":     [str, str],
    "LiveVideoPath": null
  },
  "specification":   {...},     # kaggle-environments spec
  "rewards":         [r0, r1],  #  1 win / 0 draw / -1 loss
  "statuses":        [s0, s1],  # "DONE", or a failure status
  "steps":           [[state_0, state_1], ...]
}
```

## A step

`steps[i]` holds one state per seat. Exactly one seat is normally `ACTIVE`.

```
{
  "status":      "ACTIVE" | "INACTIVE" | "DONE" | ...,
  "action":      [int, ...],   # option indices submitted for the PREVIOUS menu
  "reward":      int,
  "info":        {},
  "visualize":   null,         # non-null only on step 0
  "observation": {
    "step":                  int,
    "remainingOverageTime":  float,   # per-agent thinking-time bank, starts at 600
    "logs":                  [event, ...],   # public events since the last step
    "select":                menu | null,    # the legal-action menu to answer
    "search_begin_input":    null,
    "current":               game_state | null
  }
}
```

`action` is a list of indices into the *previous* step's `select.option`, so an
action and the menu it answers live in different steps.

## `observation.current` — one seat's view

```
{
  "yourIndex":       0 | 1,
  "turn":            int,      # the viewing seat's own turn counter
  "turnActionCount": int,
  "firstPlayer":     -1 | 0 | 1,   # -1 until the coin flip resolves
  "result":          -1,       # stays -1; the outcome lives in episode["rewards"]
  "stadium":         [card, ...],
  "stadiumPlayed":   bool,
  "supporterPlayed": bool,
  "energyAttached":  bool,
  "retreated":       bool,
  "looking":         ... | null,
  "lookingCount":    int,
  "players": [
    {
      "active":    [pokemon | null],        # null = face down during setup
      "bench":     [pokemon, ...],
      "benchMax":  5,
      "deckCount": int,
      "handCount": int,
      "hand":      [card, ...] | null,      # null for the OPPONENT
      "discard":   [card, ...],
      "prize":     [null, ...],             # length = prizes remaining
      "asleep": bool, "burned": bool, "confused": bool,
      "paralyzed": bool, "poisoned": bool
    },
    ...
  ]
}
```

### What is hidden

The replay preserves the imperfect information of the real game, which is what
makes it usable for modelling:

| Zone | Own | Opponent's |
|---|---|---|
| hand | full card list | `null`; only `handCount` |
| deck | `deckCount` only | `deckCount` only |
| prize cards | `[null] × n` | `[null] × n` |
| active / bench | full | full, except face-down setup entries (`null`) |
| discard | full | full |

`prize` is always a list of `null` — face-down by definition — so only its
length is information. Observed lengths bottom out at 1, never 0: a seat's final
observation precedes the terminal play, so the prize take that ends the game is
never reflected in a subsequent state.

## A `pokemon`

```
{
  "id": int, "serial": int, "playerIndex": int,
  "hp": int, "maxHp": int,           # hp is CURRENT hp; damage = maxHp - hp
  "energies":      [type_id, ...],   # attached energy types
  "energyCards":   [card, ...],
  "tools":         [card, ...],
  "preEvolution":  [card, ...],      # length = evolution stage (0 = basic)
  "appearThisTurn": bool
}
```

## `observation.select` — the legal-action menu

```
{
  "type":     int,   # kind of choice (see SELECT_TYPES)
  "context":  int,   # why it is being asked
  "option":   [{"type": int, "index": int, ...}, ...],
  "minCount": int, "maxCount": int,
  "contextCard": card | null, "deck": ... | null, "effect": ... | null,
  "remainEnergyCost": int, "remainDamageCounter": int
}
```

Option payload keys vary by option type: `index`, `area`, `playerIndex`,
`inPlayArea`/`inPlayIndex`, `attackId`, `number`, `energyIndex`, `count`,
`cardId`/`serial`, `toolIndex`. Enum codes for log, select and option types are
catalogued in [`src/cabt/schema.py`](../src/cabt/schema.py).

## `visualize` — ground truth, deliberately unused

Step 0 carries a `visualize` payload with the fully revealed state, including
both 60-card decklists as flat card-id arrays in `visualize[0]["action"]` and
card *names* (available nowhere else). This project reads it for exactly two
things: the card `id -> name` map, and the decklists used in the descriptive
metagame tables. It never reaches the feature matrix, because a player cannot
see it.

## Gotchas worth knowing

1. **`turn` is per-seat.** Each seat's observation carries its own turn counter,
   so the two seats of the same step can disagree by one.
2. **Setup rows are junk for modelling.** The two prize piles are dealt at
   slightly different moments, so a seat's first observations can read
   "opponent 6, me 0" — a prize differential feature would take that for a
   nearly won game. The parser drops decisions until a seat has seen both piles
   at full size.
3. **The step index leaks the outcome.** An episode ends the instant somebody
   wins, so how far through the replay a state sits is a function of the result.
   `step_index` and `n_steps` are carried for bookkeeping and excluded from the
   feature set.
4. **Repeated identical states are normal.** Several consecutive decision points
   within one turn can share a board state; they are distinct decisions, not
   duplicates.
5. **One bad file should not kill a 2,000-file run.** `parse_many` and the
   dataset builder collect failures and carry on.
