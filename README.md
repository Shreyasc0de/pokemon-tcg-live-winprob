# Live win probability for Pokémon TCG agent battles

A broadcast-style win-probability curve for the Kaggle *Card Battle* (`cabt`)
competition: instead of predicting a winner once before the match, the model
re-estimates each side's chances at **every decision point**, using only the
information that player can actually see.

![Live win probability curves for four held-out matches](reports/figures/winprob_curves.png)

The problem is a sequential-updating one under imperfect information. Each game
is a chain of ~170 decisions; the state is partly hidden (your opponent's hand
is masked, their prize cards are face down); and the forecast has to stay
coherent over time, not just accurate on average. The interesting question is
not "who wins" but **"is the number honest at every point along the way."**

---

## Headline results

Held out **110 complete games** (18,886 decision points) from the 439 games used
for training; games are split whole, never by row. Every score is quoted against
`prize_only` — logistic regression on the prize differential alone, which is the
number a human player already tracks in their head, and a much harder baseline
than the base rate.

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | ECE ↓ | Brier skill vs. prize-only |
|---|---|---|---|---|---|
| base rate (54.9%) | 0.2480 | 0.6892 | 0.500 | 0.020 | −20.2% |
| **prize differential only** | 0.2063 | 0.5960 | 0.723 | 0.036 | — |
| logistic regression, 93 features | 0.1797 | 0.5263 | 0.800 | 0.025 | +12.9% |
| gradient boosting, isotonic-calibrated | 0.1710 | 0.5133 | 0.825 | 0.030 | **+17.2%** |
| … + causal Kalman filter on the logit path | 0.1710 | 0.5133 | 0.825 | 0.033 | +17.1% |

Three findings, in descending order of how much they surprised me.

### 1. The filter buys coherence, not accuracy — and that is the point

The natural next step after a per-state model is to smooth its output, and the
natural way to sell that is "smoothing improves log loss." It does not. Once you
compare against an honest pass-through control, the filter's effect on every
scoring rule is *nil* to four decimal places:

| Smoothing | Brier | Log loss | ECE | Mean step size | Martingale slope | *p* |
|---|---|---|---|---|---|---|
| pass-through | 0.1710 | 0.5133 | 0.030 | 0.0286 | **−0.0059** | 0.00004 |
| **tuned (q = 2.5)** | 0.1710 | 0.5133 | 0.033 | 0.0226 | **+0.0016** | **0.15** |
| moderate (q = 0.5) | 0.1714 | 0.5144 | 0.035 | 0.0173 | +0.0075 | 0.0000 |
| heavy (q = 0.05) | 0.1736 | 0.5203 | 0.046 | 0.0112 | +0.0125 | 0.0000 |

What it does buy is a **21% reduction in path jitter** and, more importantly, a
forecast that stops violating the martingale property. A correctly specified
probability path must satisfy E[p<sub>t+1</sub> − p<sub>t</sub> | p<sub>t</sub>] = 0:
knowing the current level should tell you nothing about which way it moves next.
Regressing the next step on the current level, the raw per-state model has a
significantly **negative** slope — it overshoots and gets pulled back, the
signature of a memoryless model reacting to every twitch of the board. Over-smooth
it instead and the slope goes significantly **positive**, because a laggy path
keeps drifting the same direction it was already going.

The tuned filter is the only setting in the sweep that passes. That is worth
dwelling on: `q` was chosen by validation log loss, with no knowledge of the
martingale test, and the test independently picks out the same setting.

![Raw, tuned and over-smoothed paths over a 70-decision window](reports/figures/filter_zoom.png)

I had assumed smoothing was cosmetic. It isn't — it is what makes the path a
legitimate forecast rather than a sequence of unrelated point estimates. But it
is also not free accuracy, and reporting it as such would have been wrong.

### 2. Calibration is the binding constraint, not discrimination

AUC 0.82 sounds respectable and is nearly irrelevant. A win-probability curve is
only worth showing if 70% means 70%. Raw gradient boosting is over-confident
(ECE 0.049); isotonic calibration more than halves that (0.030) but overshoots
into mild under-confidence at the extremes.

![Calibration curves before and after isotonic calibration](reports/figures/calibration.png)

There is a real trade-off underneath, and the repo reports both sides rather than
picking the flattering one. Fitting a single isotonic map on dedicated held-out
games gets ECE down to **0.027** — the best-calibrated option — but costs sharpness
and pushes log loss up to 0.555, because it trains on less data and discretises
the output into a coarse staircase. The shipped model uses the cross-fitted
isotonic ensemble, which wins on both proper scoring rules.

Accuracy is also strongly phase-dependent. Before turn six the model barely beats
the prize-differential baseline; the edge arrives as the board resolves.

![Brier score by turn bucket](reports/figures/brier_by_turn.png)

### 3. Prizes and hit points carry it; tempo and status conditions carry nothing

Permutation importance is computed **by family**, not per feature, because the
features inside a family are deliberately collinear — `my_prizes`, `opp_prizes`
and `prize_diff` encode the same thing three ways, and shuffling them one at a
time lets the model read the twin and report near-zero importance for all three.

![Grouped permutation importance](reports/figures/feature_importance.png)

The prize differential dominates, board hit points are a close second, and card
economy (hand, deck and discard sizes) is a real third. Status conditions
contribute nothing measurable, and the agent's remaining thinking-time budget —
included as a deliberate control, since it is a competition artefact rather than
a game-state variable — has *negative* importance, i.e. it is noise. The ablation
tells a consistent story: history/momentum features (lags, deltas, EWMAs) do not
improve on the plain per-state feature set (0.1739 vs 0.1737 Brier), which is the
main reason the sequential structure had to be handled by the filter rather than
by feature engineering.

---

## What I would not claim

- **Generalisation to unseen agents is meaningfully worse.** Holding out whole
  *agents* rather than whole games, Brier rises to 0.202 and ECE to 0.096 — the
  model has partly learned how this specific population of bots plays, not
  Pokémon TCG in the abstract. The number to trust for "would this work on a new
  bot" is the agent-split number, and it is the one I would lead with if someone
  asked me to deploy this.
- **Card-level win rates are descriptive, not causal.** Decks are chosen, not
  assigned. `Area Zero Underdepths` appears in 5.1% of decks that won 75% of
  their games (95% CI 62–85%, n = 56 decks), but that conflates the card with the
  skill of the agents who pick it. No causal claim is made or implied.
- **First-player advantage is suggestive, not established.** The first player won
  53.9% of 549 games (95% CI 49.7–58.0%, *p* = 0.073). Directionally what you
  would expect; not significant at this sample size. Running the pipeline over
  the full 2,000-episode archive is the obvious way to settle it.
- **This run covers 549 of the archive's 2,000 episodes** (2.5 GB of the 9.5 GB).
  The pipeline processes the full archive in about two and a half minutes on a
  laptop — see *Reproducing* below. Metrics are stable at this sample size;
  card-level intervals would tighten.

---

## How it works

**Decision points, not turns.** A row is emitted wherever a player is `ACTIVE`
and holds a non-empty legal-action menu — exactly the moment a player would want
a number. Both seats contribute rows, always from the acting player's own point
of view, so `my_*` means the actor and `opp_*` their opponent and the label is
"did the actor go on to win". That doubles the data and makes seat bias
impossible to learn by accident: which seat you are is exposed explicitly as
`is_first_player` rather than smuggled in through the perspective. The test
suite pins the convention down by building a board, viewing it from each seat,
and asserting every differential is exactly negated.

**Imperfect information is preserved, not repaired.** The replays mask the
opponent's hand (`null`, with only a count) and keep prize cards face down, and
the features respect that. During setup the opponent's active Pokémon is face
down too; rather than encode "unknown HP" as zero HP, those slots become `NaN`
— read natively by the boosted model, median-imputed in-pipeline for the linear
baselines — alongside an explicit `opp_hidden_pokemon` count. Each episode also
carries a fully-revealed `visualize` payload with both decklists; it is used for
card names and the descriptive metagame tables and **never** reaches the feature
matrix.

**Two leakage traps, both closed deliberately.**

The first is the clock. An episode terminates the instant somebody wins, so
`step_index` and `n_steps` are direct functions of the outcome. Both are carried
for bookkeeping and excluded from the feature set, with a test asserting it.
`turn` stays in — that is a legitimate in-game observable.

The second is the setup phase. The two prize piles are dealt a moment apart, so
a seat's first few observations can read "opponent 6 prizes, me 0" — which the
prize-differential feature would take for a nearly-won game rather than an
un-dealt board. The parser drops decisions until a seat has seen both piles at
full size.

**Splits respect game boundaries everywhere.** Rows within a game are strongly
dependent and the two seats of a game share an outcome, so a random row split
would put near-duplicates of a test state in training and flatter every metric.
Episodes are held out whole; the isotonic calibrator's cross-fitting folds are
`GroupKFold` on episode id; the filter's `q` is tuned on a further disjoint slice
of *training* episodes, never on the test set.

**The filter.** A local-level state-space model on the win-probability logit, one
chain per (episode, seat):

x<sub>t</sub> = x<sub>t−1</sub> + w<sub>t</sub>, w ~ N(0, q) &nbsp;&nbsp;·&nbsp;&nbsp;
z<sub>t</sub> = x<sub>t</sub> + v<sub>t</sub>, v ~ N(0, r)

where z is the per-state model's output. Only the **forward** pass runs — no
smoothing pass — so the estimate at decision point *t* depends on nothing after
*t* and remains a legitimate real-time number. Tests assert the causality
directly (perturbing a later observation leaves earlier outputs bit-identical)
and that chains never bleed into each other.

### The baseline worth beating

![Empirical win rate by prize differential](reports/figures/prize_lookup.png)

Two prizes up is worth 87%; two down is 28%. Any model that cannot beat this
lookup table is not earning its complexity. The full-feature model's +17% Brier
skill over it is the headline claim of the project, and it is a modest, real
number rather than an impressive, leaky one.

### What the archive looks like

549 games, 101 distinct agents, 153 distinct cards, a median of 12 turns and 171
decision points per game, no draws, and no non-`DONE` statuses. Games end by
prizes in 91.4% of cases; 5.1% end with the loser having no Pokémon left to
promote, and 0.5% by running out of deck (2.9% stay unclassified, because the
final observation is one play stale).

---

## Reproducing

Requires Python 3.11+ and the packages in `requirements.txt` (numpy, pandas,
scipy, scikit-learn, matplotlib — no Arrow or GPU dependency; `pyarrow` is
optional and only enables `--parquet`).

```bash
pip install -r requirements.txt

# 1. Run everything against the 20 gzipped replays committed to this repo (~20s)
make sample
python scripts/train.py --data data/sample_processed --out /tmp/reports

# 2. Or against the real archive: download the Kaggle dataset, unzip, then
make dataset REPLAYS=~/Downloads/archive   # ~2.5 min for 2,000 episodes
make train                                 # ~4 min
make analyse

make test                                  # 49 tests, ~5s, no network needed
```

`make dataset` reduces 9.5 GB of JSON to a single ~25 MB table by streaming one
file at a time across a worker pool; the raw archive never needs to fit in
memory and is git-ignored. Every number and figure in this README is written to
`reports/` by `make train` and `make analyse`, and committed, so the results are
checkable without running anything.

## Repository layout

```
src/cabt/
  schema.py     replay format constants, enum codes, the non-feature blocklist
  parse.py      streaming replay -> decision-point records
  features.py   93 features from one seat's own view; NaN for the genuinely unknown
  dataset.py    parallel build, trajectory features, episode- and agent-level splits
  models.py     baselines, calibration wrapper, the causal logit Kalman filter
  evaluate.py   proper scoring rules, reliability, path volatility, martingale test
  analysis.py   descriptive tables with Wilson intervals; online Elo
  pipeline.py   the experiment: ladder, ablation, filter sweep, unseen-agent check
  plot.py       figures
scripts/        build_dataset.py · train.py · analyse.py
tests/          49 tests: parser invariants, leakage, perspective symmetry,
                filter causality, metric correctness on known cases
data/sample/    20 gzipped replays (1.3 MB) so CI runs the real pipeline
docs/           reverse-engineered notes on the replay format
reports/        generated metrics, tables and figures (committed)
```

## Data

[Kaggle Card Battle (`cabt`) episode replays](https://www.kaggle.com/) — 2,000
JSON episode files, 9.5 GB unzipped. The format is undocumented; the structural
notes, enum codes and gotchas in [`docs/data-schema.md`](docs/data-schema.md) are
reverse-engineered from the files and asserted by the test suite.

## License

MIT
