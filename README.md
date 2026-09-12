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

Built from the complete archive: 2,000 episode replays, 8.9 GB of JSON, reduced
to **332,814 decision points** across 151 agents.

---

## Headline results

Every score below is the **mean over five independent episode-level splits**,
each holding out 400 whole games, with the standard deviation across splits.
Scores are quoted against `prize_only`, logistic regression on the prize
differential alone, which is the number a human player already tracks in their
head and a much harder baseline than the base rate.

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | ECE ↓ | Brier skill vs. prize-only |
|---|---|---|---|---|---|
| **prize differential only** | 0.2167 ± 0.0044 | 0.6209 | 0.695 | 0.027 | n/a |
| gradient boosting | 0.1819 ± 0.0044 | 0.5336 | 0.798 | 0.022 | +16.1% ± 1.8 |
| … + causal Kalman filter | 0.1815 ± 0.0044 | 0.5325 | 0.799 | 0.019 | +16.2% ± 1.8 |
| … + isotonic recalibration | 0.1802 ± 0.0036 | 0.5297 | 0.801 | 0.016 | +16.9% ± 1.4 |
| **… + isotonic + filter** *(shipped)* | **0.1802 ± 0.0036** | **0.5295** | **0.801** | 0.018 | **+16.9% ± 1.4** |

Reporting five splits rather than one is not ceremony. It is the single most
important methodological decision in the repo, and I arrived at it the hard way.

### 1. Every model difference here is smaller than the split-to-split noise

Holding out 400 of 2,000 games leaves a standard deviation of **±0.0044 Brier**
between splits. All the differences worth arguing about, calibrated versus not
and filtered versus not, are 0.001 to 0.002. On a single split they are noise.

I learned this by watching my own conclusions flip. An earlier version of this
project ran on a 549-game subsample and reported that isotonic recalibration
halved calibration error. Rerunning on 1,970 games reversed it: isotonic now
appeared to *double* ECE, and I rewrote the README to say recalibration was
harmful. Adding the last 30 games reversed it a third time. Three different
answers from three defensible runs, none of them wrong arithmetic. The split was
simply doing the talking.

The fix is the paired comparison. Run the same five splits for both variants and
subtract:

| Split | Δ Brier (isotonic − raw) | Δ ECE (isotonic − raw) |
|---|---|---|
| 0 | −0.00022 | −0.0024 |
| 1 | −0.00067 | −0.0042 |
| 2 | −0.00238 | −0.0013 |
| 3 | −0.00275 | −0.0113 |
| 4 | −0.00269 | −0.0102 |

Isotonic wins **5 of 5 on both metrics**. The effect is real, it is just small
enough that a single split cannot see it. That is the finding, and it is worth
more than the number it produces.

### 2. The filter buys path coherence, and the shipped model pays for it

A correctly specified probability path must be a martingale:
E[p<sub>t+1</sub> − p<sub>t</sub> | p<sub>t</sub>] = 0. Knowing the current level
should tell you nothing about which way it moves next. Regressing the next step
on the current level, across the same five splits:

| Model | Martingale slope | *p* across the 5 splits |
|---|---|---|
| gradient boosting | −0.0157 ± 0.0014 | ~0 in all five |
| **gbdt + filter** | **+0.0008 ± 0.0004** | **0.02 to 0.67, not significant in four of five** |
| gbdt + isotonic | −0.0093 ± 0.0013 | ~0 in all five |
| gbdt + isotonic + filter *(shipped)* | +0.0043 ± 0.0006 | ~0 in all five |

The raw per-state model has a firmly negative slope: it overshoots and gets
pulled back, the signature of a memoryless model reacting to every twitch of the
board. The filter removes essentially all of that. Over-smooth instead and the
slope goes positive, because a laggy path keeps drifting the way it was already
going.

**The honest cost:** isotonic recalibration already shrinks predictions toward
the middle, so stacking the filter on top overshoots past zero to +0.0043, which
is detectable on every split. The shipped model is therefore the better forecast
and *not* the more coherent path. The uncalibrated `gbdt + filter` is the only
configuration in the sweep that is martingale-clean, at a cost of about 0.0013
Brier and 0.003 ECE. Both are in `reports/tables/repeated_splits.csv`; I ship the
one that wins the proper scoring rules and state what it gives up rather than
quietly picking whichever supports the nicer sentence.

![Raw, tuned and over-smoothed paths over a 70-decision window](reports/figures/filter_zoom.png)

*(An earlier version of this README claimed the martingale test independently
confirmed the `q` chosen by log loss. It does not. That was one split's p-value
read as a result.)*

### 3. Prizes and hit points carry it; status conditions carry nothing

Permutation importance is computed **by family**, not per feature, because the
features inside a family are deliberately collinear. `my_prizes`, `opp_prizes`
and `prize_diff` encode the same thing three ways, so shuffling them one at a
time lets the model read the twin and report near-zero importance for all three.

![Grouped permutation importance](reports/figures/feature_importance.png)

The prize differential dominates (+0.0423 Brier when shuffled), board hit points
are second (+0.0330), then energy and evolution state (+0.0180) and card economy
(+0.0174). Status conditions contribute nothing measurable at all. The ablation
is consistent: core game state 0.1827, plus the legal-action menu 0.1811, plus
history and momentum terms 0.1811 (**no gain at all from the trajectory
features**), plus the thinking-time budget 0.1798. Lags and EWMAs of the state
add nothing the per-state features do not already carry, which is the other
reason the sequential structure is handled by the filter rather than by feature
engineering.

Accuracy is strongly phase-dependent. Before turn six the model barely beats the
prize-differential baseline; the edge arrives as the board resolves.

![Brier score by turn bucket](reports/figures/brier_by_turn.png)

---

## What I would not claim

- **The model is weakest exactly where a forecast is most interesting.** In the
  first two turns it is close to a coin flip (Brier 0.239, AUC 0.620). Most of
  the headline skill is earned after turn eight, once the board has largely
  decided things anyway.
- **Card-level win rates are archetype win rates.** Decks are chosen, not
  assigned, and the top of the table gives it away: `Fan Rotom`, `Buneary` and
  `Mega Lopunny ex` all show 61.3% across exactly 455 decks, because they are the
  same deck. The table measures which lists won, not which cards cause wins.
- **Unseen-agent performance looks better, and that is suspicious.** Holding out
  whole *agents* rather than whole games gives Brier 0.1433 and AUC 0.880,
  better than the episode split rather than worse. The likely explanation is that those
  99 games are an easier subpopulation, not that the model generalises unusually
  well. Read it as "no evidence of an agent-specific overfit" and nothing more.
- **Isotonic's margin is small.** 5 of 5 paired splits is a sign test at
  *p* = 0.031. It is evidence, not proof, and five splits of 400 games are not
  independent of each other in the strict sense, since they resample the same
  2,000 games.
- **The archive is a rating-filtered sample, not a random one.** Kaggle selects
  these replays daily "ranked by average agent rating" under a 20 GiB/day cap,
  so every game here is drawn from the stronger tail of play. Everything
  descriptive is conditional on that: the first-player advantage is 53.4% *among
  highly rated agents*, the card and archetype win rates describe what strong
  decks did, and the Elo table rates agents on a filtered subset of their games
  rather than on all of them. A uniform sample of the population could give
  different numbers, and there is no way to check that from inside this dataset.
  It is also the most plausible explanation for the unseen-agent result above.

## What scaling the data settled

Running the complete archive rather than a subsample changed real conclusions.
**First-player advantage is now established**: the player moving first won 1,067
of 2,000 games, 53.4% (95% CI 51.2–55.5%, *p* = 0.0029). On 549 games the same
estimate was 53.9% with *p* = 0.073, the right answer but not yet demonstrable.
The calibration and martingale findings moved the other way, which is why the
repeated-split harness exists at all.

---

## How it works

**Decision points, not turns.** A row is emitted wherever a player is `ACTIVE`
and holds a non-empty legal-action menu, which is exactly the moment a player
would want a number. Both seats contribute rows, always from the acting player's own point
of view, so `my_*` means the actor and `opp_*` their opponent and the label is
"did the actor go on to win". That doubles the data and makes seat bias
impossible to learn by accident: which seat you are is exposed explicitly as
`is_first_player` rather than smuggled in through the perspective. The test
suite pins the convention down by building a board, viewing it from each seat,
and asserting every differential is exactly negated.

**Imperfect information is preserved, not repaired.** The replays mask the
opponent's hand (`null`, with only a count) and keep prize cards face down, and
the features respect that. During setup the opponent's active Pokémon is face
down too. Rather than encode "unknown HP" as zero HP, those slots become `NaN`,
read natively by the boosted model and median-imputed in-pipeline for the linear
baselines, alongside an explicit `opp_hidden_pokemon` count. Each episode also
carries a fully-revealed `visualize` payload with both decklists; it is used for
card names and the descriptive metagame tables and **never** reaches the feature
matrix.

**Two leakage traps, both closed deliberately.**

The first is the clock. An episode terminates the instant somebody wins, so
`step_index` and `n_steps` are direct functions of the outcome. Both are carried
for bookkeeping and excluded from the feature set, with a test asserting it.
`turn` stays in, because that is a legitimate in-game observable.

The second is the setup phase. The two prize piles are dealt a moment apart, so
a seat's first few observations can read "opponent 6 prizes, me 0", which the
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

where z is the per-state model's output. Only the **forward** pass runs, with no
smoothing pass, so the estimate at decision point *t* depends on nothing after
*t* and remains a legitimate real-time number. Tests assert the causality
directly (perturbing a later observation leaves earlier outputs bit-identical)
and that chains never bleed into each other.

### The baseline worth beating

![Empirical win rate by prize differential](reports/figures/prize_lookup.png)

Two prizes up is worth 85%; two down is 27%. Any model that cannot beat this
lookup table is not earning its complexity. The full-feature model's ~17% Brier
skill over it is the headline claim of the project, and it is a modest, real
number rather than an impressive, leaky one.

### What the archive looks like

2,000 games, 151 distinct agents, 179 distinct cards, a median of 12 turns and
167 decision points per game, no draws, and no non-`DONE` statuses. Games end by
prizes in 86.8% of cases; 5.5% end with the loser having no Pokémon left to
promote and 2.4% by running out of deck (5.4% stay unclassified, because the
final observation is one play stale). Game length has a long tail: the median is
172 replay steps, and the longest game runs 1,085.

---

## Reproducing

Requires Python 3.11+ and the packages in `requirements.txt` (numpy, pandas,
scipy, scikit-learn, matplotlib). There is no Arrow or GPU dependency; `pyarrow`
is optional and only enables `--parquet`.

```bash
pip install -r requirements.txt

# 1. Run everything against the 20 gzipped replays committed to this repo (~20s)
make sample
python scripts/train.py --data data/sample_processed --out /tmp/reports

# 2. Or against the real archive: download the Kaggle dataset, unzip, then
make dataset REPLAYS=~/Downloads/archive   # ~3 min for 2,000 episodes
make train                                 # ~20 min (five splits + ablation + sweep)
make analyse

make test                                  # 49 tests, ~5s, no network needed
make verify                                # re-derive all 91 README figures from reports/
```

`make dataset` reduces 8.9 GB of JSON to a single 24 MB table by streaming one
file at a time across a worker pool; the raw archive never needs to fit in
memory and is git-ignored. Every number and figure in this README is written to
`reports/` by `make train` and `make analyse`, and committed, so the results are
checkable without running anything. `make verify` re-derives all 91 figures
quoted in this README from those tables and fails on any drift. It runs in CI, so
a retrain that moves a number cannot silently leave the prose behind.

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
  pipeline.py   the experiment: ladder, repeated splits, ablation, filter sweep,
                calibration variants, unseen-agent check
  plot.py       figures
scripts/        build_dataset.py · train.py · analyse.py · verify_readme.py
tests/          49 tests: parser invariants, leakage, perspective symmetry,
                filter causality, metric correctness on known cases
data/sample/    20 gzipped replays (1.3 MB) so CI runs the real pipeline
docs/           reverse-engineered notes on the replay format
reports/        generated metrics, tables and figures (committed)
```

The tables worth opening first are `reports/tables/repeated_splits.csv` (the
headline numbers, with their spread) and `repeated_splits_long.csv` (per-split,
for the paired comparisons above).

## Data

"The Pokemon Company - PTCG AI Battle Challenge Simulation Episodes", a Kaggle
simulations-competition dataset of `cabt` episode replays, released under
**CC0 1.0 (public domain)**. This project uses 2,000 JSON episode files, 8.9 GB
unzipped, from a single daily release.

Two things about the dataset are worth knowing before reusing any number here.
Kaggle selects each day's replays by average agent rating under a 20 GiB cap, so
the sample is drawn from the stronger tail of play rather than uniformly (see
"What I would not claim"). And the dataset ships a `manifest.csv` listing every
included episode with its score, which this pipeline does not currently read; it
is the obvious way to quantify the selection, and to test whether forecast
accuracy varies with agent strength.

The replay format itself is undocumented. The structural notes, enum codes and
gotchas in [`docs/data-schema.md`](docs/data-schema.md) are reverse-engineered
from the files and asserted by the test suite.

## License

MIT
