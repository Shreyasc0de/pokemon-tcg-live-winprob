# Reports

Everything here is generated, and committed so the repo is readable without
running anything:

- `results.json` — headline metrics, the fitted filter parameter, the
  martingale test, and the unseen-agent check.
- `descriptive.json` — archive-level summary (episodes, agents, first-player
  advantage).
- `tables/*.csv` — every number quoted in the root README.
- `figures/*.png` — the figures in the root README.

Regenerate with `make train` and `make analyse`. `test_predictions.csv.gz`
(one row per held-out decision point, with each model's output) is written by
`make train` but git-ignored for size.
