# Pokemon TCG live win probability
#
#   make sample     parse the 20 committed replays -> data/sample_processed
#   make dataset    parse the full archive         -> data/processed
#   make train      fit the model ladder, write reports/
#   make analyse    descriptive tables and the prize-differential figure
#   make all        dataset + train + analyse
#   make test       run the test suite
#
# Point REPLAYS at the unzipped Kaggle archive:
#   make dataset REPLAYS=~/Downloads/archive

REPLAYS ?= data/raw
DATA    ?= data/processed
OUT     ?= reports
PY      ?= python3
export PYTHONPATH := src

.PHONY: all sample dataset train analyse test lint clean help

help:
	@sed -n '2,14p' Makefile | sed 's/^# \{0,1\}//'

sample:
	$(PY) scripts/build_dataset.py --replays data/sample --out data/sample_processed

dataset:
	$(PY) scripts/build_dataset.py --replays $(REPLAYS) --out $(DATA)

train:
	$(PY) scripts/train.py --data $(DATA) --out $(OUT)

analyse:
	$(PY) scripts/analyse.py --data $(DATA) --out $(OUT)

all: dataset train analyse

test:
	$(PY) -m unittest discover -s tests -v

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check src scripts tests || echo "ruff not installed; skipping"

clean:
	rm -rf data/processed data/sample_processed reports/tables reports/figures \
	       reports/results.json reports/descriptive.json reports/test_predictions.csv.gz
