NCAA bracket modeling with leakage-safe training, tournament-focused score models,
and upset-aware pick generation.

All scripts support `--help`.

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Data Workflow

### 1) Scrape missing seasons (resumable)

```bash
./fetch_missing_years.sh 2019,2020,2021,2022,2023,2024,2025
```

This wraps:
- `fetch_csvs.py get_games`
- `fetch_csvs.py get_players`
- `build_all_games_csv.py`

Per-school shards are cached in `csv/games` and `csv/players`, so reruns resume.

### 2) Consolidate and inventory

```bash
.venv/bin/python consolidate_data.py --start-year 2002 --end-year 2025
```

Outputs:
- `csv/ncaa_games_all.csv`
- `csv/ncaa_players_all.csv`
- `reports/data_inventory.json`

### 3) Tournament reference data

```bash
.venv/bin/python fetch_tourney_data.py -o external/kaggle_mania
```

Expected files include:
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MNCAATourneySlots.csv`
- `MTeams.csv`
- `MTeamSpellings.csv`

## Core Training

## Legacy winner model (player features)

```bash
.venv/bin/python train.py \
  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
  -v 2017 -t 2018 \
  --seed 42 \
  --early-stopping-patience 5 \
  --metrics-out reports/train_legacy.json \
  -o models/model_legacy.keras
```

## Tournament score pipeline (recommended)

This predicts both team scores, calibrates win probability, and avoids winner-first leakage.

```bash
.venv/bin/python train_tourney_score_pipeline.py \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
  -v 2017 -t 2018 \
  --stats-year-offset 0 \
  --cutoff-month 3 --cutoff-day 15 \
  --alignment-report-out reports/tourney_id_alignment_train.json \
  -o models/tourney_score_pipeline_prod.json
```

Notes:
- `--stats-year-offset 0` means tournament year `Y` uses season `Y` data.
- `--cutoff-month/--cutoff-day` restrict features to pre-tournament games only.

## Tournament meta model (win + upset)

```bash
.venv/bin/python train_tourney_meta_model.py \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  --score-model-in models/tourney_score_pipeline_prod.json \
  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
  -v 2017 -t 2018 \
  --stats-year-offset 0 \
  --cutoff-month 3 --cutoff-day 15 \
  -o models/tourney_meta_model_prod.json
```

## Backtesting and Sensitivity

## Tournament score rolling backtest

```bash
.venv/bin/python backtest_tourney_score_pipeline.py \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  --start-year 2008 --end-year 2018 \
  --min-train-years 6 \
  --stats-year-offset 0 \
  --cutoff-month 3 --cutoff-day 15 \
  --report-out reports/tourney_backtest.json
```

## Upset policy sensitivity sweep

```bash
.venv/bin/python backtest_tourney_sensitivity.py \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  --score-model-in models/tourney_score_pipeline_prod.json \
  --start-year 2008 --end-year 2025 \
  --min-train-years 6 \
  --stats-year-offset 0 \
  --cutoff-month 3 --cutoff-day 15 \
  --report-out reports/tourney_sensitivity_full.json
```

This reports accuracy/upset-recall tradeoffs for upset-control configs.

## Release Gates

```bash
.venv/bin/python validate_model_release.py \
  --sensitivity-report reports/tourney_sensitivity_full.json \
  --data-inventory reports/data_inventory.json
```

Fails if minimum folds/accuracy/upset recall (or missing-year limits) are not met.

## Pick Sheet and Bracket

Generate bracket JSON from Kaggle slots/seeds:

```bash
.venv/bin/python generate_bracket_json.py \
  --kaggle-dir external/kaggle_mania \
  -y 2018 \
  --name-source ncaa \
  -o brackets/2018.json
```

Validate bracket/team mappings:

```bash
.venv/bin/python validate_bracket.py \
  --bracket-file brackets/2018.json \
  --score-model-in models/tourney_score_pipeline_prod.json \
  --kaggle-dir external/kaggle_mania \
  -y 2018 \
  --report-out reports/bracket_validation_2018.json
```

Generate picks:

```bash
.venv/bin/python generate_pick_sheet.py \
  --score-model-in models/tourney_score_pipeline_prod.json \
  --meta-model-in models/tourney_meta_model_prod.json \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  --pick-style balanced \
  -y 2018 \
  --bracket-file brackets/2018.json \
  -o reports/pick_sheet_2018.csv
```

`--pick-style` options:
- `safe`
- `balanced`
- `chaos`
- `custom` (use manual upset flags)

## One-command Make targets

```bash
make data
make consolidate
make train-score
make train-meta
make sensitivity
make gate
make picks
```

## Keep Large Artifacts Out of Git

`.gitignore` is configured to ignore generated datasets/models/reports/logs, including:
- `csv/ncaa_games_*.csv`
- `csv/ncaa_players_*.csv`
- `csv/ncaa_games_all.csv`
- `csv/ncaa_players_all.csv`
- `models/*.json`, `models/*.keras`, `models/*.h5`
- `reports/*.json`, `reports/*.csv`
- `logs/`
- `external/kaggle_mania/`
