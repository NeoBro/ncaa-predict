# NCAA 2026 Runbook

This is the operational checklist for generating a 2026 bracket pick sheet.

## 0) Environment

```bash
cd "/home/michael/Documents/DEVSECOPS/NCAA Prediction/ncaa-predict"
source .venv/bin/activate
```

## 1) Refresh data

```bash
./fetch_missing_years.sh 2026
```

If interrupted, rerun the same command (scraping is resumable).

## 2) Consolidate and inventory

```bash
.venv/bin/python consolidate_data.py --start-year 2002 --end-year 2026
```

## 3) Train production tournament models (pre-tourney cutoff)

```bash
.venv/bin/python train_tourney_score_pipeline.py \
  --kaggle-dir external/kaggle_mania \
  --all-games-csv csv/ncaa_games_all.csv \
  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
  -v 2017 -t 2018 \
  --stats-year-offset 0 \
  --cutoff-month 3 --cutoff-day 15 \
  -o models/tourney_score_pipeline_prod.json
```

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

## 4) Backtest sensitivity and select policy

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

```bash
.venv/bin/python select_pick_policies.py \
  --sensitivity-report reports/tourney_sensitivity_full.json \
  --out reports/pick_policies.json
```

Optional release gate:

```bash
.venv/bin/python validate_model_release.py \
  --sensitivity-report reports/tourney_sensitivity_full.json \
  --data-inventory reports/data_inventory.json
```

## 5) Prepare 2026 seeds

Create `brackets/custom_seeds_2026.csv` from `brackets/custom_seeds_template.csv`.

Required columns:
- `season` (or `year`)
- `seed` (region-coded like `W01`, `X12`, `Y16A`)
- `team_name`

Validate:

```bash
.venv/bin/python validate_custom_seeds.py \
  --seeds-csv brackets/custom_seeds_2026.csv \
  -y 2026 \
  --report-out reports/custom_seeds_2026_validation.json \
  --strict
```

## 6) Build bracket JSON

```bash
.venv/bin/python generate_bracket_json.py \
  --kaggle-dir external/kaggle_mania \
  -y 2026 \
  --custom-seeds-csv brackets/custom_seeds_2026.csv \
  -o brackets/2026.json
```

## 7) Generate pick sheet

Balanced default:

```bash
.venv/bin/python generate_pick_sheet.py \
  --score-model-in models/tourney_score_pipeline_prod.json \
  --meta-model-in models/tourney_meta_model_prod.json \
  --kaggle-dir external/kaggle_mania \
  --custom-seeds-csv brackets/custom_seeds_2026.csv \
  --all-games-csv csv/ncaa_games_all.csv \
  --pick-style balanced \
  -y 2026 \
  --bracket-file brackets/2026.json \
  -o reports/pick_sheet_2026.csv
```

## 8) Quick sanity checks

```bash
head -n 5 reports/pick_sheet_2026.csv
python - <<'PY'
import pandas as pd
d=pd.read_csv('reports/pick_sheet_2026.csv')
print('games:',len(d),'seed20_rows:',int((d.seed_a.eq(20)|d.seed_b.eq(20)).sum()))
print(d[['team_a','seed_a','team_b','seed_b','pick']].head(10).to_string(index=False))
PY
```

If `seed20_rows > 0`, fix `brackets/custom_seeds_2026.csv` and rerun steps 5-7.
