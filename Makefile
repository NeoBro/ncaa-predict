PY=.venv/bin/python
YEARS=2019,2020,2021,2022,2023,2024,2025

.PHONY: data consolidate train-score train-meta sensitivity gate picks

data:
	./fetch_missing_years.sh $(YEARS)

consolidate:
	$(PY) consolidate_data.py --start-year 2002 --end-year 2025

train-score:
	$(PY) train_tourney_score_pipeline.py \
	  --kaggle-dir external/kaggle_mania \
	  --all-games-csv csv/ncaa_games_all.csv \
	  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
	  -v 2017 -t 2018 \
	  -o models/tourney_score_pipeline_prod.json

train-meta:
	$(PY) train_tourney_meta_model.py \
	  --kaggle-dir external/kaggle_mania \
	  --all-games-csv csv/ncaa_games_all.csv \
	  --score-model-in models/tourney_score_pipeline_prod.json \
	  -y 2008,2009,2010,2011,2012,2013,2014,2015,2016 \
	  -v 2017 -t 2018 \
	  -o models/tourney_meta_model_prod.json

sensitivity:
	$(PY) backtest_tourney_sensitivity.py \
	  --kaggle-dir external/kaggle_mania \
	  --all-games-csv csv/ncaa_games_all.csv \
	  --score-model-in models/tourney_score_pipeline_prod.json \
	  --start-year 2008 --end-year 2025 \
	  --min-train-years 6 \
	  --report-out reports/tourney_sensitivity_full.json

gate:
	$(PY) validate_model_release.py \
	  --sensitivity-report reports/tourney_sensitivity_full.json \
	  --data-inventory reports/data_inventory.json

picks:
	$(PY) generate_pick_sheet.py \
	  --score-model-in models/tourney_score_pipeline_prod.json \
	  --meta-model-in models/tourney_meta_model_prod.json \
	  --kaggle-dir external/kaggle_mania \
	  --all-games-csv csv/ncaa_games_all.csv \
	  --pick-style balanced \
	  -y 2025 \
	  --bracket-file brackets/2018.json \
	  -o reports/pick_sheet_prod.csv
