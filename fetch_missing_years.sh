#!/usr/bin/env bash
set -euo pipefail

YEARS="${1:-2019,2020,2021,2022,2023,2024,2025}"
PY="${PYTHON_BIN:-.venv/bin/python}"

echo "Using python: ${PY}"
echo "Target years: ${YEARS}"

mkdir -p logs

echo "[1/3] Fetching games..."
"${PY}" fetch_csvs.py get_games -y "${YEARS}" | tee -a logs/fetch_games.log

echo "[2/3] Fetching players..."
"${PY}" fetch_csvs.py get_players -y "${YEARS}" | tee -a logs/fetch_players.log

echo "[3/3] Rebuilding merged games CSV..."
"${PY}" build_all_games_csv.py -o csv/ncaa_games_all.csv

echo "Done."
