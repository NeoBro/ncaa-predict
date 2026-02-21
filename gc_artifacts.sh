#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

DRY_RUN=1
DO_CACHE=0
DO_LOGS=0
DO_REPORTS=0
DO_SHARDS=0
DO_GIT_IGNORED=0
DO_ALL=0

usage() {
  cat <<'EOF'
Usage: ./gc_artifacts.sh [options]

Options:
  --all                 Enable cache, logs, reports, and shard cleanup
  --cache               Remove Python cache artifacts (__pycache__, *.pyc)
  --logs                Remove logs/*.log
  --reports             Remove transient report outputs (*_check.csv, *_smoke*.json, *_validation*.json)
  --shards              Remove per-school scrape shards in csv/games and csv/players
  --git-ignored         Run git clean on ignored files (requires --force)
  --dry-run             Preview only (default)
  --force               Execute deletions
  -h, --help            Show this help
EOF
}

run_rm() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DO_ALL=1 ;;
    --cache) DO_CACHE=1 ;;
    --logs) DO_LOGS=1 ;;
    --reports) DO_REPORTS=1 ;;
    --shards) DO_SHARDS=1 ;;
    --git-ignored) DO_GIT_IGNORED=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --force) DRY_RUN=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
  shift
done

if [[ "${DO_ALL}" -eq 1 ]]; then
  DO_CACHE=1
  DO_LOGS=1
  DO_REPORTS=1
  DO_SHARDS=1
fi

if [[ "${DO_CACHE}" -eq 1 ]]; then
  run_rm "find . -type d -name '__pycache__' -prune -exec rm -rf {} +"
  run_rm "find . -type f -name '*.pyc' -delete"
fi

if [[ "${DO_LOGS}" -eq 1 ]]; then
  run_rm "rm -f logs/*.log"
fi

if [[ "${DO_REPORTS}" -eq 1 ]]; then
  run_rm "rm -f reports/*_check.csv reports/*_smoke*.json reports/*_validation*.json"
fi

if [[ "${DO_SHARDS}" -eq 1 ]]; then
  run_rm "rm -rf csv/games csv/players"
  run_rm "mkdir -p csv/games csv/players"
fi

if [[ "${DO_GIT_IGNORED}" -eq 1 ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] git clean -ndX"
    git clean -ndX
  else
    echo "[exec] git clean -fdX"
    git clean -fdX
  fi
fi

echo "GC complete. dry_run=${DRY_RUN}"
