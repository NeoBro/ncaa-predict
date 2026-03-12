#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


def read_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def run_pick_generation(year: int, pick_style: str):
    cmd = [
        ".venv/bin/python",
        "build_custom_seeds_d1.py",
        "-y",
        str(year),
        "--kaggle-dir",
        "external/kaggle_mania",
    ]
    p0 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p0.returncode != 0:
        return False, p0.stdout + "\n" + p0.stderr

    cmd = [
        ".venv/bin/python",
        "generate_bracket_json.py",
        "--kaggle-dir",
        "external/kaggle_mania",
        "-y",
        str(year),
        "--custom-seeds-csv",
        f"brackets/custom_seeds_{year}.csv",
        "-o",
        f"brackets/{year}.json",
    ]
    p1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p1.returncode != 0:
        return False, p1.stdout + "\n" + p1.stderr

    cmd = [
        ".venv/bin/python",
        "generate_pick_sheet.py",
        "--score-model-in",
        "models/tourney_score_pipeline_prod.json",
        "--meta-model-in",
        "models/tourney_meta_model_prod.json",
        "--kaggle-dir",
        "external/kaggle_mania",
        "--custom-seeds-csv",
        f"brackets/custom_seeds_{year}.csv",
        "--all-games-csv",
        "csv/ncaa_games_all.csv",
        "--pick-style",
        pick_style,
        "-y",
        str(year),
        "--bracket-file",
        f"brackets/{year}.json",
        "-o",
        f"reports/pick_sheet_{year}.csv",
    ]
    p2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p2.returncode != 0:
        return False, p2.stdout + "\n" + p2.stderr

    cmd = [
        ".venv/bin/python",
        "render_filled_bracket.py",
        "--pick-sheet",
        f"reports/pick_sheet_{year}.csv",
        "--out-md",
        f"reports/bracket_{year}_filled.md",
    ]
    p3 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p3.returncode != 0:
        return False, p3.stdout + "\n" + p3.stderr
    return True, p0.stdout + "\n" + p1.stdout + "\n" + p2.stdout + "\n" + p3.stdout


st.set_page_config(page_title="NCAA Predictor", layout="wide")
st.title("NCAA Prediction Dashboard")

year = st.sidebar.number_input("Season", min_value=2002, max_value=2100, value=2026, step=1)
pick_style = st.sidebar.selectbox("Pick Style", options=["safe", "balanced", "chaos"], index=1)

st.sidebar.markdown("### Actions")
if st.sidebar.button("Run Full D-I Bracket Pipeline"):
    ok, output = run_pick_generation(int(year), pick_style)
    if ok:
        st.sidebar.success("Pipeline completed.")
    else:
        st.sidebar.error("Pipeline failed.")
    st.sidebar.code(output.strip()[:6000] if output else "", language="text")

left, right = st.columns(2)

with left:
    st.subheader("Data Inventory")
    inv_path = ROOT / "reports" / "data_inventory.json"
    if inv_path.exists():
        with inv_path.open() as f:
            st.json(json.load(f))
    else:
        st.info("Missing reports/data_inventory.json")

    st.subheader(f"{year} Regular Season Subset")
    reg = read_csv(ROOT / "csv" / "subsets" / f"ncaa_games_{year}_regular_season.csv")
    if reg is not None:
        st.caption(f"Rows: {len(reg):,}")
        st.dataframe(reg.head(20), use_container_width=True)
    else:
        st.info(f"Missing csv/subsets/ncaa_games_{year}_regular_season.csv")

with right:
    st.subheader(f"{year} Conference Tournament Subset")
    conf = read_csv(ROOT / "csv" / "subsets" / f"ncaa_games_{year}_conference_tourney.csv")
    if conf is not None:
        st.caption(f"Rows: {len(conf):,}")
        st.dataframe(conf.head(20), use_container_width=True)
    else:
        st.info(f"Missing csv/subsets/ncaa_games_{year}_conference_tourney.csv")

    st.subheader(f"{year} Custom Seeds")
    seeds = read_csv(ROOT / "brackets" / f"custom_seeds_{year}.csv")
    if seeds is not None:
        st.caption(f"Rows: {len(seeds):,}")
        st.dataframe(seeds.head(20), use_container_width=True)
    else:
        st.info(f"Missing brackets/custom_seeds_{year}.csv")

st.subheader(f"{year} Predicted Bracket Picks")
pick_sheet = read_csv(ROOT / "reports" / f"pick_sheet_{year}.csv")
if pick_sheet is not None and len(pick_sheet) > 0:
    final_rows = pick_sheet[pick_sheet["round"] == 1]
    champion = str(final_rows.iloc[0]["pick"]) if len(final_rows) > 0 else str(pick_sheet.iloc[0]["pick"])
    st.metric("Projected Champion", champion)
    st.dataframe(pick_sheet, use_container_width=True)
else:
    st.info(f"Missing reports/pick_sheet_{year}.csv (use sidebar action to generate)")

st.subheader(f"{year} Filled Bracket")
bracket_md = ROOT / "reports" / f"bracket_{year}_filled.md"
if bracket_md.exists():
    st.markdown(bracket_md.read_text())
else:
    st.info(f"Missing reports/bracket_{year}_filled.md (run sidebar pipeline)")
