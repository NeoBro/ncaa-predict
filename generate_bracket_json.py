#!/usr/bin/env python3
import argparse
import json
import os
import re

import pandas as pd

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_kaggle_teams,
)


def round_num(slot_name):
    m = re.match(r"R(\d+)", str(slot_name))
    return int(m.group(1)) if m else -1


def slot_sort_key(slot_name):
    # Prefer Championship-like slots last at same round.
    name = str(slot_name)
    return (round_num(name), 1 if name.endswith("CH") else 0, name)


def to_lists(node):
    if isinstance(node, tuple):
        return [to_lists(node[0]), to_lists(node[1])]
    return node


def normalize_seed_code(seed_val):
    text = str(seed_val).strip().upper()
    m = re.match(r"^[WXYZ][0-1][0-9][AB]?$", text)
    if m:
        return text
    return None


def normalize_slot_ref(token):
    """Normalize seed-like slot refs (e.g., W16a -> W16A) while leaving round slots unchanged."""
    s = str(token).strip()
    norm = normalize_seed_code(s)
    return norm if norm is not None else s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument(
        "--custom-seeds-csv", default=None,
        help="Optional custom seeds CSV for target year (supports season/year, seed, and team_name or team_id).")
    parser.add_argument(
        "--slots-year", type=int, default=None,
        help="Optional slots season to use when target year slots are unavailable.")
    parser.add_argument(
        "--name-source", choices=["ncaa", "kaggle"], default="ncaa",
        help="Team-name source for leaf nodes. (default: %(default)s)")
    parser.add_argument("--out", "-o", default=None)
    args = parser.parse_args()

    seeds_path = os.path.join(args.kaggle_dir, "MNCAATourneySeeds.csv")
    slots_path = os.path.join(args.kaggle_dir, "MNCAATourneySlots.csv")
    if not os.path.exists(seeds_path):
        raise FileNotFoundError("Missing %s" % seeds_path)
    if not os.path.exists(slots_path):
        raise FileNotFoundError("Missing %s (run fetch_tourney_data.py again)" % slots_path)

    seeds = pd.read_csv(seeds_path)
    slots = pd.read_csv(slots_path)
    teams = load_kaggle_teams(args.kaggle_dir)
    kaggle_name = {int(r.TeamID): r.TeamName for r in teams.itertuples()}

    ncaa_name_by_kaggle = {}
    if args.name_source == "ncaa":
        schools = load_ncaa_schools()
        mapping, _, _ = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
        ncaa_by_id = {int(r.school_id): r.school_name for r in schools.itertuples()}
        for k_id, n_id in mapping.items():
            if n_id in ncaa_by_id:
                ncaa_name_by_kaggle[int(k_id)] = ncaa_by_id[int(n_id)]

    season_slots = slots[slots["Season"] == args.year]
    if len(season_slots) == 0:
        use_slots_year = args.slots_year
        if use_slots_year is None:
            avail = sorted(slots["Season"].dropna().astype(int).unique().tolist())
            use_slots_year = avail[-1] if avail else None
        season_slots = slots[slots["Season"] == use_slots_year]
        if len(season_slots) == 0:
            raise ValueError("No slot rows for season %s (and no fallback slots available)" % args.year)
        print("Using slots from season %s for target season %s" % (use_slots_year, args.year))

    seed_to_teams = {}
    if args.custom_seeds_csv:
        custom = pd.read_csv(args.custom_seeds_csv)
        cols = {c.lower(): c for c in custom.columns}
        if "seed" not in cols or "team_name" not in cols:
            raise ValueError("Custom bracket seeds CSV must include at least: seed, team_name")
        season_col = cols.get("season") or cols.get("year")
        if season_col:
            custom = custom[custom[season_col] == args.year]
        if len(custom) == 0:
            raise ValueError("No custom seed rows found for year %s" % args.year)
        bad = []
        for row in custom.itertuples():
            seed_code = normalize_seed_code(getattr(row, cols["seed"]))
            if seed_code is None:
                bad.append(getattr(row, cols["seed"]))
                continue
            team = str(getattr(row, cols["team_name"])).strip()
            seed_to_teams.setdefault(seed_code, []).append(team)
        if bad:
            raise ValueError(
                "Invalid custom seed codes (expected region-coded like W01/X12/Y16a): %s" % bad[:8])
    else:
        season_seeds = seeds[seeds["Season"] == args.year]
        if len(season_seeds) == 0:
            raise ValueError("No seed rows for season %s (use --custom-seeds-csv for newer seasons)" % args.year)
        for row in season_seeds.itertuples():
            seed_to_teams.setdefault(row.Seed, []).append(int(row.TeamID))

    def team_name(team_id):
        if args.name_source == "ncaa" and team_id in ncaa_name_by_kaggle:
            return ncaa_name_by_kaggle[team_id]
        return kaggle_name.get(team_id, str(team_id))

    nodes = {}
    for seed, ids in seed_to_teams.items():
        if ids and isinstance(ids[0], str):
            names = ids
        else:
            names = [team_name(i) for i in ids]
        if len(names) == 1:
            nodes[seed] = names[0]
        elif len(names) == 2:
            names = sorted(names)
            nodes[seed] = (names[0], names[1])
        else:
            # Rare/invalid; fold as binary tree in listed order.
            cur = names[0]
            for nxt in names[1:]:
                cur = (cur, nxt)
            nodes[seed] = cur

    unresolved = season_slots.copy()
    progress = True
    while len(unresolved) > 0 and progress:
        progress = False
        keep_rows = []
        for row in unresolved.itertuples():
            strong = normalize_slot_ref(row.StrongSeed)
            weak = normalize_slot_ref(row.WeakSeed)
            if strong in nodes and weak in nodes:
                nodes[row.Slot] = (nodes[strong], nodes[weak])
                progress = True
            else:
                keep_rows.append(row)
        unresolved = pd.DataFrame([r._asdict() for r in keep_rows])

    if len(unresolved) > 0:
        missing_refs = set(
            normalize_slot_ref(s) for s in unresolved["StrongSeed"].tolist()
        ).union(
            normalize_slot_ref(s) for s in unresolved["WeakSeed"].tolist()
        )
        missing = sorted(missing_refs - set(nodes.keys()))
        raise ValueError("Could not resolve bracket slots; missing nodes: %s" % missing[:20])

    final_slot = sorted(season_slots["Slot"].tolist(), key=slot_sort_key)[-1]
    bracket = nodes[final_slot]

    out = args.out or "brackets/%s.json" % args.year
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(to_lists(bracket), f, indent=2)
    print("Wrote bracket JSON: %s (final slot: %s)" % (out, final_slot))
