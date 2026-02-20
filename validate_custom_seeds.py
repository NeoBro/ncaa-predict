#!/usr/bin/env python3
import argparse
import json
import os
import re

import pandas as pd

from ncaa_predict.data_loader import load_ncaa_schools
from ncaa_predict.tourney_pipeline import normalize_team_name


SEED_RE = re.compile(r"^[WXYZ][0-1][0-9][AB]?$")


def expected_seed_slots():
    # 68-team style input seeds.
    out = []
    for region in ["W", "X", "Y", "Z"]:
        for n in range(1, 17):
            if n in (11, 16):
                out.append("%s%02dA" % (region, n))
                out.append("%s%02dB" % (region, n))
            else:
                out.append("%s%02d" % (region, n))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds-csv", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--strict", action="store_true", help="Fail on missing/extra seed slots")
    args = parser.parse_args()

    if not os.path.exists(args.seeds_csv):
        raise FileNotFoundError("Missing seeds file: %s" % args.seeds_csv)

    df = pd.read_csv(args.seeds_csv)
    cols = {c.lower(): c for c in df.columns}
    required = {"seed", "team_name"}
    missing_cols = sorted(c for c in required if c not in cols)
    if missing_cols:
        raise ValueError("Missing required columns: %s" % missing_cols)

    season_col = cols.get("season") or cols.get("year")
    if season_col is not None:
        df = df[df[season_col] == args.year]
    if len(df) == 0:
        raise ValueError("No seed rows found for year %s" % args.year)

    schools = load_ncaa_schools()
    school_name_to_ids = {}
    for row in schools.itertuples():
        key = normalize_team_name(row.school_name)
        school_name_to_ids.setdefault(key, set()).add(int(row.school_id))

    rows = []
    errors = 0
    warnings = 0
    seen_seed = {}
    seen_team = {}

    for row in df.itertuples():
        seed_raw = str(getattr(row, cols["seed"])).strip().upper()
        team_raw = str(getattr(row, cols["team_name"])).strip()
        issues = []
        status = "ok"

        if not SEED_RE.match(seed_raw):
            issues.append("invalid_seed_format")
            status = "error"
            errors += 1

        n = normalize_team_name(team_raw)
        ids = sorted(school_name_to_ids.get(n, []))
        if len(ids) == 0:
            issues.append("team_not_found")
            status = "error"
            errors += 1
        elif len(ids) > 1:
            issues.append("team_ambiguous")
            status = "error"
            errors += 1

        seen_seed[seed_raw] = seen_seed.get(seed_raw, 0) + 1
        seen_team[n] = seen_team.get(n, 0) + 1
        rows.append({
            "seed": seed_raw,
            "team_name": team_raw,
            "status": status,
            "issues": issues,
            "resolved_team_ids": ids,
        })

    dup_seed = sorted(k for k, v in seen_seed.items() if v > 1)
    dup_team = sorted(k for k, v in seen_team.items() if v > 1)
    if dup_seed:
        errors += len(dup_seed)
    if dup_team:
        errors += len(dup_team)

    expected = set(expected_seed_slots())
    actual = set(seen_seed.keys())
    missing_seed_slots = sorted(expected - actual)
    extra_seed_slots = sorted(actual - expected)
    if missing_seed_slots:
        warnings += len(missing_seed_slots)
    if extra_seed_slots:
        warnings += len(extra_seed_slots)

    summary = {
        "year": args.year,
        "rows": int(len(df)),
        "errors": int(errors),
        "warnings": int(warnings),
        "duplicate_seed_count": len(dup_seed),
        "duplicate_team_count": len(dup_team),
        "missing_seed_slots_count": len(missing_seed_slots),
        "extra_seed_slots_count": len(extra_seed_slots),
    }

    print("Custom seeds validation summary:")
    print(summary)
    if dup_seed:
        print("Duplicate seeds:", dup_seed[:10])
    if dup_team:
        print("Duplicate teams:", dup_team[:10])
    if missing_seed_slots:
        print("Missing seed slots (first 12):", missing_seed_slots[:12])
    if extra_seed_slots:
        print("Extra seed slots:", extra_seed_slots[:12])

    payload = {
        "summary": summary,
        "rows": rows,
        "duplicate_seeds": dup_seed,
        "duplicate_teams": dup_team,
        "missing_seed_slots": missing_seed_slots,
        "extra_seed_slots": extra_seed_slots,
    }
    if args.report_out:
        out_dir = os.path.dirname(args.report_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print("Wrote report:", args.report_out)

    if errors > 0:
        raise SystemExit(2)
    if args.strict and (missing_seed_slots or extra_seed_slots):
        raise SystemExit(2)
