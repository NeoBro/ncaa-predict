#!/usr/bin/env python3
import argparse
import json
import os

from ncaa_predict.bracket import load_bracket
from ncaa_predict.data_loader import load_ncaa_games, load_ncaa_schools
from ncaa_predict.score_pipeline import _team_stats, load_pipeline
from ncaa_predict.tourney_pipeline import (
    build_kaggle_to_ncaa_id_map,
    load_seed_name_map,
    load_seed_map,
)


def normalize_name(name):
    text = name.lower()
    for c in ["'", ".", ",", "(", ")", "-", "&", "/"]:
        text = text.replace(c, " ")
    text = " ".join(text.split())
    repl = {
        "saint": "st",
        "mount": "mt",
        "state": "st",
        "university": "u",
    }
    tokens = [repl.get(tok, tok) for tok in text.split()]
    out = " ".join(tokens)
    aliases = {
        "liu brooklyn": "liu",
    }
    return aliases.get(out, out)


def flatten(node):
    a, b = node
    out = []
    out += flatten(a) if isinstance(a, tuple) else [a]
    out += flatten(b) if isinstance(b, tuple) else [b]
    return out


def resolve_team_id(name, schools):
    wanted = normalize_name(name)
    matches = [
        int(r.school_id) for r in schools.itertuples()
        if normalize_name(r.school_name) == wanted
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) == 0:
        return None, "not_found"
    return None, "ambiguous"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracket-file", required=True)
    parser.add_argument("--score-model-in", required=True)
    parser.add_argument("--kaggle-dir", required=True)
    parser.add_argument("--year", "-y", required=True, type=int)
    parser.add_argument("--max-bad-ratio", type=float, default=0.06)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    bracket = load_bracket(args.bracket_file)
    teams = sorted(set(flatten(bracket)))
    payload, _, _ = load_pipeline(args.score_model_in)

    stats_year = args.year + int(payload["config"]["stats_year_offset"])
    games = load_ncaa_games(stats_year, max_bad_ratio=args.max_bad_ratio)
    stats = _team_stats(games)
    schools = load_ncaa_schools()
    kaggle_to_ncaa, _, _ = build_kaggle_to_ncaa_id_map(args.kaggle_dir, schools)
    seed_map = load_seed_map(args.kaggle_dir, kaggle_to_ncaa_id_map=kaggle_to_ncaa)
    seed_name_map = load_seed_name_map(args.kaggle_dir)

    rows = []
    errors = 0
    for name in teams:
        team_id, err = resolve_team_id(name, schools)
        status = "ok"
        issues = []
        if err is not None:
            status = "error"
            issues.append(err)
            errors += 1
        else:
            if team_id not in stats.index:
                status = "error"
                issues.append("missing_stats")
                errors += 1
            seed = seed_map.get((args.year, team_id))
            if seed is None:
                seed = seed_name_map.get((args.year, normalize_name(name)))
            if seed is None:
                status = "warn" if status != "error" else status
                issues.append("missing_seed")
        rows.append({
            "team_name": name,
            "team_id": team_id,
            "status": status,
            "issues": issues,
            "seed": seed if team_id is not None else None,
        })

    summary = {
        "bracket_file": args.bracket_file,
        "year": args.year,
        "stats_year": stats_year,
        "n_teams": len(teams),
        "errors": errors,
        "warnings": sum(1 for r in rows if r["status"] == "warn"),
    }
    print("Bracket validation summary:")
    print(summary)
    for row in rows:
        if row["status"] != "ok":
            print("%s | %s | %s" % (row["team_name"], row["status"], ",".join(row["issues"])))

    if args.report_out:
        out_dir = os.path.dirname(args.report_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({"summary": summary, "teams": rows}, f, indent=2, sort_keys=True)
        print("Wrote bracket validation report: %s" % args.report_out)

    if errors > 0:
        raise SystemExit(2)
