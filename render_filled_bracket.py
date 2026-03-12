#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


ROUND_LABELS = {
    7: "Play-In",
    6: "Round of 64",
    5: "Round of 32",
    4: "Sweet 16",
    3: "Elite 8",
    2: "Final 4",
    1: "Championship",
}


def parse_args():
    p = argparse.ArgumentParser(description="Render filled bracket from pick sheet.")
    p.add_argument("--pick-sheet", required=True)
    p.add_argument("--out-md", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pick_sheet)
    if len(df) == 0:
        raise ValueError("Empty pick sheet: %s" % args.pick_sheet)

    required = {"round", "team_a", "team_b", "seed_a", "seed_b", "pick", "win_prob_a", "win_prob_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Pick sheet missing columns: %s" % missing)

    lines = []
    lines.append("# Filled Bracket")
    final_rows = df[df["round"] == 1]
    if len(final_rows) == 0:
        raise ValueError("No championship row found (round == 1)")
    champion = str(final_rows.iloc[0]["pick"])
    lines.append("")
    lines.append("**Projected Champion:** %s" % champion)
    lines.append("")

    for r in sorted(df["round"].unique(), reverse=True):
        lines.append("## %s" % ROUND_LABELS.get(int(r), "Round %s" % r))
        round_df = df[df["round"] == r].copy().reset_index(drop=True)
        for i, row in round_df.iterrows():
            pa = float(row["win_prob_a"])
            pb = float(row["win_prob_b"])
            lines.append(
                "%s. (%s) %s vs (%s) %s -> **%s** (p=%.3f)"
                % (
                    i + 1,
                    int(row["seed_a"]),
                    row["team_a"],
                    int(row["seed_b"]),
                    row["team_b"],
                    row["pick"],
                    pa if row["pick"] == row["team_a"] else pb,
                )
            )
        lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print("Wrote:", str(out_path))
    print("Champion:", champion)


if __name__ == "__main__":
    main()
