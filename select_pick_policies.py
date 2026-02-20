#!/usr/bin/env python3
import argparse
import json
import os


def parse_key(key):
    b, t, m = key.split("|")
    return {
        "upset_bias": float(b),
        "upset_threshold": float(t),
        "min_underdog_win_prob": float(m),
    }


def pick_safe(rows):
    # Highest accuracy, then lower upset pick rate.
    return sorted(
        rows,
        key=lambda r: (
            -float(r.get("accuracy_mean", 0.0)),
            float(r.get("upset_pick_rate_mean", 1.0)),
        ),
    )[0]


def pick_balanced(rows, min_acc=0.60, min_recall=0.40):
    cands = [
        r for r in rows
        if float(r.get("accuracy_mean", 0.0)) >= min_acc
        and float(r.get("upset_recall_mean", 0.0) or 0.0) >= min_recall
    ]
    if not cands:
        cands = rows
    # Maximize a simple balance score.
    return sorted(
        cands,
        key=lambda r: (
            -(0.65 * float(r.get("accuracy_mean", 0.0)) +
              0.35 * float(r.get("upset_recall_mean", 0.0) or 0.0)),
            float(r.get("upset_pick_rate_mean", 1.0)),
        ),
    )[0]


def pick_chaos(rows, min_acc=0.52):
    cands = [r for r in rows if float(r.get("accuracy_mean", 0.0)) >= min_acc]
    if not cands:
        cands = rows
    return sorted(
        cands,
        key=lambda r: (
            -float(r.get("upset_recall_mean", 0.0) or 0.0),
            -float(r.get("accuracy_mean", 0.0)),
        ),
    )[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-report", required=True)
    parser.add_argument("--out", default="reports/pick_policies.json")
    args = parser.parse_args()

    with open(args.sensitivity_report) as f:
        report = json.load(f)
    rows = report.get("grid_summary", [])
    if not rows:
        raise ValueError("No grid_summary in %s" % args.sensitivity_report)

    safe = pick_safe(rows)
    balanced = pick_balanced(rows)
    chaos = pick_chaos(rows)

    payload = {
        "source_report": args.sensitivity_report,
        "policies": {
            "safe": {
                **parse_key(safe["config_key"]),
                "metrics": safe,
            },
            "balanced": {
                **parse_key(balanced["config_key"]),
                "metrics": balanced,
            },
            "chaos": {
                **parse_key(chaos["config_key"]),
                "metrics": chaos,
            },
        },
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("Wrote policy selections:", args.out)
    for name in ["safe", "balanced", "chaos"]:
        p = payload["policies"][name]
        print(
            "%s: bias=%.2f threshold=%.2f min_dog=%.2f | acc=%.4f upset_recall=%.4f" % (
                name,
                p["upset_bias"],
                p["upset_threshold"],
                p["min_underdog_win_prob"],
                float(p["metrics"].get("accuracy_mean", 0.0)),
                float(p["metrics"].get("upset_recall_mean", 0.0) or 0.0),
            )
        )
