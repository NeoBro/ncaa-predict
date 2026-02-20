#!/usr/bin/env python3
import argparse
import os

import requests


MIRROR_BASE_URLS = [
    "https://huggingface.co/Jensen-holm/Nigl/resolve/main/data/",
]

FILES = [
    "MNCAATourneyCompactResults.csv",
    "MNCAATourneySeeds.csv",
    "MNCAATourneySlots.csv",
    "MTeams.csv",
    "MTeamSpellings.csv",
]


def download_file(url, out_path):
    r = requests.get(url, timeout=60, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError("HTTP %s for %s" % (r.status_code, url))
    with open(out_path, "wb") as f:
        f.write(r.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", "-o", default="external/kaggle_mania",
        help="Output directory for tournament CSV files.")
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for filename in FILES:
        out_path = os.path.join(args.out_dir, filename)
        if os.path.exists(out_path) and not args.overwrite:
            print("Skip existing: %s" % out_path)
            continue

        downloaded = False
        for base in MIRROR_BASE_URLS:
            url = base + filename
            try:
                print("Downloading %s" % url)
                download_file(url, out_path)
                print("Saved %s" % out_path)
                downloaded = True
                break
            except Exception as e:
                print("Failed %s: %s" % (url, e))
        if not downloaded:
            raise SystemExit("Unable to download %s from configured mirrors" % filename)

    print("Tournament data fetch complete: %s" % args.out_dir)
