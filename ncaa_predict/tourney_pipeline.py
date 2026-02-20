import os

import pandas as pd

from ncaa_predict.score_pipeline import _team_stats, _feature_row


def normalize_team_name(name):
    if pd.isna(name):
        return ""
    text = str(name).lower()
    for ch in ["'", ".", ",", "(", ")", "-", "&", "/"]:
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    replacements = {
        "saint": "st",
        "state": "st",
        "and": "",
        "university": "u",
    }
    tokens = [replacements.get(tok, tok) for tok in text.split()]
    tokens = [tok for tok in tokens if tok]
    return " ".join(tokens)


def load_kaggle_tourney(kaggle_dir):
    compact = os.path.join(kaggle_dir, "MNCAATourneyCompactResults.csv")
    if not os.path.exists(compact):
        raise FileNotFoundError(
            "Missing Kaggle file: %s (expected in --kaggle-dir)" % compact)
    games = pd.read_csv(compact)
    expected = {"Season", "WTeamID", "WScore", "LTeamID", "LScore"}
    if not expected.issubset(set(games.columns)):
        raise ValueError("Unexpected columns in %s" % compact)
    return games


def load_kaggle_seeds(kaggle_dir):
    path = os.path.join(kaggle_dir, "MNCAATourneySeeds.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def seed_to_int(seed):
    text = str(seed)
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return int(digits[:2])


def load_kaggle_teams(kaggle_dir):
    path = os.path.join(kaggle_dir, "MTeams.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing Kaggle file: %s" % path)
    return pd.read_csv(path)


def load_kaggle_team_spellings(kaggle_dir):
    path = os.path.join(kaggle_dir, "MTeamSpellings.csv")
    if not os.path.exists(path):
        return None
    for encoding in [None, "utf-8", "latin1", "cp1252"]:
        try:
            if encoding is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "unknown", b"", 0, 1, "Could not decode MTeamSpellings.csv")


def build_kaggle_to_ncaa_id_map(kaggle_dir, ncaa_schools_df):
    teams = load_kaggle_teams(kaggle_dir)
    spellings = load_kaggle_team_spellings(kaggle_dir)

    ncaa_name_to_ids = {}
    for row in ncaa_schools_df.itertuples():
        key = normalize_team_name(row.school_name)
        ncaa_name_to_ids.setdefault(key, set()).add(int(row.school_id))

    kaggle_variants = {}
    for row in teams.itertuples():
        kaggle_variants.setdefault(int(row.TeamID), set()).add(
            normalize_team_name(row.TeamName))
    if spellings is not None:
        for row in spellings.itertuples():
            kaggle_variants.setdefault(int(row.TeamID), set()).add(
                normalize_team_name(row.TeamNameSpelling))

    mapping = {}
    unresolved = {}
    ambiguous = {}
    for team_id, variants in kaggle_variants.items():
        candidate_ids = set()
        for variant in variants:
            if variant in ncaa_name_to_ids:
                candidate_ids |= ncaa_name_to_ids[variant]
        if len(candidate_ids) == 1:
            mapping[team_id] = next(iter(candidate_ids))
        elif len(candidate_ids) == 0:
            unresolved[team_id] = sorted(v for v in variants if v)
        else:
            ambiguous[team_id] = sorted(candidate_ids)
    return mapping, unresolved, ambiguous


def load_seed_map(kaggle_dir, kaggle_to_ncaa_id_map=None):
    seeds = load_kaggle_seeds(kaggle_dir)
    if seeds is None:
        return {}
    seed_map = {}
    for row in seeds.itertuples():
        team_id = int(row.TeamID)
        if kaggle_to_ncaa_id_map is not None:
            if team_id not in kaggle_to_ncaa_id_map:
                continue
            team_id = int(kaggle_to_ncaa_id_map[team_id])
        seed_num = seed_to_int(row.Seed)
        if seed_num is None:
            continue
        seed_map[(int(row.Season), team_id)] = seed_num
    return seed_map


def load_seed_name_map(kaggle_dir):
    seeds = load_kaggle_seeds(kaggle_dir)
    if seeds is None:
        return {}
    teams = load_kaggle_teams(kaggle_dir)
    spellings = load_kaggle_team_spellings(kaggle_dir)

    variants = {}
    for row in teams.itertuples():
        variants.setdefault(int(row.TeamID), set()).add(normalize_team_name(row.TeamName))
    if spellings is not None:
        for row in spellings.itertuples():
            variants.setdefault(int(row.TeamID), set()).add(
                normalize_team_name(row.TeamNameSpelling))

    name_seed = {}
    for row in seeds.itertuples():
        season = int(row.Season)
        team_id = int(row.TeamID)
        seed_num = seed_to_int(row.Seed)
        if seed_num is None:
            continue
        for name in variants.get(team_id, set()):
            key = (season, name)
            if key not in name_seed:
                name_seed[key] = seed_num
            else:
                name_seed[key] = min(name_seed[key], seed_num)
    return name_seed


def season_team_stats_from_csv(all_games_csv, season):
    games = pd.read_csv(
        all_games_csv,
        usecols=["year", "school_id", "opponent_id", "score", "opponent_score"])
    season_games = games[games["year"] == season]
    if len(season_games) == 0:
        return None
    return _team_stats(season_games)


def load_season_team_ids(all_games_csv, season):
    games = pd.read_csv(
        all_games_csv,
        usecols=["year", "school_id", "opponent_id"])
    season_games = games[games["year"] == season]
    if len(season_games) == 0:
        return set()
    school_ids = set(season_games["school_id"].dropna().astype(int).tolist())
    opp_ids = set(season_games["opponent_id"].dropna().astype(int).tolist())
    return school_ids | opp_ids


def validate_team_id_alignment(
    kaggle_games,
    all_games_csv,
    years,
    stats_year_offset=-1,
    kaggle_to_ncaa_id_map=None,
):
    report = []
    for season in years:
        tourney = kaggle_games[kaggle_games["Season"] == season]
        kaggle_ids = set(tourney["WTeamID"].astype(int).tolist()) | \
            set(tourney["LTeamID"].astype(int).tolist())
        if kaggle_to_ncaa_id_map is not None:
            mapped_ids = set(
                kaggle_to_ncaa_id_map[i] for i in kaggle_ids
                if i in kaggle_to_ncaa_id_map)
        else:
            mapped_ids = kaggle_ids
        stats_year = season + stats_year_offset
        season_ids = load_season_team_ids(all_games_csv, stats_year)
        missing = sorted(mapped_ids - season_ids)
        coverage = 1.0
        if mapped_ids:
            coverage = 1.0 - (len(missing) / len(mapped_ids))
        report.append({
            "season": int(season),
            "stats_year": int(stats_year),
            "kaggle_team_count": int(len(kaggle_ids)),
            "mapped_team_count": int(len(mapped_ids)),
            "season_team_count": int(len(season_ids)),
            "missing_team_count": int(len(missing)),
            "coverage": float(coverage),
            "missing_team_ids": missing,
        })
    return report


def build_tourney_feature_dataset(
    kaggle_games,
    all_games_csv,
    years,
    stats_year_offset=-1,
    kaggle_to_ncaa_id_map=None,
):
    rows = []
    y_a = []
    y_b = []
    meta = []
    for year in years:
        season_games = kaggle_games[kaggle_games["Season"] == year]
        stats_year = year + stats_year_offset
        stats = season_team_stats_from_csv(all_games_csv, stats_year)
        if stats is None:
            continue
        usable = 0
        for game in season_games.itertuples():
            a_id = int(game.WTeamID)
            b_id = int(game.LTeamID)
            if kaggle_to_ncaa_id_map is not None:
                if a_id not in kaggle_to_ncaa_id_map or b_id not in kaggle_to_ncaa_id_map:
                    continue
                a_id = int(kaggle_to_ncaa_id_map[a_id])
                b_id = int(kaggle_to_ncaa_id_map[b_id])
            a_score = float(game.WScore)
            b_score = float(game.LScore)
            if a_id not in stats.index or b_id not in stats.index:
                continue
            # Add canonical orientation.
            rows.append(_feature_row(stats.loc[a_id], stats.loc[b_id]))
            y_a.append(a_score)
            y_b.append(b_score)
            meta.append({
                "season": int(year),
                "stats_year": int(stats_year),
                "team_a_id": a_id,
                "team_b_id": b_id,
                "flipped": False,
            })
            # Add flipped orientation to prevent winner-first leakage.
            rows.append(_feature_row(stats.loc[b_id], stats.loc[a_id]))
            y_a.append(b_score)
            y_b.append(a_score)
            meta.append({
                "season": int(year),
                "stats_year": int(stats_year),
                "team_a_id": b_id,
                "team_b_id": a_id,
                "flipped": True,
            })
            usable += 1
        print("Tournament dataset year %s: usable games=%s" % (year, usable))
    if not rows:
        raise ValueError("No tournament games could be featurized")
    x = pd.DataFrame(rows).to_numpy(dtype="float32")
    return x, pd.Series(y_a).to_numpy(), pd.Series(y_b).to_numpy(), meta
