"""Top-level pipeline orchestration.

Runs Elo replay -> feature building -> model training -> fast lookup table
construction, once, and bundles the results into a single object the
Streamlit app can cache with @st.cache_resource. Also provides helper lookups
used across the app's tabs (team profiles, head-to-head history, player
goal stats).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from elo_engine import compute_elo, load_goalscorers, load_results, load_shootouts
from features import build_training_table, make_match_features
from models import FastEloLookup, evaluate_outcome_accuracy, train_poisson, train_xgboost
from simulator import TournamentSimulator
from teams2026 import TEAM_BY_NAME, WC2026_TEAMS


@dataclass
class Pipeline:
    results: pd.DataFrame
    goalscorers: pd.DataFrame
    shootouts: pd.DataFrame
    history: pd.DataFrame
    elo_ratings_all: dict[str, float]          # every team in the dataset, by csv_name
    wc2026_ratings: dict[str, float]            # 48 qualified teams, by display name
    xgb_model: object
    poisson_model: object
    xgb_report: dict
    xgb_lookup: FastEloLookup
    poisson_lookup: FastEloLookup
    xgb_outcome_acc: dict
    poisson_outcome_acc: dict


def run_pipeline(num_train_rows_note: bool = True) -> Pipeline:
    results = load_results()
    goalscorers = load_goalscorers()
    shootouts = load_shootouts()

    elo_ratings_all, history = compute_elo()
    wc2026_ratings = {t.name: elo_ratings_all.get(t.csv_name, 1000.0) for t in WC2026_TEAMS}

    table = build_training_table(history)

    xgb_model, xgb_report = train_xgboost(table)
    poisson_model = train_poisson(table)

    xgb_lookup = FastEloLookup(xgb_model, is_world_cup=True, as_of_year=2026.0, neutral=True)
    poisson_lookup = FastEloLookup(poisson_model, is_world_cup=True, as_of_year=2026.0, neutral=True)

    xgb_acc = evaluate_outcome_accuracy(table, xgb_model)
    poisson_acc = evaluate_outcome_accuracy(table, poisson_model)

    return Pipeline(
        results=results,
        goalscorers=goalscorers,
        shootouts=shootouts,
        history=history,
        elo_ratings_all=elo_ratings_all,
        wc2026_ratings=wc2026_ratings,
        xgb_model=xgb_model,
        poisson_model=poisson_model,
        xgb_report=xgb_report,
        xgb_lookup=xgb_lookup,
        poisson_lookup=poisson_lookup,
        xgb_outcome_acc=xgb_acc,
        poisson_outcome_acc=poisson_acc,
    )


def make_simulator(pipeline: Pipeline, use_xgboost: bool = True, seed: int | None = None) -> TournamentSimulator:
    lookup = pipeline.xgb_lookup if use_xgboost else pipeline.poisson_lookup
    return TournamentSimulator(pipeline.wc2026_ratings, lookup.expected_goals, seed=seed)


# ---------------------------------------------------------------------------
# Team-level helpers (used by the Team Explorer tab)
# ---------------------------------------------------------------------------

def team_world_cup_matches(pipeline: Pipeline, csv_name: str) -> pd.DataFrame:
    df = pipeline.results
    wc = df[df["tournament"] == "FIFA World Cup"]
    mask = (wc["home_team"] == csv_name) | (wc["away_team"] == csv_name)
    return wc[mask].sort_values("date")


def team_all_matches(pipeline: Pipeline, csv_name: str) -> pd.DataFrame:
    df = pipeline.results
    mask = (df["home_team"] == csv_name) | (df["away_team"] == csv_name)
    return df[mask].sort_values("date")


def team_world_cup_record(pipeline: Pipeline, csv_name: str) -> dict:
    """Wins / draws / losses / goals for / against, World Cup matches only,
    plus best-ever finish if derivable from match patterns (kept simple:
    counts of matches played per edition)."""
    matches = team_world_cup_matches(pipeline, csv_name)
    played = matches.dropna(subset=["home_score", "away_score"])
    wins = draws = losses = gf = ga = 0
    for r in played.itertuples(index=False):
        is_home = r.home_team == csv_name
        own = r.home_score if is_home else r.away_score
        opp = r.away_score if is_home else r.home_score
        gf += own
        ga += opp
        if own > opp:
            wins += 1
        elif own < opp:
            losses += 1
        else:
            draws += 1
    editions = sorted(played["date"].dt.year.unique().tolist())
    return {
        "played": len(played),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": int(gf),
        "goals_against": int(ga),
        "editions": editions,
    }


def head_to_head(pipeline: Pipeline, csv_name_a: str, csv_name_b: str) -> pd.DataFrame:
    df = pipeline.results
    mask = (
        ((df["home_team"] == csv_name_a) & (df["away_team"] == csv_name_b))
        | ((df["home_team"] == csv_name_b) & (df["away_team"] == csv_name_a))
    )
    return df[mask].sort_values("date")


def team_goalscorers(pipeline: Pipeline, csv_name: str, world_cup_only: bool = True) -> pd.DataFrame:
    goals = pipeline.goalscorers
    team_goals = goals[goals["team"] == csv_name].copy()
    if world_cup_only:
        results_key = pipeline.results[["date", "home_team", "away_team", "tournament"]]
        team_goals = team_goals.merge(results_key, on=["date", "home_team", "away_team"], how="left")
        team_goals = team_goals[team_goals["tournament"] == "FIFA World Cup"]
    return team_goals


def top_scorers(pipeline: Pipeline, world_cup_only: bool = True, top_n: int = 25) -> pd.DataFrame:
    goals = pipeline.goalscorers.copy()
    if world_cup_only:
        results_key = pipeline.results[["date", "home_team", "away_team", "tournament"]]
        goals = goals.merge(results_key, on=["date", "home_team", "away_team"], how="left")
        goals = goals[goals["tournament"] == "FIFA World Cup"]
    goals = goals[goals["own_goal"] == False]  # noqa: E712
    leaderboard = (
        goals.groupby(["scorer", "team"])
        .size()
        .reset_index(name="goals")
        .sort_values("goals", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return leaderboard
