"""Data loading + Elo rating engine.

Loads the ~49k-match historical international results dataset, replays every
match in chronological order, and updates each team's Elo rating after the
result. This produces two things the rest of the app depends on:

  1. Final Elo ratings for every team today (used to seed simulations and
     head-to-head predictions).
  2. A "training table" of pre-match snapshots — each team's Elo rating *as it
     stood right before that match* — which is the leakage-free feature set
     both the Poisson and XGBoost models learn from.

Why Elo at all, instead of feeding raw match history to the ML model? Because
Elo already compresses "how good is this team right now, given everything
that happened up to today" into a single number, updated incrementally. That
turns an irregular time series (teams play wildly different numbers of
matches) into a clean, comparable feature.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
GOALSCORERS_CSV = os.path.join(DATA_DIR, "goalscorers.csv")
SHOOTOUTS_CSV = os.path.join(DATA_DIR, "shootouts.csv")

K_BASE = {
    "world_cup": 60,
    "continental": 50,   # Copa America, Euro, AFCON, Asian Cup, Gold Cup, CONCACAF NL
    "qualifier": 40,
    "nations_league": 35,
    "friendly": 20,
}

_CONTINENTAL_MARKERS = (
    "copa america", "uefa euro", "africa cup", "afc asian cup",
    "gold cup", "concacaf nations",
)


def k_factor(tournament: str) -> float:
    """Match-importance multiplier — bigger competitions move ratings more."""
    t = str(tournament).lower()
    if "fifa world cup" in t and "qualif" not in t:
        return K_BASE["world_cup"]
    if any(marker in t for marker in _CONTINENTAL_MARKERS):
        return K_BASE["continental"]
    if "qualif" in t:
        return K_BASE["qualifier"]
    if "nations league" in t or "confederation" in t:
        return K_BASE["nations_league"]
    return K_BASE["friendly"]


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo win-expectancy formula."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def goal_diff_multiplier(goal_diff: int) -> float:
    """FIFA-style margin-of-victory multiplier, capped at 1.75x."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return 1.75


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_goalscorers() -> pd.DataFrame:
    df = pd.read_csv(GOALSCORERS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_shootouts() -> pd.DataFrame:
    df = pd.read_csv(SHOOTOUTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


@dataclass
class EloEngine:
    """Replays match history chronologically and tracks Elo ratings.

    After calling `.run()`, `ratings` holds each team's current Elo and
    `history` holds one row per match with the *pre-match* Elo snapshot for
    both sides — exactly the leakage-free feature set a model should train on.
    """

    start_rating: float = 1000.0
    home_advantage: float = 75.0
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def get(self, team: str) -> float:
        return self.ratings.setdefault(team, self.start_rating)

    def run(self, results: pd.DataFrame) -> "EloEngine":
        played = results.dropna(subset=["home_score", "away_score"])
        for row in played.itertuples(index=False):
            home_elo = self.get(row.home_team)
            away_elo = self.get(row.away_team)

            self.history.append(
                {
                    "date": row.date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "neutral": bool(row.neutral),
                    "tournament": row.tournament,
                    "home_score": int(row.home_score),
                    "away_score": int(row.away_score),
                }
            )

            home_adj = 0.0 if row.neutral else self.home_advantage
            r_a, r_b = home_elo + home_adj, away_elo
            exp_a = expected_score(r_a, r_b)
            exp_b = 1.0 - exp_a

            if row.home_score > row.away_score:
                actual_a, actual_b = 1.0, 0.0
            elif row.home_score < row.away_score:
                actual_a, actual_b = 0.0, 1.0
            else:
                actual_a, actual_b = 0.5, 0.5

            k = k_factor(row.tournament)
            mult = goal_diff_multiplier(abs(row.home_score - row.away_score))

            self.ratings[row.home_team] = home_elo + k * mult * (actual_a - exp_a)
            self.ratings[row.away_team] = away_elo + k * mult * (actual_b - exp_b)
        return self

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def compute_elo() -> tuple[dict[str, float], pd.DataFrame]:
    """Convenience entry point: load data, replay it, return (ratings, history_df)."""
    results = load_results()
    engine = EloEngine().run(results)
    return engine.ratings, engine.history_df()
