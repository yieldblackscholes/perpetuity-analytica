"""Feature engineering for the goal-prediction models.

Turns the Elo-replay history into supervised training tables. Each historical
match contributes two rows (one per team's scoring perspective) so a single
model learns one consistent "strength gap -> goals" relationship for
favourites and underdogs alike — this is the same trick the original
prediction script uses for its Poisson model, extended here with a couple of
extra features for the gradient-boosted model.

Features used:
  - elo_diff      : (own Elo - opponent Elo) / 100, the dominant signal
  - is_home       : 1 if this side has home advantage (0 on neutral ground —
                     true for almost all World Cup fixtures)
  - is_world_cup  : 1 if the match is a FIFA World Cup match (knockout
                     football tends to be tighter / lower-scoring than
                     friendlies, so this gets its own coefficient)
  - days_since_2000 : a mild recency feature so the model can pick up on
                     football slowly getting more attack-minded over decades,
                     without needing a hard cutoff on older data.

Target: goals that side actually scored in the match (a non-negative integer
-> Poisson-appropriate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ELO_SCALE = 100.0
FEATURE_COLUMNS = ["elo_diff", "is_home", "is_world_cup", "days_since_2000"]


def build_training_table(history: pd.DataFrame) -> pd.DataFrame:
    """Expand match history into one row per team-perspective.

    Returns a DataFrame with FEATURE_COLUMNS plus `goals` (target) and
    `team` / `opponent` / `date` for traceability.
    """
    rows = []
    epoch = pd.Timestamp("2000-01-01")

    for r in history.itertuples(index=False):
        is_wc = 1.0 if r.tournament == "FIFA World Cup" else 0.0
        days = (r.date - epoch).days / 365.25  # years since 2000, smoother scale

        # Home team's perspective
        rows.append(
            {
                "date": r.date,
                "team": r.home_team,
                "opponent": r.away_team,
                "elo_diff": (r.home_elo - r.away_elo) / ELO_SCALE,
                "is_home": 0.0 if r.neutral else 1.0,
                "is_world_cup": is_wc,
                "days_since_2000": days,
                "goals": r.home_score,
            }
        )
        # Away team's perspective (never has home advantage)
        rows.append(
            {
                "date": r.date,
                "team": r.away_team,
                "opponent": r.home_team,
                "elo_diff": (r.away_elo - r.home_elo) / ELO_SCALE,
                "is_home": 0.0,
                "is_world_cup": is_wc,
                "days_since_2000": days,
                "goals": r.away_score,
            }
        )

    return pd.DataFrame(rows)


def to_xy(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = table[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = table["goals"].to_numpy(dtype=float)
    return X, y


def make_match_features(
    elo_a: float,
    elo_b: float,
    a_home: bool = False,
    is_world_cup: bool = True,
    as_of_year: float = 2026.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the feature row for a single hypothetical fixture, from each
    side's perspective, ready to feed into a trained model's .predict().
    World Cup fixtures are played on neutral ground, so by default neither
    side gets the home flag.
    """
    days = (as_of_year - 2000.0)
    x_a = np.array([[(elo_a - elo_b) / ELO_SCALE, 1.0 if a_home else 0.0, 1.0 if is_world_cup else 0.0, days]])
    x_b = np.array([[(elo_b - elo_a) / ELO_SCALE, 0.0, 1.0 if is_world_cup else 0.0, days]])
    return x_a, x_b
