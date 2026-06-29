"""The two goal-prediction models: XGBoost (primary) and Poisson regression
(comparison / fallback).

Both models solve the same problem — predict expected goals for a team given
its Elo gap against the opponent and a few match-context features — but with
different machinery:

  * PoissonGoalModel  : a Poisson GLM (generalized linear model). Assumes
                        log(expected goals) is *linear* in the features. Fast,
                        interpretable (each coefficient has a direct "x this
                        many goals per unit" meaning), and exactly what the
                        original prediction script used.

  * XGBoostGoalModel  : a gradient-boosted tree ensemble trained with a
                        Poisson objective. Can capture non-linear effects
                        (e.g. a 300-point Elo gap not mattering twice as much
                        as a 150-point gap) and interactions between features
                        that a linear model can't. Generally the more accurate
                        of the two on this kind of tabular sports data, which
                        is why it's the primary model here — Poisson is kept
                        alongside as an interpretable sanity check.

Both expose the same `.expected_goals(elo_a, elo_b, ...)` interface so the
simulation engine can swap between them transparently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split

from features import FEATURE_COLUMNS, build_training_table, to_xy


@dataclass
class PoissonGoalModel:
    model: PoissonRegressor

    def expected_goals(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def coefficients(self) -> dict[str, float]:
        return dict(zip(FEATURE_COLUMNS, self.model.coef_.tolist()))


@dataclass
class XGBoostGoalModel:
    model: xgb.XGBRegressor

    def expected_goals(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> dict[str, float]:
        imp = self.model.feature_importances_
        return dict(zip(FEATURE_COLUMNS, imp.tolist()))


class FastEloLookup:
    """Precomputed Elo-gap -> expected-goals lookup table.

    Calling a trained model's .predict() once per simulated match is the
    dominant cost of a 10,000-tournament Monte Carlo run (tens of millions of
    single-row predictions). Since World Cup fixtures share the same context
    (neutral ground, knockout-tournament flag, current year) and only the Elo
    gap varies, we instead batch-predict expected goals for every integer Elo
    gap once up front, then look values up by array index at simulation time.
    This keeps the *exact* model output (same trained estimator, just cached)
    while making the inner simulation loop hundreds of times faster.
    """

    def __init__(self, model, is_world_cup: bool = True, as_of_year: float = 2026.0,
                 elo_gap_range: int = 900, neutral: bool = True):
        self.is_world_cup = is_world_cup
        self.as_of_year = as_of_year
        self.neutral = neutral
        self.gap_range = elo_gap_range  # covers gaps from -range to +range

        days = as_of_year - 2000.0
        gaps = np.arange(-elo_gap_range, elo_gap_range + 1, dtype=float)
        home_flag = 0.0 if neutral else 1.0
        X = np.column_stack(
            [
                gaps / 100.0,
                np.full_like(gaps, home_flag),
                np.full_like(gaps, 1.0 if is_world_cup else 0.0),
                np.full_like(gaps, days),
            ]
        )
        self._xg_by_gap = model.expected_goals(X)  # index i -> gap (i - elo_gap_range)

    def expected_goals(self, elo_a: float, elo_b: float) -> tuple[float, float]:
        gap = elo_a - elo_b
        idx_a = int(round(min(max(gap, -self.gap_range), self.gap_range))) + self.gap_range
        idx_b = int(round(min(max(-gap, -self.gap_range), self.gap_range))) + self.gap_range
        return float(self._xg_by_gap[idx_a]), float(self._xg_by_gap[idx_b])


def train_poisson(table: pd.DataFrame) -> PoissonGoalModel:
    X, y = to_xy(table)
    model = PoissonRegressor(alpha=1e-6, max_iter=3000)
    model.fit(X, y)
    return PoissonGoalModel(model)


def train_xgboost(table: pd.DataFrame, random_state: int = 42) -> tuple[XGBoostGoalModel, dict]:
    """Train the XGBoost goal model and return it plus a small holdout-set
    evaluation report (for the "model accuracy" panel in the app)."""
    X, y = to_xy(table)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )

    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = float(np.mean(np.abs(pred_test - y_test)))
    # Poisson deviance is the natural "loss" for count-data goal predictions.
    eps = 1e-6
    deviance = float(
        2 * np.mean(
            np.where(
                y_test > 0,
                y_test * np.log((y_test + eps) / (pred_test + eps)) - (y_test - pred_test),
                pred_test,
            )
        )
    )
    report = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "mae": mae,
        "mean_poisson_deviance": deviance,
    }
    return XGBoostGoalModel(model), report


def evaluate_outcome_accuracy(table: pd.DataFrame, model, test_size: float = 0.15, random_state: int = 42) -> dict:
    """How often does the model's favourite actually win, on held-out
    matches? This regroups the per-team-perspective rows back into matches
    (every match appears twice — once per perspective — so we de-duplicate by
    date+team+opponent pairs) and compares predicted vs actual W/D/L."""
    X, y = to_xy(table)
    idx = np.arange(len(table))
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    test_rows = table.iloc[idx_test].copy()
    test_rows["pred_goals"] = model.expected_goals(X[idx_test])

    # Re-pair home/away rows that share date+team-pair (each original match
    # produced exactly two rows: team->opponent and opponent->team).
    pairs = {}
    correct = 0
    total = 0
    for r in test_rows.itertuples(index=False):
        key = tuple(sorted([r.team, r.opponent])) + (str(r.date),)
        pairs.setdefault(key, {})[r.team] = (r.pred_goals, r.goals)

    for key, sides in pairs.items():
        if len(sides) != 2:
            continue
        (team_a, (pred_a, actual_a)), (team_b, (pred_b, actual_b)) = list(sides.items())
        pred_outcome = "draw" if abs(pred_a - pred_b) < 0.15 else ("a" if pred_a > pred_b else "b")
        actual_outcome = "draw" if actual_a == actual_b else ("a" if actual_a > actual_b else "b")
        total += 1
        if pred_outcome == actual_outcome:
            correct += 1

    return {"n_matches": total, "outcome_accuracy": correct / total if total else 0.0}
