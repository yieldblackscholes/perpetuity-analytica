"""Monte Carlo tournament simulation for the 48-team, 12-group 2026 World Cup
format.

Goals are drawn from independent Poisson distributions whose means come from
whichever trained goal model (XGBoost by default, Poisson optionally) is
plugged in via `expected_goals_fn`. The full bracket — 12 groups -> Round of
32 -> Round of 16 -> quarters -> semis -> final — is replayed thousands of
times; counting how often each team reaches each stage gives the title odds.

The Round-of-32 bracket uses a fixed template that mirrors the real FIFA
draw structure (group winners can't meet each other early; a group's winner
and runner-up are kept in opposite halves) rather than a fresh random draw
each simulation, so a team's actual finishing position shapes its path
exactly as it would in the real tournament.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from teams2026 import GROUPS, R32_BRACKET, TSLOT_WINNER_GROUP, WC2026_TEAMS, WCTeam

DEFAULT_NUM_SIMULATIONS = 10_000


def poisson_sample(lam: float, rng: random.Random) -> int:
    """Knuth's algorithm for sampling a Poisson-distributed integer."""
    if lam <= 0:
        return 0
    target = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= target:
            break
    return k - 1


@dataclass
class _Standing:
    team: WCTeam
    points: int = 0
    gf: int = 0
    ga: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


class TournamentSimulator:
    """Wraps a trained goal model + Elo ratings into a repeatable Monte Carlo
    simulator for the WC2026 bracket."""

    def __init__(
        self,
        ratings: dict[str, float],
        expected_goals_fn,
        seed: int | None = None,
    ):
        """
        ratings           : {team display name -> Elo rating}
        expected_goals_fn : callable(elo_a, elo_b) -> (xg_a, xg_b), already
                             configured for "neutral ground, World Cup match"
        """
        self.ratings = ratings
        self.expected_goals_fn = expected_goals_fn
        self.rng = random.Random(seed)

    def get_elo(self, team_name: str) -> float:
        return self.ratings.get(team_name, 1000.0)

    # ---- single match ----

    def simulate_score(self, elo_a: float, elo_b: float) -> tuple[int, int]:
        xg_a, xg_b = self.expected_goals_fn(elo_a, elo_b)
        return poisson_sample(xg_a, self.rng), poisson_sample(xg_b, self.rng)

    def match_probabilities(self, elo_a: float, elo_b: float, trials: int = 20_000) -> dict:
        """Monte Carlo win/draw/loss + expected goals + most likely scoreline
        for a single hypothetical fixture (used by the head-to-head predictor
        tab, not the full tournament sim)."""
        xg_a, xg_b = self.expected_goals_fn(elo_a, elo_b)
        win_a = draw = win_b = 0
        score_freq: dict[str, dict[str, int]] = {"a": {}, "draw": {}, "b": {}}

        for _ in range(trials):
            ga = poisson_sample(xg_a, self.rng)
            gb = poisson_sample(xg_b, self.rng)
            key = f"{ga}-{gb}"
            if ga > gb:
                win_a += 1
                bucket = score_freq["a"]
            elif ga < gb:
                win_b += 1
                bucket = score_freq["b"]
            else:
                draw += 1
                bucket = score_freq["draw"]
            bucket[key] = bucket.get(key, 0) + 1

        outcome = max(("a", win_a), ("draw", draw), ("b", win_b), key=lambda kv: kv[1])[0]
        bucket = score_freq[outcome]
        most_likely = max(bucket.items(), key=lambda kv: kv[1])[0] if bucket else "1-1"

        return {
            "p_win_a": win_a / trials,
            "p_draw": draw / trials,
            "p_win_b": win_b / trials,
            "xg_a": round(xg_a, 2),
            "xg_b": round(xg_b, 2),
            "most_likely_score": most_likely,
        }

    # ---- group stage ----

    def _simulate_group(self, group_teams: list[WCTeam]) -> list[_Standing]:
        standings = {t.name: _Standing(t) for t in group_teams}
        n = len(group_teams)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = group_teams[i], group_teams[j]
                ga, gb = self.simulate_score(self.get_elo(a.name), self.get_elo(b.name))
                sa, sb = standings[a.name], standings[b.name]
                sa.gf += ga
                sa.ga += gb
                sb.gf += gb
                sb.ga += ga
                if ga > gb:
                    sa.points += 3
                elif gb > ga:
                    sb.points += 3
                else:
                    sa.points += 1
                    sb.points += 1
        ranked = sorted(standings.values(), key=lambda s: (s.points, s.gd, s.gf), reverse=True)
        return ranked

    # ---- knockout ----

    def _assign_thirds(self, thirds: list[WCTeam]) -> list[WCTeam]:
        remaining = list(range(len(thirds)))
        assigned: list[WCTeam] = []
        for i in range(len(thirds)):
            winner_group = TSLOT_WINNER_GROUP[i]
            pick = next((ti for ti in remaining if thirds[ti].group != winner_group), remaining[0])
            assigned.append(thirds[pick])
            remaining.remove(pick)
        return assigned

    def _knockout_match(self, a: WCTeam, b: WCTeam) -> WCTeam:
        elo_a, elo_b = self.get_elo(a.name), self.get_elo(b.name)
        ga, gb = self.simulate_score(elo_a, elo_b)
        if ga > gb:
            return a
        if gb > ga:
            return b
        # Tie -> penalty shootout, mildly weighted toward the stronger side.
        pen_edge = min(0.6, 0.5 + (elo_a - elo_b) / 2000)
        return a if self.rng.random() < pen_edge else b

    def _knockout_round(self, teams: list[WCTeam]) -> list[WCTeam]:
        return [self._knockout_match(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]

    # ---- full tournament ----

    def run(self, num_simulations: int = DEFAULT_NUM_SIMULATIONS, progress_callback=None) -> dict:
        names = [t.name for t in WC2026_TEAMS]
        stages = ("group_winner", "round_of_16", "quarter_finals", "semi_finals", "finals", "titles")
        result = {stage: {n: 0 for n in names} for stage in stages}
        groups_index = {g: [t for t in WC2026_TEAMS if t.group == g] for g in GROUPS}

        for sim_i in range(num_simulations):
            winners_by_group: dict[str, WCTeam] = {}
            runners_by_group: dict[str, WCTeam] = {}
            third_placers: list[_Standing] = []

            for g in GROUPS:
                standings = self._simulate_group(groups_index[g])
                winner, runner, third = standings[0], standings[1], standings[2]
                result["group_winner"][winner.team.name] += 1
                winners_by_group[g] = winner.team
                runners_by_group[g] = runner.team
                third_placers.append(third)

            third_placers.sort(key=lambda s: (s.points, s.gd, s.gf), reverse=True)
            best_thirds = third_placers[:8]
            thirds = self._assign_thirds([s.team for s in best_thirds])

            def resolve(slot: tuple) -> WCTeam:
                kind, key = slot
                if kind == "W":
                    return winners_by_group[key]
                if kind == "R":
                    return runners_by_group[key]
                return thirds[key]

            pool: list[WCTeam] = []
            for slot_a, slot_b in R32_BRACKET:
                pool.append(resolve(slot_a))
                pool.append(resolve(slot_b))

            r16 = self._knockout_round(pool)
            for t in r16:
                result["round_of_16"][t.name] += 1

            qf = self._knockout_round(r16)
            for t in qf:
                result["quarter_finals"][t.name] += 1

            sf = self._knockout_round(qf)
            for t in sf:
                result["semi_finals"][t.name] += 1

            finalists = self._knockout_round(sf)
            for t in finalists:
                result["finals"][t.name] += 1

            champion = self._knockout_round(finalists)[0]
            result["titles"][champion.name] += 1

            if progress_callback and (sim_i + 1) % max(1, num_simulations // 20) == 0:
                progress_callback((sim_i + 1) / num_simulations)

        return result


def odds_table(sim_result: dict, num_simulations: int) -> "list[dict]":
    """Flatten simulation counts into a sortable list of per-team odds."""
    rows = []
    for t in WC2026_TEAMS:
        rows.append(
            {
                "Team": t.name,
                "Flag": t.flag,
                "Group": t.group,
                "Champion %": 100 * sim_result["titles"][t.name] / num_simulations,
                "Final %": 100 * sim_result["finals"][t.name] / num_simulations,
                "Semifinal %": 100 * sim_result["semi_finals"][t.name] / num_simulations,
                "Quarterfinal %": 100 * sim_result["quarter_finals"][t.name] / num_simulations,
                "Round of 16 %": 100 * sim_result["round_of_16"][t.name] / num_simulations,
                "Win Group %": 100 * sim_result["group_winner"][t.name] / num_simulations,
            }
        )
    return sorted(rows, key=lambda r: -r["Champion %"])
