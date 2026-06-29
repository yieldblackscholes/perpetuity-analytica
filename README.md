# World Cup 2026 Predictor

A Streamlit app that predicts the 2026 FIFA World Cup using **Elo ratings +
XGBoost (gradient-boosted trees) + Monte Carlo tournament simulation**,
trained on 49,000+ historical international football matches (1872–2026).
A Poisson regression model is included alongside XGBoost for comparison.

Because the underlying dataset is live and the 2026 tournament is already
under way, every group-stage result played so far is folded into the Elo
ratings — predictions for the Round of 32 onward reflect actual current form,
not just pre-tournament rankings.

## Features

- **🏆 Tournament Odds** — full Monte Carlo simulation (configurable 1k–20k
  runs) showing every team's probability of winning the title, reaching the
  final, semis, quarters, and Round of 16, plus a group-by-group breakdown.
- **🔍 Team Explorer** — search any of the 48 qualified teams: current Elo,
  all-time World Cup record (wins/draws/losses/goals), 2026 results so far,
  full historical World Cup match list, and head-to-head lookup against any
  other team.
- **👤 Player Stats** — all-time World Cup top-scorer leaderboard, plus a
  per-team breakdown of every scorer in that nation's World Cup history.
- **⚔️ Match Predictor** — pick any two teams for a simulated neutral-ground
  result: win/draw/loss probabilities, expected goals, most likely scoreline,
  and their historical head-to-head record.
- **ℹ️ How it works** — a walkthrough of the Elo engine, both ML models
  (with live feature importances/coefficients), and the simulation mechanics.

## Data

`martj42/international_results` (GitHub) — `results.csv` (49k+ matches),
`goalscorers.csv` (player-level goal records), `shootouts.csv` (penalty
shootout winners). Already included in `data/`; delete and re-download from
`https://raw.githubusercontent.com/martj42/international_results/master/`
to refresh.

## The model, in brief

1. **Elo ratings** (`src/elo_engine.py`) — every historical match is replayed
   chronologically; ratings update based on match importance (World Cup >
   continental > qualifier > Nations League > friendly), margin of victory,
   and home advantage. This is the "current team strength" feature.
2. **Goal models** (`src/models.py`) — both trained on (Elo difference, home
   flag, World-Cup flag, recency) → goals scored:
   - **XGBoost** (primary): gradient-boosted trees with a Poisson objective.
     Captures non-linear effects a straight-line model can't.
   - **Poisson regression** (comparison): a linear GLM, fully interpretable.
   A `FastEloLookup` precomputes expected goals for every integer Elo gap up
   front, so the simulation loop doesn't pay model-inference cost per match —
   this is what makes 10,000 full-tournament simulations run in a few seconds.
3. **Monte Carlo simulation** (`src/simulator.py`) — goals are drawn from
   Poisson distributions parameterized by each model's expected-goals output.
   The full 48-team, 12-group bracket (round-robin groups → 8 best third-place
   teams → fixed Round-of-32 bracket mirroring FIFA's real draw constraints →
   knockout rounds to a champion) is replayed thousands of times to estimate
   each team's odds at every stage.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

First load trains both models on ~49k matches (a few seconds) and caches them
for the rest of the session via `@st.cache_resource`.

## Project layout

```
app.py                  Streamlit UI — all five tabs
src/
  teams2026.py           48 qualified teams, groups, bracket template
  elo_engine.py          CSV loading + chronological Elo replay
  features.py            Supervised training-table construction
  models.py               XGBoost + Poisson models, FastEloLookup cache
  simulator.py            Monte Carlo single-match & full-tournament sim
  data_pipeline.py        Orchestration + team/player query helpers
data/
  results.csv, goalscorers.csv, shootouts.csv
```

Predictions are a probabilistic analysis and entertainment tool — not
betting advice. Football's low scoring and frequent upsets mean even a
well-tuned model lands around 50–60% outcome accuracy; that's the nature of
the sport, not a modeling flaw.
