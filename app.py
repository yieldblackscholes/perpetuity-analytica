"""World Cup 2026 Predictor — Streamlit app.

Run with:  streamlit run app.py
"""
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import (
    head_to_head,
    run_pipeline,
    team_all_matches,
    team_goalscorers,
    team_world_cup_matches,
    team_world_cup_record,
    top_scorers,
)
from features import make_match_features
from models import FastEloLookup
from simulator import TournamentSimulator, odds_table
from teams2026 import GROUPS, TEAM_BY_NAME, WC2026_TEAMS, find_team

st.set_page_config(
    page_title="Perpetuity Analytica",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached pipeline — trains everything once per server process
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training models on 49,000+ historical matches...")
def get_pipeline():
    return run_pipeline()


@st.cache_data(show_spinner=False)
def get_simulation(_pipeline_token: str, use_xgboost: bool, num_sims: int, seed: int):
    pipeline = get_pipeline()
    lookup = pipeline.xgb_lookup if use_xgboost else pipeline.poisson_lookup
    sim = TournamentSimulator(pipeline.wc2026_ratings, lookup.expected_goals, seed=seed)
    result = sim.run(num_sims)
    return odds_table(result, num_sims)


pipeline = get_pipeline()

PRIMARY = "#00C2A8"
ACCENT = "#FFC857"


# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------

st.sidebar.title("Perpetuity Analytica")
st.sidebar.caption("AI-Powered FIFA World Cup Prediction Engine")
st.sidebar.markdown("---")
st.sidebar.caption("Group stage complete · Round of 32 underway")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Prediction model",
    options=["AI Model (Recommended)", "Classic Statistical Model"],
    index=0,
    help="AI Model (Recommended): XGBoost is a gradient-boosted tree ensemble — generally the stronger model on this data. "
    "Classic Statistical Model: Poisson regression is the simpler, fully interpretable baseline kept for comparison.",
)
use_xgboost = model_choice.startswith("AI Model")

num_sims = st.sidebar.select_slider(
    "Monte Carlo simulations",
    options=[1000, 2000, 5000, 10000, 20000],
    value=10000,
    help="More simulations = smoother, more stable odds, at the cost of a slightly longer run.",
)

st.sidebar.markdown("---")
with st.sidebar.expander("Model accuracy (held-out matches)"):
    acc = pipeline.xgb_outcome_acc if use_xgboost else pipeline.poisson_outcome_acc
    report = pipeline.xgb_report
    st.metric("Outcome accuracy (W/D/L)", f"{acc['outcome_accuracy']*100:.1f}%")
    st.caption(f"Evaluated on {acc['n_matches']:,} held-out historical matches.")
    if use_xgboost:
        st.caption(f"Mean abs. goal error: {report['mae']:.2f} goals/team/match")
    st.caption(
        "Football is a low-scoring, high-variance sport — even strong models "
        "land around 50-60% outcome accuracy because draws and upsets are common."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: martj42/international_results — 49,000+ international matches, 1872–2026. "
    "Models are trained for entertainment & analysis, not betting advice."
)
st.sidebar.caption("💡 🤖 AI Model = XGBoost (advanced pattern recognition) | 📊 Classic Model = Poisson regression (transparent statistics) | 📖 See Glossary tab for simple explanations")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_dashboard, tab_teams, tab_players, tab_predict, tab_about, tab_glossary = st.tabs(
    ["🏆 Tournament Odds", "🔍 Team Explorer", "👤 Player Stats", "⚔️ Match Predictor", "ℹ️ How it works", "📖 Glossary & How to Use"]
)

# ============================== TAB 1: DASHBOARD ==============================
with tab_dashboard:
    st.header("Who will win the 2026 FIFA World Cup?")
    st.caption(
        f"🎲 Monte Carlo simulation of {num_sims:,} full tournaments, using {model_choice} "
        "for expected goals and live ⚡ Elo ratings (which already reflect every group-stage result played so far)."
    )
    st.caption("💡 🎲 Monte Carlo = running simulations like rolling dice | ⚡ Elo = team strength score | 📖 See Glossary tab for simple explanations")

    # Welcome banner
    st.info("""
    🏆 **Welcome to Perpetuity Analytica!**
    We use AI trained on 49,000+ real football matches since 1872 to simulate the 2026 World Cup thousands of times
    and estimate each team's chances of winning. Think of it like running the entire tournament over and over
    to see who comes out on top most often.
    """)

    sim_rows = get_simulation("v1", use_xgboost, num_sims, seed=42)
    df_odds = pd.DataFrame(sim_rows)

    col1, col2, col3 = st.columns(3)
    leader = df_odds.iloc[0]
    runner = df_odds.iloc[1]
    third = df_odds.iloc[2]
    col1.metric(f"{leader['Flag']} {leader['Team']} — Favourite", f"{leader['Champion %']:.1f}%")
    col2.metric(f"{runner['Flag']} {runner['Team']} — 2nd", f"{runner['Champion %']:.1f}%")
    col3.metric(f"{third['Flag']} {third['Team']} — 3rd", f"{third['Champion %']:.1f}%")

    st.subheader("Title odds — top 15")
    top15 = df_odds.head(15).copy()
    top15["Label"] = top15["Flag"] + " " + top15["Team"]
    fig = px.bar(
        top15.sort_values("Champion %"),
        x="Champion %",
        y="Label",
        orientation="h",
        text=top15.sort_values("Champion %")["Champion %"].map(lambda v: f"{v:.1f}%"),
        color="Champion %",
        color_continuous_scale=["#1a3a3a", PRIMARY],
    )
    fig.update_layout(
        height=520,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="Probability of winning the title (%)",
        yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full odds table — every stage")
    st.dataframe(
        df_odds.style.format(
            {c: "{:.1f}%" for c in df_odds.columns if c.endswith("%")}
        ),
        use_container_width=True,
        height=450,
        column_config={
            "Flag": st.column_config.TextColumn(width="small"),
        },
        hide_index=True,
    )

    st.subheader("Group-by-group breakdown")
    group_sel = st.selectbox("Pick a group", GROUPS, format_func=lambda g: f"Group {g}")
    group_teams = [t for t in WC2026_TEAMS if t.group == group_sel]
    group_df = df_odds[df_odds["Team"].isin([t.name for t in group_teams])].sort_values("Win Group %", ascending=False)
    st.dataframe(
        group_df[["Flag", "Team", "Win Group %", "Round of 16 %", "Quarterfinal %", "Champion %"]].style.format(
            {c: "{:.1f}%" for c in group_df.columns if c.endswith("%")}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================== TAB 2: TEAM EXPLORER ==============================
with tab_teams:
    st.header("Team Explorer")
    team_names = sorted([t.name for t in WC2026_TEAMS])
    selected_name = st.selectbox("Search for a team", team_names, index=team_names.index("Brazil") if "Brazil" in team_names else 0)
    team = TEAM_BY_NAME[selected_name]

    elo = pipeline.wc2026_ratings.get(team.name, 1000.0)
    record = team_world_cup_record(pipeline, team.csv_name)

    st.subheader(f"{team.flag} {team.name}  ·  Group {team.group}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Elo", f"{elo:.0f}", help="Team strength score - like a video game ranking that goes up after wins and down after losses, weighted by match importance")
    c2.metric("WC matches played", record["played"])
    c3.metric("WC wins", record["wins"])
    c4.metric("WC draws", record["draws"])
    c5.metric("WC losses", record["losses"])

    c6, c7, c8 = st.columns(3)
    c6.metric("Goals scored (WC)", record["goals_for"])
    c7.metric("Goals conceded (WC)", record["goals_against"])
    win_pct = 100 * record["wins"] / record["played"] if record["played"] else 0
    c8.metric("WC win rate", f"{win_pct:.1f}%")

    st.caption(f"World Cup appearances: {', '.join(str(y) for y in record['editions'])}" if record["editions"] else "No World Cup appearances in the dataset yet.")

    st.markdown("---")
    st.subheader("2026 World Cup matches so far")
    wc26 = team_world_cup_matches(pipeline, team.csv_name)
    wc26_2026 = wc26[wc26["date"].dt.year == 2026].copy()
    if len(wc26_2026):
        wc26_2026["Result"] = wc26_2026.apply(
            lambda r: f"{r['home_team']} {int(r['home_score']) if pd.notna(r['home_score']) else '–'} - "
            f"{int(r['away_score']) if pd.notna(r['away_score']) else '–'} {r['away_team']}",
            axis=1,
        )
        st.dataframe(
            wc26_2026[["date", "Result", "city", "country"]].rename(columns={"date": "Date", "city": "City", "country": "Host"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No 2026 World Cup matches found for this team.")

    st.markdown("---")
    st.subheader("All-time World Cup match history")
    wc_all = team_world_cup_matches(pipeline, team.csv_name).copy()
    wc_all_played = wc_all.dropna(subset=["home_score", "away_score"])
    if len(wc_all_played):
        wc_all_played = wc_all_played.assign(
            Result=wc_all_played.apply(
                lambda r: f"{r['home_team']} {int(r['home_score'])} - {int(r['away_score'])} {r['away_team']}",
                axis=1,
            )
        )
        st.dataframe(
            wc_all_played[["date", "Result", "city", "country"]]
            .rename(columns={"date": "Date", "city": "City", "country": "Host"})
            .sort_values("Date", ascending=False),
            use_container_width=True,
            height=350,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Head-to-head lookup")
    opponent_name = st.selectbox(
        "Compare against",
        [n for n in team_names if n != selected_name],
        key="h2h_opponent",
    )
    opp_team = TEAM_BY_NAME[opponent_name]
    h2h = head_to_head(pipeline, team.csv_name, opp_team.csv_name).dropna(subset=["home_score", "away_score"])
    if len(h2h):
        wins_a = wins_b = draws = 0
        for r in h2h.itertuples(index=False):
            home_is_a = r.home_team == team.csv_name
            score_a = r.home_score if home_is_a else r.away_score
            score_b = r.away_score if home_is_a else r.home_score
            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
            else:
                draws += 1
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(f"{team.flag} {team.name} wins", wins_a)
        hc2.metric("Draws", draws)
        hc3.metric(f"{opp_team.flag} {opp_team.name} wins", wins_b)

        h2h_display = h2h.assign(
            Result=h2h.apply(
                lambda r: f"{r['home_team']} {int(r['home_score'])} - {int(r['away_score'])} {r['away_team']}",
                axis=1,
            )
        )
        st.dataframe(
            h2h_display[["date", "Result", "tournament", "country"]]
            .rename(columns={"date": "Date", "tournament": "Competition", "country": "Host"})
            .sort_values("Date", ascending=False),
            use_container_width=True,
            height=280,
            hide_index=True,
        )
    else:
        st.info(f"{team.name} and {opp_team.name} have never played each other in the dataset.")

# ============================== TAB 3: PLAYER STATS ==============================
with tab_players:
    st.header("Player Stats")
    st.caption("Goal-scoring records from FIFA World Cup matches, 1930–2026 (own goals excluded from leaderboards).")

    sub_overall, sub_team = st.tabs(["🌍 All-time leaderboard", "🔎 Team breakdown"])

    with sub_overall:
        n_top = st.slider("Show top N scorers", 5, 50, 20)
        leaderboard = top_scorers(pipeline, world_cup_only=True, top_n=n_top)
        leaderboard.index = leaderboard.index + 1
        fig = px.bar(
            leaderboard.head(15).sort_values("goals"),
            x="goals",
            y="scorer",
            orientation="h",
            color="goals",
            color_continuous_scale=["#1a3a3a", ACCENT],
            hover_data=["team"],
        )
        fig.update_layout(
            height=480,
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="World Cup goals",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            leaderboard.rename(columns={"scorer": "Player", "team": "Team", "goals": "WC Goals"}),
            use_container_width=True,
        )

    with sub_team:
        team_for_players = st.selectbox("Pick a team", sorted([t.name for t in WC2026_TEAMS]), key="player_team_select")
        team_obj = TEAM_BY_NAME[team_for_players]
        goals_df = team_goalscorers(pipeline, team_obj.csv_name, world_cup_only=True)
        goals_df = goals_df[goals_df["own_goal"] == False]  # noqa: E712
        if len(goals_df):
            team_leaderboard = (
                goals_df.groupby("scorer").size().reset_index(name="goals").sort_values("goals", ascending=False)
            )
            team_leaderboard.index = range(1, len(team_leaderboard) + 1)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.metric("Total WC goals", int(team_leaderboard["goals"].sum()))
                st.metric("Distinct scorers", len(team_leaderboard))
            with c2:
                top_scorer_row = team_leaderboard.iloc[0]
                st.metric(f"{team_obj.flag} Top scorer", f"{top_scorer_row['scorer']} ({top_scorer_row['goals']})")
            st.dataframe(
                team_leaderboard.rename(columns={"scorer": "Player", "goals": "WC Goals"}),
                use_container_width=True,
                height=380,
            )
            penalties = int(goals_df["penalty"].sum())
            st.caption(f"Includes {penalties} penalty goal(s) for {team_obj.name} across all World Cups.")
        else:
            st.info(f"No World Cup goal records found for {team_obj.name} in this dataset.")

# ============================== TAB 4: MATCH PREDICTOR ==============================
with tab_predict:
    st.header("Head-to-Head Match Predictor")
    st.caption("Pick any two of the 48 qualified teams to see a simulated outcome on neutral ground.")

    col_a, col_b = st.columns(2)
    team_names_sorted = sorted([t.name for t in WC2026_TEAMS])
    with col_a:
        name_a = st.selectbox("Team A", team_names_sorted, index=team_names_sorted.index("Brazil") if "Brazil" in team_names_sorted else 0)
    with col_b:
        default_b = "Argentina" if "Argentina" in team_names_sorted else team_names_sorted[1]
        name_b = st.selectbox("Team B", team_names_sorted, index=team_names_sorted.index(default_b))

    if name_a == name_b:
        st.warning("Pick two different teams.")
    else:
        team_a, team_b = TEAM_BY_NAME[name_a], TEAM_BY_NAME[name_b]
        elo_a = pipeline.wc2026_ratings.get(team_a.name, 1000.0)
        elo_b = pipeline.wc2026_ratings.get(team_b.name, 1000.0)

        lookup = pipeline.xgb_lookup if use_xgboost else pipeline.poisson_lookup
        sim = TournamentSimulator(pipeline.wc2026_ratings, lookup.expected_goals, seed=7)
        probs = sim.match_probabilities(elo_a, elo_b, trials=30_000)

        st.markdown(f"### {team_a.flag} {team_a.name} (Elo {elo_a:.0f})  vs  {team_b.flag} {team_b.name} (Elo {elo_b:.0f})")

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric(f"{team_a.name} win", f"{probs['p_win_a']*100:.1f}%")
        pc2.metric("Draw", f"{probs['p_draw']*100:.1f}%")
        pc3.metric(f"{team_b.name} win", f"{probs['p_win_b']*100:.1f}%")

        fig = go.Figure(
            go.Bar(
                x=[probs["p_win_a"] * 100, probs["p_draw"] * 100, probs["p_win_b"] * 100],
                y=[team_a.name, "Draw", team_b.name],
                orientation="h",
                marker_color=[PRIMARY, "#888", ACCENT],
                text=[f"{probs['p_win_a']*100:.1f}%", f"{probs['p_draw']*100:.1f}%", f"{probs['p_win_b']*100:.1f}%"],
                textposition="outside",
            )
        )
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"**Expected goals:** {team_a.name} {probs['xg_a']} – {probs['xg_b']} {team_b.name}  ·  "
            f"**Most likely scoreline:** {probs['most_likely_score']}"
        )

        h2h = head_to_head(pipeline, team_a.csv_name, team_b.csv_name).dropna(subset=["home_score", "away_score"])
        if len(h2h):
            st.markdown(f"**All-time head-to-head:** {len(h2h)} matches played.")
            h2h_display = h2h.assign(
                Result=h2h.apply(
                    lambda r: f"{r['home_team']} {int(r['home_score'])} - {int(r['away_score'])} {r['away_team']}",
                    axis=1,
                )
            )
            st.dataframe(
                h2h_display[["date", "Result", "tournament"]].rename(columns={"date": "Date", "tournament": "Competition"}).sort_values("Date", ascending=False),
                use_container_width=True,
                height=220,
                hide_index=True,
            )
        else:
            st.caption("These two teams have never met before in the dataset.")

# ============================== TAB 5: HOW IT WORKS ==============================
with tab_about:
    st.header("How this model works")
    st.markdown(
        """
This app predicts the 2026 FIFA World Cup using a pipeline of three layers, trained on
**49,000+ international football matches from 1872 to 2026** (martj42/international_results).

**1. Elo ratings — the "current strength" feature**
Every historical match is replayed in chronological order. After each result, both teams'
Elo ratings are nudged — winners gain, losers lose — by an amount that depends on:
- **Match importance**: World Cup (×60) > continental championships (×50) > qualifiers (×40) > Nations League (×35) > friendlies (×20)
- **Margin of victory**: bigger wins move the rating further (capped at 1.75×)
- **Home advantage**: +75 rating points for the home side, skipped on neutral ground (almost all World Cup matches)

Because the dataset is current through the 2026 tournament, every group-stage result already
played this summer is folded into these ratings.

**2. Goal-prediction models — turning a strength gap into expected goals**
Two supervised models are trained on the same historical data, each learning to map
*(Elo difference, home advantage, World Cup flag, recency)* to *goals scored*:
"""
    )
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**XGBoost (primary)**")
        st.markdown(
            "A gradient-boosted ensemble of decision trees trained with a Poisson objective. "
            "Captures non-linear effects and feature interactions a straight-line model can't."
        )
        st.json({k: round(v, 3) for k, v in pipeline.xgb_model.feature_importance().items()})
    with mc2:
        st.markdown("**Poisson regression (comparison)**")
        st.markdown(
            "A generalized linear model: log(expected goals) is a straight-line function of the features. "
            "Fast and fully interpretable — every coefficient has a direct 'goals per unit' meaning."
        )
        st.json({k: round(v, 4) for k, v in pipeline.poisson_model.coefficients().items()})

    st.markdown(
        """
**3. Monte Carlo tournament simulation**
For a given fixture, each side's expected goals feed independent Poisson distributions —
goals are naturally non-negative integers with the kind of variance real match scores show.
A single match is decided by sampling one score from each side's distribution.

The full 48-team, 12-group bracket is simulated thousands of times end-to-end: group stage
round-robins decide who advances (with the 8 best third-place teams completing the Round of
32), then a fixed knockout bracket — built to match FIFA's real draw rules, so group winners
can't meet early and a group's top two can only meet again in the final — is played out all
the way to a champion. Counting how often each team reaches each stage across every simulated
tournament gives the title odds shown on the Tournament Odds tab.

**A note on accuracy:** football is intentionally low-scoring and upset-prone — that's most of
its appeal. Even a well-tuned model lands around 50–60% accuracy on win/draw/loss outcomes,
because draws and giant-killing upsets are a real, frequent part of the sport, not model error.
These predictions are a probabilistic analysis tool, not betting advice.
"""
    )


# ============================== TAB 6: GLOSSARY & HOW TO USE ==============================
with tab_glossary:
    st.header("📖 Glossary & How to Use This App")
    st.markdown("""
    Welcome to Perpetuity Analytica! This guide explains everything you see in plain, everyday language — no data science degree required.
    """)

    st.subheader("📊 Key Terms Explained Simply")

    with st.expander("🏆 Elo Rating - Team Strength Score", expanded=False):
        st.markdown("""
        Think of Elo like a **video game ranking** or **chess rating** for national football teams:

        - Every team starts with a base score (around 1000 points)
        - When teams play, the winner gains points and the loser loses points
        - **Big matches** (World Cup finals) exchange **more points** than friendly matches
        - **Big wins** earn more points than close wins
        - Playing **at home** gives a small bonus (like home-field advantage in other sports)

        Because we update these ratings with every match ever played (since 1872!), today's Elo ratings show **current form** - not just historical reputation.
        """)

    with st.expander("🤖 XGBoost - The AI Model", expanded=False):
        st.markdown("""
        This is our **"smart pattern-finding" AI** that learns from history:

        - Looks at **thousands of past matches** to find hidden patterns
        - Not just a simple formula - it notices complex combinations (like "Team A tends to score more when they have high Elo AND are playing in a World Cup qualifier AND it's been less than 30 days since their last match")
        - Think of it like a **super-experienced coach** who's seen every match ever played and can spot subtle trends humans might miss
        - We use it as our **primary prediction engine** because it usually catches more nuances than simpler methods
        """)

    with st.expander("📊 Poisson Regression - The Classic Model", expanded=False):
        st.markdown("""
        This is the **straightforward, transparent method** - like a trusted rule of thumb:

        - Uses a clear mathematical formula: if you know the Elo difference, home advantage, etc., you can plug numbers in and get a prediction
        - Every factor has a clear, understandable weight (e.g., "100 Elo points difference = 0.3 more goals expected")
        - Great for **double-checking** the AI model - if both agree, we can be more confident
        - Think of it as the **"show your work"** version of the prediction
        """)

    with st.expander("🎲 Monte Carlo Simulation - Rolling the Dice 10,000 Times", expanded=False):
        st.markdown("""
        Imagine **re-running the entire World Cup tournament thousands of times** to see what happens most often:

        - For each simulated tournament, we "play" every match using dice rolls influenced by team strengths
        - Some tournaments Brazil wins easily, others they get upset early - that's football!
        - After 10,000+ simulations, we count: "In how many of these did Argentina reach the final?"
        - The **percentage** shown is exactly that: "Argentina reached the final in 3,200 out of 10,000 simulations = 32%"
        - More simulations = smoother, more reliable percentages (like flipping a coin 10,000 times vs 10 times)
        """)

    with st.expander("🎯 Expected Goals (xG) - The Predicted Score Average", expanded=False):
        st.markdown("""
        This is the **mathematical expectation** - not what will happen, but what we'd expect on average:

        - If two teams have an xG of 2.1 - 0.8, it means we expect Team A to score about 2 goals and Team B about 1 goal *on average*
        - But remember: football scores are always **whole numbers** (0, 1, 2, 3...) so actual results will vary around this average
        - Think of it like weather forecasting: "We expect 0.2 inches of rain" doesn't mean you'll get exactly 0.2 inches - it means sometimes you'll get 0, sometimes 0.5, etc.
        - Higher xG means a team is more likely to score, but doesn't guarantee any specific number
        """)

    with st.expander("📈 Tournament Stage Probabilities - What the % Means", expanded=False):
        st.markdown("""
        Simple percentage breakdown - think "out of 100 tournaments":

        - **Champion %**: "Out of 100 simulated World Cups, this team won the trophy X times"
        - **Final %**: "Made it to the final match X times out of 100"
        - **Semifinal %**: "Made it to the semi-finals X times out of 100"
        - **Round of 16 %**: "Made it past the group stage X times out of 100"
        - Higher percentages = stronger tournament performance across our simulations
        """)

    with st.expander("🎯 Outcome Accuracy - How Well We Predicted Past Matches", expanded=False):
        st.markdown('''
        Our "report card" on predicting games we'"'"'ve never seen:

        - We train our models on **older matches**, then test them on **more recent matches** they haven'"'"'t seen
        - **Outcome accuracy** = "What percentage of the time did we correctly predict win/draw/loss?"
        - In football, even the best models typically score **50-60%** - not because they'"'"'re bad, but because:
          - Low scoring (1-0 wins are common, hard to predict exact winner)
          - Frequent upsets (underdogs win more often than in other sports)
          - Draws happen surprisingly often
        - Think of it like weather forecasting: being right 60% of the time is actually pretty good for something as chaotic as football!
        ''')

    st.divider()

    st.subheader("🚀 How to Get the Most Out of This App")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🏆 **Tournament Odds Tab**
        - See each team's chances to win it all, reach the final, etc.
        - Use the slider to adjust how many tournament simulations run (more = smoother but slower)
        - Switch between AI model and classic model to compare predictions
        - Check group-by-group tabs to see how teams stack up in their specific paths
        """)

    with col2:
        st.markdown("""
        ### 🔍 **Team Explorer Tab**
        - Search for any team to see their current Elo rating and World Cup history
        - Check their 2026 match results so far
        - Compare head-to-head records against any other team
        - See their all-time World Cup scoring history
        """)

    st.markdown("""
    ### ⚔️ **Match Predictor Tab**
    - Pick any two teams to see how they would match up on neutral ground
    - Get win/draw/loss probabilities and expected scores
    - View their historical head-to-head record

    ### 👤 **Player Stats Tab**
    - Explore all-time World Cup goal scorers
    - Dive deep into any team's scoring history

    ### ℹ️ **How it Works Tab**
    - Technical deep-dive into the mathematics and models (for the curious)
    """)

    st.info("""
    💡 **Quick Tip**: All predictions update automatically when you change settings in the sidebar (model type, number of simulations).
    The app remembers your preferences during your browsing session!
    """)
