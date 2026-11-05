"""
NBA Private Simulator
---------------------
Personal Monte Carlo player projection tool.
Uses nba_api for live data. 10,000 simulations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from functools import lru_cache

# nba_api imports
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    leaguedashplayerstats,
    scoreboardv2,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

# ------------------------------------------------------------
# CONFIG & THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="NBA Private Simulator",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark, smooth, mobile-friendly UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(180deg, #0f0f13 0%, #1a1a24 100%);
    }
    
    .stApp {
        background-color: #0f0f13;
    }
    
    h1, h2, h3 {
        color: #f0f0f5 !important;
        font-weight: 700;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e1e2a, #252533);
        border-radius: 16px;
        padding: 1.2rem;
        border: 1px solid #2e2e3e;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d4aa;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0a0b0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .player-card {
        background: linear-gradient(145deg, #1e1e2a, #252533);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #2e2e3e;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d4aa, #00b894);
        color: #0f0f13;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: #14141c;
    }
    
    div[data-testid="stMetricValue"] {
        color: #00d4aa;
    }
    
    .stSelectbox label, .stSlider label {
        color: #c0c0d0 !important;
    }
    
    /* Mobile tweaks */
    @media (max-width: 768px) {
        .metric-value { font-size: 1.4rem; }
        h1 { font-size: 1.6rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
CURRENT_SEASON = "2026-27"
PREV_SEASON = "2025-26"
N_SIMS = 10000

@st.cache_data(ttl=3600)
def get_all_players():
    return players.get_active_players()

@st.cache_data(ttl=3600)
def get_all_teams():
    return teams.get_teams()

def get_player_headshot(player_id: int) -> str:
    return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

def get_team_logo(team_id: int) -> str:
    return f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_player_gamelog(player_id: int, season: str = CURRENT_SEASON, last_n: int = 15):
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=str(player_id),
            season=season,
            season_type_all_star="Regular Season"
        )
        df = gl.get_data_frames()[0]
        if df.empty and season == CURRENT_SEASON:
            # fallback to previous season
            gl = playergamelog.PlayerGameLog(
                player_id=str(player_id),
                season=PREV_SEASON,
                season_type_all_star="Regular Season"
            )
            df = gl.get_data_frames()[0]
        if not df.empty:
            df = df.head(last_n).copy()
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE")
            for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
            df["PR"] = df["PTS"] + df["REB"]
            df["PA"] = df["PTS"] + df["AST"]
            df["TD"] = ((df["PTS"] >= 10) & (df["REB"] >= 10) & (df["AST"] >= 10)).astype(int)
        return df
    except Exception as e:
        st.warning(f"Could not fetch game log: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_info(player_id: int):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=str(player_id))
        return info.get_data_frames()[0]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_todays_games():
    try:
        sb = live_scoreboard.ScoreBoard()
        data = sb.get_dict()
        games = data.get("scoreboard", {}).get("games", [])
        return games
    except Exception as e:
        return []

def run_monte_carlo(recent_df: pd.DataFrame, n_sims: int = N_SIMS, weight_recent: bool = True):
    """Simple but effective Monte Carlo using empirical mean/std with recent weighting."""
    if recent_df.empty or len(recent_df) < 3:
        return None

    stats = ["PTS", "REB", "AST"]
    results = {}

    # Optional: weight more recent games higher
    n = len(recent_df)
    if weight_recent and n > 5:
        weights = np.linspace(0.5, 1.5, n)
        weights = weights / weights.sum()
    else:
        weights = np.ones(n) / n

    for stat in stats:
        values = recent_df[stat].values.astype(float)
        mu = np.average(values, weights=weights)
        # Weighted std
        var = np.average((values - mu)**2, weights=weights)
        sigma = np.sqrt(var) if var > 0 else max(mu * 0.25, 1.0)

        # Sample (truncated normal at 0)
        samples = np.random.normal(mu, sigma, n_sims)
        samples = np.clip(samples, 0, None)
        results[stat] = samples

    # Combos (independence assumption – common for prop work)
    results["PRA"] = results["PTS"] + results["REB"] + results["AST"]
    results["PR"] = results["PTS"] + results["REB"]
    results["PA"] = results["PTS"] + results["AST"]
    results["TD"] = (
        (results["PTS"] >= 10) &
        (results["REB"] >= 10) &
        (results["AST"] >= 10)
    ).astype(float)

    return results

def summary_stats(samples: np.ndarray):
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "p10": float(np.percentile(samples, 10)),
        "p25": float(np.percentile(samples, 25)),
        "p75": float(np.percentile(samples, 75)),
        "p90": float(np.percentile(samples, 90)),
        "over_10": float(np.mean(samples >= 10)),
        "over_15": float(np.mean(samples >= 15)),
        "over_20": float(np.mean(samples >= 20)),
        "over_25": float(np.mean(samples >= 25)),
        "over_30": float(np.mean(samples >= 30)),
    }

# ------------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------------
st.sidebar.title("🏀 NBA Private Sim")
st.sidebar.markdown("Personal Monte Carlo projections")
page = st.sidebar.radio(
    "Navigate",
    ["Today's Games", "Player Simulator", "Player Profiles", "Trends & Form", "Props Edge Checker"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Season: {CURRENT_SEASON} • Sims: {N_SIMS:,}")
st.sidebar.caption("Data via nba_api • Private use only")

# ------------------------------------------------------------
# PAGE: TODAY'S GAMES
# ------------------------------------------------------------
if page == "Today's Games":
    st.title("Tonight's Games")
    st.markdown("Live scoreboard + quick access to simulate key players")

    games = get_todays_games()

    if not games:
        st.info("No games scheduled or data temporarily unavailable.")
    else:
        cols = st.columns(2)
        for i, game in enumerate(games):
            with cols[i % 2]:
                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})
                status = game.get("gameStatusText", "")
                home_score = home.get("score", 0)
                away_score = away.get("score", 0)

                home_id = home.get("teamId")
                away_id = away.get("teamId")

                st.markdown(f"""
                <div class="player-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="text-align:center; flex:1;">
                            <img src="{get_team_logo(away_id)}" width="48" style="margin-bottom:4px;"><br>
                            <strong>{away.get('teamTricode', away.get('teamName', 'AWAY'))}</strong><br>
                            <span style="font-size:1.6rem; font-weight:700; color:#00d4aa;">{away_score}</span>
                        </div>
                        <div style="text-align:center; color:#888; padding:0 12px;">
                            <div style="font-size:0.75rem;">{status}</div>
                            <div style="font-size:1.1rem; font-weight:600;">VS</div>
                        </div>
                        <div style="text-align:center; flex:1;">
                            <img src="{get_team_logo(home_id)}" width="48" style="margin-bottom:4px;"><br>
                            <strong>{home.get('teamTricode', home.get('teamName', 'HOME'))}</strong><br>
                            <span style="font-size:1.6rem; font-weight:700; color:#00d4aa;">{home_score}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE: PLAYER SIMULATOR
# ------------------------------------------------------------
elif page == "Player Simulator":
    st.title("Player Monte Carlo Simulator")
    st.markdown(f"Run **{N_SIMS:,}** simulations based on recent form. Projects PTS / REB / AST / combos / triple-doubles.")

    all_players = get_all_players()
    player_options = {f"{p['full_name']} ({p['id']})": p['id'] for p in all_players}

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox("Search player", options=list(player_options.keys()), index=None, placeholder="Type a name...")
    with col2:
        last_n = st.slider("Games for form", 5, 25, 12)

    if selected_label:
        player_id = player_options[selected_label]
        player_name = selected_label.split(" (")[0]

        # Header with photo
        info_df = fetch_player_info(player_id)
        team_id = None
        if not info_df.empty:
            team_id = info_df.iloc[0].get("TEAM_ID")

        hcol1, hcol2 = st.columns([1, 4])
        with hcol1:
            st.image(get_player_headshot(player_id), width=140)
        with hcol2:
            st.subheader(player_name)
            if team_id:
                st.image(get_team_logo(int(team_id)), width=60)
            st.caption(f"Player ID: {player_id}")

        # Fetch data
        with st.spinner("Pulling recent game logs..."):
            recent = fetch_player_gamelog(player_id, last_n=last_n)

        if recent.empty:
            st.error("No recent game data found for this player.")
        else:
            st.markdown("### Recent Form")
            display_cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "PRA", "PR", "PA"]
            st.dataframe(
                recent[display_cols].sort_values("GAME_DATE", ascending=False).style.format({
                    "PTS": "{:.0f}", "REB": "{:.0f}", "AST": "{:.0f}",
                    "PRA": "{:.0f}", "PR": "{:.0f}", "PA": "{:.0f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Run sims
            if st.button("🚀 Run 10,000 Simulations", type="primary"):
                with st.spinner(f"Simulating {N_SIMS:,} outcomes..."):
                    sims = run_monte_carlo(recent, n_sims=N_SIMS)

                if sims is None:
                    st.error("Not enough data to simulate.")
                else:
                    st.success("Simulations complete")

                    # Summary metrics
                    st.markdown("### Projection Summary")
                    mcols = st.columns(4)
                    metrics = [
                        ("PTS", sims["PTS"]),
                        ("REB", sims["REB"]),
                        ("AST", sims["AST"]),
                        ("PRA", sims["PRA"]),
                    ]
                    for i, (label, arr) in enumerate(metrics):
                        s = summary_stats(arr)
                        with mcols[i]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{s['mean']:.1f}</div>
                                <div style="color:#888; font-size:0.8rem;">± {s['std']:.1f} • Med {s['median']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Combos + TD
                    st.markdown("### Combo & Triple-Double Projections")
                    ccols = st.columns(4)
                    for i, (label, arr) in enumerate([
                        ("PTS+REB", sims["PR"]),
                        ("PTS+AST", sims["PA"]),
                        ("PRA", sims["PRA"]),
                        ("Triple-Double %", sims["TD"] * 100),
                    ]):
                        s = summary_stats(arr)
                        with ccols[i]:
                            if label == "Triple-Double %":
                                st.metric(label, f"{s['mean']:.1f}%")
                            else:
                                st.metric(label, f"{s['mean']:.1f}", help=f"10th–90th: {s['p10']:.1f} – {s['p90']:.1f}")

                    # Distribution charts
                    st.markdown("### Simulation Distributions")
                    fig = make_subplots(rows=2, cols=2, subplot_titles=["Points", "Rebounds", "Assists", "PRA"])
                    for idx, (stat, row, col) in enumerate([
                        ("PTS", 1, 1), ("REB", 1, 2), ("AST", 2, 1), ("PRA", 2, 2)
                    ]):
                        fig.add_trace(
                            go.Histogram(x=sims[stat], nbinsx=40, name=stat, marker_color="#00d4aa", opacity=0.85),
                            row=row, col=col
                        )
                        mean_val = np.mean(sims[stat])
                        fig.add_vline(x=mean_val, line_dash="dash", line_color="#ff6b6b", row=row, col=col)
                    fig.update_layout(
                        height=560,
                        showlegend=False,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(30,30,42,0.6)",
                        font_color="#c0c0d0",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Over probabilities table
                    st.markdown("### Over Probability Table (from sims)")
                    over_data = []
                    for stat in ["PTS", "REB", "AST", "PRA", "PR", "PA"]:
                        arr = sims[stat]
                        over_data.append({
                            "Stat": stat,
                            "Mean": f"{np.mean(arr):.1f}",
                            "P(≥10)": f"{np.mean(arr >= 10)*100:.1f}%",
                            "P(≥15)": f"{np.mean(arr >= 15)*100:.1f}%",
                            "P(≥20)": f"{np.mean(arr >= 20)*100:.1f}%",
                            "P(≥25)": f"{np.mean(arr >= 25)*100:.1f}%",
                            "P(≥30)": f"{np.mean(arr >= 30)*100:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(over_data), use_container_width=True, hide_index=True)

                    # Store in session for props page
                    st.session_state["last_sims"] = sims
                    st.session_state["last_player"] = player_name
                    st.session_state["last_recent"] = recent

# ------------------------------------------------------------
# PAGE: PLAYER PROFILES
# ------------------------------------------------------------
elif page == "Player Profiles":
    st.title("Player Profiles")
    st.markdown("Headshots, team logos, recent trends, and season context.")

    all_players = get_all_players()
    player_options = {f"{p['full_name']}": p['id'] for p in all_players}

    selected = st.selectbox("Select player", options=sorted(player_options.keys()), index=None, placeholder="Search...")

    if selected:
        pid = player_options[selected]
        info = fetch_player_info(pid)
        recent = fetch_player_gamelog(pid, last_n=15)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(get_player_headshot(pid), width=180)
            if not info.empty:
                team_id = info.iloc[0].get("TEAM_ID")
                if team_id:
                    st.image(get_team_logo(int(team_id)), width=80)
        with col2:
            st.subheader(selected)
            if not info.empty:
                row = info.iloc[0]
                st.write(f"**Team:** {row.get('TEAM_NAME', '—')} ({row.get('TEAM_ABBREVIATION', '')})")
                st.write(f"**Position:** {row.get('POSITION', '—')} • **Height:** {row.get('HEIGHT', '—')} • **Weight:** {row.get('WEIGHT', '—')}")
                st.write(f"**Draft:** {row.get('DRAFT_YEAR', '—')} • Round {row.get('DRAFT_ROUND', '—')} Pick {row.get('DRAFT_NUMBER', '—')}")
                st.write(f"**Country:** {row.get('COUNTRY', '—')} • **School:** {row.get('SCHOOL', '—')}")

        if not recent.empty:
            st.markdown("### Last 15 Games – Trend Bars")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=recent["GAME_DATE"], y=recent["PTS"], name="PTS", marker_color="#00d4aa"))
            fig.add_trace(go.Bar(x=recent["GAME_DATE"], y=recent["REB"], name="REB", marker_color="#4ecdc4"))
            fig.add_trace(go.Bar(x=recent["GAME_DATE"], y=recent["AST"], name="AST", marker_color="#ff6b6b"))
            fig.update_layout(
                barmode="group",
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,42,0.5)",
                font_color="#c0c0d0",
                legend=dict(orientation="h", y=1.1),
                xaxis_title="",
                yaxis_title="Stat",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Game Log")
            st.dataframe(
                recent[["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "PRA"]].sort_values("GAME_DATE", ascending=False),
                use_container_width=True,
                hide_index=True
            )

# ------------------------------------------------------------
# PAGE: TRENDS
# ------------------------------------------------------------
elif page == "Trends & Form":
    st.title("Player & Team Trends")
    st.markdown("Visualize form over recent games.")

    all_players = get_all_players()
    player_options = {f"{p['full_name']}": p['id'] for p in all_players}
    selected = st.selectbox("Player", options=sorted(player_options.keys()), index=None)

    if selected:
        pid = player_options[selected]
        recent = fetch_player_gamelog(pid, last_n=20)
        if not recent.empty:
            # Rolling averages
            recent = recent.sort_values("GAME_DATE")
            recent["PTS_roll"] = recent["PTS"].rolling(5, min_periods=1).mean()
            recent["REB_roll"] = recent["REB"].rolling(5, min_periods=1).mean()
            recent["AST_roll"] = recent["AST"].rolling(5, min_periods=1).mean()
            recent["PRA_roll"] = recent["PRA"].rolling(5, min_periods=1).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent["GAME_DATE"], y=recent["PTS"], mode="lines+markers", name="PTS", line=dict(color="#00d4aa")))
            fig.add_trace(go.Scatter(x=recent["GAME_DATE"], y=recent["PTS_roll"], mode="lines", name="PTS 5G Avg", line=dict(color="#00d4aa", dash="dot")))
            fig.add_trace(go.Scatter(x=recent["GAME_DATE"], y=recent["REB"], mode="lines+markers", name="REB", line=dict(color="#4ecdc4")))
            fig.add_trace(go.Scatter(x=recent["GAME_DATE"], y=recent["AST"], mode="lines+markers", name="AST", line=dict(color="#ff6b6b")))
            fig.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,42,0.5)",
                font_color="#c0c0d0",
                title=f"{selected} – Recent Form",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Rolling Averages (5-game)")
            st.line_chart(recent.set_index("GAME_DATE")[["PTS_roll", "REB_roll", "AST_roll", "PRA_roll"]])

# ------------------------------------------------------------
# PAGE: PROPS EDGE
# ------------------------------------------------------------
elif page == "Props Edge Checker":
    st.title("Props Edge Checker")
    st.markdown("""
    Compare your simulation results against lines from **Underdog**, **PrizePicks**, **Kalshi**, etc.
    
    > **Note:** These platforms do not offer free public APIs.  
    > Enter the line manually after checking the app. The simulator will tell you the fair probability from 10k sims.
    """)

    if "last_sims" not in st.session_state:
        st.info("👉 First run a simulation on the **Player Simulator** page. Results will appear here automatically.")
    else:
        sims = st.session_state["last_sims"]
        player = st.session_state.get("last_player", "Player")
        st.success(f"Using latest sims for **{player}**")

        stat = st.selectbox("Stat", ["PTS", "REB", "AST", "PRA", "PR", "PA", "TD"])
        line = st.number_input("Book / App Line", min_value=0.0, max_value=80.0, value=20.5, step=0.5)

        if stat == "TD":
            prob = float(np.mean(sims["TD"]))
            st.metric("Simulated Triple-Double Probability", f"{prob*100:.1f}%")
            st.caption("Compare this % to the implied probability from the price on Kalshi / Underdog etc.")
        else:
            arr = sims[stat]
            over_prob = float(np.mean(arr > line))
            under_prob = 1 - over_prob
            mean = float(np.mean(arr))

            c1, c2, c3 = st.columns(3)
            c1.metric("Sim Mean", f"{mean:.1f}")
            c2.metric(f"P(Over {line})", f"{over_prob*100:.1f}%")
            c3.metric(f"P(Under {line})", f"{under_prob*100:.1f}%")

            # Simple edge framing
            st.markdown("#### Quick Edge Read")
            if over_prob > 0.55:
                st.success(f"Sims lean **Over** ({over_prob*100:.1f}%). Potential value if the book is pricing it closer to 50/50.")
            elif under_prob > 0.55:
                st.success(f"Sims lean **Under** ({under_prob*100:.1f}%). Potential value if the book is pricing it closer to 50/50.")
            else:
                st.info("Sims are close to a coin flip around this line.")

            # Distribution vs line
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=arr, nbinsx=50, marker_color="#00d4aa", opacity=0.8, name="Sims"))
            fig.add_vline(x=line, line_dash="dash", line_color="#ff6b6b", annotation_text=f"Line {line}")
            fig.add_vline(x=mean, line_dash="dot", line_color="#feca57", annotation_text=f"Mean {mean:.1f}")
            fig.update_layout(
                height=360,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,30,42,0.5)",
                font_color="#c0c0d0",
                title=f"{player} – {stat} Distribution vs Line",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.caption("Always cross-check injuries, minutes projections, and matchup context. This is a statistical tool, not financial advice.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built for private use • Data © NBA")
