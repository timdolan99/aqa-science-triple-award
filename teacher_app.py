import os
import json
import sqlite3
import collections
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "Socratic Learning Assistant",
        "level": "GCSE/A-Level"
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "Socratic Coach")
LEVEL = COURSE_SPEC.get("level", "GCSE/A-Level")
DB_NAME = "analytics.db"

# --- Page Setup & Dynamic Styling ---
st.set_page_config(page_title=f"{COURSE_TITLE} - Teacher Console", layout="wide", page_icon="📊")

THEME_PRIMARY = "#0284c7"
THEME_GRADIENT_START = "#0369a1"
THEME_GRADIENT_END = "#0284c7"

st.markdown(f"""
    <style>
    .stApp {{ background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
    div[data-testid="stSidebar"] {{ background-color: #ffffff; border-right: 1px solid #e2e8f0; }}
    h1, h2, h3 {{ color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }}
    .console-header {{ 
        background: linear-gradient(135deg, {THEME_GRADIENT_START} 0%, {THEME_GRADIENT_END} 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.4em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(2, 132, 199, 0.25);
        margin-bottom: 24px;
    }}
    .metric-card {{
        background: #ffffff; padding: 18px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; text-align: center;
    }}
    .afl-callout {{
        background: #f0f9ff; border-left: 5px solid {THEME_PRIMARY}; padding: 16px 20px;
        border-radius: 8px; color: #0369a1; font-size: 0.98em; line-height: 1.5; margin-top: 15px;
    }}
    </style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="console-header">📊 {COURSE_TITLE} ({LEVEL}) Teacher Analytics Console</div>', unsafe_allow_html=True)

# --- Database Fetch Helper ---
def load_data():
    if not os.path.exists(DB_NAME):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM activity_logs", conn)
    conn.close()
    return df

df_raw = load_data()

if df_raw.empty:
    st.warning("⚠️ No analytics data found in `analytics.db`. Please run `generate_test_data.py` first to populate test logs.")
    st.stop()

# --- Dynamic Filter Generation ---
st.sidebar.header("🎯 Filter Cohort Data")

# Place near the top of your sidebar in teacher_app.py
if st.sidebar.button("🔄 Refresh Analytics Data"):
    st.cache_data.clear()
    st.rerun()

subjects = ["All"] + sorted(list(df_raw["subject"].unique()))
selected_subject = st.sidebar.selectbox("Filter by Subject / Branch:", options=subjects)

filtered_df = df_raw.copy()
if selected_subject != "All":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]

units = ["All"] + sorted(list(filtered_df["unit"].unique()))
selected_unit = st.sidebar.selectbox("Filter by Unit:", options=units)

if selected_unit != "All":
    filtered_df = filtered_df[filtered_df["unit"] == selected_unit]

modes = ["All"] + sorted(list(filtered_df["app_mode"].unique()))
selected_mode = st.sidebar.selectbox("Filter by Activity Mode:", options=modes)

if selected_mode != "All":
    filtered_df = filtered_df[filtered_df["app_mode"] == selected_mode]

st.sidebar.write("---")
st.sidebar.caption("🔒 **Zero-PII Compliance:** Data is logged statelessly at session level. No individual pupil identities are tracked.")

# --- Macro KPI Cards ---
total_sessions = len(filtered_df)
avg_score = round(filtered_df["score_pct"].mean(), 1) if total_sessions > 0 else 0.0
misconception_count = int(filtered_df["misconception_flag"].sum()) if total_sessions > 0 else 0

all_missed_keywords = []
for kw_str in filtered_df["keywords_missed"].dropna():
    try:
        kw_list = json.loads(kw_str)
        all_missed_keywords.extend(kw_list)
    except Exception:
        pass

kw_counter = collections.Counter(all_missed_keywords)
top_missed_term = kw_counter.most_common(1)[0][0] if kw_counter else "None"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h3>{total_sessions}</h3><p>Total Practice Sessions</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><h3>{avg_score}%</h3><p>Cohort Avg Score</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><h3>{misconception_count}</h3><p>Misconception Flags</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3>{top_missed_term}</h3><p>Top Missed Term</p></div>', unsafe_allow_html=True)

st.write("")

# --- Visualizations ---
tab1, tab2 = st.tabs(["📌 Keyword Gap Analysis", "📈 Engagement & Performance"])

with tab1:
    st.subheader("Top Specification Keywords Missed Across Cohort")
    if kw_counter:
        top_kws = pd.DataFrame(kw_counter.most_common(10), columns=["Keyword", "Frequency"])
        fig_kw = px.bar(
            top_kws, x="Frequency", y="Keyword", orientation="h",
            color="Frequency", color_continuous_scale="Reds",
            title="Specification Terms Requiring Targeted Remediation"
        )
        fig_kw.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_kw, use_container_width=True)
        
        st.markdown(f"""
            <div class="afl-callout">
                💡 <b>Actionable Assessment for Learning (AfL) Insight:</b><br>
                The most frequently missed term in this selection is <b>"{top_missed_term}"</b> (missed <b>{kw_counter[top_missed_term]}</b> times). 
                Consider starting your next lesson with a 5-minute retrieval starter focused on this specific concept.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No missed keywords recorded for this selection.")

with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Activity Mode Distribution")
        mode_counts = filtered_df["app_mode"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Sessions"]
        fig_mode = px.pie(mode_counts, names="Mode", values="Sessions", hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_mode, use_container_width=True)
        
    with col_right:
        st.subheader("Performance Score Distribution")
        fig_score = px.histogram(filtered_df, x="score_pct", nbins=10, title="Cohort Score Range (%)", color_discrete_sequence=[THEME_PRIMARY])
        st.plotly_chart(fig_score, use_container_width=True)