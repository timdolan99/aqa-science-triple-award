import collections
import io
import json
import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine, text
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# --- Helper to safely extract string text from LangChain content ---
def extract_text(content):
    """Safely extracts plain text from LangChain message content (handles str or list)."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {
        "course_title": "OCR A-Level Computer Science",
        "level": "A-Level",
    }

COURSE_TITLE = COURSE_SPEC.get("course_title", "OCR A-Level Computer Science")
LEVEL = COURSE_SPEC.get("level", "A-Level")


# --- Database Connection Helper ---
@st.cache_resource
def get_db_engine():
    """Retrieves cached SQLAlchemy engine using Supabase credentials in secrets.toml."""
    try:
        db_url = st.secrets["postgres"]["url"]
        return create_engine(db_url, pool_pre_ping=True)
    except Exception as e:
        st.error(f"❌ Database secret configuration error: {e}")
        return None


# --- Game Schema File Generators ---

def create_blooket_csv(questions_data: list) -> bytes:
    """Formats raw question dicts into Blooket's exact required CSV schema,
    including the required sample row so Blooket does not skip Question #1.
    """
    df_raw = pd.DataFrame(questions_data)

    # Add sequential question numbers
    df_raw.insert(0, "Question #", range(1, len(df_raw) + 1))

    column_mapping = {
        "Question #": "Question #",
        "Question": "Question Text",
        "Answer 1": "Answer 1",
        "Answer 2": "Answer 2",
        "Answer 3": "Answer 3 \n(Optional)",
        "Answer 4": "Answer 4 \n(Optional)",
        "Time Limit (sec)": "Time Limit (sec) \n(Max: 300 Seconds)",
        "Correct Answer(s)": "Correct Answer(s) \n(Only Include Answer #)",
    }

    df_formatted = df_raw[list(column_mapping.keys())].rename(
        columns=column_mapping
    )

    # Insert Blooket's dummy/sample row at index 0 so Blooket skips it instead of Q1
    sample_row = pd.DataFrame(
        [
            {
                "Question #": "1 (Sample)",
                "Question Text": "What's 2 + 2?",
                "Answer 1": "4",
                "Answer 2": "3",
                "Answer 3 \n(Optional)": "1",
                "Answer 4 \n(Optional)": "2",
                "Time Limit (sec) \n(Max: 300 Seconds)": 20,
                "Correct Answer(s) \n(Only Include Answer #)": "1",
            }
        ]
    )

    # Re-number actual questions starting from 1 after sample row
    df_formatted["Question #"] = range(1, len(df_formatted) + 1)

    # Combine sample row + generated questions
    df_final = pd.concat([sample_row, df_formatted], ignore_index=False)

    return df_final.to_csv(index=False).encode("utf-8")


def create_gimkit_csv(questions_data: list) -> bytes:
    """Formats raw question dicts into Gimkit's exact required CSV schema."""
    df_raw = pd.DataFrame(questions_data)

    column_mapping = {
        "Question": "Question",
        "Correct Answer": "Correct Answer",
        "Incorrect Answer 1": "Incorrect Answer 1",
        "Incorrect Answer 2": "Incorrect Answer 2 (Optional)",
        "Incorrect Answer 3": "Incorrect Answer 3 (Optional)",
    }

    df_formatted = df_raw[list(column_mapping.keys())].rename(
        columns=column_mapping
    )
    return df_formatted.to_csv(index=False).encode("utf-8")


# --- Page Setup & Dynamic Styling ---
LOGO_PATH = "zp_logo.png"

st.set_page_config(
    page_title=f"{COURSE_TITLE} - ZeroPi Console",
    layout="wide",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "📊",
)

THEME_PRIMARY = "#0284c7"
THEME_GRADIENT_START = "#0369a1"
THEME_GRADIENT_END = "#0284c7"

st.markdown(
    f"""
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
        min-height: 120px; display: flex; flex-direction: column; justify-content: center;
    }}
    .afl-callout {{
        background: #f0f9ff; border-left: 5px solid {THEME_PRIMARY}; padding: 16px 20px;
        border-radius: 8px; color: #0369a1; font-size: 0.98em; line-height: 1.5; margin-top: 15px;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# Render main header banner with logo alignment

st.markdown(
    f'<div class="console-header">📊 {COURSE_TITLE} ({LEVEL}) ZeroPi Teacher Console</div>',
    unsafe_allow_html=True,
)

# --- Database Fetch Helper ---
@st.cache_data(ttl=60)
def load_data():
    engine = get_db_engine()
    if engine is None:
        return pd.DataFrame()
    try:
        query = "SELECT * FROM activity_logs ORDER BY timestamp DESC;"
        df = pd.read_sql_query(query, engine)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data from Supabase: {e}")
        return pd.DataFrame()


df_raw = load_data()

if df_raw.empty:
    st.warning(
        "⚠️ No analytics data found in Supabase. Please run `generate_test_data.py` first to populate test logs."
    )
    st.stop()

# --- Dynamic Filter Generation ---


# --- Sidebar (Centered Logo) ---
if os.path.exists(LOGO_PATH):
    col1, col2, col3 = st.sidebar.columns([1, 6, 1])
    with col2:
        st.image(LOGO_PATH, use_container_width=True)

st.sidebar.header("🎯 Filter Cohort Data")

if st.sidebar.button("🔄 Refresh Analytics Data"):
    st.cache_data.clear()
    st.rerun()

subjects = ["All"] + sorted(list(df_raw["subject"].unique()))
selected_subject = st.sidebar.selectbox(
    "Filter by Subject / Branch:", options=subjects
)

filtered_df = df_raw.copy()
if selected_subject != "All":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]

units = ["All"] + sorted(list(filtered_df["unit"].unique()))
selected_unit = st.sidebar.selectbox("Filter by Unit:", options=units)

if selected_unit != "All":
    filtered_df = filtered_df[filtered_df["unit"] == selected_unit]

modes = ["All"] + sorted(list(filtered_df["app_mode"].unique()))
selected_mode = st.sidebar.selectbox(
    "Filter by Activity Mode:", options=modes
)

if selected_mode != "All":
    filtered_df = filtered_df[filtered_df["app_mode"] == selected_mode]

st.sidebar.write("---")
st.sidebar.caption(
    "🔒 **Zero-PII Compliance:** Data is logged statelessly at session level. No individual pupil identities or tokens are stored."
)

# --- Macro KPI Cards ---
total_sessions = len(filtered_df)
avg_score = (
    round(filtered_df["score_pct"].mean(), 1) if total_sessions > 0 else 0.0
)
misconception_count = (
    int(filtered_df["misconception_flag"].sum()) if total_sessions > 0 else 0
)

# Extract 2 Most Recent Distinct Explored Topics
recent_topics = []
if "subtopic" in filtered_df.columns and not filtered_df.empty:
    for topic in filtered_df["subtopic"].dropna().tolist():
        if topic and topic not in recent_topics:
            recent_topics.append(topic)
        if len(recent_topics) == 2:
            break

if len(recent_topics) == 0:
    trending_str = "<p style='font-size: 0.9em; color: #64748b;'>None Recorded</p>"
elif len(recent_topics) == 1:
    trending_str = f"<p style='font-size: 0.85em; font-weight: 600; color: #0284c7; margin:0;'>• {recent_topics[0]}</p>"
else:
    trending_str = (
        f"<p style='font-size: 0.8em; font-weight: 600; color: #0284c7; margin:0; line-height: 1.2;'>• {recent_topics[0]}</p>"
        f"<p style='font-size: 0.8em; font-weight: 600; color: #0284c7; margin:3px 0 0 0; line-height: 1.2;'>• {recent_topics[1]}</p>"
    )

all_missed_keywords = []
for kw_raw in filtered_df["keywords_missed"].dropna():
    if not kw_raw:
        continue
    if kw_raw.startswith("["):
        try:
            kw_list = json.loads(kw_raw)
            all_missed_keywords.extend([k.strip() for k in kw_list if k.strip()])
        except Exception:
            pass
    else:
        all_missed_keywords.extend(
            [k.strip() for k in kw_raw.split(",") if k.strip()]
        )

kw_counter = collections.Counter(all_missed_keywords)
top_missed_term = kw_counter.most_common(1)[0][0] if kw_counter else "None"

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(
        f'<div class="metric-card"><h3>{total_sessions}</h3><p style="margin:0; font-size:0.85em;">Total Practice Sessions</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><h3>{avg_score}%</h3><p style="margin:0; font-size:0.85em;">Cohort Avg Score</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><h3>{misconception_count}</h3><p style="margin:0; font-size:0.85em;">Misconception Flags</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><h3 style="font-size:1.2em; word-break:break-word;">{top_missed_term}</h3><p style="margin:0; font-size:0.85em;">Top Missed Term</p></div>',
        unsafe_allow_html=True,
    )

# Helper regex to strip leading specification section numbers (e.g., "1.1.1 ", "2.3.1 ")
clean_topics = [re.sub(r"^\d+(\.\d+)*\s*", "", t) for t in recent_topics]

if len(clean_topics) == 0:
    trending_str = "<p style='font-size: 0.85em; color: #64748b; margin:0;'>None Recorded</p>"
elif len(clean_topics) == 1:
    trending_str = (
        f"<p style='font-size: 0.8em; font-weight: 600; color: #0284c7; margin:0; "
        f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='#{clean_topics[0]}'>"
        f"#{clean_topics[0]}</p>"
    )
else:
    trending_str = (
        f"<p style='font-size: 0.8em; font-weight: 600; color: #0284c7; margin:0; line-height: 1.2; "
        f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='#{clean_topics[0]}'>"
        f"#{clean_topics[0]}</p>"
        f"<p style='font-size: 0.8em; font-weight: 600; color: #0284c7; margin:3px 0 0 0; line-height: 1.2; "
        f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='#{clean_topics[1]}'>"
        f"#{clean_topics[1]}</p>"
    )

with c5:
    st.markdown(
        f'<div class="metric-card">{trending_str}<p style="margin:4px 0 0 0; font-size:0.85em;">Trending Topics</p></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --- Visualizations & Game Studio ---
tab1, tab2, tab3 = st.tabs(
    [
        "📌 Keyword Gap Analysis",
        "📈 Engagement & Performance",
        "🎮 Game Export Studio",
    ]
)

with tab1:
    st.subheader("Top Specification Keywords Missed Across Cohort")
    if kw_counter:
        # Get top missed term for the AfL callout below
        top_missed_term = kw_counter.most_common(1)[0][0] if kw_counter else "N/A"

        # 1. Create a 50/50 column layout
        col_chart, col_cloud = st.columns(2)

        # --- LEFT COLUMN: Most Active Revision Topics Bar Chart ---
        with col_chart:
            st.markdown("##### 📊 Most Active Revision Topics (Last 100 Sessions)")

            # Fetch Last 100 Entries for Active Filtered Subject
            df_recent_100 = filtered_df.sort_values(by="timestamp", ascending=False).head(100)

            if not df_recent_100.empty:
                # Aggregate Topic Frequencies
                df_recent_100["topic_label"] = df_recent_100["unit"] + " - " + df_recent_100["subtopic"]
                
                topic_counts = (
                    df_recent_100["topic_label"]
                    .value_counts()
                    .reset_index()
                )
                topic_counts.columns = ["Topic", "Session Count"]
                
                # Select top 10 most selected targets
                top_topics = topic_counts.head(10).sort_values(by="Session Count", ascending=True)

                # Read dynamic qualification level from course spec
                active_level = COURSE_SPEC.get("level", "GCSE") if "COURSE_SPEC" in globals() else "GCSE"
                chart_title = f"{active_level} Specification: Most Selected Revision Targets"

                # Build Horizontal Bar Chart
                fig_topics = px.bar(
                    top_topics,
                    x="Session Count",
                    y="Topic",
                    orientation="h",
                    title=chart_title,
                    color="Session Count",
                    color_continuous_scale="Viridis",
                    labels={"Session Count": "Number of Sessions", "Topic": "Unit / Subtopic"},
                )

                fig_topics.update_layout(
                    xaxis_title="Session Count (out of last 100)",
                    yaxis_title="Specification Subtopic",
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=380,
                )

                st.plotly_chart(fig_topics, use_container_width=True)

            else:
                st.info("No session activity recorded yet for the selected filters.")

        # --- RIGHT COLUMN: Matching Height HTML Word Cloud ---
        with col_cloud:
            st.markdown("##### Keyword Cloud")
            
            if kw_counter:
                import math
                import random

                top_terms = kw_counter.most_common(20)
                
                # Shuffle order so top terms scatter naturally throughout the cloud
                shuffled_terms = top_terms.copy()
                random.seed(42)  # Fixed seed prevents cloud jump on rerender
                random.shuffle(shuffled_terms)

                # Calculate max and min for scaling
                max_freq = max(freq for _, freq in top_terms) if top_terms else 1
                min_freq = min(freq for _, freq in top_terms) if top_terms else 1

                html_tags = []
                for term, freq in shuffled_terms:
                    # Logarithmic scaling prevents top items from dominating row height
                    if max_freq > min_freq:
                        norm_score = math.log(freq - min_freq + 1) / math.log(max_freq - min_freq + 1)
                    else:
                        norm_score = 0.5

                    # Font size ranges between 0.8em and 1.6em
                    font_size = round(0.8 + (norm_score * 0.8), 2)
                    
                    # Color hierarchy matching dark red palette
                    if norm_score > 0.7:
                        color = "#7f1d1d"  # Very dark red
                    elif norm_score > 0.4:
                        color = "#b91c1c"  # Medium red
                    else:
                        color = "#ef4444"  # Light red/coral

                    html_tags.append(
                        f'<span style="font-size: {font_size}em; color: {color}; font-weight: 700; '
                        f'margin: 6px 10px; padding: 2px 4px; display: inline-block; '
                        f'line-height: 1.2; vertical-align: middle;">'
                        f'{term} <sub style="font-size:0.55em; color:#94a3b8; font-weight:400;">({freq})</sub></span>'
                    )

                cloud_html = f"""
                <div style="background-color: #ffffff; padding: 24px; border-radius: 12px; 
                            border: 1px solid #e2e8f0; height: 380px; box-sizing: border-box;
                            display: flex; flex-wrap: wrap; align-content: center; 
                            align-items: center; justify-content: center; text-align: center; 
                            overflow-y: auto;">
                    {"".join(html_tags)}
                </div>
                """
                st.markdown(cloud_html, unsafe_allow_html=True)
            else:
                st.info("No missed keywords captured for this selection.")

        # --- BELOW BOTH VISUALS: Full-width AfL Insight Callout ---
        st.markdown(
            f"""
            <div class="afl-callout">
                💡 <b>Actionable Assessment for Learning (AfL) Insight:</b><br>
                The most frequently missed technical term in this view is <b>"{top_missed_term}"</b> (missed <b>{kw_counter[top_missed_term]}</b> times across student practice rounds). 
                Consider opening your next lesson with a 5-minute retrieval starter or whiteboard trace task targeting this specific concept.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No missed keywords recorded for this selection.")

with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Activity Mode Distribution")
        mode_counts = filtered_df["app_mode"].value_counts().reset_index()
        mode_counts.columns = ["Mode", "Sessions"]
        fig_mode = px.pie(
            mode_counts,
            names="Mode",
            values="Sessions",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=380,  # Matched height
        )
        fig_mode.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_mode, use_container_width=True)

    with col_right:
        st.subheader("Performance Score Distribution")
        fig_score = px.histogram(
            filtered_df,
            x="score_pct",
            nbins=10,
            title="Cohort Score Range (%)",
            color_discrete_sequence=[THEME_PRIMARY],
            height=380,  # Matched height
        )
        fig_score.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_score, use_container_width=True)

with tab3:
    st.subheader("🎮 Classroom Game Export Studio")
    st.write(
        "Generate ready-to-import CSV files (`.csv`) formatted specifically for classroom game platforms."
    )

    game_platform = st.radio(
        "Select Target Platform:",
        ["Blooket", "Gimkit"],
        horizontal=True,
        key="game_platform_selector",
    )

    st.write("---")

    # =========================================================
    # PLATFORM 1: BLOOKET
    # =========================================================
    if game_platform == "Blooket":
        blooket_tab1, blooket_tab2 = st.tabs(
            ["📚 Structured Topic Revision", "🎯 AfL Intervention Prompting"]
        )

        with blooket_tab1:
            st.markdown("### Generate Blooket Sets from Specification Topics")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                rev_unit = st.selectbox(
                    "Select Target Unit:",
                    options=units[1:],
                    key="blooket_unit_select",
                )
            with col_b2:
                num_q_rev = st.slider(
                    "Number of Revision Questions:",
                    min_value=5,
                    max_value=25,
                    value=10,
                    key="blooket_rev_q_slider",
                )

            if st.button("Generate Revision Blooket", key="btn_rev_blooket"):
                with st.spinner("Compiling specification revision set..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.6-flash", temperature=0.2
                    )
                    prompt = ChatPromptTemplate.from_template(
                        """You are an AQA/OCR A-Level Computer Science examiner.
                        Create {num_q} standard revision questions for the unit: '{unit}'.
                        Focus on core terminology, key facts, and standard exam definitions.
                        
                        Return ONLY a JSON array with exact keys:
                        [
                          {{
                            "Question": "Question text?",
                            "Answer 1": "Correct Answer",
                            "Answer 2": "Wrong Option 1",
                            "Answer 3": "Wrong Option 2",
                            "Answer 4": "Wrong Option 3",
                            "Time Limit (sec)": 30,
                            "Correct Answer(s)": 1
                          }}
                        ]
                        Do NOT include markdown formatting or backticks in your output."""
                    )
                    chain = prompt | llm
                    res = chain.invoke({"unit": rev_unit, "num_q": num_q_rev})

                    raw_text = extract_text(res.content)
                    clean_res = (
                        raw_text.strip().strip("`").replace("json\n", "").strip()
                    )
                    data = json.loads(clean_res)

                    csv_data = create_blooket_csv(data)
                    st.success("Blooket revision file ready for download!")
                    st.download_button(
                        label="📥 Download Revision Blooket (.csv)",
                        data=csv_data,
                        file_name=f"blooket_revision_{rev_unit[:10].replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="dl_blooket_rev",
                    )

        with blooket_tab2:
            st.markdown("### Generate Targeted Blooket Intervention Sets")
            st.info(
                "Paste diagnostic marking notes or student misconceptions to design distractor-focused diagnostic games."
            )

            default_afl_prompt = (
                f"Top missed term in cohort data: '{top_missed_term}'. "
                f"Students frequently miss this keyword across practice sessions."
                if top_missed_term != "None"
                else ""
            )

            intervention_context = st.text_area(
                "Intervention Prompt / Misconception Notes:",
                value=default_afl_prompt,
                height=120,
                placeholder="e.g., Students are confusing encapsulation with abstraction.",
                key="blooket_afl_text_area",
            )

            num_q_int = st.number_input(
                "Number of Targeted Questions:",
                min_value=3,
                max_value=25,
                value=5,
                key="afl_num_questions_input",
            )

            if st.button("Generate Intervention Blooket", key="btn_int_blooket"):
                if not intervention_context.strip():
                    st.warning("Please enter intervention details first.")
                else:
                    with st.spinner(
                        "Designing diagnostic distractors based on student misconceptions..."
                    ):
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-3.6-flash", temperature=0.2
                        )
                        prompt = ChatPromptTemplate.from_template(
                            """You are an expert Computer Science teacher designing a targeted diagnostic intervention.
                            Use the following AfL assessment notes / student misconceptions:
                            "{context}"
                            
                            Generate {num_q} multiple choice questions specifically designed to address these misunderstandings.
                            Make the incorrect options (distractors) represent the specific student mistakes mentioned.
                            
                            Return ONLY a JSON array with exact keys:
                            [
                              {{
                                "Question": "Question text?",
                                "Answer 1": "Correct Answer",
                                "Answer 2": "Common Misconception Distractor 1",
                                "Answer 3": "Common Misconception Distractor 2",
                                "Answer 4": "Plausible Distractor",
                                "Time Limit (sec)": 45,
                                "Correct Answer(s)": 1
                              }}
                            ]
                            Do NOT include markdown formatting or backticks in your output."""
                        )
                        chain = prompt | llm
                        res = chain.invoke(
                            {
                                "context": intervention_context,
                                "num_q": num_q_int,
                            }
                        )

                        raw_text = extract_text(res.content)
                        clean_res = (
                            raw_text.strip().strip("`").replace("json\n", "").strip()
                        )
                        data = json.loads(clean_res)

                        csv_data = create_blooket_csv(data)
                        st.success("Diagnostic intervention set generated!")
                        st.download_button(
                            label="📥 Download Intervention Blooket (.csv)",
                            data=csv_data,
                            file_name="blooket_intervention_afl.csv",
                            mime="text/csv",
                            key="dl_blooket_int",
                        )

    # =========================================================
    # PLATFORM 2: GIMKIT
    # =========================================================
    elif game_platform == "Gimkit":
        gimkit_tab1, gimkit_tab2 = st.tabs(
            ["📚 Structured Topic Revision", "🎯 AfL Intervention Prompting"]
        )

        with gimkit_tab1:
            st.markdown("### Generate Gimkit Sets from Specification Topics")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                rev_unit_g = st.selectbox(
                    "Select Target Unit:",
                    options=units[1:],
                    key="gimkit_unit_select",
                )
            with col_g2:
                num_q_g_rev = st.slider(
                    "Number of Revision Questions:",
                    min_value=5,
                    max_value=25,
                    value=10,
                    key="gimkit_rev_q_slider",
                )

            if st.button("Generate Revision Gimkit", key="btn_rev_gimkit"):
                with st.spinner("Compiling Gimkit specification set..."):
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.6-flash", temperature=0.2
                    )
                    prompt = ChatPromptTemplate.from_template(
                        """You are an AQA/OCR A-Level Computer Science examiner.
                        Create {num_q} standard revision questions for the unit: '{unit}'.
                        Focus on core terminology, key facts, and standard exam definitions.
                        
                        Return ONLY a JSON array with exact keys:
                        [
                          {{
                            "Question": "Question text?",
                            "Correct Answer": "Exact correct answer text",
                            "Incorrect Answer 1": "Wrong choice 1",
                            "Incorrect Answer 2": "Wrong choice 2",
                            "Incorrect Answer 3": "Wrong choice 3"
                          }}
                        ]
                        Do NOT include markdown formatting or backticks in your output."""
                    )
                    chain = prompt | llm
                    res = chain.invoke({"unit": rev_unit_g, "num_q": num_q_g_rev})

                    raw_text = extract_text(res.content)
                    clean_res = (
                        raw_text.strip().strip("`").replace("json\n", "").strip()
                    )
                    data = json.loads(clean_res)

                    csv_data = create_gimkit_csv(data)
                    st.success("Gimkit revision file ready for download!")
                    st.download_button(
                        label="📥 Download Revision Gimkit (.csv)",
                        data=csv_data,
                        file_name=f"gimkit_revision_{rev_unit_g[:10].replace(' ', '_')}.csv",
                        mime="text/csv",
                        key="dl_gimkit_rev",
                    )

        with gimkit_tab2:
            st.markdown("### Generate Targeted Gimkit Intervention Sets")
            st.info(
                "Paste diagnostic marking notes or student misconceptions to design distractor-focused Gimkit games."
            )

            default_gimkit_prompt = (
                f"Top missed term in cohort data: '{top_missed_term}'. "
                f"Students frequently miss this keyword across practice sessions."
                if top_missed_term != "None"
                else ""
            )

            intervention_context_g = st.text_area(
                "Intervention Prompt / Misconception Notes:",
                value=default_gimkit_prompt,
                height=120,
                placeholder="e.g., Students confuse encapsulation with abstraction.",
                key="gimkit_afl_text_area",
            )

            num_q_g_int = st.number_input(
                "Number of Targeted Questions:",
                min_value=3,
                max_value=25,
                value=5,
                key="gimkit_num_questions_input",
            )

            if st.button("Generate Intervention Gimkit", key="btn_int_gimkit"):
                if not intervention_context_g.strip():
                    st.warning("Please enter intervention details first.")
                else:
                    with st.spinner(
                        "Designing diagnostic distractors based on student misconceptions..."
                    ):
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-3.6-flash", temperature=0.2
                        )
                        prompt = ChatPromptTemplate.from_template(
                            """You are an expert Computer Science teacher designing a targeted diagnostic intervention.
                            Use the following AfL assessment notes / student misconceptions:
                            "{context}"
                            
                            Generate {num_q} multiple choice questions specifically designed to address these misunderstandings.
                            Make the incorrect options (Incorrect Answer 1, 2, 3) represent the specific student mistakes mentioned.
                            
                            Return ONLY a JSON array with exact keys:
                            [
                              {{
                                "Question": "Question text?",
                                "Correct Answer": "Exact correct answer text",
                                "Incorrect Answer 1": "Common Misconception Distractor 1",
                                "Incorrect Answer 2": "Common Misconception Distractor 2",
                                "Incorrect Answer 3": "Plausible Distractor"
                              }}
                            ]
                            Do NOT include markdown formatting or backticks in your output."""
                        )
                        chain = prompt | llm
                        res = chain.invoke(
                            {
                                "context": intervention_context_g,
                                "num_q": num_q_g_int,
                            }
                        )

                        raw_text = extract_text(res.content)
                        clean_res = (
                            raw_text.strip().strip("`").replace("json\n", "").strip()
                        )
                        data = json.loads(clean_res)

                        csv_data = create_gimkit_csv(data)
                        st.success("Diagnostic intervention Gimkit set generated!")
                        st.download_button(
                            label="📥 Download Intervention Gimkit (.csv)",
                            data=csv_data,
                            file_name="gimkit_intervention_afl.csv",
                            mime="text/csv",
                            key="dl_gimkit_int",
                        )

st.write("---")

# --- Interactive AI Chatbot (Multi-Turn Conversational Analytics) ---
st.subheader("💬 Interactive AI Telemetry Assistant")

if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []

col_info, col_clear = st.columns([5, 1])
with col_clear:
    if st.button("🗑️ Reset Chat"):
        st.session_state.ai_chat_history = []
        st.rerun()


def clean_html_formatting(text):
    if isinstance(text, list):
        extracted_parts = []
        for item in text:
            if isinstance(item, str):
                extracted_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                extracted_parts.append(item["text"])
        text = "".join(extracted_parts)

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    return re.sub(r"</?[a-zA-Z0-9]+[^>]*>", "", text)


def stream_ai_synthesis(prompt_text):
    """Generates streaming tokens directly from Gemini for real-time response rendering."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", temperature=0.2, streaming=True
    )
    for chunk in llm.stream(prompt_text):
        if chunk.content:
            yield clean_html_formatting(chunk.content)


def ask_ai_about_data(user_question, chat_history):
    schema_info = """
    Table name: activity_logs (PostgreSQL)
    Columns:
    - id (SERIAL PRIMARY KEY)
    - timestamp (TIMESTAMPTZ)
    - subject (TEXT) - e.g., 'OCR A-Level Computer Science'
    - level (TEXT) - e.g., 'A-Level'
    - unit (TEXT) - e.g., '1.1 The characteristics of contemporary processors...'
    - subtopic (TEXT) - e.g., '1.1.1 Structure and function of the processor'
    - app_mode (TEXT) - 'socratic', 'quiz', 'extended', 'rewrite'
    - score_pct (REAL) - percentage score (0.0 to 100.0)
    - keywords_used (TEXT) - comma-separated list of technical CS terms used correctly
    - keywords_missed (TEXT) - comma-separated list of technical CS terms missed
    - misconception_flag (INTEGER) - 1 if fundamental CS misconception detected (<45%), else 0
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

    full_transcript = ""
    for msg in chat_history:
        full_transcript += f"Teacher: {msg['question']}\nAI Assistant: {msg['summary']}\n\n"

    # --- Step 1: Classify Intent ---
    intent_prompt = f"""
    Analyze the teacher's latest message based on the conversation transcript.
    
    Transcript:
    {full_transcript}
    
    Latest Message: "{user_question}"
    
    Is the teacher asking to:
    A) Query/fetch NEW data from the database (e.g., scores, counts, top missed CS terms, subtopic search)?
    B) Refine, reformat, expand, or follow up on previously generated text/questions (e.g., remove options, format answers, bold terms, rephrase response)?
    
    Respond with ONLY the single letter 'A' or 'B'.
    """

    intent_resp = llm.invoke(intent_prompt)
    intent_raw = intent_resp.content
    intent_text = (
        "".join(
            [
                str(b.get("text", b)) if isinstance(b, dict) else str(b)
                for b in intent_raw
            ]
        )
        if isinstance(intent_raw, list)
        else str(intent_raw)
    )
    intent = intent_text.strip().upper()

    # --- PATH B: Refinement / Content Follow-up (No SQL Executed) ---
    if "B" in intent and chat_history:
        refine_prompt = f"""
        You are an expert A-Level Computer Science pedagogical assistant.
        
        Full Conversation History:
        {full_transcript}
        
        Teacher's Refinement Request: {user_question}
        
        CRITICAL FORMATTING INSTRUCTION:
        Do NOT output raw HTML tags (e.g., do NOT use <u>, <i>, <b>). Use standard Markdown bolding (**key term**) for emphasis.

        Directly address the request by building on the EXACT questions or text generated in the previous assistant message.
        Do NOT query new database terms. Maintain complete continuity with the previous output.
        """

        last_sql = chat_history[-1].get(
            "sql", "N/A (Refinement of previous response)"
        )
        last_df = chat_history[-1].get("df", pd.DataFrame())
        return (
            refine_prompt,
            last_df,
            f"-- [Refinement Task: Reusing previous context]\n-- Previous SQL:\n{last_sql}",
        )

    # --- PATH A: New Data Retrieval (Text-to-SQL for PostgreSQL) ---
    sql_prompt = f"""
    You are an expert PostgreSQL data analyst for an A-Level Computer Science department.
    Given the PostgreSQL table schema below and previous context, write a SINGLE valid, read-only SELECT query to answer the teacher's latest question.
    Use PostgreSQL syntax. Do NOT include markdown fences (like ```sql), code blocks, or commentary—output ONLY the plain SQL query text.

    Schema:
    {schema_info}

    Latest Question: {user_question}
    """

    response_obj = llm.invoke(sql_prompt)
    raw_content = response_obj.content
    raw_text = (
        "".join(
            [
                str(block.get("text", block)) if isinstance(block, dict) else str(block)
                for block in raw_content
            ]
        )
        if isinstance(raw_content, list)
        else str(raw_content)
    )

    sql_query = raw_text.strip()
    if sql_query.startswith("```"):
        sql_query = sql_query.split("\n", 1)[-1]
    if sql_query.endswith("```"):
        sql_query = sql_query.rsplit("```", 1)[0]
    sql_query = sql_query.replace("```sql", "").strip()

    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df_result = pd.read_sql_query(text(sql_query), conn)
    except Exception as e:
        return (
            f"❌ Query execution error: `{sql_query}`\n\nDetails: {str(e)}",
            None,
            sql_query,
        )

    synthesis_prompt = f"""
    You are a Head of Computer Science evaluating learning analytics and generating targeted A-Level revision resources.
    Synthesize the query results into a concise, professional response that directly fulfills the teacher's request.
    
    CRITICAL FORMATTING INSTRUCTION:
    Do NOT use HTML tags (e.g., <u>, <b>). Use standard Markdown bolding (**term**) for key terms.

    Teacher Question: {user_question}
    Query Results:
    {df_result.to_string(index=False)}
    """

    return synthesis_prompt, df_result, sql_query


# --- Render Active Chat Stream ---
for idx, entry in enumerate(st.session_state.ai_chat_history):
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(f"**AI Insight:**\n\n{entry['summary']}")

        st.download_button(
            label="📥 Download Generated Starter / Resource (.txt)",
            data=entry["summary"],
            file_name=f"retrieval_starter_{idx + 1}.txt",
            mime="text/plain",
            key=f"dl_{idx}",
        )

        with st.expander("📊 View SQL Query & Raw Data Table"):
            st.code(entry["sql"], language="sql")
            st.dataframe(entry["df"], use_container_width=True)

# --- Chat Input for Questions & Follow-ups ---
teacher_query = st.chat_input(
    "Ask an initial or follow-up question about A-Level cohort telemetry..."
)

if teacher_query:
    with st.chat_message("user"):
        st.write(teacher_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Supabase database..."):
            prompt_or_err, df_res, sql_used = ask_ai_about_data(
                teacher_query, st.session_state.ai_chat_history
            )

        if df_res is not None:
            st.markdown("**AI Insight:**")
            summary = st.write_stream(stream_ai_synthesis(prompt_or_err))

            st.download_button(
                label="📥 Download Generated Starter / Resource (.txt)",
                data=summary,
                file_name=f"retrieval_starter_{len(st.session_state.ai_chat_history) + 1}.txt",
                mime="text/plain",
                key=f"dl_live_{len(st.session_state.ai_chat_history)}",
            )

            with st.expander("📊 View SQL Query & Raw Data Table"):
                st.code(sql_used, language="sql")
                st.dataframe(df_res, use_container_width=True)

            st.session_state.ai_chat_history.append(
                {
                    "question": teacher_query,
                    "summary": summary,
                    "sql": sql_used,
                    "df": df_res,
                }
            )
        else:
            st.error(prompt_or_err)