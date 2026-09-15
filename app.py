import importlib
import json
import os
import re
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st

# Ensure API Key is set BEFORE loading socratic_fsm
if "GOOGLE_API_KEY" in st.secrets:
  os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Reload socratic_fsm FIRST to ensure newly added functions are picked up
import socratic_fsm

importlib.reload(socratic_fsm)

from socratic_fsm import (
    generate_extended_question,
    generate_layman_transformation_prompt,
    generate_quiz_questions,
    grade_disciplinary_rewrite,
    grade_extended_response,
    grade_quiz_responses,
    workflow,
)

# --- Load Dynamic Course Spec ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
  with open(SPEC_PATH, "r", encoding="utf-8") as f:
    COURSE_SPEC = json.load(f)
else:
  COURSE_SPEC = {}

COURSE_TITLE = COURSE_SPEC.get("course_title", "AQA GCSE Separate Sciences")
LEVEL = COURSE_SPEC.get("level", "GCSE")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 5)
SCIENCE_STRUCTURE = (
    COURSE_SPEC.get("subjects") or COURSE_SPEC.get("topics") or {}
)


# --- Helpers ---
def extract_clean_text(response) -> str:
  if isinstance(response, str):
    return response
  if hasattr(response, "content"):
    return extract_clean_text(response.content)
  if isinstance(response, list) and len(response) > 0:
    first_item = response[0]
    if isinstance(first_item, dict):
      return first_item.get("text", str(first_item))
    elif hasattr(first_item, "text"):
      return first_item.text
    return extract_clean_text(first_item)
  if isinstance(response, dict):
    if "text" in response:
      return response["text"]
    elif "content" in response:
      return extract_clean_text(response["content"])
  return str(response)


def clean_latex(text: str) -> str:
  text = re.sub(r"\$([^\$]+)\$", r"<b>\1</b>", text)
  return text.replace("$", "")


def md_to_html(text: str) -> str:
  text = re.sub(
      r"^####\s+(.*$)",
      r'<h4 style="margin: 4px 0 1px 0; font-size: 1.05em; color:'
      r' inherit;">\1</h4>',
      text,
      flags=re.MULTILINE,
  )
  text = re.sub(
      r"^###\s+(.*$)",
      r'<h3 style="margin: 6px 0 2px 0; font-size: 1.1em; color:'
      r' inherit;">\1</h3>',
      text,
      flags=re.MULTILINE,
  )
  text = re.sub(
      r"^##\s+(.*$)",
      r'<h2 style="margin: 8px 0 2px 0; font-size: 1.2em; color:'
      r' inherit;">\1</h2>',
      text,
      flags=re.MULTILINE,
  )
  text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
  text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
  text = re.sub(
      r"^\s*[-*]\s+(.*$)",
      r'<div style="margin: 1px 0;">• \1</div>',
      text,
      flags=re.MULTILINE,
  )
  text = re.sub(r"(</(div|h2|h3|h4)>)\s*\n+", r"\1", text)
  text = re.sub(r"\n{2,}", "<br>", text)
  text = text.replace("\n", "<br>")
  return re.sub(r"(<br\s*/?>\s*)+", "<br>", text)


# --- CSS Styling ---
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    div[data-testid="stProgress"] > div > div > div { background-color: #2563eb !important; }
    h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 700; }
    .chat-header { 
        background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); 
        color: white; padding: 22px; font-weight: 700; text-align: center; 
        font-size: 1.3em; border-radius: 16px; box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 24px;
    }
    .selection-card {
        background: #ffffff; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0; margin-bottom: 20px;
    }
    .tutor-msg { 
        background-color: #ffffff; color: #1e293b; padding: 14px 18px; 
        border-radius: 18px 18px 18px 4px; margin-bottom: 12px; max-width: 82%; 
        line-height: 1.35; border: 1px solid #e2e8f0;
    }
    .student-msg { 
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
        color: white; padding: 14px 18px; border-radius: 18px 18px 4px 18px; 
        margin-bottom: 12px; max-width: 82%; margin-left: auto; line-height: 1.35;
    }
    .summary-box { 
        background: #fefce8; border-left: 5px solid #eab308; padding: 10px 14px; 
        border-radius: 12px; color: #713f12; font-size: 0.93em; margin: 8px 0; max-width: 85%; 
    }
    button[kind="primary"] {
        background-color: #ff4b4b !important;
        border-color: #ff4b4b !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Session State ---
if "active_subject" not in st.session_state:
  st.session_state.active_subject = None
if "active_topic" not in st.session_state:
  st.session_state.active_topic = None
if "active_subtopic" not in st.session_state:
  st.session_state.active_subtopic = None
if "messages" not in st.session_state:
  st.session_state.messages = []

if "graph_state" not in st.session_state:
  st.session_state.graph_state = {
      "messages": [],
      "sub_topic": None,
      "turn_count": 0,
      "is_final_turn": False,
  }

if "app_mode" not in st.session_state:
  st.session_state.app_mode = None  # "socratic", "quiz", "extended", "rewrite"
if "quiz_questions" not in st.session_state:
  st.session_state.quiz_questions = []
if "quiz_results" not in st.session_state:
  st.session_state.quiz_results = None

# Extended Question State
if "extended_question" not in st.session_state:
  st.session_state.extended_question = None
if "extended_results" not in st.session_state:
  st.session_state.extended_results = None

# Rewrite Answer State
if "rewrite_data" not in st.session_state:
  st.session_state.rewrite_data = None
if "rewrite_results" not in st.session_state:
  st.session_state.rewrite_results = None


def reset_session():
  st.session_state.app_mode = None
  st.session_state.quiz_questions = []
  st.session_state.quiz_results = None
  st.session_state.extended_question = None
  st.session_state.extended_results = None
  st.session_state.rewrite_data = None
  st.session_state.rewrite_results = None
  st.session_state.active_subject = None
  st.session_state.active_topic = None
  st.session_state.active_subtopic = None
  st.session_state.messages = []
  st.session_state.graph_state = {
      "messages": [],
      "sub_topic": None,
      "turn_count": 0,
      "is_final_turn": False,
  }
  st.rerun()


# --- Screen Router ---

# 1. Selection Screen with 4 Target Buttons
if st.session_state.active_subtopic is None:
  st.markdown(
      f'<div class="chat-header">🎓 {COURSE_TITLE} Socratic Coach</div>',
      unsafe_allow_html=True,
  )

  # App Purpose banner
  st.info(
      "💡 **App Purpose:** The aim of this app is to help you develop your use"
      " of disciplinary literacy. To gain a top grade, you need to talk and"
      " write like a Scientist!"
  )

  # Native subheader and target selections (no unclosed div wrapper)
  st.subheader("🎯 Select Revision Target")
  st.write(
      "Choose a subject, topic, and subtopic to begin your practice session:"
  )

  subjects = (
      list(SCIENCE_STRUCTURE.keys())
      if isinstance(SCIENCE_STRUCTURE, dict)
      else []
  )
  selected_subject = st.selectbox(
      "🔬 Step 1: Choose Science Subject:", options=subjects
  )

  topics_dict = (
      SCIENCE_STRUCTURE.get(selected_subject, {}) if selected_subject else {}
  )
  if isinstance(topics_dict, dict):
    topic_options = list(topics_dict.keys())
  elif isinstance(topics_dict, list):
    topic_options = topics_dict
    topics_dict = {t: [t] for t in topics_dict}
  else:
    topic_options = []

  selected_topic = st.selectbox(
      "📘 Step 2: Choose Unit / Topic:", options=topic_options
  )

  if isinstance(topics_dict, dict) and selected_topic in topics_dict:
    subtopics = topics_dict[selected_topic]
    if not isinstance(subtopics, list):
      subtopics = [str(subtopics)]
  else:
    subtopics = [selected_topic] if selected_topic else []

  selected_subtopic = st.selectbox(
      "🔍 Step 3: Choose Specific Subtopic:", options=subtopics
  )

  # Button Group
  if st.button(
      "🚀 Start a Socratic Session",
      type="primary",
      use_container_width=True,
      disabled=not selected_subtopic,
  ):
    st.session_state.app_mode = "socratic"
    st.session_state.active_subject = selected_subject
    st.session_state.active_topic = selected_topic
    st.session_state.active_subtopic = selected_subtopic
    st.session_state.graph_state["sub_topic"] = (
        f"{selected_subject} - {selected_topic}: {selected_subtopic}"
    )
    st.rerun()

  if st.button(
      "📝 Take a Retrieval Quiz",
      use_container_width=True,
      disabled=not selected_subtopic,
  ):
    st.session_state.app_mode = "quiz"
    st.session_state.active_subject = selected_subject
    st.session_state.active_topic = selected_topic
    st.session_state.active_subtopic = selected_subtopic
    full_topic_name = (
        f"{selected_subject} - {selected_topic}: {selected_subtopic}"
    )
    with st.spinner("Generating 10 specification retrieval questions..."):
      st.session_state.quiz_questions = generate_quiz_questions(
          full_topic_name, COURSE_TITLE, LEVEL
      )
    st.rerun()

  if st.button(
      "🧠 Answer an Extended Question",
      use_container_width=True,
      disabled=not selected_subtopic,
  ):
    st.session_state.app_mode = "extended"
    st.session_state.active_subject = selected_subject
    st.session_state.active_topic = selected_topic
    st.session_state.active_subtopic = selected_subtopic
    full_topic_name = (
        f"{selected_subject} - {selected_topic}: {selected_subtopic}"
    )
    with st.spinner("Generating high-tier extended response scenario..."):
      st.session_state.extended_question = generate_extended_question(
          full_topic_name, COURSE_TITLE, LEVEL
      )
    st.rerun()

  if st.button(
      "✍️ Rewrite an Answer",
      use_container_width=True,
      disabled=not selected_subtopic,
  ):
    st.session_state.app_mode = "rewrite"
    st.session_state.active_subject = selected_subject
    st.session_state.active_topic = selected_topic
    st.session_state.active_subtopic = selected_subtopic
    full_topic_name = (
        f"{selected_subject} - {selected_topic}: {selected_subtopic}"
    )
    with st.spinner("Generating informal response scenario..."):
      st.session_state.rewrite_data = generate_layman_transformation_prompt(
          full_topic_name, COURSE_TITLE, LEVEL
      )
    st.rerun()

# 2. Retrieval Quiz View (Forms-based)
elif st.session_state.app_mode == "quiz":
  st.markdown(
      f'<div class="chat-header">📝 {COURSE_TITLE} Retrieval Quiz</div>',
      unsafe_allow_html=True,
  )

  with st.sidebar:
    st.subheader("📌 Active Target")
    st.info(
        f"**Subject:** {st.session_state.active_subject}\n\n**Topic:**"
        f" {st.session_state.active_topic}\n\n**Subtopic:**"
        f" {st.session_state.active_subtopic}"
    )
    st.write("---")
    if st.button("🔄 New Session / Change Topic", use_container_width=True):
      reset_session()

  if not st.session_state.quiz_results:
    with st.form("retrieval_quiz_form"):
      st.subheader(
          f"Short Retrieval Quiz: {st.session_state.active_subtopic} (10"
          " Marks)"
      )
      st.write(
          "Answer all questions precisely using exact specification keywords."
      )

      user_answers = {}
      for i, q in enumerate(st.session_state.quiz_questions):
        user_answers[i] = st.text_area(
            f"**Q{i+1}: {q}**",
            key=f"q_{i}",
            height=100,
            placeholder="Type key terms and precise specification definition...",
        )

      submitted = st.form_submit_button(
          "Submit Quiz for Examination",
          type="primary",
          use_container_width=True,
      )

      if submitted:
        full_topic_name = (
            f"{st.session_state.active_subject} -"
            f" {st.session_state.active_topic}:"
            f" {st.session_state.active_subtopic}"
        )
        with st.spinner("Grading against Mark Scheme keywords..."):
          results = grade_quiz_responses(
              full_topic_name,
              st.session_state.quiz_questions,
              user_answers,
              COURSE_TITLE,
              LEVEL,
          )
          st.session_state.quiz_results = results
        st.rerun()

  else:
    results = st.session_state.quiz_results
    score = results.get("total_score", 0)

    st.success(
        f"### 🎉 Quiz Complete! Total Score: {score} / 10\nReview your keyword"
        " accuracy breakdown below:"
    )

    for item in results.get("breakdown", []):
      with st.expander(
          f"Q{item['question_num']}: {item['question']} — Score:"
          f" {item['score']}/1"
      ):
        st.markdown(f"**Your Answer:**\n> {item['student_answer']}")
        st.write(f"**Model Answer:** {item['model_answer']}")
        st.write(f"**Key Terms Used:** {', '.join(item['keywords_used'])}")
        st.write(f"**Missed Keywords:** {', '.join(item['keywords_missed'])}")
        st.info(f"**Examiner Note:** {item['explanation']}")

    if st.button("Try Another Topic", type="primary"):
      reset_session()

# 3. Productive Struggle & Disciplinary Language View
elif st.session_state.app_mode == "extended":
  st.markdown(
      f'<div class="chat-header">🧠 {COURSE_TITLE} Disciplinary Language'
      " Challenge</div>",
      unsafe_allow_html=True,
  )

  with st.sidebar:
    st.subheader("📌 Active Target")
    st.info(
        f"**Subject:** {st.session_state.active_subject}\n\n**Topic:**"
        f" {st.session_state.active_topic}\n\n**Subtopic:**"
        f" {st.session_state.active_subtopic}"
    )
    st.write("---")
    if st.button("🔄 New Session / Change Topic", use_container_width=True):
      reset_session()

  if not st.session_state.extended_results:
    with st.form("extended_question_form"):
      st.subheader("Extended Answer Challenge (6 Marks)")
      st.markdown(
          f"### 📋 Question:\n**{st.session_state.extended_question}**"
      )
      st.info(
          "💡 **Productive Struggle Focus:** Write a detailed explanation."
          " Focus on cause-and-effect reasoning and credit-bearing"
          " specification terminology."
      )

      student_entry = st.text_area(
          "Your Response:",
          height=220,
          placeholder=(
              "Construct your explanation using precise domain terms..."
          ),
      )

      submitted = st.form_submit_button(
          "Submit Response for Evaluation",
          type="primary",
          use_container_width=True,
      )

      if submitted:
        full_topic_name = (
            f"{st.session_state.active_subject} -"
            f" {st.session_state.active_topic}:"
            f" {st.session_state.active_subtopic}"
        )
        with st.spinner(
            "Evaluating disciplinary terminology and reasoning..."
        ):
          res = grade_extended_response(
              full_topic_name,
              st.session_state.extended_question,
              student_entry,
              COURSE_TITLE,
              LEVEL,
          )
          st.session_state.extended_results = res
        st.rerun()

  else:
    res = st.session_state.extended_results
    st.success(
        f"### 📊 Assessment Complete! Score: {res.get('score', 0)} /"
        f" {res.get('max_score', 6)}"
    )

    st.markdown(
        "**Disciplinary Mastery Level:**"
        f" `{res.get('disciplinary_level', 'N/A')}`"
    )

    col1, col2 = st.columns(2)
    with col1:
      st.markdown("#### ✅ Specification Terms Used")
      st.write(", ".join(res.get("keywords_used", [])) or "None detected")
    with col2:
      st.markdown("#### ⚠️ Key Terms Missed")
      st.write(", ".join(res.get("keywords_missed", [])) or "None")

    st.write("---")
    st.markdown("### 🔍 Detailed Feedback & Guidance")
    st.write(f"**Strengths:** {res.get('strengths')}")
    st.info(f"**Productive Struggle Advice:** {res.get('struggle_advice')}")

    with st.expander("📖 Exemplar Specification Model Answer"):
      st.write(res.get("model_answer"))

    if st.button("Try Another Topic", type="primary"):
      reset_session()

# 4. Disciplinary Language Upgrade (Rewrite the Answer)
elif st.session_state.app_mode == "rewrite":
  st.markdown(
      f'<div class="chat-header">✍️ {COURSE_TITLE} Disciplinary Language'
      " Upgrade</div>",
      unsafe_allow_html=True,
  )

  with st.sidebar:
    st.subheader("📌 Active Target")
    st.info(
        f"**Subject:** {st.session_state.active_subject}\n\n**Topic:**"
        f" {st.session_state.active_topic}\n\n**Subtopic:**"
        f" {st.session_state.active_subtopic}"
    )
    st.write("---")
    if st.button("🔄 New Session / Change Topic", use_container_width=True):
      reset_session()

  data = st.session_state.rewrite_data or {}
  question = data.get("question", "Question missing")
  layman_answer = data.get("layman_answer", "Layman text missing")

  if not st.session_state.rewrite_results:
    st.subheader("Rewrite the Answer (4 Marks)")
    st.markdown(f"### 📋 Question:\n**{question}**")
    st.write("---")

    # Row 1: Informal Draft (Top)
    st.subheader("💬 Informal Draft (Layman's Terms)")
    st.warning(f'"{layman_answer}"')
    st.info(
        "💡 **Task:** Read the draft above. Rewrite it using precise GCSE"
        " scientific key terms, correct phrasing, and formal academic tone."
    )

    st.write("")

    # Row 2: Student Scientific Rewrite Input (Stacked Below for Mobile)
    st.subheader("🔬 Your Scientific Rewrite")
    with st.form("rewrite_answer_form"):
      student_rewrite = st.text_area(
          "Write your upgraded response here:",
          height=180,
          placeholder=(
              "Type your response using precise disciplinary language..."
          ),
      )

      submitted = st.form_submit_button(
          "Submit Scientific Rewrite",
          type="primary",
          use_container_width=True,
      )

      if submitted and student_rewrite.strip():
        full_topic_name = (
            f"{st.session_state.active_subject} -"
            f" {st.session_state.active_topic}:"
            f" {st.session_state.active_subtopic}"
        )
        with st.spinner("Evaluating disciplinary upgrade..."):
          res = grade_disciplinary_rewrite(
              full_topic_name,
              question,
              layman_answer,
              student_rewrite,
              COURSE_TITLE,
              LEVEL,
          )
          st.session_state.rewrite_results = res
        st.rerun()

  else:
    res = st.session_state.rewrite_results
    st.success(
        f"### 📊 Assessment Complete! Score: {res.get('score', 0)} /"
        f" {res.get('max_score', 4)}"
    )

    st.write(f"**Examiner Feedback:** {res.get('feedback')}")

    # Results stacked for mobile
    st.markdown("#### ✅ Scientific Terms Included")
    st.write(", ".join(res.get("key_terms_used", [])) or "None detected")

    st.markdown("#### ⚠️ Key Terms to Add Next Time")
    st.write(", ".join(res.get("missed_terms", [])) or "None")

    if st.button("Try Another Topic", type="primary"):
      reset_session()

# 5. Socratic Dialogue View
else:
  st.markdown(
      f'<div class="chat-header">🎓 {COURSE_TITLE} Coach</div>',
      unsafe_allow_html=True,
  )

  student_turns = sum(
      1 for m in st.session_state.messages if m.get("role") == "student"
  )

  with st.sidebar:
    st.subheader("📌 Active Target")
    st.info(
        f"**Subject:** {st.session_state.active_subject}\n\n**Topic:**"
        f" {st.session_state.active_topic}\n\n**Subtopic:**"
        f" {st.session_state.active_subtopic}"
    )

    st.metric(label="Turn Counter", value=f"{student_turns} / {TARGET_TURNS}")
    st.progress(min(student_turns / TARGET_TURNS, 1.0))

    st.write("---")
    if st.button("🔄 New Session / Change Topic", use_container_width=True):
      reset_session()

  if len(st.session_state.messages) == 0:
    initial_greeting = (
        f"Welcome! We're exploring **{st.session_state.active_subtopic}** today."
        " To get started, what core concept or term in this topic would you"
        " like to review?"
    )
    st.session_state.messages.append(
        {"role": "tutor", "content": initial_greeting, "style": "tutor-msg"}
    )
    st.session_state.graph_state["messages"].append(
        AIMessage(content=initial_greeting)
    )

  for msg in st.session_state.messages:
    html_content = md_to_html(msg["content"])
    if msg["role"] == "tutor":
      div_class = msg.get("style", "tutor-msg")
      header = (
          "💡 <b>Summary Note</b>"
          if div_class == "summary-box"
          else "🎓 <b>Tutor</b>"
      )
      st.markdown(
          f'<div class="{div_class}">{header}<br><br>{html_content}</div>',
          unsafe_allow_html=True,
      )
    else:
      st.markdown(
          '<div class="student-msg">🎒'
          f" <b>Student</b><br><br>{html_content}</div>",
          unsafe_allow_html=True,
      )

  if student_turns >= TARGET_TURNS:
    st.info(
        f"🎉 **Session Complete!** You completed all {TARGET_TURNS} turns of"
        f" the {LEVEL} Socratic dialogue."
    )

  is_disabled = student_turns >= TARGET_TURNS
  placeholder = (
      "Session complete. Select a new topic in the sidebar."
      if is_disabled
      else "Type your response here..."
  )

  if user_input := st.chat_input(placeholder, disabled=is_disabled):
    st.session_state.messages.append({"role": "student", "content": user_input})
    st.session_state.graph_state["messages"].append(
        HumanMessage(content=user_input)
    )

    current_student_turns = sum(
        1 for m in st.session_state.messages if m.get("role") == "student"
    )
    st.session_state.graph_state["turn_count"] = current_student_turns
    st.session_state.graph_state["is_final_turn"] = (
        current_student_turns >= TARGET_TURNS
    )

    with st.spinner("Analyzing response and generating feedback..."):
      input_payload = {
          "messages": st.session_state.graph_state["messages"],
          "sub_topic": (
              f"{st.session_state.active_subject} -"
              f" {st.session_state.active_topic}:"
              f" {st.session_state.active_subtopic}"
          ),
          "turn_count": current_student_turns,
          "is_final_turn": current_student_turns >= TARGET_TURNS,
      }
      updated_state = workflow.invoke(input_payload)

    last_msg = updated_state["messages"][-1]
    ai_reply = clean_latex(extract_clean_text(last_msg))

    split_match = re.split(
        r"={3,}\s*SPLIT\s*={3,}", ai_reply, flags=re.IGNORECASE
    )
    if len(split_match) > 1:
      st.session_state.messages.append({
          "role": "tutor",
          "content": split_match[0].strip(),
          "style": "tutor-msg",
      })
      st.session_state.messages.append({
          "role": "tutor",
          "content": split_match[1].strip(),
          "style": "summary-box",
      })
    else:
      st.session_state.messages.append(
          {"role": "tutor", "content": ai_reply, "style": "tutor-msg"}
      )

    st.session_state.graph_state = updated_state
    st.rerun()