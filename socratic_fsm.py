import json
import os
import re
from typing import List, TypedDict

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

# --- Load dynamic course specification ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
  with open(SPEC_PATH, "r", encoding="utf-8") as f:
    COURSE_SPEC = json.load(f)
else:
  COURSE_SPEC = {}

COURSE_TITLE = COURSE_SPEC.get("course_title", "AQA GCSE Separate Sciences")
LEVEL = COURSE_SPEC.get("level", "GCSE")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 5)


class ChatState(TypedDict, total=False):
  messages: List[BaseMessage]
  sub_topic: str
  turn_count: int
  is_final_turn: bool


def get_context(sub_topic: str, user_query: str) -> str:
  try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    results = db.similarity_search(user_query, k=3)
    return "\n\n".join([doc.page_content for doc in results]) if results else ""
  except Exception:
    return "No specific syllabus context found."


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


# --- Socratic Nodes ---
def socratic_tutor(state: ChatState) -> dict:
  sub_topic = state.get("sub_topic", COURSE_TITLE)
  user_query = state["messages"][-1].content if state.get("messages") else ""
  context = get_context(sub_topic, user_query)

  system_prompt = f"""You are an expert Socratic {COURSE_TITLE} ({LEVEL}) Tutor.
    Topic Focus: {sub_topic}
    Syllabus Context:
    {context}

    Guide the student step-by-step using probing questions and constructive hints. Never give away full answers directly."""

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
  messages_to_send = [SystemMessage(content=system_prompt)] + list(
      state["messages"]
  )
  response = llm.invoke(messages_to_send)

  return {"messages": state["messages"] + [response]}


def didactic_fallback(state: ChatState) -> dict:
  sub_topic = state.get("sub_topic", "")
  user_query = state["messages"][-1].content if state.get("messages") else ""
  context = get_context(sub_topic, user_query)

  system_prompt = f"""You are a strict {COURSE_TITLE} ({LEVEL}) Senior Examiner wrapping up a Socratic revision session.
    Topic Focus: {sub_topic}
    Syllabus Context:
    {context}

    CRITICAL MANDATE: This is the FINAL turn. You MUST NOT ask any follow-up questions. Conclude immediately and provide the performance assessment and summary card.

    STRICT FORMATTING & LATEX RULES:
    - NEVER use LaTeX math delimiters like $, $$, \\(, or \\). Write all complexity, matrices, and variables in plain text or Markdown bold/code.

    OBJECTIVE MARKING RUBRIC:
    1. Base accuracy strictly on exact specification keyword hit-rate derived from the Syllabus Context.
    2. Ignore Setup Words: Do not count the initial topic name chosen by the student as a keyword hit.
    3. Strict Terminology: Only credit official domain terms found in the context. Layperson words get 0% keyword credit.
    4. Misconception Penalty: Cap the overall score at 20% maximum if the student expresses a fundamental factual error.

    INSTRUCTIONS FOR SESSION ENDING:
    1. **Validate Final Answer:** Directly validate the student's final input in detail first.
    2. **Performance Assessment:** Apply the rubric above, list technical terms used well vs. missed across the dialogue, give 1–2 targeted feedback points, and recommend next sub-topics.
    3. **Structured Summary Note:** Provide a clean, comprehensive topic summary data drop.

    FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS (Include the exact separator string ===SPLIT=== on its own line):

    [Your validation of the student's final answer and corrected solution]

    ### 📊 Session Performance
    - **Overall Accuracy:** [X]%
    - **Targeted Feedback:** 
      - [Constructive feedback point 1]
      - [Constructive feedback point 2]
    - **Exam Terminology:**
      - **Keywords Used Well:** [Term 1, Term 2]
      - **Missed Terms to Learn:** [Term 3, Term 4]
    - **Recommended Next Revision Topic(s):** [Sub-topic 1 / Sub-topic 2]

    ===SPLIT===

    ### 💡 Topic Summary: [Topic Name]
    [Detailed structured summary data drop in plain text/Markdown only, NO $ symbols]

    CRITICAL RULE: DO NOT ask any follow-up questions anywhere in your response. Conclude cleanly."""

  final_command = HumanMessage(
      content=(
          f"[SYSTEM DIRECTIVE: This is turn {TARGET_TURNS} (FINAL TURN). Do NOT"
          " ask any follow-up questions. Provide the final answer validation,"
          " performance evaluation, the exact ===SPLIT=== delimiter, and"
          " topic summary now.]"
      )
  )

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
  messages_to_send = (
      [SystemMessage(content=system_prompt)]
      + list(state["messages"])
      + [final_command]
  )
  response = llm.invoke(messages_to_send)

  return {"messages": state["messages"] + [response]}


def route_turn(state: ChatState) -> str:
  if (
      state.get("is_final_turn", False)
      or state.get("turn_count", 0) >= TARGET_TURNS
  ):
    return "didactic_fallback"

  messages = state.get("messages", [])
  human_count = sum(
      1
      for m in messages
      if getattr(m, "type", None) == "human"
      or "Human" in m.__class__.__name__
      or isinstance(m, HumanMessage)
  )

  if human_count >= TARGET_TURNS:
    return "didactic_fallback"

  return "socratic_tutor"


builder = StateGraph(ChatState)
builder.add_node("socratic_tutor", socratic_tutor)
builder.add_node("didactic_fallback", didactic_fallback)

builder.set_conditional_entry_point(
    route_turn,
    {
        "socratic_tutor": "socratic_tutor",
        "didactic_fallback": "didactic_fallback",
    },
)

builder.add_edge("socratic_tutor", END)
builder.add_edge("didactic_fallback", END)

workflow = builder.compile()


# --- Standalone Quiz Helpers ---
def generate_quiz_questions(
    sub_topic: str, course_title: str, level: str
) -> list:
  context = get_context(sub_topic, sub_topic)
  prompt = f"""You are an expert {course_title} ({level}) Senior Examiner.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

Generate exactly 10 short-answer exam questions testing precise definitions and technical terminology for this subtopic.
Output ONLY a valid JSON array of 10 question strings, with no additional text or formatting:
["Question 1 text...", "Question 2 text...", ...]"""

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)
  response = llm.invoke([HumanMessage(content=prompt)])

  raw_text = extract_clean_text(response)
  clean_text = re.sub(r"```json|```", "", raw_text).strip()
  return json.loads(clean_text)


def grade_quiz_responses(
    sub_topic: str, questions: list, answers: dict, course_title: str, level: str
) -> dict:
  context = get_context(sub_topic, sub_topic)
  qa_pairs = "\n".join([
      f"Q{i+1}: {q}\nStudent Answer: {answers.get(i, 'No answer provided')}\n"
      for i, q in enumerate(questions)
  ])

  prompt = f"""You are a strict {course_title} ({level}) Senior Examiner grading a 10-question retrieval quiz.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

STRICT MARKING RUBRIC:
- Base accuracy strictly on exact specification keywords derived from Syllabus Context.
- Award 1 mark per question ONLY if exact domain terms are present. Layperson terms get 0 marks.
- Provide concise, actionable feedback focusing on missing exam terminology.

Return your assessment strictly as a single JSON object with no extra commentary:
{{
  "total_score": 8,
  "breakdown": [
    {{
      "question_num": 1,
      "question": "Question text...",
      "student_answer": "Student text...",
      "score": 1,
      "model_answer": "Model specification definition...",
      "keywords_used": ["Term 1"],
      "keywords_missed": ["Term 2"],
      "explanation": "Short sentence explaining mark allocation..."
    }}
  ]
}}

Student Submission:
{qa_pairs}"""

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
  response = llm.invoke([HumanMessage(content=prompt)])

  raw_text = extract_clean_text(response)
  clean_text = re.sub(r"```json|```", "", raw_text).strip()
  return json.loads(clean_text)


# --- Disciplinary Language / Productive Struggle Helpers ---
def generate_extended_question(
    sub_topic: str, course_title: str, level: str
) -> str:
  """Generates a high-order, multi-mark extended answer question requiring disciplinary language."""
  context = get_context(sub_topic, sub_topic)
  prompt = f"""You are an expert {course_title} ({level}) Senior Examiner designing an extended-response exam question.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

Generate ONE high-tier 6-mark extended explanation question (e.g., "Describe and explain...", "Evaluate the impact of...") that forces students to apply precise disciplinary terminology, cause-and-effect reasoning, and domain vocabulary.
Return ONLY the question text as a plain string."""

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.4)
  response = llm.invoke([HumanMessage(content=prompt)])
  return extract_clean_text(response).strip()


def grade_extended_response(
    sub_topic: str,
    question: str,
    student_answer: str,
    course_title: str,
    level: str,
) -> dict:
  """Grades an extended response focusing on disciplinary language, keyword usage, and conceptual accuracy."""
  context = get_context(sub_topic, sub_topic)
  prompt = f"""You are a strict {course_title} ({level}) Senior Examiner marking a 6-mark extended-response question.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

Question: {question}
Student Submission: {student_answer}

STRICT DISCIPLINARY LANGUAGE RUBRIC:
- Grade out of 6 marks based on scientific accuracy and use of domain-specific vocabulary.
- Award 0 marks for vague layperson explanations.
- Identify exact specification keywords successfully used and critical missing terms.
- Provide actionable advice on improving scientific phrasing and productive struggle strategies.

Return your evaluation strictly as a single JSON object:
{{
  "score": 5,
  "max_score": 6,
  "disciplinary_level": "Developing / Proficient / Master",
  "keywords_used": ["Term 1", "Term 2"],
  "keywords_missed": ["Term 3", "Term 4"],
  "model_answer": "Comprehensive 6-mark model answer...",
  "strengths": "Feedback on terminology used well...",
  "struggle_advice": "Actionable guidance on refining disciplinary language and cause-and-effect structure..."
}}"""

  llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
  response = llm.invoke([HumanMessage(content=prompt)])

  raw_text = extract_clean_text(response)
  clean_text = re.sub(r"```json|```", "", raw_text).strip()
  return json.loads(clean_text)