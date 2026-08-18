import os, json
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END

# --- Load dynamic course specification ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {}

COURSE_TITLE = COURSE_SPEC.get("course_title", "General Subject")
LEVEL = COURSE_SPEC.get("level", "GCSE/A-Level")
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
    messages_to_send = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages_to_send)

    return {
        "messages": state["messages"] + [response]
    }


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

    # Trailing message forces Gemini to stop questioning and render the summary
    final_command = HumanMessage(
        content="[SYSTEM DIRECTIVE: This is turn 5 (FINAL TURN). Do NOT ask any follow-up questions. Provide the final answer validation, performance evaluation, the exact ===SPLIT=== delimiter, and topic summary now.]"
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    messages_to_send = [SystemMessage(content=system_prompt)] + list(state["messages"]) + [final_command]
    response = llm.invoke(messages_to_send)

    return {
        "messages": state["messages"] + [response]
    }


def route_turn(state: ChatState) -> str:
    if state.get("is_final_turn", False) or state.get("turn_count", 0) >= TARGET_TURNS:
        return "didactic_fallback"

    messages = state.get("messages", [])
    human_count = sum(1 for m in messages if getattr(m, "type", None) == "human" or "Human" in m.__class__.__name__ or isinstance(m, HumanMessage))

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
    }
)

builder.add_edge("socratic_tutor", END)
builder.add_edge("didactic_fallback", END)

workflow = builder.compile()