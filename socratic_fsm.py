import json
import os
import re
from typing import Any, Dict, List, TypedDict

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, StateGraph

# --- Robust Global API Key Retrieval for Streamlit Cloud & Local ---
def _get_google_api_key():
    key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if isinstance(key, dict) or hasattr(key, "get"):
        return key.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key

API_KEY = _get_google_api_key()

# --- Load dynamic course specification ---
SPEC_PATH = "course_spec.json"
if os.path.exists(SPEC_PATH):
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        COURSE_SPEC = json.load(f)
else:
    COURSE_SPEC = {}

COURSE_TITLE = COURSE_SPEC.get("course_title", "AQA GCSE Separate Sciences")
SUBJECT = COURSE_SPEC.get("subject", "Science")
LEVEL = COURSE_SPEC.get("level", "GCSE")
TARGET_TURNS = COURSE_SPEC.get("target_turns", 5)

# Primary database table target for Science telemetry
DB_TABLE_NAME = "activity_logs_science"


class ChatState(TypedDict, total=False):
    messages: List[BaseMessage]
    sub_topic: str
    turn_count: int
    is_final_turn: bool
    is_freestyle: bool


# --- PERFORMANCE OPTIMIZATION: CACHED MODELS & VECTOR DB ---
@st.cache_resource
def get_embeddings():
    """Cache the HuggingFace embedding model in memory to avoid reload latency."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_vector_db():
    """Cache persistent ChromaDB instance for Science across session renders."""
    embeddings = get_embeddings()
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


@st.cache_resource
def get_llm(temperature: float = 0.2, streaming: bool = False):
    """Cache Gemini client instances to avoid repeated connection setup."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=API_KEY,
        temperature=temperature,
        streaming=streaming
    )


def get_context(sub_topic: str, user_query: str) -> str:
    """Retrieves relevant specification chunks from local ChromaDB store."""
    try:
        db = get_vector_db()
        results = db.similarity_search(f"{sub_topic} {user_query}", k=3)
        return "\n\n".join([doc.page_content for doc in results]) if results else ""
    except Exception:
        return "No specific syllabus context found."


def extract_clean_text(response) -> str:
    """Extracts clean text strings from LLM response payload objects."""
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


def verify_topic_alignment(custom_prompt: str) -> Dict[str, Any]:
    """Guardrail: Fast check ensuring custom science prompts align with curriculum scope."""
    context = get_context(custom_prompt, custom_prompt)
    prompt = f"""You are an expert {COURSE_TITLE} ({LEVEL}) examiner.
Verify if the following student question or topic aligns with the specification or broader science curriculum scope.

Student Custom Prompt: "{custom_prompt}"
Syllabus Context snippet:
{context}

Return ONLY a JSON object:
{{
  "is_aligned": true,
  "reasoning": "Brief note on how it connects to core scientific processes, concepts, formulas, or experimental methods.",
  "suggested_pivot": "If not aligned, suggest an aligned topic/question close to their entry."
}}"""

    llm = get_llm(temperature=0.1, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    clean_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", clean_text).strip()
    return json.loads(clean_text)


# --- Live Streaming Generators ---
def stream_socratic_response(messages: list, sub_topic: str):
    """Streams response tokens directly using cached streaming client."""
    user_query = messages[-1].content if messages else ""
    context = get_context(sub_topic, user_query)

    system_prompt = f"""You are an expert Socratic {COURSE_TITLE} ({LEVEL}) Tutor.
Target Revision Topic: {sub_topic}
Syllabus Context:
{context}

CRITICAL TOPIC BOUNDARY RULE:
- The student MUST stay focused on the Target Revision Topic: '{sub_topic}'.
- If the student attempts to switch to an unrelated topic:
  1. Politely acknowledge their input.
  2. Clarify that today's revision focus is strictly on **{sub_topic}**.
  3. Pivot the conversation back by connecting their comment to **{sub_topic}** (if a logical link exists) OR explicitly redirect them with a probing question about **{sub_topic}**.

PEDAGOGICAL & DISCIPLINARY LITERACY DIRECTIVES:
- Guide the student step-by-step using probing questions and constructive hints. Never give away full answers directly.
- Command-Word Compliance:
  * "Describe": Require students to state WHAT happens step-by-step or identify trends.
  * "Explain": Require students to state WHY or HOW something happens using cause-and-effect language ("because", "this causes", "therefore").
  * "Evaluate": Require students to balance pros and cons before making a conclusion.
- Disciplinary Precision: Enforce exact specification vocabulary, accurate mathematical/unit notation, and correct underlying scientific mechanisms while trapping common misconceptions."""

    llm = get_llm(temperature=0.2, streaming=True)
    messages_to_send = [SystemMessage(content=system_prompt)] + list(messages)

    for chunk in llm.stream(messages_to_send):
        if chunk.content:
            yield chunk.content


def stream_freestyle_socratic_response(messages: list, custom_prompt: str, focus_mode: str = "Scientific Explanations & Cause/Effect"):
    """Streams response tokens for uncapped Socratic sessions using cached streaming client."""
    user_query = messages[-1].content if messages else ""
    context = get_context(custom_prompt, user_query)

    if "Keywords" in focus_mode or "Vocabulary" in focus_mode:
        pedagogical_focus = """
        PEDAGOGICAL FOCUS: KEYWORDS & TECHNICAL VOCABULARY
        - Focus on developing precise domain terminology, key definitions, proper SI units, formulas, and structural representations.
        - Gently challenge informal, vague, or layperson language; ask the student to rephrase explanations using precise scientific terms.
        - Conduct a conversational check to reinforce conceptual retention.
        """
    else:
        pedagogical_focus = """
        PEDAGOGICAL FOCUS: SCIENTIFIC EXPLANATIONS & CAUSE/EFFECT
        - Focus on helping the student articulate multi-step scientific mechanisms (e.g., 'because...', 'this causes...', 'resulting in...').
        - Guide them to structure explanations logically: State the cause, explain the mechanism, and describe the outcome or effect.
        - Encourage them to make connections between scientific principles and real-world phenomena or experimental data.
        """

    system_prompt = f"""You are an expert Socratic Disciplinary Literacy Coach for {COURSE_TITLE} ({LEVEL}).
Student Focus Topic / Question: {custom_prompt}
Relevant Specification Context:
{context}

{pedagogical_focus}

CRITICAL RULES:
- DO NOT replicate or mimic standard exam board questions or mark schemes.
- Guide the student step-by-step using probing questions and short, accessible hints.
- Focus strictly on developing the student's scientific literacy and vocabulary.
- Never give away full answers directly."""

    llm = get_llm(temperature=0.2, streaming=True)
    messages_to_send = [SystemMessage(content=system_prompt)] + list(messages)

    for chunk in llm.stream(messages_to_send):
        if chunk.content:
            yield chunk.content


# --- Socratic Nodes ---
def socratic_tutor(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", COURSE_TITLE)
    user_query = state["messages"][-1].content if state.get("messages") else ""
    context = get_context(sub_topic, user_query)

    system_prompt = f"""You are an expert Socratic {COURSE_TITLE} ({LEVEL}) Tutor.
Target Revision Topic: {sub_topic}
Syllabus Context:
{context}

CRITICAL TOPIC BOUNDARY RULE:
- The student MUST stay focused on the Target Revision Topic: '{sub_topic}'.
- If the student attempts to switch to an unrelated topic:
  1. Politely acknowledge their input.
  2. Clarify that today's revision focus is strictly on **{sub_topic}**.
  3. Pivot the conversation back by connecting their comment to **{sub_topic}** (if a logical link exists) OR explicitly redirect them with a probing question about **{sub_topic}**.

PEDAGOGICAL & DISCIPLINARY LITERACY DIRECTIVES:
- Guide the student step-by-step using probing questions and constructive hints. Never give away full answers directly.
- Command-Word Compliance:
  * "Describe": Require students to state WHAT happens step-by-step or identify trends.
  * "Explain": Require students to state WHY or HOW something happens using cause-and-effect language ("because", "this causes", "therefore").
  * "Evaluate": Require students to balance pros and cons before making a conclusion.
- Disciplinary Precision: Enforce exact specification vocabulary, accurate mathematical/unit notation, and correct underlying scientific mechanisms while trapping common misconceptions."""

    llm = get_llm(temperature=0.2, streaming=False)
    messages_to_send = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages_to_send)

    return {"messages": state["messages"] + [response]}


def didactic_fallback(state: ChatState) -> dict:
    sub_topic = state.get("sub_topic", "")
    user_query = state["messages"][-1].content if state.get("messages") else ""
    context = get_context(sub_topic, user_query)

    system_prompt = f"""You are a strict {COURSE_TITLE} ({LEVEL}) Senior Science Examiner concluding a Socratic revision session.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

CRITICAL MANDATE: This is turn {TARGET_TURNS} (FINAL TURN). You MUST NOT ask any follow-up questions. Conclude the dialogue phase and provide qualitative feedback.

STRICT FORMATTING & LATEX RULES:
- Write all variables, units, formulas, and chemical symbols in clean plain text or Markdown bold/code. Avoid raw LaTeX delimiters ($ / $$).

INSTRUCTIONS FOR SESSION ENDING:
1. **Validate Final Answer:** Directly validate the student's final input in detail first.
2. **Qualitative Performance Assessment:** 
   - Highlight subject-specific technical terms used effectively during the chat.
   - List key specification terms that were missed or need review.
   - Provide 1–2 targeted feedback points.
   - Reiterate that they will now move on to their 6-mark independent synthesis question.

FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:

[Your detailed validation of the student's final answer and corrected solution]

### 📊 Dialogue Phase Complete
- **Targeted Feedback:** 
  - [Constructive feedback point 1]
  - [Constructive feedback point 2]
- **Subject Terminology:**
  - **Keywords Used Well:** [Term 1, Term 2]
  - **Missed Terms to Learn:** [Term 3, Term 4]

CRITICAL RULE: DO NOT include a percentage score, DO NOT include a summary data box, and DO NOT ask any follow-up questions."""

    final_command = HumanMessage(
        content=(
            f"[SYSTEM DIRECTIVE: This is turn {TARGET_TURNS} (FINAL TURN). Do NOT"
            " ask any follow-up questions. Provide the final answer validation and"
            " qualitative dialogue assessment now.]"
        )
    )

    llm = get_llm(temperature=0.1, streaming=False)
    messages_to_send = (
        [SystemMessage(content=system_prompt)]
        + list(state["messages"])
        + [final_command]
    )
    response = llm.invoke(messages_to_send)

    return {"messages": state["messages"] + [response]}


def route_turn(state: ChatState) -> str:
    if state.get("is_freestyle", False):
        return "socratic_tutor"

    if state.get("is_final_turn", False) or state.get("turn_count", 0) >= TARGET_TURNS:
        return "didactic_fallback"

    messages = state.get("messages", [])
    human_count = sum(
        1 for m in messages
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


# --- Socratic 6-Mark Synthesis & Disciplinary Literacy Helpers ---
def generate_socratic_synthesis_prompt(sub_topic: str, chat_history: list, course_title: str = COURSE_TITLE, level: str = LEVEL) -> dict:
    """Generates an accessible 6-mark synthesis prompt dynamically bounded by the chat history."""
    
    transcript_lines = []
    for msg in chat_history:
        role = "Student" if msg.get("role") == "student" or isinstance(msg, HumanMessage) else "Tutor"
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        transcript_lines.append(f"{role}: {content}")
    full_transcript = "\n".join(transcript_lines)

    prompt = f"""You are an expert {course_title} ({level}) educator.
Target Subtopic: {sub_topic}

Below is the transcript of a Socratic revision discussion between a student and a tutor:
---
TRANSCRIPT:
{full_transcript}
---

CRITICAL MANDATE:
Create a 6-mark synthesis exercise bounded STRICTLY by the specific concepts, processes, or mechanisms discussed in the transcript above.
Do NOT include specification terms or concepts UNLESS they were explicitly mentioned or introduced in the chat transcript.

Task:
Instruct the student to write a structured summary connecting the key technical terms and mechanisms explored during their discussion.

Return ONLY a JSON object:
{{
    "question": "Construct a coherent 6-mark explanation summarizing the specific mechanisms and technical terms explored in today's discussion on '{sub_topic}'.",
    "mark_scheme": [
        "Mark 1: [Specific concept/term 1 covered in transcript]",
        "Mark 2: [Specific concept/term 2 covered in transcript]",
        "Mark 3: [Specific concept/term 3 covered in transcript]",
        "Mark 4: [Specific concept/term 4 covered in transcript]",
        "Mark 5: [Specific concept/term 5 covered in transcript]",
        "Mark 6: [Coherent synthesis connecting these concepts]"
    ]
}}"""

    llm = get_llm(temperature=0.2, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])

    raw_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(clean_text)


def generate_freestyle_synthesis_prompt(
    custom_prompt: str,
    chat_history: list,
    focus_mode: str = "Scientific Explanations & Cause/Effect",
    course_title: str = COURSE_TITLE,
    level: str = LEVEL,
) -> dict:
    """Generates a structured 6-point disciplinary literacy synthesis task based on the freestyle dialogue."""
    transcript_lines = []
    for msg in chat_history:
        role = "Student" if msg.get("role") == "student" or isinstance(msg, HumanMessage) else "Tutor"
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        transcript_lines.append(f"{role}: {content}")
    full_transcript = "\n".join(transcript_lines)

    if "Keywords" in focus_mode or "Vocabulary" in focus_mode:
        task_instruction = f"Construct a coherent 6-mark summary connecting the key scientific terms, definitions, and concepts explored during today's discussion on '{custom_prompt}'."
        rubric_focus = "Focus points strictly on precise scientific terms, units, and factual accuracy developed in the chat."
    else:
        task_instruction = f"Using the mechanisms and explanations developed in today's dialogue, write a structured 6-mark cause-and-effect outline explaining: '{custom_prompt}'."
        rubric_focus = "Focus points on cause, mechanism, effect, scientific terms, and logical links."

    prompt = f"""You are an expert {course_title} ({level}) educator.
Student Topic / Question: "{custom_prompt}"

Discussion Transcript:
{full_transcript}

Task:
Create a 6-mark Disciplinary Literacy Synthesis Task based on the concepts explored in the discussion.
{rubric_focus}

Return ONLY a JSON object:
{{
    "question": "{task_instruction}",
    "mark_scheme": [
        "Mark 1: [Specific point/term from transcript]",
        "Mark 2: [Specific point/term from transcript]",
        "Mark 3: [Specific point/term from transcript]",
        "Mark 4: [Specific point/term from transcript]",
        "Mark 5: [Specific point/term from transcript]",
        "Mark 6: [Coherent synthesis connecting these points]"
    ]
}}"""

    llm = get_llm(temperature=0.2, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(clean_text)


def evaluate_socratic_synthesis(question: str, mark_scheme: list, student_answer: str, chat_history: list, course_title: str = COURSE_TITLE, level: str = LEVEL) -> dict:
    """Evaluates student's synthesis response prioritizing DISCIPLINARY LITERACY over binary mark-scheme ticking."""
    ms_text = "\n".join([f"- Mark {i+1}: {pt}" for i, pt in enumerate(mark_scheme)])
    
    prompt = f"""You are an expert {course_title} ({level}) disciplinary literacy coach evaluating a 6-mark student topic synthesis.
Task Prompt: {question}

Session Reference Rubric:
{ms_text}

Student Response:
{student_answer}

EVALUATION GOAL:
Primary focus is DISCIPLINARY LITERACY (accurate use of domain vocabulary, scientific cause-and-effect reasoning, and conceptual connections) rather than doing an all-or-nothing binary tick-box.

SCORING MATRIX (Max 6 Marks):
- 2 Marks (Domain Vocabulary): Award up to 2 marks generously for using key domain terms correctly (e.g. naming relevant structures, organelles, or processes), even if a minor factual error or misnomer is present elsewhere.
- 2 Marks (Mechanistic & Cause-and-Effect Logic): Award up to 2 marks for linking structure to function or explaining mechanisms accurately (e.g. using 'because', 'which leads to', or connecting an organelle to its role).
- 2 Marks (Completeness & Synthesis): Award up to 2 marks for synthesizing multiple core concepts from the discussion into a structured summary.

FEEDBACK DIRECTIVE:
1. Praise the precise domain terms and mechanisms the student used effectively first.
2. If a misconception occurs (e.g. chitin vs cellulose), politely clarify the correction as a "refinement point" without wiping out all credit earned for other correct terminology.
3. Suggest 1–2 actionable terms or omitted concepts to help them upgrade to 6/6.

Return ONLY a JSON object:
{{
    "score": 4,
    "max_score": 6,
    "awarded_points": ["Point 1 praising correct vocabulary/mechanisms", "Point 2 praising structural logic"],
    "missed_points": ["Refinement point (e.g. cell wall is cellulose, not chitin)", "Omitted topic point"],
    "keywords_used": ["Term 1", "Term 2"],
    "keywords_missed": ["Term 3", "Term 4"],
    "feedback": "Encouraging advice praising terminology use and suggesting specific additions for 6/6..."
}}"""

    llm = get_llm(temperature=0.0, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])

    raw_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(clean_text)


def prepare_history_telemetry_payload(
    unit: str,
    subtopic: str,
    synthesis_eval: dict,
    course_title: str = COURSE_TITLE,
    level: str = LEVEL,
) -> dict:
    """Formats payload strictly for insertion into 'activity_logs_science' under 'socratic' mode."""
    score_pct = synthesis_eval.get("score_pct")
    if score_pct is None and "score" in synthesis_eval:
        score_pct = round((synthesis_eval["score"] / synthesis_eval.get("max_score", 6)) * 100, 1)

    misconception = synthesis_eval.get("misconception_flag")
    if misconception is None:
        misconception = 1 if (score_pct is not None and score_pct < 45.0) else 0

    return {
        "table_name": DB_TABLE_NAME,
        "subject": course_title,
        "level": level,
        "unit": unit,
        "subtopic": subtopic,
        "app_mode": "socratic",
        "score_pct": score_pct or 0.0,
        "keywords_used": ", ".join(synthesis_eval.get("keywords_used", [])),
        "keywords_missed": ", ".join(synthesis_eval.get("keywords_missed", [])),
        "misconception_flag": misconception,
    }


# --- Standalone Quiz Helpers ---
def generate_quiz_questions(
    sub_topic: str, course_title: str = COURSE_TITLE, level: str = LEVEL
) -> list:
    context = get_context(sub_topic, sub_topic)
    prompt = f"""You are an expert {course_title} ({level}) Senior Examiner.
Topic Focus: {sub_topic}
Syllabus Context:
{context}

Generate exactly 10 short-answer exam questions testing precise definitions, scientific mechanisms, processes, or calculations.
Output ONLY a valid JSON array of 10 question strings, with no additional text or formatting:
["Question 1 text...", "Question 2 text...", ...]"""

    llm = get_llm(temperature=0.3, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])

    raw_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(clean_text)


def grade_quiz_responses(
    sub_topic: str, questions: list, answers: dict, course_title: str = COURSE_TITLE, level: str = LEVEL
) -> dict:
    context = get_context(sub_topic, sub_topic)
    qa_pairs = "\n".join([
        f"Q{i+1}: {q}\nStudent Answer: {answers.get(i+1, 'No answer provided')}\n"
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

    llm = get_llm(temperature=0.0, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])

    raw_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(clean_text)


def evaluate_quiz_answers(
    questions: list,
    user_answers: dict,
    topic: str,
    course_title: str = COURSE_TITLE,
    level: str = LEVEL,
) -> dict:
    return grade_quiz_responses(topic, questions, user_answers, course_title, level)


# --- Extended & Rewrite Mode Helpers ---
def generate_extended_question(sub_topic: str, course_title: str = COURSE_TITLE, level: str = LEVEL) -> str:
    prompt = f"""You are an expert {level} lead examiner for {course_title}.
Create ONE high-tier 6-mark extended evaluation question for the subtopic: '{sub_topic}'.
Use standard command words like 'Describe', 'Explain', or 'Evaluate'.
Return ONLY the raw question text."""
    llm = get_llm(temperature=0.3, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw_text = extract_clean_text(response)
    return raw_text.strip()


def grade_extended_response(sub_topic: str, question: str, student_answer: str, course_title: str = COURSE_TITLE, level: str = LEVEL) -> Dict[str, Any]:
    prompt = f"""You are a senior {level} examiner evaluating an extended response for {course_title}.
Topic: {sub_topic}
Question: {question}
Student Response: {student_answer}

Evaluate based on subject disciplinary literacy: structural logic, precise terminology, command-word compliance, and scientific cause-and-effect reasoning.

Return a JSON object:
{{
  "score": 5,
  "max_score": 6,
  "disciplinary_level": "Competent",
  "keywords_used": ["term 1", "term 2"],
  "keywords_missed": ["term 3"],
  "strengths": "Summary of strengths...",
  "struggle_advice": "Advice for improvement...",
  "model_answer": "Exemplar top-band response..."
}}"""
    llm = get_llm(temperature=0.0, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    clean_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", clean_text).strip()
    return json.loads(clean_text)


def generate_layman_transformation_prompt(sub_topic: str, course_title: str = COURSE_TITLE, level: str = LEVEL) -> Dict[str, str]:
    prompt = f"""Create an exercise for {level} level {course_title} on '{sub_topic}'.
1. Write a typical 4-mark exam question.
2. Write a poorly phrased, informal student response using vague everyday language (e.g., 'gets hot', 'makes more stuff', 'goes faster' without precise technical terms or scientific cause-and-effect).

Return ONLY JSON:
{{
  "question": "Exam question...",
  "layman_answer": "Informal student draft..."
}}"""
    llm = get_llm(temperature=0.4, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    clean_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", clean_text).strip()
    return json.loads(clean_text)


def grade_disciplinary_rewrite(sub_topic: str, question: str, layman_answer: str, student_rewrite: str, course_title: str = COURSE_TITLE, level: str = LEVEL) -> Dict[str, Any]:
    prompt = f"""Grade a student's upgraded response in {level} {course_title}.
Topic: {sub_topic}
Question: {question}
Original Informal Draft: {layman_answer}
Student Upgraded Rewrite: {student_rewrite}

Assess whether the student replaced informal language with precise scientific terminology, accurate mechanisms, or command-word precision.

Return JSON:
{{
  "score": 4,
  "max_score": 4,
  "key_terms_used": ["term 1"],
  "missed_terms": ["term 2"],
  "feedback": "Constructive examiner note..."
}}"""
    llm = get_llm(temperature=0.0, streaming=False)
    response = llm.invoke([HumanMessage(content=prompt)])
    clean_text = extract_clean_text(response)
    clean_text = re.sub(r"```json|```", "", clean_text).strip()
    return json.loads(clean_text)