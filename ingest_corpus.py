import os, json, tomllib
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

secrets_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_path) and "GOOGLE_API_KEY" not in os.environ:
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
        os.environ["GOOGLE_API_KEY"] = secrets.get("GOOGLE_API_KEY") or secrets.get("GEMINI_API_KEY", "")

class SubjectSpec(BaseModel):
    course_title: str = Field(description="Overall course title, e.g. AQA GCSE Separate Sciences")
    subject: str = Field(description="Subject name, e.g. Biology, Chemistry, Physics")
    level: str = Field(description="GCSE or A-Level")
    target_turns: int = Field(description="5 for GCSE, 7 for A-Level")
    topics: Dict[str, List[str]] = Field(description="Map of units to subtopics")

def process_corpus():
    syllabus_dir = "./syllabus"
    pdf_files = sorted([f for f in os.listdir(syllabus_dir) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("⚠️ No PDFs found in ./syllabus/")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    master_spec = {"topics": {}}

    for pdf in pdf_files:
        print(f"🔍 Processing {pdf}...")
        docs = PyPDFLoader(os.path.join(syllabus_dir, pdf)).load()
        
        sampled_pages = [docs[i].page_content for i in range(0, len(docs), 2)]
        full_overview_text = "\n--- PAGE ---\n".join(sampled_pages)


        prompt = (
            "Analyze this science specification PDF.\n"
            "Extract the general course suite title (e.g., 'AQA GCSE Separate Sciences' rather than single subject name), "
            "the specific subject name (e.g. Biology), qualification level, target turns, and ALL units with subtopics.\n\n"
            f"Specification Sample:\n{full_overview_text[:120000]}"
        )

        try:
            spec = llm.with_structured_output(SubjectSpec).invoke(prompt)
            
            if "course_title" not in master_spec:
                master_spec.update({
                    "course_id": spec.course_title.lower().replace(" ", "_").replace(":", ""),
                    "course_title": spec.course_title,
                    "level": spec.level,
                    "target_turns": spec.target_turns
                })

            master_spec["topics"][spec.subject] = spec.topics
            print(f"   ✅ Merged {spec.subject} topics into master spec!")
        except Exception as e:
            print(f"   ⚠️ Spec extraction error on {pdf}: {e}")

        # Index document chunks into shared ChromaDB
        content_docs = docs[15:] if len(docs) > 15 else docs
        chunks = splitter.split_documents(content_docs)
        for c in chunks:
            c.metadata["source_file"] = pdf
        
        for i in range(0, len(chunks), 100):
            vectorstore.add_documents(chunks[i:i + 100])
        print(f"   ✅ Indexed {len(chunks)} chunks into ChromaDB.")

    # Save dynamic 3-tier master JSON
    with open("course_spec.json", "w", encoding="utf-8") as f:
        json.dump(master_spec, f, indent=2)
    print("\n🚀 Saved master course_spec.json successfully!")

if __name__ == "__main__":
    process_corpus()