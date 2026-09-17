import os
import json
import sqlite3
import random
from datetime import datetime, timedelta

DB_NAME = "analytics.db"
SPEC_PATH = "course_spec.json"

def setup_database():
    """Creates the SQLite database file and activity_logs table if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            subject TEXT NOT NULL,
            level TEXT NOT NULL,
            unit TEXT NOT NULL,
            subtopic TEXT NOT NULL,
            app_mode TEXT NOT NULL,
            score_pct REAL NOT NULL,
            keywords_used TEXT,
            keywords_missed TEXT,
            misconception_flag INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def extract_spec_hierarchy(spec_data):
    """Recursively extracts subject, unit, and subtopic records from course_spec.json."""
    raw_structure = spec_data.get("subjects") or spec_data.get("topics", {})
    course_title = spec_data.get("course_title", "Socratic Subject")
    records = []
    
    first_key = next(iter(raw_structure), None)
    is_three_tier = isinstance(raw_structure.get(first_key), dict) if first_key else False

    if is_three_tier:
        for subj, units in raw_structure.items():
            for unit_name, subtopics in units.items():
                for sub in subtopics:
                    records.append({
                        "subject": f"{course_title} ({subj})",
                        "unit": unit_name,
                        "subtopic": sub
                    })
    else:
        for unit_name, subtopics in raw_structure.items():
            for sub in subtopics:
                records.append({
                    "subject": course_title,
                    "unit": unit_name,
                    "subtopic": sub
                })
    return records

def generate_synthetic_data(num_entries=200):
    setup_database()

    if not os.path.exists(SPEC_PATH):
        print(f"❌ Error: {SPEC_PATH} not found. Please ensure course_spec.json exists.")
        return

    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec_data = json.load(f)

    level = spec_data.get("level", "GCSE/A-Level")
    spec_records = extract_spec_hierarchy(spec_data)

    if not spec_records:
        print("❌ Error: No units or subtopics found in course_spec.json.")
        return

    modes = ["socratic", "quiz", "extended", "rewrite"]
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now()

    for _ in range(num_entries):
        target = random.choice(spec_records)
        mode = random.choice(modes)
        score_pct = round(random.uniform(30.0, 100.0), 1)
        
        days_ago = random.randint(0, 14)
        hours_ago = random.randint(0, 23)
        timestamp = (now - timedelta(days=days_ago, hours=hours_ago)).strftime("%Y-%m-%d %H:%M:%S")

        words = [w.strip(",.()").lower() for w in target["subtopic"].split() if len(w) > 3]
        if not words:
            words = ["specification", "concept", "process", "theory"]
            
        sample_size = min(len(words), random.randint(2, 5))
        chosen_kws = random.sample(words, sample_size)
        
        split_idx = int(len(chosen_kws) * (score_pct / 100.0))
        used = chosen_kws[:split_idx]
        missed = chosen_kws[split_idx:]
        misconception_flag = 1 if score_pct < 50.0 and random.random() > 0.35 else 0

        cursor.execute("""
            INSERT INTO activity_logs 
            (timestamp, subject, level, unit, subtopic, app_mode, score_pct, keywords_used, keywords_missed, misconception_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, 
            target["subject"], 
            level, 
            target["unit"], 
            target["subtopic"],
            mode, 
            score_pct, 
            json.dumps(used), 
            json.dumps(missed), 
            misconception_flag
        ))

    conn.commit()
    conn.close()
    print(f"✅ Created {DB_NAME} and inserted {num_entries} synthetic log entries successfully!")

if __name__ == "__main__":
    user_input = input("Enter number of synthetic records to generate [default: 200]: ").strip()
    
    if user_input.isdigit():
        target_count = int(user_input)
    else:
        target_count = 200
        
    generate_synthetic_data(target_count)