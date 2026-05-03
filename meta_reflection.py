import sqlite3
import logging
import os
from agent import HermesAgent
from tracker import PaperTracker

_DATA_DIR = os.getenv("DATA_DIR", ".")
_DB_PATH = os.path.join(_DATA_DIR, "paper_trading.db")
_RULEBOOK_PATH = os.path.join(_DATA_DIR, "master_rulebook.md")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def consolidate_memory():
    """定期執行大腦突觸修剪與分類總結 (Meta-Reflection)"""
    tracker = PaperTracker()

    # Only run if there are enough new unconsolidated lessons to justify a 405B API call
    unconsolidated_count = tracker.get_unconsolidated_lesson_count()
    if unconsolidated_count < 5:
        logging.info(f"Meta-Reflection skipped: only {unconsolidated_count} new lessons (need 5+).")
        return

    logging.info(f"Starting Meta-Reflection Memory Consolidation ({unconsolidated_count} new lessons)...")
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT market_category, lesson FROM lessons_learned WHERE is_consolidated = 0 ORDER BY id DESC LIMIT 20")
        recent_lessons = cursor.fetchall()
    except Exception as e:
        logging.error(f"Error reading lessons: {e}")
        conn.close()
        return

    if not recent_lessons:
        logging.info("No lessons to consolidate.")
        conn.close()
        return

    # 將教訓按分類整理
    category_lessons = {}
    for cat, lesson in recent_lessons:
        category = cat if cat else "General"
        if category not in category_lessons:
            category_lessons[category] = []
        category_lessons[category].append(lesson)

    # 讀取現有的 master_rulebook
    rulebook_path = _RULEBOOK_PATH
    existing_rulebook = ""
    if os.path.exists(rulebook_path):
        with open(rulebook_path, "r", encoding="utf-8") as f:
            existing_rulebook = f.read()

    # 準備合併的文本
    new_experiences_text = ""
    for cat, lessons in category_lessons.items():
        new_experiences_text += f"\n### Category: {cat}\n"
        for i, l in enumerate(lessons):
            new_experiences_text += f"{i+1}. {l}\n"

    sys_p = (
        "You are Hermes, a master quantitative trader and prediction market expert. "
        "Your task is to perform 'Meta-Reflection' (Memory Consolidation). "
        "You will be given the CURRENT Master Rulebook, and a list of RECENT lessons learned from recent trades. "
        "You must output a NEW, updated Master Rulebook in Markdown format. "
        "Merge the core concepts from the recent lessons into the rulebook. "
        "Organize the rulebook by categories (e.g. Crypto, Geopolitics, General Psychology). "
        "Keep the rulebook extremely concise, actionable, and profound. Remove redundant or outdated rules."
    )

    usr_p = (
        f"--- CURRENT MASTER RULEBOOK ---\n{existing_rulebook}\n\n"
        f"--- RECENT EXPERIENCES TO INTEGRATE ---\n{new_experiences_text}\n\n"
        f"Please output ONLY the updated Master Rulebook in Markdown format. Start immediately with '# Hermes Master Rulebook'."
    )

    agent = HermesAgent(model_name="llama3.1")
    providers_chain = [
        ("nvidia", "meta/llama-3.1-405b-instruct"),
        ("groq", "llama-3.3-70b-versatile"),
        ("ollama", agent.model_name)
    ]
    
    try:
        updated_rulebook = agent._call_llm_with_fallback(sys_p, usr_p, json_mode=False, providers=providers_chain)
        if updated_rulebook:
            with open(rulebook_path, "w", encoding="utf-8") as f:
                f.write(updated_rulebook)
            logging.info("Master Rulebook successfully updated and saved to master_rulebook.md")
            tracker.mark_lessons_consolidated()
        else:
            logging.warning("Failed to generate updated rulebook.")
    except Exception as e:
        logging.error(f"Error during Meta-Reflection LLM call: {e}")

    conn.close()

if __name__ == "__main__":
    consolidate_memory()
