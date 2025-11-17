import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # goes up to sql-micro-brain/
SPIDER_DIR = ROOT / "data" / "raw" / "spider"
OUTPUT_PATH = ROOT / "data" / "processed" / "train_spider_sqlmb.jsonl"

TRAIN_FILES = ["train_spider.json", "train_others.json"]

SYSTEM_PROMPT = (
    "You are SQL Micro-Brain, an expert PostgreSQL assistant. "
    "Given a natural language task and a database schema, you output a JSON object "
    "with actions, migrations, rls_policies, indexes, queries, error_explanations, "
    "explanations, and safe_to_execute. You MUST produce valid PostgreSQL SQL."
)

def load_schema_sql(db_id: str) -> str:
    """
    Load Spider's schema.sql for a given database id.
    """
    schema_path = SPIDER_DIR / "database" / db_id / "schema.sql"
    if not schema_path.exists():
        return ""
    return schema_path.read_text(encoding="utf-8").strip()

def load_spider_examples():
    examples = []
    for fname in TRAIN_FILES:
        path = SPIDER_DIR / fname
        with path.open("r", encoding="utf-8") as f:
            examples.extend(json.load(f))
    return examples

def main():
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    examples = load_spider_examples()

    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for ex in examples:
            question = ex["question"]           # natural language
            sql = ex["query"]                  # ground truth SQL
            db_id = ex["db_id"]                # which database
            schema_sql = load_schema_sql(db_id)

            user_content = (
                f"Task: {question}\n\n"
                f"Database: {db_id}\n\n"
                f"""Schema:\n{schema_sql}\n"""
            )

            # This is the *output format* that matches your SQL Micro-Brain style.
            # You can tweak 'actions' or add indexes if you want.
            assistant_payload = {
                "actions": ["select"],  # Most Spider queries are SELECTs
                "migrations": [],
                "rls_policies": [],
                "indexes": [],
                "queries": [
                    {
                        "description": question,
                        "sql": sql
                    }
                ],
                "error_explanations": [],
                "explanations": [],
                "safe_to_execute": True,
            }

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant_payload, ensure_ascii=False),
                    },
                ]
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} training examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
