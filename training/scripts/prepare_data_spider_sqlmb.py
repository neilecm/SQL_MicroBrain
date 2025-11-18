#!/usr/bin/env python3
"""
Prepare Spider dataset for SQL Micro-Brain training.
Converts Spider JSON format to chat-style JSONL format compatible with train_lora_spider.py.
"""
import json
import os
import re
from pathlib import Path

# Hardcoded relative paths (run from repo root)
SPIDER_TRAIN_PATH = "data/raw/spider/evaluation_examples/examples/train_spider.json"
SPIDER_TABLES_PATH = "data/raw/spider/evaluation_examples/examples/tables.json"
OUTPUT_PATH = "data/processed/train_spider_sqlmb.jsonl"

# Type mapping: Spider types to Postgres types
TYPE_MAPPING = {
    "text": "TEXT",
    "varchar": "TEXT",
    "number": "NUMERIC",
    "time": "TIME",
    "date": "DATE",
    "boolean": "BOOLEAN"
}

def normalize_sql(sql: str) -> str:
    """Minimal SQL normalization: strip trailing semicolon, collapse whitespace."""
    sql = sql.rstrip(';')
    # Collapse multiple spaces/newlines into single space
    sql = re.sub(r'\s+', ' ', sql)
    return sql.strip()

def build_schema_from_tables(tables_data: list) -> dict:
    """Build mapping of db_id to schema SQL string."""
    db_schemas = {}
    for db_entry in tables_data:
        db_id = db_entry["db_id"]
        table_list = db_entry["table_names_original"]
        column_types = db_entry["column_types"]
        column_names = db_entry["column_names_original"]  # list of (table_idx, col_name)
        primary_keys = db_entry.get("primary_keys", [])  # list of column indices

        table_schemas = []
        for i, table_name in enumerate(table_list):
            # Get column indices for this table
            table_cols = [j for j, (tid, _) in enumerate(column_names) if tid == i]
            if not table_cols:
                continue

            columns = []
            pk_col_names = []
            for j in table_cols:
                _, col_name = column_names[j]
                col_type = column_types[j]
                col_name = col_name.lower()  # Convert to snake_case
                postgres_type = TYPE_MAPPING.get(col_type.lower(), "TEXT")
                columns.append(f"    {col_name} {postgres_type}")
                # Check if this column is in primary_keys
                if j in primary_keys:
                    pk_col_names.append(col_name)

            create_stmt = f"CREATE TABLE {table_name.lower()} (\n" + ",\n".join(columns)
            if pk_col_names:
                pk_str = ", ".join(pk_col_names)
                create_stmt += f",\n    PRIMARY KEY ({pk_str})"
            create_stmt += "\n);"
            table_schemas.append(create_stmt)

        db_schemas[db_id] = "\n".join(table_schemas)
    return db_schemas

def main():
    print("Loading tables.json for schema generation...")
    with open(SPIDER_TABLES_PATH, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    db_schemas = build_schema_from_tables(tables_data)
    print(f"Built schemas for {len(db_schemas)} databases.")

    print("Loading train_spider.json for training examples...")
    with open(SPIDER_TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    print("Processing training examples...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for example in train_data:
            question = example["question"]
            db_id = example["db_id"]
            gold_sql = normalize_sql(example["query"])
            schema_sql = db_schemas.get(db_id, "-- Schema not found")

            # Build messages
            user_content = {
                "current_schema": schema_sql,
                "error_message": "",
                "mode": "write_sql",
                "natural_language_task": question,
                "preferences": {
                    "db_engine": "postgres",
                    "naming": "snake_case",
                    "supabase_style": False
                },
                "sql_snippets": {},
                "db_id": db_id  # Small improvement: add db_id
            }

            assistant_content = {
                "actions": ["select"],
                "error_explanations": [],
                "explanations": ["Auto-generated from Spider dataset as a supervised example."],
                "indexes": [],
                "migrations": [],
                "queries": [
                    {
                        "description": question,
                        "sql": gold_sql
                    }
                ],
                "rls_policies": [],
                "safe_to_execute": True
            }

            messages = [
                {
                    "role": "system",
                    "content": "SQL Micro-Brain v0"
                },
                {
                    "role": "user",
                    "content": "USER_JSON:\n" + json.dumps(user_content, indent=2, ensure_ascii=False)
                },
                {
                    "role": "assistant",
                    "content": json.dumps(assistant_content, indent=2, ensure_ascii=False, sort_keys=False)
                }
            ]

            record = {"messages": messages}
            out_f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")

    print("\nUsing:")
    print(f"  train: {SPIDER_TRAIN_PATH}")
    print(f"  tables: {SPIDER_TABLES_PATH}")
    print(f"  out: {OUTPUT_PATH}")
    print(f"Wrote {len(train_data)} training examples to {OUTPUT_PATH}")

    # Sanity check
    print("Running sanity check...")
    sanity_passed = True
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 lines
                break
            try:
                obj = json.loads(line)
                if "messages" not in obj or not isinstance(obj["messages"], list):
                    sanity_passed = False
                    break
                for msg in obj["messages"]:
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        sanity_passed = False
                        break
                if not sanity_passed:
                    break
            except json.JSONDecodeError:
                sanity_passed = False
                break

    if sanity_passed:
        print("Sanity check passed.")
    else:
        print("Sanity check failed!")

if __name__ == "__main__":
    main()
