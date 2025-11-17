#!/usr/bin/env python3
"""
Script to prepare LoRA fine-tuning data from JSONL training examples.

Reads all JSONL files from data/raw/ and converts them to LoRA format:
  {"prompt": "SYSTEM:\n...\n\nUSER_JSON:\n<json>\n\nASSISTANT_JSON:\n", "completion": "<json>"}

Usage: python training/scripts/prepare_data.py [--output path]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl_record(line: str) -> Dict[str, Any]:
    """Parse a single JSONL line, returning the dict or None on error."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None


def build_prompt(system: str, user_obj: Dict[str, Any]) -> str:
    """Build the prompt string for the model."""
    user_json = json.dumps(user_obj, ensure_ascii=False, indent=2, sort_keys=True)
    return f"SYSTEM:\n{system}\n\nUSER_JSON:\n{user_json}\n\nASSISTANT_JSON:\n"


def build_completion(assistant_obj: Dict[str, Any]) -> str:
    """Build the completion string (just the assistant JSON pretty-printed)."""
    return json.dumps(assistant_obj, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training data from JSONL examples."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/train.jsonl"),
        help="Path to output JSONL file (default: data/processed/train.jsonl)"
    )

    args = parser.parse_args(argv)

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = repo_root / "data" / "raw"
    output_path = repo_root / args.output

    # Check if raw directory exists
    if not raw_dir.exists():
        sys.stderr.write(f"ERROR: Raw data directory does not exist: {raw_dir}\n")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all JSONL files in data/raw/
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        sys.stderr.write(f"ERROR: No .jsonl files found in {raw_dir}\n")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files in {raw_dir}")

    # Process all files
    processed_count = 0
    with output_path.open("w", encoding="utf-8") as f_out:
        for file_path in jsonl_files:
            print(f"Processing {file_path.name}...")
            with file_path.open("r", encoding="utf-8") as f_in:
                for line_num, line in enumerate(f_in, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    record = load_jsonl_record(line)
                    if record is None:
                        sys.stderr.write(f"WARNING: Skipping {file_path.name} line {line_num}: invalid JSON\n")
                        continue

                    # Handle different formats (with or without system key)
                    if "system" in record and "user" in record and "assistant" in record:
                        # Old format with system, user, assistant
                        try:
                            prompt = build_prompt(record["system"], record["user"])
                            completion = build_completion(record["assistant"])

                            output_record = {
                                "prompt": prompt,
                                "completion": completion
                            }

                            json.dump(output_record, f_out, ensure_ascii=False)
                            f_out.write("\n")
                            processed_count += 1

                        except Exception as e:
                            sys.stderr.write(f"WARNING: Skipping {file_path.name} line {line_num}: error processing - {e}\n")
                            continue

                    elif "id" in record and "mode" in record and record["mode"] == "write_sql":
                        # New simplified format
                        try:
                            # Convert to prompt format
                            user_content = {
                                "mode": "write_sql",
                                "natural_language_task": record["task"],
                                "current_schema": record["schema"],
                                "preferences": {
                                    "db_engine": "postgres",
                                    "naming": "snake_case",
                                    "supabase_style": False
                                },
                                "sql_snippets": {},
                                "error_message": ""
                            }
                            assistant_content = {
                                "actions": ["write_sql"],
                                "migrations": [],
                                "rls_policies": [],
                                "indexes": [],
                                "queries": [{"description": "SQL query response", "sql": record["sql"]}],
                                "error_explanations": [],
                                "explanations": [f"Query complexity: {record['difficulty']}, tags: {', '.join(record['tags'])}"],
                                "safe_to_execute": True
                            }

                            system = "SQL Micro-Brain v0 – Training example"
                            prompt = build_prompt(system, user_content)
                            completion = build_completion(assistant_content)

                            output_record = {
                                "prompt": prompt,
                                "completion": completion
                            }

                            json.dump(output_record, f_out, ensure_ascii=False)
                            f_out.write("\n")
                            processed_count += 1

                        except Exception as e:
                            sys.stderr.write(f"WARNING: Skipping {file_path.name} line {line_num}: error processing simplified format - {e}\n")
                            continue

                    elif "output" in record:
                        # Our synthetic format: {"id": "...", "schema": "...", "task": "...", "output": {...}}
                        try:
                            # Map to user content
                            user_content = {
                                "mode": "write_sql" if "select" in record["output"]["actions"] or "insert" in record["output"]["actions"] else "design_schema",
                                "natural_language_task": record["task"],
                                "current_schema": record["schema"],
                                "preferences": {
                                    "db_engine": "postgres",
                                    "naming": "snake_case",
                                    "supabase_style": False
                                },
                                "sql_snippets": {},
                                "error_message": ""
                            }
                            assistant_content = record["output"]

                            system = "SQL Micro-Brain v0"
                            prompt = build_prompt(system, user_content)
                            completion = build_completion(assistant_content)

                            output_record = {
                                "prompt": prompt,
                                "completion": completion
                            }

                            json.dump(output_record, f_out, ensure_ascii=False)
                            f_out.write("\n")
                            processed_count += 1

                        except Exception as e:
                            sys.stderr.write(f"WARNING: Skipping {file_path.name} line {line_num}: error processing output format - {e}\n")
                            continue

                    else:
                        sys.stderr.write(f"WARNING: Skipping {file_path.name} line {line_num}: unknown format\n")
                        continue

    print(f"Processed {processed_count} total examples to {output_path}")


if __name__ == "__main__":
    main()
