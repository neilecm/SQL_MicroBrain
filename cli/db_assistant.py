#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add current directory to path for relative imports
sys.path.append('.')


# ---------- Paths ----------

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.sql-micro-brain.md"


# ---------- LLM Wrapper (for SQL Micro-Brain v0: local Qwen model) ----------

def call_llm(messages: List[Dict[str, str]]) -> str:
    """
    For SQL Micro-Brain v0, we use the local Qwen2.5-Coder-1.5B-Instruct model.

    - Extract the user JSON payload from the last message (assuming it's the user content)
    - Pass that dict to run_sql_microbrain()
    - Return json.dumps(...) of the result dict
    """
    from server.model_loader import run_sql_microbrain

    # Find the last user message (typically the one with the input payload)
    user_content = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            user_content = msg["content"]
            break

    if user_content is None:
        raise ValueError("No user message found in messages")

    try:
        input_payload = json.loads(user_content)
    except json.JSONDecodeError:
        raise ValueError("User content is not valid JSON payload")

    result_dict = run_sql_microbrain(input_payload)
    return json.dumps(result_dict, ensure_ascii=False)


# ---------- Validation Helpers ----------

REQUIRED_TOP_LEVEL_KEYS = [
    "actions",
    "migrations",
    "rls_policies",
    "indexes",
    "queries",
    "error_explanations",
    "explanations",
    "safe_to_execute",
]

def validate_response_shape(obj: Dict[str, Any]) -> None:
    """Basic sanity check: required keys + types."""
    missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in obj]
    if missing:
        raise ValueError(f"Missing top-level keys in response JSON: {missing}")

    if not isinstance(obj["actions"], list):
        raise ValueError("actions must be a list")

    if not isinstance(obj["migrations"], list):
        raise ValueError("migrations must be a list")

    if not isinstance(obj["rls_policies"], list):
        raise ValueError("rls_policies must be a list")

    if not isinstance(obj["indexes"], list):
        raise ValueError("indexes must be a list")

    if not isinstance(obj["queries"], list):
        raise ValueError("queries must be a list")

    if not isinstance(obj["error_explanations"], list):
        raise ValueError("error_explanations must be a list")

    if not isinstance(obj["explanations"], list):
        raise ValueError("explanations must be a list")

    if not isinstance(obj["safe_to_execute"], (bool,)):
        raise ValueError("safe_to_execute must be a boolean")


# ---------- CLI Logic ----------

def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found at {SYSTEM_PROMPT_PATH}")
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

def build_input_json(
    mode: str,
    task: str,
    current_schema: str = "",
    db_engine: str = "supabase",
    supabase_style: bool = True,
    naming: str = "snake_case",
    id_type: str = "uuid",
    multi_tenant: bool = True,
) -> Dict[str, Any]:
    """Build the JSON input payload described in our spec."""
    return {
        "mode": mode,
        "natural_language_task": task,
        "current_schema": current_schema,
        "preferences": {
            "db_engine": db_engine,
            "supabase_style": supabase_style,
            "naming": naming,
            "id_type": id_type,
            "multi_tenant": multi_tenant,
        },
        "sql_snippets": {
            "problem_query": "",
            "explain_analyze": ""
        },
        "error_message": ""
    }

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="SQL Micro-Brain v0 CLI – send a JSON task to an LLM and get SQL JSON back."
    )
    parser.add_argument(
        "--mode",
        choices=["design_schema", "write_sql", "fix_error", "optimize_query", "design_rls"],
        default="design_schema",
        help="Mode of operation for SQL Micro-Brain.",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Natural language description of what you want (if no --input-file).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a JSON file with a full input payload (overrides --task & prefs).",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="",
        help="Optional: schema SQL as string for current_schema.",
    )
    parser.add_argument(
        "--current-schema-file",
        type=str,
        help="Optional: path to a .sql file with the current schema to include (alternative to --schema).",
    )

    args = parser.parse_args(argv)

    system_prompt = load_system_prompt()

    # Build or load input JSON
    if args.input_file:
        input_payload = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    else:
        if not args.task:
            print("Error: either --task or --input-file is required.", file=sys.stderr)
            return 1

        current_schema = args.schema  # Default from --schema arg
        if args.current_schema_file:
            current_schema = Path(args.current_schema_file).read_text(encoding="utf-8")

        input_payload = build_input_json(
            mode=args.mode,
            task=args.task,
            current_schema=current_schema,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(input_payload)},
    ]

    # Call the LLM
    try:
        raw = call_llm(messages)
    except NotImplementedError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Parse & validate JSON
    try:
        resp_obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print("ERROR: LLM did not return valid JSON:", file=sys.stderr)
        print(raw, file=sys.stderr)
        print(f"JSON error: {e}", file=sys.stderr)
        return 1

    try:
        validate_response_shape(resp_obj)
    except ValueError as e:
        print("ERROR: JSON shape validation failed:", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(json.dumps(resp_obj, indent=2), file=sys.stderr)
        return 1

    # Pretty-print result
    print(json.dumps(resp_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
