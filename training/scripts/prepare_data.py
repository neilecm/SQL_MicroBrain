#!/usr/bin/env python3
"""
Script to prepare LoRA fine-tuning data from JSONL training examples.

Reads input JSONL with records like:
  {"system": "...", "user": {...}, "assistant": {...}}
Converts to LoRA format:
  {"prompt": "SYSTEM:\n...\n\nUSER_JSON:\n<json>\n\nASSISTANT_JSON:\n", "completion": "<json>"}

Usage: python training/scripts/prepare_data.py [--input path] [--output path]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


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
        "--input", type=Path, default=Path("data/raw/pagila_training.jsonl"),
        help="Path to input JSONL file (default: data/raw/pagila_training.jsonl)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/train_pagila.jsonl"),
        help="Path to output JSONL file (default: data/processed/train_pagila.jsonl)"
    )

    args = parser.parse_args(argv)

    # Resolve paths relative to repo root
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / args.input
    output_path = repo_root / args.output

    # Check if input exists
    if not input_path.exists():
        sys.stderr.write(f"ERROR: Input file does not exist: {input_path}\n")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    processed_count = 0
    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            record = load_jsonl_record(line)
            if record is None:
                sys.stderr.write(f"WARNING: Skipping line {line_num}: invalid JSON\n")
                continue

            # Validate required keys
            if "system" not in record or "user" not in record or "assistant" not in record:
                sys.stderr.write(f"WARNING: Skipping line {line_num}: missing required keys\n")
                continue

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
                sys.stderr.write(f"WARNING: Skipping line {line_num}: error processing - {e}\n")
                continue

    print(f"Processed {processed_count} examples from {input_path} to {output_path}")


if __name__ == "__main__":
    main()
