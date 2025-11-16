# SQL Micro-Brain

A 1B-parameter specialist model for PostgreSQL + Supabase:
- schema design
- migrations
- RLS
- natural-language-to-SQL
- query optimization

This repo contains:
- `prompts/` – system prompt and few-shot examples
- `data/` – training/eval data in JSONL format
- `training/` – LoRA fine-tuning scripts
- `server/` – small HTTP API wrapper
- `cli/` – local command-line assistant

## Quick Start

### Installation

```bash
cd sql-micro-brain
pip install -r requirements.txt
```

Note: For Apple Silicon (M4), `torch` should automatically use MPS acceleration. If needed, upgrade `torch` for nightly builds.

### CLI Usage

```bash
# Install deps first
pip install -r requirements.txt

# Run schema design task
python cli/db_assistant.py --mode design_schema --task "Create a multi-tenant rental database for salons renting wedding veils and dresses"

# Output: JSON object with migrations, indexes, etc.
```

The CLI downloads and loads Qwen2.5-Coder-1.5B-Instruct locally on first run (may take a few minutes).

### Server (HTTP API)

```bash
# Start server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Test health
curl http://localhost:8000/healthz

# Post a task
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"mode": "design_schema", "natural_language_task": "Design tables for books and authors"}'
```

### Fine-Tuning with LoRA

SQL Micro-Brain supports fine-tuning Qwen2.5-Coder-1.5B-Instruct on prepared data for enhanced performance.

#### Data Preparation
```bash
# Process raw examples to LoRA format
python training/scripts/prepare_data.py --input data/raw/pagila_training.jsonl --output data/processed/train_custom.jsonl
```

#### Training
```bash
# Install PEFT and datasets first
pip install peft datasets

# Fine-tune on Pagila data (will download base model first time)
python training/scripts/train_lora_pagila.py --num_epochs 3 --batch_size 1

# Custom training
python training/scripts/train_lora_pagila.py \
  --train_file data/processed/train_custom.jsonl \
  --output_dir models/sql-micro-brain-custom-lora \
  --num_epochs 5
```

#### Using Fine-Tuned Models

Set environment variables to load a LoRA adapter:

```bash
# Base model only (default)
export SQL_MB_LORA_PATH=""

# Load fine-tuned adapter
export SQL_MB_LORA_PATH="models/sql-micro-brain-qwen2.5-pagila-lora"

# Optional: customize inference
export SQL_MB_MAX_NEW_TOKENS="1024"
export SQL_MB_TEMPERATURE="0.1"

# Then run CLI or server
python cli/db_assistant.py --mode write_sql --task "Show me top rental films"
# Or
uvicorn server:app --reload
```

If LoRA path is invalid, it automatically falls back to base model.
