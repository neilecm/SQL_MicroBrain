#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-Coder-1.5B-Instruct with LoRA on Pagila SQL Micro-Brain data.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


def load_pagila_dataset(path: Path) -> List[Dict[str, str]]:
    """Load training data from JSONL."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                data.append({"prompt": record["prompt"], "completion": record["completion"]})
    return data


def prepare_texts(examples: List[Dict[str, str]]) -> List[str]:
    """Build full training texts by concatenating prompt + completion."""
    texts = []
    for ex in examples:
        # Ensure there's a clear separation
        full_text = ex["prompt"] + ex["completion"]
        texts.append(full_text)
    return texts


def tokenize_function(tokenizer, max_length: int):
    """Tokenizer function for datasets."""
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    return tokenize_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder with LoRA on Pagila data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--train_file", type=str, default="data/processed/train_pagila.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--output_dir", type=str, default="models/sql-micro-brain-qwen2.5-pagila-lora",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")

    args = parser.parse_args()

    # Determine device - use MPS by default but optimize for memory
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon) with memory optimizations")
        # Force memory-efficient settings for MPS
        args.batch_size = 1  # Force small batch
        args.max_length = 512  # Short sequences for memory efficiency
        print(f"  → Forced batch_size=1, max_length={args.max_length} for MPS memory")
    else:
        device = "cpu"
        print("Using CPU")
        args.max_length = min(args.max_length, 1024)  # Reasonable limit for CPU

    # Load data
    repo_root = Path(__file__).resolve().parents[2]
    train_path = repo_root / args.train_file
    examples = load_pagila_dataset(train_path)
    texts = prepare_texts(examples)
    print(f"Loaded {len(texts)} training examples from {train_path}")

    # Create dataset
    dataset_dict = [{"text": t} for t in texts]
    dataset = Dataset.from_list(dataset_dict)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function(tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )

    # Load model (avoid device_map for MPS compatibility)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # MPS compatible
        trust_remote_code=True
    )

    # Explicitly move to device (after applying LoRA)
    model.to(device)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Count parameters
    model.print_trainable_parameters()

    # Training arguments - optimized for low-memory iMac
    training_args = TrainingArguments(
        output_dir=str(repo_root / args.output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0,  # Skip warmup for speed
        weight_decay=0.0,  # Skip regularization
        logging_steps=1,  # Log every step for small datasets
        save_strategy="no",  # Don't save checkpoints to save disk I/O
        eval_strategy="no",  # No validation set
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=False,  # MPS compatibility
        bf16=False,  # Disable to avoid MPS issues
        dataloader_pin_memory=False,  # Avoid MPS warnings
        dataloader_num_workers=0  # Single thread to avoid process overhead
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    print(f"Starting training for {args.num_epochs} epochs...")
    trainer.train()

    # Save
    output_path = repo_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    model.save_pretrained(str(output_path))

    print(f"Fine-tuned model saved to {output_path}")


if __name__ == "__main__":
    main()
