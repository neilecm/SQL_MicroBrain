#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for SQL Micro-Brain on Spider Dataset
Optimized for Google Colab T4 GPU
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from pathlib import Path


def setup_device():
    """Setup GPU/CPU device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)  # Use first GPU in Colab
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")
    else:
        print("No GPU available, using CPU (not recommended)")
    return device


def main():
    # Setup paths
    ROOT = Path(__file__).resolve().parents[2]
    training_data_path = ROOT / "data" / "processed" / "train_spider_sqlmb.jsonl"
    model_save_path = ROOT / "models" / "sql-micro-brain-spider-lora"

    os.makedirs(model_save_path, exist_ok=True)

    device = setup_device()

    # Model configuration
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    MAX_SEQ_LENGTH = 512

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Handle special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")

    # LoRA configuration for efficient fine-tuning
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,  # Rank
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load training data
    print(f"Loading training data from: {training_data_path}")
    dataset = load_dataset("json", data_files=str(training_data_path))

    def preprocess_function(examples):
        """Tokenize the conversations"""
        conversations = examples["messages"]
        formatted_texts = []

        for conv in conversations:
            chat_text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(chat_text)

        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        # Create labels for training (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names
    )

    # Training arguments optimized for T4 GPU (16GB VRAM)
    training_args = TrainingArguments(
        output_dir=str(model_save_path),
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small batch size for T4
        gradient_accumulation_steps=8,  # Effective batch size = 16
        gradient_checkpointing=True,  # Memory optimization
        optim="adamw_torch_fused",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        bf16=True if device == "cuda" else False,  # Mixed precision if CUDA
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard logging
    )

    # Data collator for seq2seq-like tasks, but we use it for causal LM
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=MAX_SEQ_LENGTH,
    )

    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to: {model_save_path}")
    trainer.save_model()
    tokenizer.save_pretrained(model_save_path)

    print("Training completed successfully!")
    print(f"Model saved at: {model_save_path}")


if __name__ == "__main__":
    main()
