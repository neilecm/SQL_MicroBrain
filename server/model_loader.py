import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.sql-micro-brain.md"

# Default model config
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2

# Device detection
if torch.backends.mps.is_available():
    DEVICE = "mps"
    logger.info("Using MPS (Apple Silicon)")
else:
    DEVICE = "cpu"
    logger.info("Using CPU (MPS not available)")


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt not found at {SYSTEM_PROMPT_PATH}")
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _try_parse_json(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON, trying to extract a JSON object substring if needed."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON between first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            substring = text[start:end+1]
            try:
                return json.loads(substring)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse JSON from: {text[:200]}...")


@lru_cache(maxsize=1)
def get_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer, optionally applying LoRA adapter."""
    base_model_name = os.getenv("SQL_MB_BASE_MODEL_NAME", DEFAULT_MODEL_NAME)
    lora_path = os.getenv("SQL_MB_LORA_PATH", "").strip()

    logger.info("Loading base model: %s", base_model_name)
    logger.info("LoRA path: %s", lora_path or "(none, using base model only)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # MPS stable
    )

    # Apply LoRA if configured
    if lora_path and os.path.isdir(lora_path):
        logger.info("Applying LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        if lora_path:
            logger.warning("LoRA path %s is not a valid directory; using base model only.", lora_path)
        model = base_model

    # Move to device
    model.to(DEVICE)
    model.eval()

    logger.info("Model loaded and moved to device: %s", DEVICE)
    return model, tokenizer


def run_sql_microbrain(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs SQL Micro-Brain inference using Qwen2.5-Coder-1.5B-Instruct (with optional LoRA fine-tuning).

    - Loads system prompt from prompts/system.sql-micro-brain.md
    - Builds chat messages: system + user input
    - Generates JSON response using configured model (base + optional LoRA)
    - Parses response as SQL Micro-Brain JSON schema
    - Returns validated dict
    """
    model, tokenizer = get_model_and_tokenizer()

    system_prompt = load_system_prompt()
    user_content = json.dumps(input_payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(DEVICE)

    # Generate
    max_new_tokens = int(os.getenv("SQL_MB_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS)))
    temperature = float(os.getenv("SQL_MB_TEMPERATURE", str(DEFAULT_TEMPERATURE)))

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode new tokens
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generated_text = generated_text.strip()

    # Try to parse JSON
    try:
        resp_obj = _try_parse_json(generated_text)
        logger.debug("JSON parsed successfully on first attempt")
    except ValueError:
        logger.warning("Initial JSON parse failed, attempting repair generation")
        # Repair prompt
        repair_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": generated_text},
            {"role": "user", "content": "The following is supposed to be a JSON object for SQL Micro-Brain. Fix it so it is valid JSON with double quotes and no trailing commas. Return ONLY the JSON:\n\n" + generated_text}
        ]
        repair_input_text = tokenizer.apply_chat_template(
            repair_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        repair_inputs = tokenizer(
            repair_input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(DEVICE)

        with torch.no_grad():
            repair_outputs = model.generate(
                **repair_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=False,  # Deterministic for repair
                pad_token_id=tokenizer.eos_token_id,
            )
        repair_generated = tokenizer.decode(repair_outputs[0][repair_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        repair_generated = repair_generated.strip()

        try:
            resp_obj = _try_parse_json(repair_generated)
            logger.debug("JSON parsed successfully after repair")
        except ValueError:
            raise ValueError(f"Model failed to produce valid JSON after repair. Raw output: {repair_generated[:500]}...")

    # Validate shape (reuse existing validation)
    validate_response_shape(resp_obj)

    return resp_obj


def validate_response_shape(obj: Dict[str, Any]) -> None:
    """Same validation as in cli/db_assistant.py"""
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
    if not isinstance(obj["safe_to_execute"], bool):
        raise ValueError("safe_to_execute must be a boolean")
