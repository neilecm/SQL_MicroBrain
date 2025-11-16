"""
Tests for SQL Micro-Brain v0: sample inference tasks and shape validation.
"""

import pytest
from server.model_loader import run_sql_microbrain


def test_basic_schema_design_task():
    """
    Test a simple schema design task: ensure model returns JSON with correct shape.
    """
    input_payload = {
        "mode": "design_schema",
        "natural_language_task": "Create tables for books and authors",
        "current_schema": "",
        "preferences": {
            "db_engine": "supabase",
            "supabase_style": True,
            "naming": "snake_case",
            "id_type": "uuid",
            "multi_tenant": False,
        },
        "sql_snippets": {
            "problem_query": "",
            "explain_analyze": ""
        },
        "error_message": ""
    }

    # Run inference (this will load the model if not already loaded)
    result = run_sql_microbrain(input_payload)

    # Assert it's a dict
    assert isinstance(result, dict)

    # Assert required keys are present (shape validation is done inside run_sql_microbrain)
    required_keys = [
        "actions", "migrations", "rls_policies", "indexes", "queries",
        "error_explanations", "explanations", "safe_to_execute"
    ]
    for key in required_keys:
        assert key in result

    # Assert types
    assert isinstance(result["actions"], list)
    assert isinstance(result["migrations"], list)
    assert isinstance(result["rls_policies"], list)
    assert isinstance(result["indexes"], list)
    assert isinstance(result["queries"], list)
    assert isinstance(result["error_explanations"], list)
    assert isinstance(result["explanations"], list)
    assert isinstance(result["safe_to_execute"], bool)

    # For this task, expect migrations to have something
    assert len(result["migrations"]) > 0, "Should generate at least one migration for schema design"
