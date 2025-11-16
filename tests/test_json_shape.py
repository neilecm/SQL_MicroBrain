import json
from pathlib import Path
import pytest

# We import from cli.db_assistant
from cli.db_assistant import validate_response_shape

def test_valid_shape_passes():
    good = {
        "actions": [],
        "migrations": [],
        "rls_policies": [],
        "indexes": [],
        "queries": [],
        "error_explanations": [],
        "explanations": [],
        "safe_to_execute": False,
    }
    # Should not raise
    validate_response_shape(good)

def test_missing_key_fails():
    bad = {
        "actions": [],
        "migrations": [],
        # rls_policies missing
        "indexes": [],
        "queries": [],
        "error_explanations": [],
        "explanations": [],
        "safe_to_execute": False,
    }
    with pytest.raises(ValueError):
        validate_response_shape(bad)
