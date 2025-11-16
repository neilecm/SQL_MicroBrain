"""
SQL Micro-Brain v0 HTTP API

Serves the Qwen2.5-Coder-1.5B-Instruct model for PostgreSQL/Supabase schema and SQL tasks.

Endpoints:
- POST /infer: Accepts input payload, returns SQL Micro-Brain JSON response
- GET /healthz: Simple health check
"""

import json
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Add current directory to path for relative imports
sys.path.append('.')

from server.model_loader import run_sql_microbrain

app = FastAPI(
    title="SQL Micro-Brain v0",
    description="Specialist AI for PostgreSQL/Supabase schema design, migrations, RLS, and SQL generation",
    version="0.1.0",
)

# Pydantic models for request/response

class SQLMicroBrainRequest(BaseModel):
    mode: str
    natural_language_task: str
    current_schema: Optional[str] = ""
    preferences: Optional[Dict[str, Any]] = None
    sql_snippets: Optional[Dict[str, str]] = None
    error_message: Optional[str] = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "natural_language_task": self.natural_language_task,
            "current_schema": self.current_schema or "",
            "preferences": self.preferences or {},
            "sql_snippets": self.sql_snippets or {},
            "error_message": self.error_message or "",
        }

@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/infer")
async def infer(request: SQLMicroBrainRequest):
    """
    Run SQL Micro-Brain inference on the provided input payload.

    Returns the full JSON response conforming to the required schema.
    """
    try:
        input_payload = request.to_dict()

        # Validate mode
        allowed_modes = ["design_schema", "write_sql", "fix_error", "optimize_query", "design_rls"]
        if input_payload["mode"] not in allowed_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode. Allowed: {', '.join(allowed_modes)}"
            )

        result = run_sql_microbrain(input_payload)
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Example usage:
# curl -X POST "http://localhost:8000/infer" -H "Content-Type: application/json" -d '{
#   "mode": "design_schema",
#   "natural_language_task": "Create tables for books and authors"
# }'
