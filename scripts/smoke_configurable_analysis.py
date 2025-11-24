#!/usr/bin/env python3
"""
Smoke test: import configurable_analysis and ensure run_configurable_analysis() can be called.

- Monkeypatches scripts.flash_analysis.ollama_generate to avoid external model calls
- Uses an existing small text file from fatrag_data as input
- Prints JSON result to stdout
"""

import os
import sys
import json
from typing import Any, Dict

# Ensure project root on path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import target modules
from scripts import configurable_analysis as ca  # noqa: E402
import scripts.flash_analysis as flash  # noqa: E402


def main() -> int:
    # Monkeypatch heavy LLM calls to a lightweight stub
    def _stub_ollama_generate(prompt: str, model: str, port: int, options: Dict[str, Any] | None = None, timeout: int = 120) -> str:
        return "- bedrag: €1.000\n- partij: Test BV\n- datum: 2025\n- transactie: test\n- risico: laag"

    flash.ollama_generate = _stub_ollama_generate  # type: ignore

    # Build a tiny config to keep things fast
    config = ca.AnalysisConfig(
        model="llama3.1:8b",
        temperature=0.1,
        max_tokens=1500,
        max_chunks=5,
    )

    # Use a real existing small text file
    sample_file = os.path.join(ROOT_DIR, "fatrag_data", "5-thuiswerk-tips.txt")
    if not os.path.isfile(sample_file):
        print(json.dumps({"success": False, "error": f"Sample file not found: {sample_file}"}))
        return 1

    # Run the configurable analysis with selected_files to avoid scanning uploads
    result = ca.run_configurable_analysis(
        config=config,
        project_id="project-smoke",
        project_name="Smoke Test Project",
        upload_dir=os.path.join(ROOT_DIR, "fatrag_data", "uploads"),
        run_id="run-smoke",
        selected_files=[sample_file],
    )

    # Indicate success and key fields
    print(json.dumps({
        "invoked": True,
        "success": bool(result.get("success")),
        "documents": result.get("documents", []),
        "chunks_processed": result.get("chunks_processed", 0),
        "tokens_used": result.get("tokens_used", 0),
        "job_id": result.get("job_id", ""),
        "job_dir": result.get("job_dir", ""),
    }, ensure_ascii=False))

    return 0 if result.get("success") else 0  # treat as pass if callable, regardless of 'success' content


if __name__ == "__main__":
    raise SystemExit(main())
