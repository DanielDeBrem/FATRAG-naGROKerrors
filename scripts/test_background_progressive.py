#!/usr/bin/env python3
"""
Background smoke test for _background_progressive_test

- Monkeypatches heavy deps (Ollama calls, QualityRatings store, project loader)
- Copies a small sample file into fatrag_data/uploads
- Invokes the background function directly (async) and prints outcome
"""

import os
import sys
import json
import shutil
import asyncio
from typing import Any, Dict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Prepare uploads dir and a small sample file
uploads_dir = os.path.join(ROOT_DIR, "fatrag_data", "uploads")
os.makedirs(uploads_dir, exist_ok=True)
sample_src = os.path.join(ROOT_DIR, "fatrag_data", "5-thuiswerk-tips.txt")
sample_dst = os.path.join(uploads_dir, "sample_upload.txt")
if os.path.isfile(sample_src):
    try:
        shutil.copy(sample_src, sample_dst)
    except Exception:
        pass

# Monkeypatch QualityRatings and clients_projects before importing main
import quality_ratings as qr  # noqa: E402
import clients_projects as cp  # noqa: E402

class DummyQR:
    def __init__(self) -> None:
        self.events = []

    def update_test_run_status(
        self,
        run_id: str,
        status: str,
        progress: int = 0,
        stage: str = "",
        error_message: str | None = None,
        output_preview: str = "",
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        self.events.append({
            "run_id": run_id,
            "status": status,
            "progress": progress,
            "stage": stage,
            "error_message": error_message,
            "output_preview": output_preview[:120] if output_preview else "",
            "metrics": metrics or {}
        })

# Patch the factory
qr.QualityRatings = lambda: DummyQR()  # type: ignore[assignment]

# Patch project loader to point to our uploaded sample
def fake_get_project_with_documents(project_id: str) -> Dict[str, Any]:
    return {
        "project_id": project_id,
        "name": "BG Smoke Project",
        "documents": [{"filename": "sample_upload.txt"} if os.path.isfile(sample_dst) else {}]
    }

cp.get_project_with_documents = fake_get_project_with_documents  # type: ignore[assignment]

# Patch ollama_generate to avoid external calls
import scripts.flash_analysis as flash  # noqa: E402
def _stub_ollama_generate(prompt: str, model: str, port: int, options: Dict[str, Any] | None = None, timeout: int = 120) -> str:
    return "- bedrag: €1.000\n- partij: Test BV\n- datum: 2025\n- transactie: test\n- risico: laag"
flash.ollama_generate = _stub_ollama_generate  # type: ignore[assignment]

# Now import the target function (this imports main and FastAPI app)
from main import _background_progressive_test  # noqa: E402


async def run_test() -> Dict[str, Any]:
    run_id = "run-bg-smoke"
    project_id = "project-bg-smoke"
    config = {
        "model": "llama3.1:8b",
        "temperature": 0.1,
        "max_tokens": 1500,
        "max_chunks": 5
    }

    # Execute background task
    await _background_progressive_test(run_id, config, project_id)

    # Collect simple artifacts
    log_path = os.path.join(ROOT_DIR, "logs", "progressive_test.log")
    log_exists = os.path.isfile(log_path)
    latest_log_tail = ""
    if log_exists:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                latest_log_tail = "".join(lines[-10:])  # last 10 lines
        except Exception:
            pass

    return {
        "invoked": True,
        "log_exists": log_exists,
        "log_tail": latest_log_tail[-500:],  # tail preview
        "uploads_has_sample": os.path.isfile(sample_dst)
    }


def main() -> int:
    try:
        result = asyncio.run(run_test())
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as e:
        print(json.dumps({"invoked": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
