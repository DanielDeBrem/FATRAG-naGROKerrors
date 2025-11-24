#!/usr/bin/env python3
"""
Smoke test for the multi-GPU tuner endpoint.

- Calls /api/progressive-test/tune with a tiny search space
- Prints best_config, score, and a compact trials summary
- Does not persist changes by default (set persist=True to write FLASH_TUNING)
- Requires the FastAPI server to be running on port 8020 and 8 Ollama workers started

Usage:
  python3 scripts/smoke_tuner.py --project-id project-XXXX [--persist]
"""

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test tuner endpoint")
    p.add_argument("--project-id", required=True, help="Target project id")
    p.add_argument("--host", default="http://127.0.0.1:8020", help="API base url")
    p.add_argument("--persist", action="store_true", help="Persist best config to config/config.json")
    args = p.parse_args()

    url = f"{args.host}/api/progressive-test/tune"
    body: Dict[str, Any] = {
        "project_id": args.project_id,
        "search_space": {
            "model": ["llama3.1:8b"],          # keep small for smoke
            "temperature": [0.1, 0.15],
            "max_tokens": [1536, 2048],
            "max_chunks": [15, 25],
            "chunk_size": [600, 800],
            "chunk_overlap": [25, 50],
            "concurrency": [1]                 # low to avoid VRAM thrash
        },
        "objective": "maximize_chunks_per_second",
        "budget": {"max_trials": 3, "max_total_runtime_seconds": 600, "early_stopping_rounds": 2},
        "persist": bool(args.persist),
    }

    try:
        resp = requests.post(url, json=body, timeout=30)
        if resp.status_code != 200:
            print(f"❌ Tuner request failed: {resp.status_code} {resp.text}")
            return 1
        data = resp.json()
    except Exception as e:
        print(f"❌ HTTP error calling tuner: {e}")
        return 1

    best = data.get("best_config", {})
    score = data.get("score")
    trials = data.get("trials", [])

    print("✅ Tuner completed")
    print(f"Best score: {score}")
    print(f"Best config: {json.dumps(best, ensure_ascii=False)}")
    print(f"Trials: {len(trials)}")
    for t in trials:
        cfg = t.get("config", {})
        met = t.get("metrics", {})
        print(f"- status={t.get('status')} score={t.get('score')} gpu={t.get('gpu_index')} port={t.get('port')} "
              f"chunks={met.get('chunks_processed')} dur={met.get('duration'):.1f}s tokens={met.get('tokens_used')} "
              f"cfg={{temp={cfg.get('temperature')}, tok={cfg.get('max_tokens')}, max_chunks={cfg.get('max_chunks')}, "
              f"chunk_size={cfg.get('chunk_size')}, overlap={cfg.get('chunk_overlap')}, conc={cfg.get('concurrency')}}}")

    if args.persist:
        print("ℹ️ Persist requested. FLASH_TUNING should now be stored in config/config.json.")
    else:
        print("ℹ️ Persistence disabled. Run with --persist to save best FLASH_TUNING to config.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
