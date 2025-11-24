#!/usr/bin/env python3
"""
Reduce-only resume helper.

Reads existing map_chunks from a job directory and synthesizes per-document
summaries sequentially using available Ollama models (no new pulls). This avoids
spawning a large parallel workload and lets you pick models via environment
variables.

Usage:
  python3 scripts/reduce_resume.py --job-dir outputs/job-YYYYMMDD_HHMMSS [--limit N] [--port 11434]

Environment (optional):
  REDUCE_MODEL            e.g. "gemma2:27b" (default if unset)
  FALLBACK_REDUCE_MODEL   e.g. "llama3.1:8b" (default if unset)

Notes:
- Writes doc_summaries/<doc>.md files; existing ones are skipped.
- Timeouts are conservative to avoid long-hanging calls.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, List, Tuple

# Reuse helpers and prompts from the main pipeline
# Ensure project root on sys.path so this script can import top-level modules when run directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import analysis_pipeline as ap


def pick_models_from_env() -> Tuple[str, str]:
    reduce_model = os.getenv("REDUCE_MODEL", "gemma2:27b")
    fallback = os.getenv("FALLBACK_REDUCE_MODEL", "llama3.1:8b")
    return reduce_model, fallback


def reduce_one(
    project_name: str,
    doc_name: str,
    bullets: List[str],
    port: int,
    reduce_model: str,
    fallback_model: str,
    out_path: str,
    num_ctx: int,
    temperature: float,
    num_predict: int,
    timeout_s: int,
) -> None:
    prompt = ap.build_reduce_prompt(project_name, doc_name, bullets)
    options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict}
    # Shorter timeouts to stay responsive
    try:
        resp = ap.ollama_generate(prompt, reduce_model, port, options=options, timeout=timeout_s)
    except RuntimeError as e:
        msg = str(e).lower()
        if "404" in msg or ("model" in msg and "not found" in msg):
            resp = ap.ollama_generate(prompt, fallback_model, port, options=options, timeout=timeout_s)
        else:
            raise
    # Persist
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp)


def main() -> int:
    p = argparse.ArgumentParser(description="Reduce-only resume on existing job dir (sequential)")
    p.add_argument("--job-dir", required=True, help="Path to outputs/job-YYYYMMDD_HHMMSS")
    p.add_argument("--project-name", default="Project", help="Project name used in prompts")
    p.add_argument("--port", type=int, default=11434, help="Worker port to use (default 11434)")
    p.add_argument("--limit", type=int, default=0, help="Process at most N documents (0 = all)")
    p.add_argument("--num-ctx", type=int, default=2048, help="Context window for reduce model (tokens)")
    p.add_argument("--num-predict", type=int, default=256, help="Max tokens to generate")
    p.add_argument("--timeout", type=int, default=180, help="Timeout (seconds) per reduce call")
    p.add_argument("--max-bullets", type=int, default=200, help="Use at most N bullets per document (0 = no limit)")
    args = p.parse_args()

    job_dir = os.path.abspath(args.job_dir)
    if not os.path.isdir(job_dir):
        print(f"[ERR] job dir not found: {job_dir}", file=sys.stderr)
        return 1

    map_dir = os.path.join(job_dir, "map_chunks")
    sum_dir = os.path.join(job_dir, "doc_summaries")
    os.makedirs(sum_dir, exist_ok=True)

    # Load bullets per doc
    grouped: Dict[str, List[str]] = ap.load_map_bullets_from_dir(job_dir)
    if not grouped:
        print(f"[ERR] no map_chunks found in {map_dir}", file=sys.stderr)
        return 1

    reduce_model, fallback_model = pick_models_from_env()
    print(f"[INFO] Using REDUCE_MODEL={reduce_model} FALLBACK_REDUCE_MODEL={fallback_model}")
    print(f"[INFO] Using worker port :{args.port}")
    print(f"[INFO] Documents in scope: {len(grouped)}")

    # Process sequentially to keep resource usage modest
    processed = 0
    keys = sorted(grouped.keys())
    for i, doc_key in enumerate(keys, 1):
        out_path = os.path.join(sum_dir, f"{doc_key}.md")
        if os.path.isfile(out_path):
            print(f"[SKIP] already exists: {os.path.relpath(out_path, job_dir)}")
        else:
            bullets = grouped.get(doc_key, []) or []
            bul = bullets[:args.max_bullets] if args.max_bullets and args.max_bullets > 0 else bullets
            try:
                reduce_one(
                    project_name=args.project_name,
                    doc_name=doc_key,
                    bullets=bul,
                    port=args.port,
                    reduce_model=reduce_model,
                    fallback_model=fallback_model,
                    out_path=out_path,
                    num_ctx=args.num_ctx,
                    temperature=ap.TEMPERATURE,
                    num_predict=args.num_predict,
                    timeout_s=args.timeout,
                )
                print(f"[OK] wrote {os.path.relpath(out_path, job_dir)}")
                processed += 1
            except Exception as e:
                # Friendly, short error; continue
                err_log = os.path.join(sum_dir, "errors.log")
                with open(err_log, "a", encoding="utf-8") as ef:
                    ef.write(f"[REDUCE-ERROR] {doc_key}: {str(e)}\n")
                print(f"[ERR] {doc_key}: {str(e)}", file=sys.stderr)

        if args.limit and processed >= args.limit:
            break

    print(f"[DONE] processed={processed} (limit={args.limit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
