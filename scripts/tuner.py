#!/usr/bin/env python3
"""
Multi-GPU Parameter Tuner for Progressive Tests (Flash Analysis-backed)

- Distributes trials across 8 RTX 3060 Ti GPUs (1 GPU per trial), bound via worker_ports 11434..11441
- Tunes small-model params for speed and quality proxies
- Polls the existing Progressive Test API to remain consistent with admin UI
- Persists the best config into config/config.json under FLASH_TUNING and refreshes app.state.config via config_store_mysql

Notes:
- No NVLink: we keep 70B usage optional and prefer parallel small models.
- Each HTTP call uses ≤30s timeout (project rule).
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import random
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root importable (for config_store_mysql)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Safe imports from project
from config_store_mysql import load_config, save_config, update_runtime_from_env, backup_config  # type: ignore


# Fixed mapping: 8 GPUs → 8 Ollama worker ports
DEFAULT_WORKER_PORTS = [11434, 11435, 11436, 11437, 11438, 11439, 11440, 11441]


def _detect_gpu_count() -> int:
    """Detect GPU count via nvidia-smi; return 0 on failure."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return 0
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip() != ""]
        return len(lines)
    except Exception:
        return 0


def _gpu_stats_snapshot() -> str:
    """Return a one-line GPU stats snapshot string for logging."""
    try:
        q = [
            "index",
            "name",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "temperature.gpu",
        ]
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(q)}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return "nvidia-smi unavailable"
        return " | ".join(ln.strip() for ln in proc.stdout.splitlines() if ln.strip())
    except Exception:
        return "nvidia-smi unavailable"


def _port_for_gpu_index(i: int) -> int:
    """Map GPU index to Ollama worker port. 0→11434, 7→11441."""
    i = int(i)
    base = DEFAULT_WORKER_PORTS[0]
    return base + i if 0 <= i < len(DEFAULT_WORKER_PORTS) else base


def _score_trial(objective: str, metrics: Dict[str, Any], output_text: str) -> float:
    """Compute score per objective with light quality proxies."""
    duration = float(metrics.get("duration", 0.0) or 0.0)
    chunks = int(metrics.get("chunks_processed", 0) or 0)
    tokens = int(metrics.get("tokens_used", 0) or 0)
    docs = int(metrics.get("documents", 0) or 0)

    # Quality proxies (very light): check sections and currency presence
    text = output_text or ""
    has_sections = 0
    for kw in ["Financieel", "Partijen", "Termijnen", "Aandachtspunten", "TL;DR", "Hoofdzaken"]:
        if kw.lower() in text.lower():
            has_sections += 1
    eur_hits = text.count("€") + text.lower().count("eur")
    quality_bonus = min(has_sections, 5) * 0.02 + min(eur_hits, 10) * 0.005  # small bonus

    if objective == "maximize_chunks_per_second":
        base = chunks / max(duration, 1e-6)
        return base * (1.0 + quality_bonus)
    elif objective == "maximize_documents":
        base = float(docs)
        return base * (1.0 + quality_bonus)
    elif objective == "maximize_quality_proxy":
        base = tokens / math.sqrt(duration + 1.0)
        return base * (1.0 + quality_bonus)
    else:
        # Default to chunks/sec
        base = chunks / max(duration, 1e-6)
        return base * (1.0 + quality_bonus)


async def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Simple HTTP helper using requests in a thread to avoid blocking the event loop.
    """
    import requests  # local import; requests already used elsewhere in repo
    def _do():
        if method == "GET":
            resp = requests.get(url, timeout=timeout)
        else:
            resp = requests.post(url, json=payload or {}, timeout=timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {}
    return await asyncio.to_thread(_do)


async def _start_trial(base_url: str, project_id: str, config: Dict[str, Any]) -> str:
    data = await _http_json(
        "POST",
        f"{base_url}/api/progressive-test/start",
        payload={"project_id": project_id, "config": config},
        timeout=30,
    )
    run_id = data.get("run_id", "")
    if not run_id:
        raise RuntimeError(f"Failed to start trial: {data}")
    return run_id


async def _poll_trial(base_url: str, run_id: str, max_seconds: int = 1200, interval: float = 2.0) -> Dict[str, Any]:
    """
    Poll until completed or failed. Returns status payload.
    """
    start = time.time()
    while True:
        status = await _http_json("GET", f"{base_url}/api/progressive-test/status/{run_id}", timeout=30)
        st = (status or {}).get("status", "")
        if st in ("completed", "failed"):
            return status
        if time.time() - start > max_seconds:
            raise TimeoutError(f"Trial {run_id} timed out after {max_seconds}s")
        await asyncio.sleep(interval)


async def _get_trial_result(base_url: str, run_id: str) -> Dict[str, Any]:
    return await _http_json("GET", f"{base_url}/api/progressive-test/result/{run_id}", timeout=30)


def _build_candidates(search_space: Dict[str, Any], max_trials: int) -> List[Dict[str, Any]]:
    """
    Build a candidate list from search space; cap at max_trials by random sampling if needed.
    Keys supported (aligned with configurable_analysis + flash overrides):
      - model, temperature, max_tokens, max_chunks, chunk_size, chunk_overlap, concurrency
    """
    # Defaults if keys missing
    def vals(key, default):
        v = search_space.get(key)
        if isinstance(v, list) and v:
            return v
        return default

    grid = []
    models = vals("model", ["llama3.1:8b"])
    temps = vals("temperature", [0.1, 0.15, 0.2])
    toks = vals("max_tokens", [1536, 2048, 3072])
    chunks = vals("max_chunks", [15, 25, 35])
    csize = vals("chunk_size", [600, 800, 1000, 1200])
    cover = vals("chunk_overlap", [25, 50, 100, 200])
    concs = vals("concurrency", [1, 2])

    for m in models:
        for t in temps:
            for tok in toks:
                for mk in chunks:
                    for cs in csize:
                        for co in cover:
                            for cc in concs:
                                grid.append({
                                    "model": m,
                                    "temperature": t,
                                    "max_tokens": tok,
                                    "max_chunks": mk,
                                    "chunk_size": cs,
                                    "chunk_overlap": co,
                                    "concurrency": cc,
                                })

    if len(grid) <= max_trials:
        return grid
    random.seed(42)
    return random.sample(grid, max_trials)


async def _run_single_trial(base_url: str, project_id: str, cfg: Dict[str, Any], gpu_index: int, log: List[str]) -> Dict[str, Any]:
    """
    Run one trial bound to a single GPU via worker_ports=[port_for_gpu].
    """
    port = _port_for_gpu_index(gpu_index)
    cfg = dict(cfg)  # copy
    cfg["worker_ports"] = [port]

    # GPU snapshot (before)
    before = _gpu_stats_snapshot()
    log.append(f"[trial@gpu{gpu_index}/port{port}] BEFORE: {before}")

    try:
        run_id = await _start_trial(base_url, project_id, cfg)
        status = await _poll_trial(base_url, run_id)
        st = status.get("status", "")
        if st != "completed":
            return {
                "run_id": run_id,
                "status": st or "failed",
                "metrics": {},
                "output_text": "",
                "score": -1.0,
                "config": cfg,
                "gpu_index": gpu_index,
                "port": port,
            }

        result = await _get_trial_result(base_url, run_id)
        metrics = result.get("metrics", {}) or {}
        output_text = result.get("output_text", "") or ""
        return {
            "run_id": run_id,
            "status": "completed",
            "metrics": metrics,
            "output_text": output_text,
            "config": cfg,
            "gpu_index": gpu_index,
            "port": port,
        }
    except Exception as e:
        return {
            "run_id": "",
            "status": "failed",
            "error": str(e),
            "metrics": {},
            "output_text": "",
            "config": cfg,
            "gpu_index": gpu_index,
            "port": port,
        }
    finally:
        # GPU snapshot (after)
        after = _gpu_stats_snapshot()
        log.append(f"[trial@gpu{gpu_index}/port{port}] AFTER : {after}")


async def tune_fatrag_pipeline_params(
    project_id: str,
    search_space: Dict[str, Any],
    objective: str,
    budget: Dict[str, Any],
    base_url: str = "http://127.0.0.1:8020",
) -> Dict[str, Any]:
    """
    Main entry: run parallel trials across GPUs, score, choose best, return ledger.
    """
    # Budget defaults
    max_trials = int(budget.get("max_trials", 8))
    max_total_runtime = int(budget.get("max_total_runtime_seconds", 1800))
    early_rounds = int(budget.get("early_stopping_rounds", 3))

    start_overall = time.time()
    # Discover GPUs
    gpu_count = _detect_gpu_count()
    if gpu_count <= 0:
        gpu_count = 8  # assume 8x3060 Ti setup per environment description (fallback)

    # Build candidates
    candidates = _build_candidates(search_space or {}, max_trials=max_trials)
    if not candidates:
        raise RuntimeError("Search space produced no candidates")

    # Prepare logging
    log_lines: List[str] = []
    os.makedirs(os.path.join(ROOT_DIR, "logs"), exist_ok=True)
    tuner_log_path = os.path.join(ROOT_DIR, "logs", "tuner.log")
    with open(tuner_log_path, "a", encoding="utf-8") as lf:
        lf.write(f"==== TUNER START {time.strftime('%Y-%m-%d %H:%M:%S')} project={project_id} objective={objective} max_trials={max_trials} gpus={gpu_count}\n")

    best_score = -1e18
    best_trial: Dict[str, Any] = {}
    trials_ledger: List[Dict[str, Any]] = []
    no_improve = 0

    # Concurrency: up to gpu_count trials in-flight
    sem = asyncio.Semaphore(gpu_count)

    async def _do_trial(i: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            # Assign GPU index by round-robin
            gpu_idx = i % gpu_count
            res = await _run_single_trial(base_url, project_id, cfg, gpu_idx, log_lines)
            # Compute score if completed
            if res.get("status") == "completed":
                metrics = res.get("metrics") or {}
                output_text = res.get("output_text") or ""
                score = _score_trial(objective, metrics, output_text)
                res["score"] = float(score)
            else:
                res["score"] = -1.0
            return res

    tasks = []
    for i, cfg in enumerate(candidates):
        # Respect overall budget
        if time.time() - start_overall > max_total_runtime:
            break
        tasks.append(asyncio.create_task(_do_trial(i, cfg)))

        # If we've scheduled as many as max_trials, stop scheduling
        if len(tasks) >= max_trials:
            break

    # Consume tasks as they complete and perform early stopping
    for coro in asyncio.as_completed(tasks):
        res = await coro
        trials_ledger.append(res)

        # Evaluate improvement
        score = float(res.get("score", -1.0))
        if score > best_score:
            best_score = score
            best_trial = res
            no_improve = 0
        else:
            no_improve += 1

        # Early stop condition
        if early_rounds > 0 and no_improve >= early_rounds:
            break

        # Time budget check
        if time.time() - start_overall > max_total_runtime:
            break

    # Flush tuner log with GPU snapshots and brief trial summaries
    try:
        with open(tuner_log_path, "a", encoding="utf-8") as lf:
            for ln in log_lines:
                lf.write(ln + "\n")
            lf.write(f"==== TUNER END best_score={best_score:.4f} best_cfg={json.dumps(best_trial.get('config', {}))}\n")
    except Exception:
        pass

    # Craft response
    return {
        "best_config": best_trial.get("config") or {},
        "best_metrics": best_trial.get("metrics") or {},
        "score": float(best_score),
        "trials": [
            {
                "config": t.get("config"),
                "metrics": t.get("metrics"),
                "status": t.get("status"),
                "score": t.get("score", -1.0),
                "run_id": t.get("run_id"),
                "gpu_index": t.get("gpu_index"),
                "port": t.get("port"),
                "error": t.get("error"),
            }
            for t in trials_ledger
        ],
        "gpu_count": gpu_count,
        "log_path": tuner_log_path,
        "duration_seconds": time.time() - start_overall,
    }


async def persist_best_to_config(best_config: Dict[str, Any]) -> None:
    """
    Persist winning parameters to config/config.json under FLASH_TUNING and refresh runtime.
    """
    # Map to a stable structure
    tun = {
        "model": best_config.get("model"),
        "temperature": best_config.get("temperature"),
        "max_tokens": best_config.get("max_tokens"),
        "max_chunks": best_config.get("max_chunks"),
        "chunk_size": best_config.get("chunk_size"),
        "chunk_overlap": best_config.get("chunk_overlap"),
        "concurrency": best_config.get("concurrency"),
        "worker_ports": best_config.get("worker_ports"),  # typically single-port per task
    }

    # Load, update, backup, save, refresh
    cfg = load_config()
    cfg["FLASH_TUNING"] = tun
    # Optional: update global TEMPERATURE for QA chain only if explicitly using small model
    # Leave LLM_MODEL as-is to avoid changing main QA defaults unexpectedly.

    # Backup before write
    try:
        backup_config("pre-tuning")
    except Exception:
        pass

    save_config(cfg)
    # Refresh runtime (used by main)
    update_runtime_from_env(load_config())


# CLI usage for debugging
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Tune FATRAG parameters via Progressive Tests")
    p.add_argument("--project-id", required=True)
    p.add_argument("--objective", default="maximize_chunks_per_second")
    p.add_argument("--max-trials", type=int, default=8)
    p.add_argument("--budget-seconds", type=int, default=1800)
    p.add_argument("--base-url", default="http://127.0.0.1:8020")
    args = p.parse_args()

    # Small default space
    space = {
        "model": ["llama3.1:8b", "qwen2.5:7b"],
        "temperature": [0.1, 0.15],
        "max_tokens": [1536, 2048, 3072],
        "max_chunks": [15, 25],
        "chunk_size": [600, 800, 1000],
        "chunk_overlap": [25, 50, 100],
        "concurrency": [1, 2],
    }
    budget = {"max_trials": args.max_trials, "max_total_runtime_seconds": args.budget_seconds, "early_stopping_rounds": 3}

    async def _main():
        res = await tune_fatrag_pipeline_params(
            project_id=args.project_id,
            search_space=space,
            objective=args.objective,
            budget=budget,
            base_url=args.base_url,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(_main())
