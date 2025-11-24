#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timedelta
import sys
import os
import traceback

# Ensure project root on sys.path so we can import top-level modules when running from scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import job_store_mysql as js

RUNNING_STATUSES = {
    "queued",
    "initializing",
    "preempting_gpu",
    "running",
    "running_pipeline",
    "running_l1_analysis",
    "generating",
    "saving",
    "saving_outputs",
}

def dt_to_str(dt):
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)

def list_jobs(project_id: str, limit: int = 200):
    return js.list_jobs(project_id=project_id, limit=limit)

def summarize_jobs(jobs):
    total = len(jobs)
    by_status = {}
    for j in jobs:
        s = j.get("status")
        by_status[s] = by_status.get(s, 0) + 1
    return {"total": total, "by_status": by_status}

def find_stuck(jobs, max_age_minutes: int = 20):
    now = datetime.now()
    stuck = []
    for j in jobs:
        status = j.get("status")
        if status in ("completed", "failed"):
            continue
        # treat running-like statuses as potentially stuck
        if status in RUNNING_STATUSES or status not in ("completed", "failed"):
            updated_at = j.get("updated_at")
            try:
                age_min = (now - updated_at).total_seconds() / 60.0 if isinstance(updated_at, datetime) else 0
            except Exception:
                age_min = 0
            if age_min >= max_age_minutes:
                stuck.append(j)
    return stuck

def mark_failed(job):
    job_id = job.get("job_id")
    msg = f"Auto-marked stale at {datetime.now().isoformat()}"
    return js.update_job(job_id, status="failed", progress=100, error_message=msg)

def main():
    p = argparse.ArgumentParser(description="Inspect and cleanup FATRAG jobs per project")
    p.add_argument("--project-id", required=True, help="project id, e.g., project-06cfd7c8")
    p.add_argument("--limit", type=int, default=200, help="max jobs to fetch")
    p.add_argument("--max-age-min", type=int, default=20, help="age threshold (minutes) to consider a running job stale")
    p.add_argument("--resolve-stuck", action="store_true", help="mark stale running jobs as failed")
    p.add_argument("--json", action="store_true", help="output JSON for machine reading")
    args = p.parse_args()

    try:
        jobs = list_jobs(args.project_id, args.limit)
        summary = summarize_jobs(jobs)
        stuck = find_stuck(jobs, args.max_age_min)

        out = {
            "project_id": args.project_id,
            "summary": summary,
            "total_jobs": len(jobs),
            "stuck_count": len(stuck),
            "stuck_jobs": [
                {
                    "job_id": j.get("job_id"),
                    "job_type": j.get("job_type"),
                    "status": j.get("status"),
                    "progress": j.get("progress"),
                    "updated_at": dt_to_str(j.get("updated_at")),
                    "created_at": dt_to_str(j.get("created_at")),
                }
                for j in stuck
            ],
        }

        if args.resolve_stuck and stuck:
            resolved = []
            for j in stuck:
                try:
                    r = mark_failed(j)
                    if r:
                        resolved.append(r.get("job_id"))
                except Exception:
                    traceback.print_exc()
            out["resolved_job_ids"] = resolved

        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"Project: {args.project_id}")
            print(f"Total jobs: {out['total_jobs']}")
            print("By status:", out["summary"]["by_status"])
            print(f"Stuck jobs (>{args.max_age_min} min not completed/failed): {out['stuck_count']}")
            for j in out["stuck_jobs"]:
                print(f" - {j['job_id']} {j['job_type']} status={j['status']} progress={j['progress']} updated_at={j['updated_at']}")
            if args.resolve_stuck and out.get("resolved_job_ids"):
                print("Resolved (marked failed):", out["resolved_job_ids"])

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
        else:
            print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
