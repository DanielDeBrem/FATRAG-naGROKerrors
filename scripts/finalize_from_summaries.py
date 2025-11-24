#!/usr/bin/env python3
"""
Finalize from existing doc_summaries and evidence.csv.

- Reads all *.md under outputs/<job>/doc_summaries
- Reads evidence.csv if present
- Builds final prompt (NL, bullets) and calls BIG server model (default llama3.1:70b)
- Writes final.txt and final.json into the job directory

Usage:
  python3 scripts/finalize_from_summaries.py --job-dir outputs/job-YYYYMMDD_HHMMSS \
    [--project-name "name"] [--num-ctx 8192] [--num-predict 300] [--timeout 30]
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import analysis_pipeline as ap  # reuse prompts, http client, and constants


def load_doc_summaries(sum_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isdir(sum_dir):
        return out
    for fname in sorted(os.listdir(sum_dir)):
        if not fname.endswith(".md"):
            continue
        fpath = os.path.join(sum_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
        except Exception:
            text = ""
        key = fname[:-3]  # strip .md
        out[key] = text
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Finalize from doc_summaries + evidence.csv")
    p.add_argument("--job-dir", required=True, help="Path to outputs/job-YYYYMMDD_HHMMSS")
    p.add_argument("--project-name", default="Project", help="Project name used in prompts")
    p.add_argument("--num-ctx", type=int, default=8192, help="Context window for final model")
    p.add_argument("--num-predict", type=int, default=300, help="Max tokens to generate")
    p.add_argument("--timeout", type=int, default=30, help="Timeout (seconds) for final call")
    p.add_argument("--port", type=int, default=ap.BIG_PORT, help="Ollama worker port (default BIG_PORT)")
    p.add_argument("--max-docs", type=int, default=0, help="Use at most N doc summaries (0 = all)")
    p.add_argument("--max-chars-per-doc", type=int, default=1500, help="Trim each doc summary to first N chars (0 = no trim)")
    args = p.parse_args()

    job_dir = os.path.abspath(args.job_dir)
    sum_dir = os.path.join(job_dir, "doc_summaries")
    if not os.path.isdir(sum_dir):
        print(f"[ERR] doc_summaries directory not found: {sum_dir}", file=sys.stderr)
        return 1

    doc_summaries = load_doc_summaries(sum_dir)
    if not doc_summaries:
        print(f"[ERR] no doc_summaries *.md files found in {sum_dir}", file=sys.stderr)
        return 1

    # Apply limits to keep request small and within 30s timeout
    if args.max_docs and args.max_docs > 0:
        limited = {}
        for k in sorted(doc_summaries.keys())[:args.max_docs]:
            limited[k] = doc_summaries[k]
        doc_summaries = limited
    if args.max_chars_per_doc and args.max_chars_per_doc > 0:
        for k in list(doc_summaries.keys()):
            v = doc_summaries[k] or ""
            if len(v) > args.max_chars_per_doc:
                doc_summaries[k] = v[:args.max_chars_per_doc]

    evidence_rows = ap.load_evidence_csv(job_dir)  # best-effort

    # Build final prompt and call BIG server model
    final_prompt = ap.build_final_prompt(args.project_name, doc_summaries, evidence_rows)
    model = os.getenv("FINAL_MODEL", ap.FINAL_MODEL)
    options = {"temperature": ap.TEMPERATURE, "num_ctx": args.num_ctx, "num_predict": args.num_predict}

    try:
        final_text = ap.ollama_generate(final_prompt, model, args.port, options=options, timeout=args.timeout)
    except Exception as e:
        print(f"[ERR] final generation failed: {e}", file=sys.stderr)
        return 2

    # Save final artifacts
    txt_path = os.path.join(job_dir, "final.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    # Build a simple structured JSON similar to analysis_pipeline
    sections = {
        "tldr": [],
        "financiele_kernpunten": [],
        "fiscale_impact": [],
        "juridische_structuur_partijen": [],
        "risicos_aannames_onvoldoende_data": [],
        "aanpak_stappenplan": [],
        "open_vragen": [],
        "raw": final_text,
    }
    tag_map = {
        "tl;dr": "tldr",
        "financiële kernpunten": "financiele_kernpunten",
        "fiscale impact": "fiscale_impact",
        "juridische structuur": "juridische_structuur_partijen",
        "risico's": "risicos_aannames_onvoldoende_data",
        "aanpak": "aanpak_stappenplan",
        "open vragen": "open_vragen",
    }
    cur_key = None
    for line in final_text.splitlines():
        l = line.strip()
        low = l.lower()
        for k, v in tag_map.items():
            if low.startswith(k):
                cur_key = v
                break
        if l.startswith("-") and cur_key:
            sections[cur_key].append(l)

    json_path = os.path.join(job_dir, "final.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(sections, jf, ensure_ascii=False, indent=2)

    print(f"[OK] final artifacts written:\n - {txt_path}\n - {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
