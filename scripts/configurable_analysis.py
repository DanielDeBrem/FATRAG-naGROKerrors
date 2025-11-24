#!/usr/bin/env python3
"""
Configurable Analysis Wrapper for Progressive Testing.

Wraps flash_analysis.py with configurable parameters and metrics tracking.
"""

from __future__ import annotations

import os
import sys
import time
import json
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flash_analysis module
import scripts.flash_analysis as flash


class AnalysisConfig:
    """Configuration for analysis run"""
    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.15,
        max_tokens: int = 3072,
        max_chunks: int = 25,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_chunks = max_chunks


def run_configurable_analysis(
    config: AnalysisConfig,
    project_id: str,
    project_name: str,
    upload_dir: str,
    run_id: str,
    selected_files: Optional[list] = None,
    worker_ports: Optional[list[int]] = None,
    concurrency: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run analysis with custom configuration and track metrics.
    
    Args:
        config: AnalysisConfig with model/temp/tokens/chunks settings
        project_id: Project identifier
        project_name: Human-readable project name
        upload_dir: Directory containing documents
        run_id: Test run identifier for tracking
        selected_files: Optional list of specific files to analyze
    
    Returns:
        Dictionary with output_text, duration, tokens_used, chunks_processed, metadata
    """
    
    start_time = time.time()
    
    # Temporarily override global settings
    original_model = flash.FLASH_MODEL
    original_temp = flash.TEMPERATURE
    original_ctx = flash.FLASH_NUM_CTX
    
    try:
        # Apply configuration
        flash.FLASH_MODEL = config.model
        flash.TEMPERATURE = config.temperature
        flash.FLASH_NUM_CTX = config.max_tokens
        
        # Patch sample_chunks to use custom max_chunks
        original_sample = flash.sample_chunks
        
        def custom_sample(chunks, max_chunks=None):
            return original_sample(chunks, max_chunks=config.max_chunks)
        
        flash.sample_chunks = custom_sample
        
        # Run analysis
        result = flash.run_flash_analysis(
            project_name=project_name,
            upload_dir=upload_dir,
            selected_files=selected_files,
            worker_ports=worker_ports,
            concurrency=concurrency,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        duration = time.time() - start_time
        
        # Read output text
        report_path = result.get("report_path", "")
        output_text = ""
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        
        # Count chunks processed (approximate from output)
        chunks_processed = 0
        metadata_path = os.path.join(result.get("job_dir", ""), "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                chunks_processed = meta.get("total_chunks", 0)
        
        # Estimate tokens (rough: 4 chars per token)
        tokens_used = len(output_text) // 4 if output_text else 0
        
        return {
            "success": True,
            "output_text": output_text,
            "duration": duration,
            "tokens_used": tokens_used,
            "chunks_processed": chunks_processed,
            "job_id": result.get("job_id", ""),
            "job_dir": result.get("job_dir", ""),
            "documents": result.get("documents", []),
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "max_chunks": config.max_chunks,
        }
    
    except Exception as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "error": str(e),
            "duration": duration,
            "tokens_used": 0,
            "chunks_processed": 0,
        }
    
    finally:
        # Restore original settings
        flash.FLASH_MODEL = original_model
        flash.TEMPERATURE = original_temp
        flash.FLASH_NUM_CTX = original_ctx
        flash.sample_chunks = original_sample


def get_preset_config(level: int) -> AnalysisConfig:
    """
    Get preset configuration for testing levels.
    
    Level 1: Fast, deterministic
    Level 2: Balanced
    Level 3: Comprehensive
    """
    presets = {
        1: AnalysisConfig(
            model="llama3.1:8b",
            temperature=0.1,
            max_tokens=1500,
            max_chunks=15,
        ),
        2: AnalysisConfig(
            model="llama3.1:8b",
            temperature=0.15,
            max_tokens=2500,
            max_chunks=25,
        ),
        3: AnalysisConfig(
            model="llama3.1:8b",
            temperature=0.2,
            max_tokens=3000,
            max_chunks=35,
        ),
    }
    
    return presets.get(level, presets[2])


if __name__ == "__main__":
    import argparse
    
    p = argparse.ArgumentParser(description="Run configurable analysis")
    p.add_argument("--project-id", required=True, help="Project ID")
    p.add_argument("--project-name", default="Test Project", help="Project name")
    p.add_argument("--upload-dir", required=True, help="Upload directory")
    p.add_argument("--run-id", required=True, help="Test run ID")
    p.add_argument("--level", type=int, default=2, help="Preset level (1-3)")
    p.add_argument("--model", help="Model override")
    p.add_argument("--temp", type=float, help="Temperature override")
    p.add_argument("--tokens", type=int, help="Max tokens override")
    p.add_argument("--chunks", type=int, help="Max chunks override")
    
    args = p.parse_args()
    
    # Get preset or build custom config
    if args.model or args.temp or args.tokens or args.chunks:
        config = AnalysisConfig(
            model=args.model or "llama3.1:8b",
            temperature=args.temp or 0.15,
            max_tokens=args.tokens or 2500,
            max_chunks=args.chunks or 25,
        )
    else:
        config = get_preset_config(args.level)
    
    result = run_configurable_analysis(
        config=config,
        project_id=args.project_id,
        project_name=args.project_name,
        upload_dir=args.upload_dir,
        run_id=args.run_id,
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
