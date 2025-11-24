"""
Performance Metrics Tracking voor FATRAG Analyses
Tracks timing, token usage, quality scores voor alle analyse types
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

class MetricsStore:
    """Track performance metrics voor analyse optimization"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
    def log_analysis_start(self, job_id: str, analysis_type: str, config: Dict[str, Any]) -> str:
        """Start tracking een analyse run"""
        run_id = f"{analysis_type}_{int(time.time())}"
        
        metrics = {
            "run_id": run_id,
            "job_id": job_id,
            "analysis_type": analysis_type,
            "config": config,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "stages": [],
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "quality_scores": {},
            "errors": [],
            "status": "running"
        }
        
        # Save initial metrics
        metrics_file = self.metrics_dir / f"{run_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return run_id
    
    def log_stage(self, run_id: str, stage_name: str, duration: float, tokens_used: int = 0, metadata: Dict = None):
        """Log a stage within the analysis"""
        metrics = self._load_metrics(run_id)
        
        stage_data = {
            "name": stage_name,
            "duration_seconds": duration,
            "tokens_used": tokens_used,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        metrics["stages"].append(stage_data)
        metrics["tokens"]["total"] += tokens_used
        
        self._save_metrics(run_id, metrics)
    
    def log_analysis_complete(self, run_id: str, result: Dict, quality_scores: Dict = None):
        """Mark analysis as complete and log final metrics"""
        metrics = self._load_metrics(run_id)
        
        metrics["end_time"] = datetime.now().isoformat()
        start = datetime.fromisoformat(metrics["start_time"])
        end = datetime.fromisoformat(metrics["end_time"])
        metrics["duration_seconds"] = (end - start).total_seconds()
        
        if quality_scores:
            metrics["quality_scores"] = quality_scores
            
        metrics["status"] = "completed"
        metrics["result_summary"] = {
            "output_length": len(str(result)),
            "has_error": bool(metrics.get("errors"))
        }
        
        self._save_metrics(run_id, metrics)
        
    def log_error(self, run_id: str, error: str, stage: str = None):
        """Log an error during analysis"""
        metrics = self._load_metrics(run_id)
        
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error": error
        }
        
        metrics["errors"].append(error_data)
        metrics["status"] = "failed"
        
        self._save_metrics(run_id, metrics)
    
    def get_metrics(self, run_id: str) -> Dict:
        """Get metrics for a specific run"""
        return self._load_metrics(run_id)
    
    def get_analysis_stats(self, analysis_type: str, limit: int = 50) -> Dict:
        """Get aggregate stats for an analysis type"""
        runs = []
        
        for metrics_file in self.metrics_dir.glob(f"{analysis_type}_*.json"):
            with open(metrics_file) as f:
                runs.append(json.load(f))
        
        # Sort by start time, most recent first
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        runs = runs[:limit]
        
        if not runs:
            return {"error": "No runs found"}
        
        completed = [r for r in runs if r["status"] == "completed"]
        
        if not completed:
            return {
                "total_runs": len(runs),
                "completed_runs": 0,
                "failed_runs": len([r for r in runs if r["status"] == "failed"])
            }
        
        durations = [r["duration_seconds"] for r in completed if r.get("duration_seconds")]
        tokens = [r["tokens"]["total"] for r in completed]
        
        stats = {
            "analysis_type": analysis_type,
            "total_runs": len(runs),
            "completed_runs": len(completed),
            "failed_runs": len([r for r in runs if r["status"] == "failed"]),
            "duration": {
                "mean": statistics.mean(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0,
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "stdev": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "tokens": {
                "mean": statistics.mean(tokens) if tokens else 0,
                "median": statistics.median(tokens) if tokens else 0,
                "total": sum(tokens)
            },
            "recent_runs": [
                {
                    "run_id": r["run_id"],
                    "duration": r.get("duration_seconds"),
                    "tokens": r["tokens"]["total"],
                    "status": r["status"],
                    "config": r["config"]
                }
                for r in runs[:10]
            ]
        }
        
        return stats
    
    def compare_configs(self, analysis_type: str, config_key: str) -> Dict:
        """Compare performance across different config values"""
        runs = []
        
        for metrics_file in self.metrics_dir.glob(f"{analysis_type}_*.json"):
            with open(metrics_file) as f:
                data = json.load(f)
                if data["status"] == "completed":
                    runs.append(data)
        
        # Group by config value
        groups = {}
        for run in runs:
            config_value = run["config"].get(config_key, "unknown")
            config_value = str(config_value)
            
            if config_value not in groups:
                groups[config_value] = []
            groups[config_value].append(run)
        
        # Calculate stats per group
        comparison = {}
        for config_value, group_runs in groups.items():
            durations = [r["duration_seconds"] for r in group_runs if r.get("duration_seconds")]
            tokens = [r["tokens"]["total"] for r in group_runs]
            
            comparison[config_value] = {
                "count": len(group_runs),
                "avg_duration": statistics.mean(durations) if durations else 0,
                "avg_tokens": statistics.mean(tokens) if tokens else 0,
                "median_duration": statistics.median(durations) if durations else 0
            }
        
        return {
            "config_key": config_key,
            "comparison": comparison,
            "recommendation": self._recommend_config(comparison)
        }
    
    def _recommend_config(self, comparison: Dict) -> str:
        """Recommend best config based on balance of speed and quality"""
        if not comparison:
            return "Insufficient data"
        
        # Find config with best duration
        best_config = min(comparison.items(), key=lambda x: x[1]["avg_duration"])
        
        return f"Recommended: {best_config[0]} (avg: {best_config[1]['avg_duration']:.1f}s)"
    
    def _load_metrics(self, run_id: str) -> Dict:
        """Load metrics from file"""
        metrics_file = self.metrics_dir / f"{run_id}.json"
        
        if not metrics_file.exists():
            raise ValueError(f"Metrics not found for run_id: {run_id}")
        
        with open(metrics_file) as f:
            return json.load(f)
    
    def _save_metrics(self, run_id: str, metrics: Dict):
        """Save metrics to file"""
        metrics_file = self.metrics_dir / f"{run_id}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
