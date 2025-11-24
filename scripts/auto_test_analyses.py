#!/usr/bin/env python3
"""
Auto-Test Framework voor FATRAG Analyses
Test verschillende configuraties en vind optimale settings
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import asyncio
from typing import Dict, List
from datetime import datetime
import clients_projects as cp
import job_store_mysql as js
from metrics_store import MetricsStore

# Test configuraties voor elk analyse type
FLASH_CONFIGS = [
    {"model": "llama3.1:8b", "temperature": 0.1, "max_tokens": 2000},
    {"model": "llama3.1:8b", "temperature": 0.2, "max_tokens": 3000},
    {"model": "llama3:8b", "temperature": 0.15, "max_tokens": 2500},
]

GRONDIGE_CONFIGS = [
    {"model": "llama3.1:70b", "temperature": 0.1, "chunk_size": 2000, "overlap": 200},
    {"model": "llama3.1:70b", "temperature": 0.15, "chunk_size": 3000, "overlap": 300},
    {"model": "llama3.1:70b", "temperature": 0.05, "chunk_size": 2500, "overlap": 250},
]

TEMPLATE_CONFIGS = [
    {"model": "llama3.1:70b", "temperature": 0.1, "sections_parallel": True},
    {"model": "llama3.1:70b", "temperature": 0.15, "sections_parallel": True},
    {"model": "llama3.1:70b", "temperature": 0.1, "sections_parallel": False},
]

class AnalysisAutoTester:
    """Automatically test different analysis configurations"""
    
    def __init__(self, project_id: str = None):
        self.metrics_store = MetricsStore()
        self.project_id = project_id or self._get_test_project()
        
        print(f"🧪 Auto-Tester initialized for project: {self.project_id}")
        
    def _get_test_project(self) -> str:
        """Get or create a test project"""
        projects = cp.list_projects()
        
        # Look for existing test project
        for proj in projects:
            if "test" in proj["name"].lower():
                return proj["project_id"]
        
        # Find project with documents
        for proj in projects:
            docs = cp.list_project_documents(proj["project_id"])
            if docs:
                return proj["project_id"]
        
        raise ValueError("No suitable test project found. Create a project with documents first.")
    
    async def test_flash_analysis(self, configs: List[Dict] = None) -> Dict:
        """Test Flash Analysis with different configs"""
        configs = configs or FLASH_CONFIGS
        
        print("\n📊 Testing Flash Analysis...")
        print(f"   Testing {len(configs)} configurations")
        
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n   Config {i}/{len(configs)}: {config}")
            
            try:
                # Start metrics tracking
                job_id = f"flash_test_{int(time.time())}"
                run_id = self.metrics_store.log_analysis_start(
                    job_id=job_id,
                    analysis_type="flash_analysis",
                    config=config
                )
                
                # Run analysis (placeholder - you'll replace with actual call)
                start_time = time.time()
                
                print(f"      Starting analysis...")
                # TODO: Actually trigger flash analysis with this config
                # For now, simulate
                await asyncio.sleep(2)  # Simulate work
                
                duration = time.time() - start_time
                
                # Log completion
                self.metrics_store.log_stage(
                    run_id=run_id,
                    stage_name="flash_generation",
                    duration=duration,
                    tokens_used=2500  # Placeholder
                )
                
                self.metrics_store.log_analysis_complete(
                    run_id=run_id,
                    result={"status": "test_completed"},
                    quality_scores={"test": True}
                )
                
                results.append({
                    "config": config,
                    "run_id": run_id,
                    "duration": duration,
                    "status": "completed"
                })
                
                print(f"      ✅ Completed in {duration:.1f}s")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                self.metrics_store.log_error(run_id, str(e))
                results.append({
                    "config": config,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {"test_type": "flash_analysis", "results": results}
    
    async def test_grondige_analysis(self, configs: List[Dict] = None) -> Dict:
        """Test Grondige Analyse with different configs"""
        configs = configs or GRONDIGE_CONFIGS
        
        print("\n📊 Testing Grondige Analyse...")
        print(f"   Testing {len(configs)} configurations")
        
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n   Config {i}/{len(configs)}: {config}")
            
            try:
                job_id = f"grondige_test_{int(time.time())}"
                run_id = self.metrics_store.log_analysis_start(
                    job_id=job_id,
                    analysis_type="grondige_analysis",
                    config=config
                )
                
                start_time = time.time()
                print(f"      Starting analysis...")
                
                # TODO: Actually trigger grondige analysis
                await asyncio.sleep(5)  # Simulate longer work
                
                duration = time.time() - start_time
                
                self.metrics_store.log_stage(
                    run_id=run_id,
                    stage_name="map_reduce",
                    duration=duration,
                    tokens_used=15000  # Placeholder
                )
                
                self.metrics_store.log_analysis_complete(
                    run_id=run_id,
                    result={"status": "test_completed"},
                    quality_scores={"test": True}
                )
                
                results.append({
                    "config": config,
                    "run_id": run_id,
                    "duration": duration,
                    "status": "completed"
                })
                
                print(f"      ✅ Completed in {duration:.1f}s")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                self.metrics_store.log_error(run_id, str(e))
                results.append({
                    "config": config,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {"test_type": "grondige_analysis", "results": results}
    
    async def test_template_reports(self, configs: List[Dict] = None) -> Dict:
        """Test Template Reports with different configs"""
        configs = configs or TEMPLATE_CONFIGS
        
        print("\n📊 Testing Template Reports...")
        print(f"   Testing {len(configs)} configurations")
        
        results = []
        templates = ["holding_analysis", "estate_planning"]  # Test 2 templates
        
        for i, config in enumerate(configs, 1):
            for template_key in templates:
                print(f"\n   Config {i}/{len(configs)}, Template: {template_key}")
                
                try:
                    job_id = f"template_test_{template_key}_{int(time.time())}"
                    run_id = self.metrics_store.log_analysis_start(
                        job_id=job_id,
                        analysis_type=f"template_{template_key}",
                        config=config
                    )
                    
                    start_time = time.time()
                    print(f"      Starting generation...")
                    
                    # TODO: Actually trigger template generation
                    await asyncio.sleep(10)  # Simulate long work (7 sections)
                    
                    duration = time.time() - start_time
                    
                    self.metrics_store.log_stage(
                        run_id=run_id,
                        stage_name="template_generation",
                        duration=duration,
                        tokens_used=25000  # Placeholder
                    )
                    
                    self.metrics_store.log_analysis_complete(
                        run_id=run_id,
                        result={"status": "test_completed"},
                        quality_scores={"test": True}
                    )
                    
                    results.append({
                        "config": config,
                        "template": template_key,
                        "run_id": run_id,
                        "duration": duration,
                        "status": "completed"
                    })
                    
                    print(f"      ✅ Completed in {duration:.1f}s")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    self.metrics_store.log_error(run_id, str(e))
                    results.append({
                        "config": config,
                        "template": template_key,
                        "status": "failed",
                        "error": str(e)
                    })
        
        return {"test_type": "template_reports", "results": results}
    
    async def run_full_test_suite(self):
        """Run all tests and generate report"""
        print("\n" + "="*60)
        print("🚀 FATRAG Auto-Test Suite")
        print("="*60)
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: {self.project_id}")
        
        all_results = {}
        
        # Test each analysis type
        all_results["flash"] = await self.test_flash_analysis()
        all_results["grondige"] = await self.test_grondige_analysis()
        all_results["templates"] = await self.test_template_reports()
        
        # Generate summary report
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60)
        
        for test_type, data in all_results.items():
            print(f"\n{test_type.upper()}:")
            results = data["results"]
            completed = [r for r in results if r["status"] == "completed"]
            failed = [r for r in results if r["status"] == "failed"]
            
            print(f"  Total: {len(results)}")
            print(f"  ✅ Completed: {len(completed)}")
            print(f"  ❌ Failed: {len(failed)}")
            
            if completed:
                durations = [r["duration"] for r in completed]
                avg = sum(durations) / len(durations)
                print(f"  ⏱️  Avg Duration: {avg:.1f}s")
                print(f"  ⏱️  Min Duration: {min(durations):.1f}s")
                print(f"  ⏱️  Max Duration: {max(durations):.1f}s")
        
        # Save full results
        results_file = f"test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📄 Full results saved to: {results_file}")
        print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_results
    
    def analyze_results(self, analysis_type: str):
        """Analyze historical results and recommend config"""
        print(f"\n📈 Analysis for: {analysis_type}")
        
        stats = self.metrics_store.get_analysis_stats(analysis_type)
        
        if "error" in stats:
            print(f"   {stats['error']}")
            return
        
        print(f"   Total runs: {stats['total_runs']}")
        print(f"   Completed: {stats['completed_runs']}")
        print(f"   Failed: {stats['failed_runs']}")
        print(f"\n   Duration:")
        print(f"   - Mean: {stats['duration']['mean']:.1f}s")
        print(f"   - Median: {stats['duration']['median']:.1f}s")
        print(f"   - Range: {stats['duration']['min']:.1f}s - {stats['duration']['max']:.1f}s")
        print(f"\n   Tokens:")
        print(f"   - Mean: {stats['tokens']['mean']:.0f}")
        print(f"   - Total: {stats['tokens']['total']:.0f}")
        
        # Try to compare model configs
        comparison = self.metrics_store.compare_configs(analysis_type, "model")
        if comparison.get("comparison"):
            print(f"\n   Model Comparison:")
            for model, metrics in comparison["comparison"].items():
                print(f"   - {model}: {metrics['avg_duration']:.1f}s avg ({metrics['count']} runs)")
            print(f"\n   {comparison['recommendation']}")

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-test FATRAG analyses")
    parser.add_argument("--project", help="Project ID to test with")
    parser.add_argument("--flash-only", action="store_true", help="Test only Flash Analysis")
    parser.add_argument("--grondige-only", action="store_true", help="Test only Grondige Analyse")
    parser.add_argument("--templates-only", action="store_true", help="Test only Templates")
    parser.add_argument("--analyze", help="Analyze results for analysis type")
    
    args = parser.parse_args()
    
    tester = AnalysisAutoTester(project_id=args.project)
    
    if args.analyze:
        tester.analyze_results(args.analyze)
        return
    
    if args.flash_only:
        await tester.test_flash_analysis()
    elif args.grondige_only:
        await tester.test_grondige_analysis()
    elif args.templates_only:
        await tester.test_template_reports()
    else:
        await tester.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main())
