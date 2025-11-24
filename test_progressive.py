#!/usr/bin/env python3
"""
Quick test of the Progressive Testing system.
Tests the full flow: start test → poll status → get result → rate
"""

import requests
import time
import json

BASE_URL = "http://localhost:8020"
PROJECT_ID = "project-06cfd7c8"

def test_progressive_system():
    print("🧪 Testing Progressive Testing System\n")
    
    # Test config - Level 1 (Mini)
    config = {
        "model": "llama3.1:8b",
        "temperature": 0.1,
        "max_tokens": 1500,
        "max_chunks": 15
    }
    
    print(f"📋 Project: {PROJECT_ID}")
    print(f"⚙️  Config: {json.dumps(config, indent=2)}\n")
    
    # Step 1: Start test
    print("1️⃣ Starting progressive test...")
    start_response = requests.post(
        f"{BASE_URL}/api/progressive-test/start",
        json={
            "project_id": PROJECT_ID,
            "config": config,
            "notes": "Test run from test_progressive.py"
        }
    )
    
    if start_response.status_code != 200:
        print(f"❌ Failed to start test: {start_response.text}")
        return False
    
    start_data = start_response.json()
    run_id = start_data.get("run_id")
    print(f"✅ Test started! Run ID: {run_id}\n")
    
    # Step 2: Poll status
    print("2️⃣ Polling status...")
    max_polls = 60  # 2 minutes max
    poll_count = 0
    
    while poll_count < max_polls:
        time.sleep(2)
        poll_count += 1
        
        status_response = requests.get(f"{BASE_URL}/api/progressive-test/status/{run_id}")
        if status_response.status_code != 200:
            print(f"❌ Failed to get status: {status_response.text}")
            return False
        
        status_data = status_response.json()
        status = status_data.get("status")
        progress = status_data.get("progress", 0)
        stage = status_data.get("stage", "")
        
        print(f"   Status: {status} | Progress: {progress}% | Stage: {stage}")
        
        if status == "completed":
            print("✅ Test completed!\n")
            break
        elif status == "failed":
            error = status_data.get("error_message", "Unknown error")
            print(f"❌ Test failed: {error}")
            return False
    else:
        print("❌ Test timed out")
        return False
    
    # Step 3: Get result
    print("3️⃣ Getting full result...")
    result_response = requests.get(f"{BASE_URL}/api/progressive-test/result/{run_id}")
    if result_response.status_code != 200:
        print(f"❌ Failed to get result: {result_response.text}")
        return False
    
    result_data = result_response.json()
    metrics = result_data.get("metrics", {})
    output_preview = result_data.get("output_text", "")[:200]
    
    print(f"✅ Results retrieved:")
    print(f"   Duration: {metrics.get('duration', 0):.1f}s")
    print(f"   Tokens: {metrics.get('tokens_used', 0):,}")
    print(f"   Chunks: {metrics.get('chunks_processed', 0)}")
    print(f"   Output preview: {output_preview}...\n")
    
    # Step 4: Save rating
    print("4️⃣ Saving quality rating...")
    rating_response = requests.post(
        f"{BASE_URL}/api/progressive-test/rate/{run_id}",
        json={
            "accuracy": 4,
            "completeness": 4,
            "relevance": 5,
            "clarity": 4,
            "notes": "Automated test rating - looks good!"
        }
    )
    
    if rating_response.status_code != 200:
        print(f"❌ Failed to save rating: {rating_response.text}")
        return False
    
    rating_data = rating_response.json()
    print(f"✅ Rating saved! ID: {rating_data.get('rating_id')}\n")
    
    print("="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print(f"\n📊 View results at:")
    print(f"   Progressive Tester: {BASE_URL}/static/admin/progressive-test.html")
    print(f"   Review Output: {BASE_URL}/static/admin/review-output.html?run_id={run_id}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_progressive_system()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
