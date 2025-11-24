#!/usr/bin/env python3
"""
Test MySQL database connectivity and models
"""
import sys
from db_models import test_connection, SessionLocal, Config
from config_store_mysql import load_config, save_config
from feedback_store_mysql import submit_feedback, list_feedback

def test_basic_connection():
    """Test basic database connection"""
    print("Testing basic MySQL connection...")
    if test_connection():
        print("✓ MySQL connection successful")
        return True
    else:
        print("✗ MySQL connection failed")
        return False

def test_config_operations():
    """Test configuration read/write"""
    print("\nTesting configuration operations...")
    try:
        # Load config
        cfg = load_config()
        print(f"✓ Loaded config: {len(cfg)} keys")
        print(f"  LLM_MODEL: {cfg.get('LLM_MODEL')}")
        print(f"  RETRIEVER_K: {cfg.get('RETRIEVER_K')}")
        
        # Test save (update temperature)
        cfg['TEMPERATURE'] = 0.8
        save_config(cfg)
        print("✓ Saved config with updated TEMPERATURE")
        
        # Reload and verify
        cfg_reloaded = load_config()
        if cfg_reloaded['TEMPERATURE'] == 0.8:
            print("✓ Config save/load verified")
            
            # Reset to original
            cfg_reloaded['TEMPERATURE'] = 0.7
            save_config(cfg_reloaded)
            print("✓ Reset TEMPERATURE to 0.7")
        else:
            print("✗ Config verification failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Config operations failed: {e}")
        return False

def test_feedback_operations():
    """Test feedback submission and retrieval"""
    print("\nTesting feedback operations...")
    try:
        # Submit test feedback
        feedback = submit_feedback(
            question="Test question?",
            answer="Test answer",
            rating="up",
            tags=["test"],
            user_role="tester"
        )
        print(f"✓ Submitted feedback: {feedback['id']}")
        
        # List feedback
        items = list_feedback(status="pending")
        print(f"✓ Retrieved {len(items)} pending feedback items")
        
        # Find our test feedback
        found = False
        for item in items:
            if item['id'] == feedback['id']:
                found = True
                print(f"✓ Found test feedback in pending list")
                break
        
        if not found:
            print("✗ Test feedback not found in list")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Feedback operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_query():
    """Test direct SQL query"""
    print("\nTesting direct SQL queries...")
    try:
        db = SessionLocal()
        try:
            # Count config entries
            config_count = db.query(Config).count()
            print(f"✓ Found {config_count} config entries in database")
            
            # List config keys
            configs = db.query(Config).all()
            print("  Config keys:")
            for cfg in configs:
                print(f"    - {cfg.config_key}: {cfg.config_value}")
            
            return True
        finally:
            db.close()
    except Exception as e:
        print(f"✗ Direct query failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FATRAG MySQL Database Tests")
    print("=" * 60)
    
    tests = [
        test_basic_connection,
        test_direct_query,
        test_config_operations,
        test_feedback_operations,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
