"""
Quality Ratings System
Manage ratings and test runs for progressive testing
"""
import mysql.connector
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class QualityRatings:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'fatrag'),
            password=os.getenv('MYSQL_PASSWORD', 'fatrag_pw'),
            database=os.getenv('MYSQL_DATABASE', 'fatrag')
        )
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # ==================== TEST RUNS ====================
    
    def create_test_run(self, run_id: str, project_id: str, analysis_type: str, 
                       level: int, config: dict) -> bool:
        """Create a new test run record"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO progressive_test_runs 
                (run_id, project_id, analysis_type, level, config, status, start_time)
                VALUES (%s, %s, %s, %s, %s, 'pending', NOW())
            """, (run_id, project_id, analysis_type, level, json.dumps(config)))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error creating test run: {e}")
            return False
    
    def update_test_run_status(self, run_id: str, status: str, 
                               progress: int = None, current_stage: str = None) -> bool:
        """Update test run status"""
        try:
            cursor = self.conn.cursor()
            updates = ["status = %s"]
            params = [status]
            
            if progress is not None:
                updates.append("progress = %s")
                params.append(progress)
            
            if current_stage:
                updates.append("current_stage = %s")
                params.append(current_stage)
            
            params.append(run_id)
            
            cursor.execute(f"""
                UPDATE progressive_test_runs 
                SET {', '.join(updates)}
                WHERE run_id = %s
            """, params)
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating test run: {e}")
            return False
    
    def complete_test_run(self, run_id: str, output: str, 
                         duration: int, tokens: int, error: str = None) -> bool:
        """Mark test run as completed"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE progressive_test_runs 
                SET status = %s, 
                    output = %s, 
                    error = %s,
                    end_time = NOW(),
                    duration_seconds = %s,
                    tokens_used = %s,
                    progress = 100
                WHERE run_id = %s
            """, ('failed' if error else 'completed', output, error, duration, tokens, run_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error completing test run: {e}")
            return False
    
    def get_test_run(self, run_id: str) -> Optional[Dict]:
        """Get test run by ID"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM progressive_test_runs WHERE run_id = %s
            """, (run_id,))
            result = cursor.fetchone()
            
            if result and result['config']:
                result['config'] = json.loads(result['config'])
            
            return result
        except Exception as e:
            print(f"Error getting test run: {e}")
            return None
    
    def get_recent_test_runs(self, project_id: str = None, limit: int = 20) -> List[Dict]:
        """Get recent test runs"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            
            if project_id:
                cursor.execute("""
                    SELECT * FROM progressive_test_runs 
                    WHERE project_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (project_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM progressive_test_runs 
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cursor.fetchall()
            
            for r in results:
                if r['config']:
                    r['config'] = json.loads(r['config'])
            
            return results
        except Exception as e:
            print(f"Error getting test runs: {e}")
            return []
    
    # ==================== QUALITY RATINGS ====================
    
    def save_rating(self, run_id: str, project_id: str, analysis_type: str,
                   level: int, config: dict, accuracy: int, completeness: int,
                   relevance: int, speed: int, notes: str = "",
                   duration: int = None, tokens: int = None) -> bool:
        """Save quality rating"""
        try:
            overall = (accuracy + completeness + relevance + speed) / 4.0
            
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO quality_ratings 
                (run_id, project_id, analysis_type, level, config,
                 accuracy_score, completeness_score, relevance_score, speed_score,
                 overall_score, notes, duration_seconds, tokens_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, project_id, analysis_type, level, json.dumps(config),
                  accuracy, completeness, relevance, speed, overall, notes,
                  duration, tokens))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving rating: {e}")
            return False
    
    def get_ratings_for_run(self, run_id: str) -> Optional[Dict]:
        """Get rating for a specific run"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM quality_ratings WHERE run_id = %s
            """, (run_id,))
            result = cursor.fetchone()
            
            if result and result['config']:
                result['config'] = json.loads(result['config'])
            
            return result
        except Exception as e:
            print(f"Error getting rating: {e}")
            return None
    
    def get_best_configs(self, analysis_type: str, limit: int = 5) -> List[Dict]:
        """Get best rated configurations for an analysis type"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    config,
                    AVG(overall_score) as avg_score,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(completeness_score) as avg_completeness,
                    AVG(relevance_score) as avg_relevance,
                    AVG(speed_score) as avg_speed,
                    AVG(duration_seconds) as avg_duration,
                    COUNT(*) as rating_count
                FROM quality_ratings
                WHERE analysis_type = %s
                GROUP BY config
                ORDER BY avg_score DESC, avg_duration ASC
                LIMIT %s
            """, (analysis_type, limit))
            
            results = cursor.fetchall()
            
            for r in results:
                if r['config']:
                    r['config'] = json.loads(r['config'])
            
            return results
        except Exception as e:
            print(f"Error getting best configs: {e}")
            return []
    
    def get_ratings_summary(self, project_id: str = None) -> Dict:
        """Get summary statistics of ratings"""
        try:
            cursor = self.conn.cursor(dictionary=True)
            
            where_clause = "WHERE project_id = %s" if project_id else ""
            params = (project_id,) if project_id else ()
            
            cursor.execute(f"""
                SELECT 
                    analysis_type,
                    level,
                    COUNT(*) as total_ratings,
                    AVG(overall_score) as avg_score,
                    AVG(duration_seconds) as avg_duration,
                    SUM(tokens_used) as total_tokens
                FROM quality_ratings
                {where_clause}
                GROUP BY analysis_type, level
                ORDER BY analysis_type, level
            """, params)
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting ratings summary: {e}")
            return {}


# Quick test
if __name__ == "__main__":
    qr = QualityRatings()
    
    # Test create run
    test_config = {"model": "llama3.1:8b", "temp": 0.1, "tokens": 1500}
    qr.create_test_run("test_run_001", "test_project", "flash", 1, test_config)
    
    # Test update
    qr.update_test_run_status("test_run_001", "running", 50, "retrieval")
    
    # Test get
    run = qr.get_test_run("test_run_001")
    print("Test run:", run)
    
    print("✅ Quality ratings module working!")
