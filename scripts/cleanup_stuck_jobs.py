#!/usr/bin/env python3
"""
Cleanup stuck jobs that are marked as running but are no longer active.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import job_store_mysql as js
from datetime import datetime, timedelta


def cleanup_stuck_jobs(max_age_hours=2):
    """
    Mark jobs as failed if they've been running for longer than max_age_hours.
    
    Args:
        max_age_hours: Maximum hours a job can be in 'running' status
    """
    try:
        import pymysql
        from clients_projects import get_db_connection
        
        conn = get_db_connection()
        
        # Find jobs that are stuck (running for more than max_age_hours)
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with conn.cursor() as cursor:
            # Find stuck jobs
            cursor.execute("""
                SELECT job_id, job_type, project_id, status, created_at, updated_at
                FROM jobs
                WHERE status IN ('running', 'queued', 'initializing', 'preempting_gpu', 
                                'running_pipeline', 'running_l1_analysis', 'saving_outputs',
                                'generating', 'saving', 'analyzing')
                AND updated_at < %s
            """, (cutoff_time,))
            
            stuck_jobs = cursor.fetchall()
            
            if not stuck_jobs:
                print("✓ No stuck jobs found")
                return
            
            print(f"Found {len(stuck_jobs)} stuck job(s):")
            print()
            
            for job in stuck_jobs:
                job_id = job['job_id']
                job_type = job['job_type']
                project_id = job['project_id']
                status = job['status']
                updated_at = job['updated_at']
                
                age = datetime.now() - updated_at
                hours = age.total_seconds() / 3600
                
                print(f"  Job ID: {job_id}")
                print(f"    Type: {job_type}")
                print(f"    Project: {project_id}")
                print(f"    Status: {status}")
                print(f"    Age: {hours:.1f} hours")
                print()
            
            # Ask for confirmation
            response = input(f"Mark these {len(stuck_jobs)} job(s) as failed? [y/N]: ").strip().lower()
            
            if response != 'y':
                print("Cancelled")
                return
            
            # Mark jobs as failed
            for job in stuck_jobs:
                job_id = job['job_id']
                cursor.execute("""
                    UPDATE jobs
                    SET status = 'failed',
                        error_message = 'Job timed out / stuck (cleaned up automatically)',
                        progress = 0,
                        updated_at = NOW()
                    WHERE job_id = %s
                """, (job_id,))
            
            conn.commit()
            print(f"✓ Marked {len(stuck_jobs)} job(s) as failed")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup stuck jobs")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=2,
        help="Maximum hours a job can be running (default: 2)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically clean up without asking for confirmation"
    )
    
    args = parser.parse_args()
    
    if args.auto:
        # Bypass confirmation
        import pymysql
        from clients_projects import get_db_connection
        
        conn = get_db_connection()
        cutoff_time = datetime.now() - timedelta(hours=args.max_age_hours)
        
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE jobs
                SET status = 'failed',
                    error_message = 'Job timed out / stuck (cleaned up automatically)',
                    progress = 0,
                    updated_at = NOW()
                WHERE status IN ('running', 'queued', 'initializing', 'preempting_gpu',
                                'running_pipeline', 'running_l1_analysis', 'saving_outputs',
                                'generating', 'saving', 'analyzing')
                AND updated_at < %s
            """, (cutoff_time,))
            
            count = cursor.rowcount
            conn.commit()
        
        conn.close()
        print(f"✓ Automatically cleaned up {count} stuck job(s)")
    else:
        cleanup_stuck_jobs(max_age_hours=args.max_age_hours)
