-- Migration 003: Quality Ratings System
-- Voor het beoordelen van verschillende analyse configuraties

CREATE TABLE IF NOT EXISTS quality_ratings (
    id INT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(100) NOT NULL,
    project_id VARCHAR(100) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    level INT NOT NULL,
    config JSON NOT NULL,
    
    -- Quality scores (1-5)
    accuracy_score INT CHECK (accuracy_score BETWEEN 1 AND 5),
    completeness_score INT CHECK (completeness_score BETWEEN 1 AND 5),
    relevance_score INT CHECK (relevance_score BETWEEN 1 AND 5),
    speed_score INT CHECK (speed_score BETWEEN 1 AND 5),
    
    -- Overall rating
    overall_score DECIMAL(3,2),
    
    -- Free text feedback
    notes TEXT,
    
    -- Metrics
    duration_seconds INT,
    tokens_used INT,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_run_id (run_id),
    INDEX idx_project_id (project_id),
    INDEX idx_analysis_type (analysis_type),
    INDEX idx_overall_score (overall_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table for storing test run results
CREATE TABLE IF NOT EXISTS progressive_test_runs (
    run_id VARCHAR(100) PRIMARY KEY,
    project_id VARCHAR(100) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL,
    level INT NOT NULL,
    config JSON NOT NULL,
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    progress INT DEFAULT 0, -- 0-100
    current_stage VARCHAR(100),
    
    -- Results
    output TEXT,
    error TEXT,
    
    -- Metrics
    start_time TIMESTAMP NULL,
    end_time TIMESTAMP NULL,
    duration_seconds INT,
    tokens_used INT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_project_id (project_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
