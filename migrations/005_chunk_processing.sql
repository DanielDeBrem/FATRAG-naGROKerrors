-- Migration 005: Chunk Processing Tables
-- Supports chunked progressive analysis with fine-grained tracking

CREATE TABLE IF NOT EXISTS chunk_analysis (
    chunk_id VARCHAR(64) PRIMARY KEY,
    job_id VARCHAR(64) NOT NULL,
    doc_name VARCHAR(255) NOT NULL,
    chunk_index INT NOT NULL,
    total_chunks INT NOT NULL,
    chunk_text TEXT,
    status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending',
    result_json TEXT,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    processing_time_sec FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    INDEX idx_job_id (job_id),
    INDEX idx_status (status),
    INDEX idx_doc_name (doc_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chunk_job_metadata (
    job_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64),
    doc_path VARCHAR(512),
    chunk_size INT,
    chunk_overlap INT,
    total_chunks INT,
    chunks_completed INT DEFAULT 0,
    chunks_failed INT DEFAULT 0,
    status ENUM('initializing', 'chunking', 'processing', 'aggregating', 'completed', 'failed') DEFAULT 'initializing',
    model_name VARCHAR(64),
    worker_count INT,
    eta_minutes FLOAT,
    throughput_chunks_per_min FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    output_dir VARCHAR(512),
    error_message TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
