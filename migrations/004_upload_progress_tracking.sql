-- Upload Progress Tracking
-- Tracks detailed progress for each uploaded file through the ingestion pipeline

CREATE TABLE IF NOT EXISTS upload_progress (
    upload_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64),
    client_id VARCHAR(64),
    filename VARCHAR(512) NOT NULL,
    file_size BIGINT DEFAULT 0,
    file_path TEXT,
    
    -- Progress stages
    status ENUM('queued', 'uploading', 'uploaded', 'extracting', 'tokenizing', 'embedding', 'indexing', 'completed', 'failed') DEFAULT 'queued',
    progress_percent INT DEFAULT 0,
    current_stage VARCHAR(64),
    
    -- Step-by-step metrics
    extraction_time_ms INT DEFAULT 0,
    tokenization_time_ms INT DEFAULT 0,
    embedding_time_ms INT DEFAULT 0,
    indexing_time_ms INT DEFAULT 0,
    
    -- Results
    total_chunks INT DEFAULT 0,
    total_tokens INT DEFAULT 0,
    embedding_dimensions INT DEFAULT 0,
    doc_id VARCHAR(64),
    
    -- Error tracking
    error_message TEXT,
    error_stage VARCHAR(64),
    retry_count INT DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE,
    
    INDEX idx_project_uploads (project_id, status),
    INDEX idx_status (status),
    INDEX idx_created (created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Batch upload tracking for multiple files
CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64),
    total_files INT DEFAULT 0,
    completed_files INT DEFAULT 0,
    failed_files INT DEFAULT 0,
    status ENUM('queued', 'processing', 'completed', 'partial_failure', 'failed') DEFAULT 'queued',
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
    INDEX idx_project_batches (project_id, status),
    INDEX idx_created (created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
