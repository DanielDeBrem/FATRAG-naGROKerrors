-- FATRAG Database Schema
-- Fresh start for Financial Advisory Tool

-- Documents table: store uploaded document metadata
CREATE TABLE documents (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doc_id VARCHAR(128) NOT NULL UNIQUE,
    filename VARCHAR(512) NOT NULL,
    source_type VARCHAR(64) NOT NULL COMMENT 'upload, scrape, api',
    file_path VARCHAR(1024) DEFAULT NULL,
    file_size INT DEFAULT 0,
    checksum VARCHAR(128) DEFAULT NULL,
    chunk_count INT DEFAULT 0,
    status VARCHAR(32) DEFAULT 'pending' COMMENT 'pending, processing, indexed, failed',
    error_message TEXT DEFAULT NULL,
    metadata JSON DEFAULT NULL COMMENT 'Custom metadata like category, tags, etc',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP NULL DEFAULT NULL,
    deleted_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_doc_id (doc_id),
    INDEX idx_status (status),
    INDEX idx_source_type (source_type),
    INDEX idx_deleted_at (deleted_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Queries table: store all user queries and responses
CREATE TABLE queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_id VARCHAR(128) NOT NULL UNIQUE,
    question TEXT NOT NULL,
    answer TEXT DEFAULT NULL,
    language VARCHAR(10) DEFAULT 'nl' COMMENT 'nl, en, de, fr',
    retriever_k INT DEFAULT 5,
    temperature FLOAT DEFAULT 0.7,
    model_used VARCHAR(128) DEFAULT NULL,
    response_time_ms INT DEFAULT NULL,
    chunks_retrieved JSON DEFAULT NULL COMMENT 'Array of doc_ids and scores',
    confidence_score FLOAT DEFAULT NULL,
    status VARCHAR(32) DEFAULT 'completed' COMMENT 'completed, failed, timeout',
    error_message TEXT DEFAULT NULL,
    user_role VARCHAR(64) DEFAULT NULL,
    session_id VARCHAR(128) DEFAULT NULL,
    metadata JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_query_id (query_id),
    INDEX idx_language (language),
    INDEX idx_status (status),
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    FULLTEXT idx_question (question)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Feedback table: store user feedback on query responses
CREATE TABLE feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feedback_id VARCHAR(128) NOT NULL UNIQUE,
    query_id VARCHAR(128) DEFAULT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating VARCHAR(10) DEFAULT NULL COMMENT 'up, down',
    corrected_answer TEXT DEFAULT NULL,
    tags JSON DEFAULT NULL COMMENT 'Array of tags',
    user_role VARCHAR(64) DEFAULT NULL,
    status VARCHAR(32) DEFAULT 'pending' COMMENT 'pending, approved, rejected',
    moderator_notes TEXT DEFAULT NULL,
    metadata JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_feedback_id (feedback_id),
    INDEX idx_query_id (query_id),
    INDEX idx_status (status),
    INDEX idx_rating (rating),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (query_id) REFERENCES queries(query_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Configuration table: store system configuration (replacing config.json)
CREATE TABLE config (
    id INT AUTO_INCREMENT PRIMARY KEY,
    config_key VARCHAR(128) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    value_type VARCHAR(32) DEFAULT 'string' COMMENT 'string, int, float, bool, json',
    description TEXT DEFAULT NULL,
    is_secret BOOLEAN DEFAULT FALSE COMMENT 'Mark sensitive values',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_config_key (config_key),
    INDEX idx_is_secret (is_secret)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Jobs table: background task processing queue
CREATE TABLE jobs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(128) NOT NULL UNIQUE,
    job_type VARCHAR(64) NOT NULL COMMENT 'document_ingestion, index_rebuild, etc',
    status VARCHAR(32) DEFAULT 'pending' COMMENT 'pending, running, completed, failed',
    priority INT DEFAULT 5 COMMENT '1=highest, 10=lowest',
    params JSON DEFAULT NULL,
    result JSON DEFAULT NULL,
    error_message TEXT DEFAULT NULL,
    progress INT DEFAULT 0 COMMENT 'Percentage 0-100',
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    started_at TIMESTAMP NULL DEFAULT NULL,
    completed_at TIMESTAMP NULL DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_job_id (job_id),
    INDEX idx_job_type (job_type),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Audit log: system events and actions
CREATE TABLE audit_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    event VARCHAR(128) NOT NULL,
    event_type VARCHAR(32) DEFAULT 'info' COMMENT 'info, warning, error',
    user_id VARCHAR(128) DEFAULT NULL,
    payload JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event (event),
    INDEX idx_event_type (event_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Clients table: store client information
CREATE TABLE clients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(128) NOT NULL UNIQUE,
    name VARCHAR(256) NOT NULL,
    type VARCHAR(64) DEFAULT 'individual' COMMENT 'individual, business, trust, foundation, etc',
    tax_id VARCHAR(128) DEFAULT NULL COMMENT 'BSN, RSIN, KvK nummer',
    contact_info JSON DEFAULT NULL COMMENT 'Email, telefoon, adres',
    notes TEXT DEFAULT NULL,
    metadata JSON DEFAULT NULL COMMENT 'Extra velden zoals branche, adviseur, tags',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    archived_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_client_id (client_id),
    INDEX idx_name (name),
    INDEX idx_type (type),
    INDEX idx_archived (archived_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Projects table: organize work per client
CREATE TABLE projects (
    id INT AUTO_INCREMENT PRIMARY KEY,
    project_id VARCHAR(128) NOT NULL UNIQUE,
    client_id VARCHAR(128) NOT NULL,
    name VARCHAR(256) NOT NULL,
    type VARCHAR(64) DEFAULT 'general' COMMENT 'tax_planning, valuation, restructuring, estate_planning, compliance, etc',
    status VARCHAR(32) DEFAULT 'active' COMMENT 'active, completed, on_hold, archived',
    description TEXT DEFAULT NULL,
    metadata JSON DEFAULT NULL COMMENT 'Deadlines, bedragen, milestones',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    archived_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_project_id (project_id),
    INDEX idx_client_id (client_id),
    INDEX idx_status (status),
    INDEX idx_type (type),
    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Organograms table: store interactive organizational structures
CREATE TABLE organograms (
    id INT AUTO_INCREMENT PRIMARY KEY,
    organogram_id VARCHAR(128) NOT NULL UNIQUE,
    project_id VARCHAR(128) NOT NULL,
    name VARCHAR(256) NOT NULL,
    structure_data JSON NOT NULL COMMENT 'vis.js format: {nodes: [...], edges: [...]}',
    version INT DEFAULT 1,
    notes TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_organogram_id (organogram_id),
    INDEX idx_project_id (project_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Extend documents table to link to clients and projects
ALTER TABLE documents ADD COLUMN project_id VARCHAR(128) DEFAULT NULL;
ALTER TABLE documents ADD COLUMN client_id VARCHAR(128) DEFAULT NULL;
ALTER TABLE documents ADD INDEX idx_project_id (project_id);
ALTER TABLE documents ADD INDEX idx_client_id (client_id);

-- Insert default configuration values
INSERT INTO config (config_key, config_value, value_type, description) VALUES
('LLM_MODEL', 'llama3.1:70b', 'string', 'Ollama LLM model for generation'),
('EMBED_MODEL', 'gemma2:2b', 'string', 'Ollama embedding model'),
('OLLAMA_BASE_URL', 'http://localhost:11434', 'string', 'Ollama API base URL'),
('CHROMA_DIR', './fatrag_chroma_db', 'string', 'ChromaDB persistence directory'),
('CHROMA_COLLECTION', 'fatrag', 'string', 'ChromaDB collection name'),
('RETRIEVER_K', '5', 'int', 'Number of chunks to retrieve'),
('TEMPERATURE', '0.7', 'float', 'LLM temperature for generation'),
('FEEDBACK_ENABLED', 'true', 'bool', 'Enable feedback collection');
