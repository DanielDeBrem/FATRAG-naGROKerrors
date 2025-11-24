-- Add documents table for tracking uploaded files per project
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(50) PRIMARY KEY,
    project_id VARCHAR(50) NOT NULL,
    client_id VARCHAR(50),
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500),
    file_size INT,
    file_type VARCHAR(50),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by VARCHAR(100),
    deleted_at TIMESTAMP NULL,
    metadata JSON,
    FOREIGN KEY (project_id) REFERENCES projects(project_id),
    FOREIGN KEY (client_id) REFERENCES clients(client_id),
    INDEX idx_project (project_id),
    INDEX idx_client (client_id),
    INDEX idx_uploaded (uploaded_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
