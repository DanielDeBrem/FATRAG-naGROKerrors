-- Migration: Add clients, projects, and organograms tables
-- Date: 2025-01-09

-- Clients table: store client information
CREATE TABLE IF NOT EXISTS clients (
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
CREATE TABLE IF NOT EXISTS projects (
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
CREATE TABLE IF NOT EXISTS organograms (
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
-- Check if columns exist before adding
SET @exist_project_id = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA='fatrag' AND TABLE_NAME='documents' AND COLUMN_NAME='project_id');
SET @exist_client_id = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA='fatrag' AND TABLE_NAME='documents' AND COLUMN_NAME='client_id');

SET @sql_project = IF(@exist_project_id = 0, 
    'ALTER TABLE documents ADD COLUMN project_id VARCHAR(128) DEFAULT NULL, ADD INDEX idx_project_id (project_id)', 
    'SELECT "Column project_id already exists"');
PREPARE stmt_project FROM @sql_project;
EXECUTE stmt_project;
DEALLOCATE PREPARE stmt_project;

SET @sql_client = IF(@exist_client_id = 0, 
    'ALTER TABLE documents ADD COLUMN client_id VARCHAR(128) DEFAULT NULL, ADD INDEX idx_client_id (client_id)', 
    'SELECT "Column client_id already exists"');
PREPARE stmt_client FROM @sql_client;
EXECUTE stmt_client;
DEALLOCATE PREPARE stmt_client;
