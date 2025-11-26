-- Migration 006: Document prompts table
-- Opslag van prompts per documenttype en prompt-soort (extract/summary/risk)

CREATE TABLE IF NOT EXISTS document_prompts (
  id                 BIGINT AUTO_INCREMENT PRIMARY KEY,
  doc_type           VARCHAR(64) NOT NULL,    -- bijv. 'jaarrekening', 'factuur', 'bankafschrift'
  prompt_kind        VARCHAR(32) NOT NULL,    -- 'extract' | 'summary' | 'risk'
  template           MEDIUMTEXT NOT NULL,     -- volledige prompttekst
  max_context_tokens INT NULL,
  active             TINYINT(1) NOT NULL DEFAULT 1,
  created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                     ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY uq_document_prompts_doc_type_kind (doc_type, prompt_kind)
);
