-- Migration: Create documents table
-- PR3 v2 - Hard Reset RAG Architecture
-- Date: 2025-01-30

-- Habilita extensão para UUIDs
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Tabela principal de registro de documentos
CREATE TABLE IF NOT EXISTS documents (
    -- Identificação
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(200) NOT NULL,
    version INT NOT NULL DEFAULT 1,

    -- Status do pipeline
    status VARCHAR(50) NOT NULL DEFAULT 'uploaded',

    -- Hashes para integridade
    sha256_source VARCHAR(64),
    sha256_canonical_md VARCHAR(64),

    -- Referências ao MinIO
    minio_source_key TEXT,
    minio_canonical_key TEXT,

    -- Métricas
    chunk_count INT,
    edge_count INT,

    -- Rastreamento de execução
    ingest_run_id UUID,
    pipeline_version VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    -- Erro (se status == 'failed')
    error_message TEXT,

    -- Constraints
    CONSTRAINT uq_document_version UNIQUE (document_id, version)
);

-- Índices para queries frequentes
CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_ingest_run_id ON documents(ingest_run_id);
CREATE INDEX IF NOT EXISTS idx_documents_updated_at ON documents(updated_at);

-- Comentários
COMMENT ON TABLE documents IS 'Registro de documentos no pipeline de ingestão (PR3 v2)';
COMMENT ON COLUMN documents.document_id IS 'ID canônico do documento (ex: LEI-14133-2021)';
COMMENT ON COLUMN documents.version IS 'Versão do documento (para re-ingestões)';
COMMENT ON COLUMN documents.status IS 'Status: uploaded, processed, embedded, indexed, graph_synced, failed';
COMMENT ON COLUMN documents.sha256_source IS 'Hash SHA256 do PDF fonte';
COMMENT ON COLUMN documents.sha256_canonical_md IS 'Hash SHA256 do markdown canônico';
COMMENT ON COLUMN documents.ingest_run_id IS 'UUID da execução de ingestão (cola entre sinks)';
COMMENT ON COLUMN documents.pipeline_version IS 'Versão do pipeline (git SHA ou tag)';
