-- NAH Memory Unified PostgreSQL Schema
-- Run: psql -h localhost -U human -d notahuman -f setup_schema.sql

-- Create schema
CREATE SCHEMA IF NOT EXISTS nah_remember;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ==================== SESSIONS ====================
CREATE TABLE IF NOT EXISTS nah_remember.sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL UNIQUE,
    application VARCHAR(100) NOT NULL,
    user_id VARCHAR(255) DEFAULT 'default_user',
    project_context TEXT,
    working_directory TEXT,
    git_branch VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    summary TEXT,
    total_messages INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON nah_remember.sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON nah_remember.sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_application ON nah_remember.sessions(application);

-- ==================== MESSAGES ====================
CREATE TABLE IF NOT EXISTS nah_remember.messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL REFERENCES nah_remember.sessions(session_id) ON DELETE CASCADE,
    message_index INTEGER NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls JSONB,
    tool_results JSONB,
    token_count INTEGER DEFAULT 0,
    importance_score FLOAT DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),  -- mistral-embed (1024-dim); keep in sync with server EMBEDDING_DIM
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, message_index)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON nah_remember.messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_embedding ON nah_remember.messages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==================== CONTEXT SNAPSHOTS ====================
CREATE TABLE IF NOT EXISTS nah_remember.context_snapshots (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES nah_remember.sessions(session_id) ON DELETE CASCADE,
    snapshot_type VARCHAR(50) NOT NULL CHECK (snapshot_type IN ('checkpoint', 'milestone', 'completion', 'context', 'handoff')),
    context_data JSONB NOT NULL,
    working_directory TEXT,
    active_files TEXT[],
    git_state JSONB,
    todo_state JSONB,
    task_master_state JSONB,
    context_summary TEXT,
    key_entities TEXT[],
    action_items TEXT[],
    decisions_made TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_snapshots_session ON nah_remember.context_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_type ON nah_remember.context_snapshots(snapshot_type);

-- ==================== ENTITIES (Knowledge Graph) ====================
CREATE TABLE IF NOT EXISTS nah_remember.entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    source_session_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    embedding vector(1024),
    importance_score FLOAT DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON nah_remember.entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON nah_remember.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON nah_remember.entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==================== RELATIONSHIPS ====================
CREATE TABLE IF NOT EXISTS nah_remember.relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES nah_remember.entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES nah_remember.entities(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    description TEXT,
    strength FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source ON nah_remember.relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON nah_remember.relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON nah_remember.relationships(relationship_type);

-- ==================== SEMANTIC MEMORIES ====================
CREATE TABLE IF NOT EXISTS nah_remember.semantic_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) DEFAULT 'default_user',
    agent_id VARCHAR(255),
    content TEXT NOT NULL,
    memory_type VARCHAR(100),
    category VARCHAR(100),
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    importance_score FLOAT DEFAULT 0.5,
    embedding vector(1024),
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_semantic_user ON nah_remember.semantic_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_semantic_agent ON nah_remember.semantic_memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_semantic_type ON nah_remember.semantic_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_semantic_embedding ON nah_remember.semantic_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ==================== MEMORY HISTORY (Version Tracking) ====================
CREATE TABLE IF NOT EXISTS nah_remember.memory_history (
    id SERIAL PRIMARY KEY,
    memory_id UUID NOT NULL,
    version INTEGER NOT NULL,
    previous_content TEXT,
    new_content TEXT,
    change_type VARCHAR(50) NOT NULL CHECK (change_type IN ('create', 'update', 'delete')),
    changed_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_history_memory ON nah_remember.memory_history(memory_id);

-- ==================== PROCEDURES (Learned Workflows) ====================
CREATE TABLE IF NOT EXISTS nah_remember.procedures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    steps JSONB NOT NULL DEFAULT '[]',
    trigger_patterns TEXT[],
    success_rate FLOAT DEFAULT 0.0,
    execution_count INTEGER DEFAULT 0,
    embedding vector(1024),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_procedures_name ON nah_remember.procedures(name);

-- ==================== PROCEDURE EXECUTIONS ====================
CREATE TABLE IF NOT EXISTS nah_remember.procedure_executions (
    id SERIAL PRIMARY KEY,
    procedure_id UUID NOT NULL REFERENCES nah_remember.procedures(id) ON DELETE CASCADE,
    session_id VARCHAR(255),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(50) DEFAULT 'running',
    result JSONB,
    error_message TEXT
);

-- ==================== HELPER FUNCTIONS ====================

-- Function to get session context
CREATE OR REPLACE FUNCTION nah_remember.get_session_context(
    p_session_id VARCHAR(255),
    p_message_limit INTEGER DEFAULT 50
)
RETURNS TABLE (
    session_info JSONB,
    recent_messages JSONB,
    latest_snapshot JSONB,
    relevant_entities JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT row_to_json(s.*) FROM nah_remember.sessions s WHERE s.session_id = p_session_id)::JSONB,
        (SELECT COALESCE(json_agg(row_to_json(m.*) ORDER BY m.message_index DESC), '[]'::json)
         FROM (SELECT * FROM nah_remember.messages WHERE session_id = p_session_id ORDER BY message_index DESC LIMIT p_message_limit) m)::JSONB,
        (SELECT row_to_json(cs.*) FROM nah_remember.context_snapshots cs 
         WHERE cs.session_id = p_session_id ORDER BY cs.created_at DESC LIMIT 1)::JSONB,
        (SELECT COALESCE(json_agg(row_to_json(e.*)), '[]'::json)
         FROM nah_remember.entities e WHERE e.source_session_id = p_session_id)::JSONB;
END;
$$ LANGUAGE plpgsql;

-- Function for unified semantic search
CREATE OR REPLACE FUNCTION nah_remember.semantic_search(
    p_embedding vector(1024),
    p_memory_types TEXT[],
    p_limit_per_type INTEGER DEFAULT 5
)
RETURNS TABLE (
    memory_type TEXT,
    id TEXT,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    
    -- Search messages
    SELECT 'messages'::TEXT, m.message_id::TEXT, m.content,
           1 - (m.embedding <=> p_embedding)::FLOAT, m.metadata, m.created_at
    FROM nah_remember.messages m
    WHERE 'messages' = ANY(p_memory_types) AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> p_embedding
    LIMIT p_limit_per_type
    
    UNION ALL
    
    -- Search entities
    SELECT 'entities'::TEXT, e.id::TEXT, e.description,
           1 - (e.embedding <=> p_embedding)::FLOAT, e.metadata, e.created_at
    FROM nah_remember.entities e
    WHERE 'entities' = ANY(p_memory_types) AND e.embedding IS NOT NULL
    ORDER BY e.embedding <=> p_embedding
    LIMIT p_limit_per_type
    
    UNION ALL
    
    -- Search semantic memories
    SELECT 'semantic'::TEXT, sm.id::TEXT, sm.content,
           1 - (sm.embedding <=> p_embedding)::FLOAT, sm.metadata, sm.created_at
    FROM nah_remember.semantic_memories sm
    WHERE 'semantic' = ANY(p_memory_types) AND sm.embedding IS NOT NULL
    ORDER BY sm.embedding <=> p_embedding
    LIMIT p_limit_per_type
    
    UNION ALL
    
    -- Search procedures
    SELECT 'procedures'::TEXT, p.id::TEXT, p.description,
           1 - (p.embedding <=> p_embedding)::FLOAT, p.metadata, p.created_at
    FROM nah_remember.procedures p
    WHERE 'procedures' = ANY(p_memory_types) AND p.embedding IS NOT NULL
    ORDER BY p.embedding <=> p_embedding
    LIMIT p_limit_per_type;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA nah_remember TO human;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA nah_remember TO human;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA nah_remember TO human;

-- Verification
DO $$
BEGIN
    RAISE NOTICE 'NAH Memory Unified schema created successfully!';
    RAISE NOTICE 'Tables: sessions, messages, context_snapshots, entities, relationships, semantic_memories, memory_history, procedures, procedure_executions';
END $$;
