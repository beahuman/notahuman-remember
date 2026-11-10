-- Migration 003: Negative Constraints Parity & All-Layers Search Alignment
-- Backfill existing constraints from semantic_memories and ensure canonical failure invariants

DO $$
BEGIN
    -- 1. Ensure table exists with correct schema
    CREATE TABLE IF NOT EXISTS nah_remember.negative_constraints (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        pattern TEXT NOT NULL,
        trigger_context TEXT NOT NULL,
        severity VARCHAR(50) DEFAULT 'warning',
        weight FLOAT DEFAULT 0.8,
        source_session_id VARCHAR(255) REFERENCES nah_remember.sessions(session_id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        hit_count INTEGER DEFAULT 0,
        last_hit_at TIMESTAMPTZ,
        CONSTRAINT uq_negative_constraints_pattern_context UNIQUE (pattern, trigger_context)
    );

    CREATE INDEX IF NOT EXISTS idx_negative_constraints_context
        ON nah_remember.negative_constraints
        USING gin(to_tsvector('english', trigger_context));

    CREATE INDEX IF NOT EXISTS idx_negative_constraints_pattern
        ON nah_remember.negative_constraints
        USING gin(to_tsvector('english', pattern));

    -- 2. Backfill from semantic_memories tagged as negative_constraint
    INSERT INTO nah_remember.negative_constraints (pattern, trigger_context, severity, weight)
    SELECT
        REPLACE(content, 'NEGATIVE CONSTRAINT: ', '') as pattern,
        'agent runtime, omo platform, llm routing, model selection, bedrock, anthropic' as trigger_context,
        'critical' as severity,
        1.0 as weight
    FROM nah_remember.semantic_memories
    WHERE category = 'negative_constraint' OR 'negative_constraint' = ANY(tags)
    ON CONFLICT (pattern, trigger_context) DO NOTHING;

    -- 3. Seed canonical ecosystem invariants
    INSERT INTO nah_remember.negative_constraints (pattern, trigger_context, severity, weight)
    VALUES
        ('Never state or assume Omo runs on OpenCode. Omo is a Hermes agent. Humano, Astro, and Kodo are also Hermes.', 'agent roster, omo platform, runtime environment, hermes, opencode', 'critical', 1.0),
        ('Amazon Bedrock is temporarily unavailable and disabled in OmniRoute.', 'amazon bedrock, aws models, model selection, provider routing', 'critical', 1.0),
        ('Anthropic commercial models are forbidden due to cost, unless explicitly free.', 'anthropic models, claude models, commercial llms, token pricing', 'critical', 1.0),
        ('Never use weak or pointless smoke tests; only deep disruptive adversarial and fault-injection verification is permitted.', 'test verification, smoke tests, quality assurance, test suites', 'high', 0.9),
        ('Never bypass access controls, permission checks, credentials, or sandboxes.', 'security, permissions, sandboxes, credentials, access control', 'critical', 1.0)
    ON CONFLICT (pattern, trigger_context) DO NOTHING;

END $$;

-- 4. Update helper function to include negative_constraints and episodes in search_all_memories
CREATE OR REPLACE FUNCTION nah_remember.search_all_memories(
    p_embedding vector(1024),
    p_limit_per_type INTEGER DEFAULT 5,
    p_memory_types TEXT[] DEFAULT ARRAY['messages', 'entities', 'semantic', 'procedures', 'episodes', 'negative_constraints']
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
    (SELECT 'messages'::TEXT, m.message_id::TEXT, m.content,
           (1 - (m.embedding <=> p_embedding))::FLOAT, m.metadata, m.created_at
    FROM nah_remember.messages m
    WHERE 'messages' = ANY(p_memory_types) AND m.embedding IS NOT NULL
    ORDER BY m.embedding <=> p_embedding
    LIMIT p_limit_per_type)
    
    UNION ALL
    
    -- Search entities
    (SELECT 'entities'::TEXT, e.id::TEXT, e.description,
           (1 - (e.embedding <=> p_embedding))::FLOAT, e.metadata, e.created_at
    FROM nah_remember.entities e
    WHERE 'entities' = ANY(p_memory_types) AND e.embedding IS NOT NULL
    ORDER BY e.embedding <=> p_embedding
    LIMIT p_limit_per_type)
    
    UNION ALL
    
    -- Search semantic memories
    (SELECT 'semantic'::TEXT, sm.id::TEXT, sm.content,
           (1 - (sm.embedding <=> p_embedding))::FLOAT, sm.metadata, sm.created_at
    FROM nah_remember.semantic_memories sm
    WHERE 'semantic' = ANY(p_memory_types) AND sm.embedding IS NOT NULL
    ORDER BY sm.embedding <=> p_embedding
    LIMIT p_limit_per_type)
    
    UNION ALL
    
    -- Search procedures
    (SELECT 'procedures'::TEXT, p.id::TEXT, p.description,
           (1 - (p.embedding <=> p_embedding))::FLOAT, p.metadata, p.created_at
    FROM nah_remember.procedures p
    WHERE 'procedures' = ANY(p_memory_types) AND p.embedding IS NOT NULL
    ORDER BY p.embedding <=> p_embedding
    LIMIT p_limit_per_type)

    UNION ALL

    -- Search episodes
    (SELECT 'episodes'::TEXT, ep.id::TEXT, ep.narrative,
           (1 - (ep.embedding <=> p_embedding))::FLOAT, ep.metadata, ep.created_at
    FROM nah_remember.episodes ep
    WHERE 'episodes' = ANY(p_memory_types) AND ep.embedding IS NOT NULL
    ORDER BY ep.embedding <=> p_embedding
    LIMIT p_limit_per_type)

    UNION ALL

    -- Search negative constraints
    (SELECT 'negative_constraints'::TEXT, nc.id::TEXT, nc.pattern || ' (Trigger: ' || nc.trigger_context || ')',
           0.9::FLOAT, jsonb_build_object('severity', nc.severity, 'weight', nc.weight)::JSONB, nc.created_at
    FROM nah_remember.negative_constraints nc
    WHERE 'negative_constraints' = ANY(p_memory_types)
    LIMIT p_limit_per_type);
END;
$$ LANGUAGE plpgsql;
