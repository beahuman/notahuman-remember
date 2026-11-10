-- Migration 002: Reconcile Schema Parity & SSoT Alignment
-- Safe, idempotent migration to align existing databases with setup_schema.sql

DO $$
BEGIN
    -- 1. semantic_memories columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'semantic_memories' AND column_name = 'retrieval_count') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN retrieval_count INTEGER DEFAULT 0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'semantic_memories' AND column_name = 'reinforcement_score') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN reinforcement_score FLOAT DEFAULT 0.0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'semantic_memories' AND column_name = 'last_retrieved_at') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN last_retrieved_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'semantic_memories' AND column_name = 'archived') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN archived BOOLEAN DEFAULT FALSE;
    END IF;

    -- 2. procedures columns & unique constraint
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'procedure_type') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN procedure_type VARCHAR(50) DEFAULT 'playbook';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'trigger_conditions') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN trigger_conditions JSONB DEFAULT '[]';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'prerequisites') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN prerequisites JSONB DEFAULT '[]';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'avg_duration_ms') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN avg_duration_ms INTEGER DEFAULT 0;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'agent_id') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN agent_id VARCHAR(50) DEFAULT 'system';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'status') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN status VARCHAR(20) DEFAULT 'active';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'procedures_name_key') THEN
        ALTER TABLE nah_remember.procedures ADD CONSTRAINT procedures_name_key UNIQUE (name);
    END IF;

    -- 3. procedure_executions columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedure_executions' AND column_name = 'duration_ms') THEN
        ALTER TABLE nah_remember.procedure_executions ADD COLUMN duration_ms INTEGER;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedure_executions' AND column_name = 'input_data') THEN
        ALTER TABLE nah_remember.procedure_executions ADD COLUMN input_data JSONB;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedure_executions' AND column_name = 'output_data') THEN
        ALTER TABLE nah_remember.procedure_executions ADD COLUMN output_data JSONB;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedure_executions' AND column_name = 'error_message') THEN
        ALTER TABLE nah_remember.procedure_executions ADD COLUMN error_message TEXT;
    END IF;

    -- 4. negative_constraints table & constraints
    CREATE TABLE IF NOT EXISTS nah_remember.negative_constraints (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        pattern TEXT NOT NULL,
        trigger_context TEXT NOT NULL,
        severity VARCHAR(50) DEFAULT 'warning',
        weight FLOAT DEFAULT 0.8,
        source_session_id VARCHAR(255) REFERENCES nah_remember.sessions(session_id) ON DELETE SET NULL,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        hit_count INTEGER DEFAULT 0,
        last_hit_at TIMESTAMPTZ
    );

    -- If table already existed with legacy source_session_id UUID column, migrate it to VARCHAR(255)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'nah_remember' 
          AND table_name = 'negative_constraints' 
          AND column_name = 'source_session_id' 
          AND data_type = 'uuid'
    ) THEN
        ALTER TABLE nah_remember.negative_constraints DROP CONSTRAINT IF EXISTS negative_constraints_source_session_id_fkey;
        ALTER TABLE nah_remember.negative_constraints ALTER COLUMN source_session_id TYPE VARCHAR(255) USING source_session_id::text;
    END IF;

    -- Ensure foreign key constraint points to sessions(session_id)
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'negative_constraints_source_session_id_fkey') THEN
        ALTER TABLE nah_remember.negative_constraints ADD CONSTRAINT negative_constraints_source_session_id_fkey 
            FOREIGN KEY (source_session_id) REFERENCES nah_remember.sessions(session_id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_negative_constraints_pattern_context') THEN
        ALTER TABLE nah_remember.negative_constraints ADD CONSTRAINT uq_negative_constraints_pattern_context UNIQUE (pattern, trigger_context);
    END IF;

    -- 5. Expand episode_events check constraint
    ALTER TABLE nah_remember.episode_events DROP CONSTRAINT IF EXISTS episode_events_event_type_check;
    ALTER TABLE nah_remember.episode_events ADD CONSTRAINT episode_events_event_type_check 
        CHECK (event_type IN ('action', 'observation', 'error', 'failure', 'timeout', 'retry', 'pivot', 'breakthrough', 'decision'));

    -- 6. Ensure tsvector columns and indexes
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'semantic_memories' AND column_name = 'tsv') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED;
    END IF;
    CREATE INDEX IF NOT EXISTS idx_semantic_memories_tsv ON nah_remember.semantic_memories USING gin (tsv);

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'procedures' AND column_name = 'tsv') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', (COALESCE(name, '') || ' ') || COALESCE(description, ''))) STORED;
    END IF;
    CREATE INDEX IF NOT EXISTS idx_procedures_tsv ON nah_remember.procedures USING gin (tsv);

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'nah_remember' AND table_name = 'episodes' AND column_name = 'tsv') THEN
        ALTER TABLE nah_remember.episodes ADD COLUMN tsv tsvector 
            GENERATED ALWAYS AS (to_tsvector('english', ((((COALESCE(title, '') || ' ') || COALESCE(goal, '')) || ' ') || COALESCE(narrative, '')) || ' ' || COALESCE(retrospective, ''))) STORED;
    END IF;
    CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON nah_remember.episodes USING gin (tsv);

    RAISE NOTICE 'Migration 002: Schema parity reconciliation completed successfully!';
END $$;
