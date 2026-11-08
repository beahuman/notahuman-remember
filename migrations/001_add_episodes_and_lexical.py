#!/usr/bin/env python3
"""
Migration 001: Add true episodic primitives and lexical TSVECTOR support
Ecosystem: nah-neural / Subsystem: nah-remember
"""

import asyncio
import os
import asyncpg

DB_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")

SQL = """
-- 1. Create EPISODES Table
CREATE TABLE IF NOT EXISTS nah_remember.episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255),
    agent_id VARCHAR(255),
    title VARCHAR(500) NOT NULL,
    goal TEXT NOT NULL,
    task_type VARCHAR(100),
    pressure FLOAT DEFAULT 0.5,
    hunt_context VARCHAR(255),
    outcome VARCHAR(50) DEFAULT 'in_progress' CHECK (outcome IN ('success', 'failure', 'partial', 'aborted', 'in_progress')),
    narrative TEXT,
    retrospective TEXT,
    embedding vector(1024),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    reinforcement_score FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON nah_remember.episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_agent ON nah_remember.episodes(agent_id);
CREATE INDEX IF NOT EXISTS idx_episodes_task_type ON nah_remember.episodes(task_type);
CREATE INDEX IF NOT EXISTS idx_episodes_hunt_context ON nah_remember.episodes(hunt_context);
CREATE INDEX IF NOT EXISTS idx_episodes_outcome ON nah_remember.episodes(outcome);
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON nah_remember.episodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 2. Create EPISODE_EVENTS Table
CREATE TABLE IF NOT EXISTS nah_remember.episode_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES nah_remember.episodes(id) ON DELETE CASCADE,
    event_index INTEGER NOT NULL,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('action', 'observation', 'error', 'pivot', 'breakthrough', 'decision')),
    description TEXT NOT NULL,
    causal_parent_id UUID REFERENCES nah_remember.episode_events(id) ON DELETE SET NULL,
    tool_name VARCHAR(255),
    tool_input JSONB,
    tool_result JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(episode_id, event_index)
);

CREATE INDEX IF NOT EXISTS idx_episode_events_episode ON nah_remember.episode_events(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_events_type ON nah_remember.episode_events(event_type);

-- 3. Add reinforcement_score to semantic_memories if not exists
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='nah_remember' AND table_name='semantic_memories' AND column_name='reinforcement_score') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN reinforcement_score FLOAT DEFAULT 0.0;
    END IF;
END $$;

-- 4. Full-Text Search TSVECTOR columns for Bet 2 lexical matching
DO $$ 
BEGIN 
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='nah_remember' AND table_name='episodes' AND column_name='tsv') THEN
        ALTER TABLE nah_remember.episodes ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(goal, '') || ' ' || coalesce(narrative, '') || ' ' || coalesce(retrospective, ''))
        ) STORED;
        CREATE INDEX IF NOT EXISTS idx_episodes_tsv ON nah_remember.episodes USING gin(tsv);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='nah_remember' AND table_name='semantic_memories' AND column_name='tsv') THEN
        ALTER TABLE nah_remember.semantic_memories ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content, ''))
        ) STORED;
        CREATE INDEX IF NOT EXISTS idx_semantic_memories_tsv ON nah_remember.semantic_memories USING gin(tsv);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema='nah_remember' AND table_name='procedures' AND column_name='tsv') THEN
        ALTER TABLE nah_remember.procedures ADD COLUMN tsv tsvector GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
        ) STORED;
        CREATE INDEX IF NOT EXISTS idx_procedures_tsv ON nah_remember.procedures USING gin(tsv);
    END IF;
END $$;
"""


async def run_migration():
    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)
    try:
        print("Executing Migration 001...")
        await conn.execute(SQL)
        print("Migration 001 successfully applied!")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migration())
