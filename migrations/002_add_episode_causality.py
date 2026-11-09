#!/usr/bin/env python3
"""
Migration 002: Add Layer 2 episode_causality table to nah_remember schema
Ecosystem: nah-neural / Subsystem: nah-remember
"""

import asyncio
import os
import asyncpg

DB_URL = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")

SQL = """
CREATE TABLE IF NOT EXISTS nah_remember.episode_causality (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_episode_id UUID NOT NULL REFERENCES nah_remember.episodes(id) ON DELETE CASCADE,
    target_episode_id UUID NOT NULL REFERENCES nah_remember.episodes(id) ON DELETE CASCADE,
    causal_type VARCHAR(100) NOT NULL,
    description TEXT,
    confidence FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_episode_id, target_episode_id, causal_type)
);

CREATE INDEX IF NOT EXISTS idx_episode_causality_source ON nah_remember.episode_causality(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_causality_target ON nah_remember.episode_causality(target_episode_id);
"""


async def run_migration():
    print(f"Connecting to {DB_URL}...")
    conn = await asyncpg.connect(DB_URL)
    try:
        print("Applying migration 002: episode_causality...")
        await conn.execute(SQL)
        print("Migration 002 applied successfully.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(run_migration())
