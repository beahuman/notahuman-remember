#!/usr/bin/env python3
"""
Migrate data from legacy memory systems to unified nah_remember schema.

Sources:
1. RAG Memory: public.entities (notahuman db), public.relationships
2. Episodic Memory: public.chat_sessions, conversation_messages, memory_snapshots (notahuman_episodic_memory db)

Target: notahuman.nah_remember.*
"""

import subprocess
import sys
import json
from datetime import datetime


def run_sql(database: str, sql: str) -> str:
    """Execute SQL and return output."""
    result = subprocess.run(
        ["psql", "-h", "localhost", "-U", "human", "-d", database, "-t", "-A", "-c", sql],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  SQL Error: {result.stderr.strip()[:200]}")
        return ""
    return result.stdout.strip()


def migrate_entities():
    """Migrate RAG memory entities to nah_remember.entities."""
    print("\n=== Migrating Entities from RAG Memory ===")

    # Get existing entities from public schema
    sql = """
    SELECT json_agg(row_to_json(e))
    FROM (
        SELECT name, entity_type, description, metadata, created_at, updated_at
        FROM public.entities
    ) e;
    """
    result = run_sql("notahuman", sql)

    if not result or result == "" or result == "null":
        print("  No entities found in source table")
        return 0

    try:
        entities = json.loads(result)
        if not entities:
            print("  No entities to migrate")
            return 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse entities: {e}")
        return 0

    migrated = 0
    for entity in entities:
        name = entity["name"].replace("'", "''")
        entity_type = (entity.get("entity_type") or "unknown").replace("'", "''")
        description = (entity.get("description") or "").replace("'", "''")
        metadata = json.dumps(entity.get("metadata") or {}).replace("'", "''")
        created = entity.get("created_at", datetime.now().isoformat())
        updated = entity.get("updated_at", datetime.now().isoformat())

        # Insert into nah_remember.entities (matching actual schema)
        insert_sql = f"""
        INSERT INTO nah_remember.entities (name, entity_type, description, metadata, created_at, updated_at)
        VALUES ('{name}', '{entity_type}', '{description}', '{metadata}'::jsonb, '{created}'::timestamptz, '{updated}'::timestamptz)
        ON CONFLICT (name, entity_type) DO UPDATE SET
            description = EXCLUDED.description,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at;
        """

        run_sql("notahuman", insert_sql)
        migrated += 1
        print(f"  Migrated entity: {entity['name'][:50]}...")

    print(f"  Total entities migrated: {migrated}")
    return migrated


def migrate_relationships():
    """Migrate RAG memory relationships to nah_remember.relationships."""
    print("\n=== Migrating Relationships from RAG Memory ===")

    # First check if public.relationships exists
    check_sql = """
    SELECT COUNT(*) FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'relationships';
    """
    exists = run_sql("notahuman", check_sql)
    if exists != "1":
        print("  No relationships table in public schema")
        return 0

    # Get column names
    col_sql = """
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'relationships';
    """
    cols = run_sql("notahuman", col_sql)
    print(f"  Found columns: {cols.replace(chr(10), ', ')}")

    # Query based on actual columns (source_entity_id, target_entity_id based on foreign keys)
    sql = """
    SELECT json_agg(row_to_json(r))
    FROM (
        SELECT r.id, r.source_entity_id, r.target_entity_id, r.relationship_type,
               r.description, r.metadata, r.created_at,
               se.name as source_name, te.name as target_name
        FROM public.relationships r
        JOIN public.entities se ON r.source_entity_id = se.id
        JOIN public.entities te ON r.target_entity_id = te.id
    ) r;
    """
    result = run_sql("notahuman", sql)

    if not result or result == "" or result == "null":
        print("  No relationships found")
        return 0

    try:
        relationships = json.loads(result)
        if not relationships:
            print("  No relationships to migrate")
            return 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse: {e}")
        return 0

    migrated = 0
    for rel in relationships:
        # Get source and target entity IDs from nah_remember.entities
        source_name = rel["source_name"].replace("'", "''")
        target_name = rel["target_name"].replace("'", "''")

        source_id_sql = (
            f"SELECT id FROM nah_remember.entities WHERE name = '{source_name}' LIMIT 1;"
        )
        target_id_sql = (
            f"SELECT id FROM nah_remember.entities WHERE name = '{target_name}' LIMIT 1;"
        )

        source_id = run_sql("notahuman", source_id_sql)
        target_id = run_sql("notahuman", target_id_sql)

        if not source_id or not target_id:
            print(
                f"  Skipping (entities not found): {rel['source_name'][:20]} -> {rel['target_name'][:20]}"
            )
            continue

        rel_type = (rel.get("relationship_type") or "related").replace("'", "''")
        description = (rel.get("description") or "").replace("'", "''")
        metadata = json.dumps(rel.get("metadata") or {}).replace("'", "''")
        created = rel.get("created_at", datetime.now().isoformat())

        insert_sql = f"""
        INSERT INTO nah_remember.relationships (source_entity_id, target_entity_id, relationship_type, description, metadata, created_at)
        VALUES ('{source_id}'::uuid, '{target_id}'::uuid, '{rel_type}', '{description}', '{metadata}'::jsonb, '{created}'::timestamptz)
        ON CONFLICT DO NOTHING;
        """

        run_sql("notahuman", insert_sql)
        migrated += 1
        print(f"  Migrated: {source_name[:20]} -> {target_name[:20]}")

    print(f"  Total relationships migrated: {migrated}")
    return migrated


def migrate_episodic_sessions():
    """Migrate episodic memory sessions and messages."""
    print("\n=== Migrating Sessions from Episodic Memory ===")

    # Get sessions using actual column names
    sql = """
    SELECT json_agg(row_to_json(s))
    FROM (
        SELECT session_id, application, project_context, created_at, ended_at, metadata
        FROM chat_sessions
    ) s;
    """
    result = run_sql("notahuman_episodic_memory", sql)

    if not result or result == "" or result == "null":
        print("  No sessions found")
        return 0

    try:
        sessions = json.loads(result)
        if not sessions:
            print("  No sessions to migrate")
            return 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse: {e}")
        return 0

    migrated_sessions = 0
    for session in sessions:
        old_session_id = session["session_id"]
        application = (session.get("application") or "unknown").replace("'", "''")
        project = (session.get("project_context") or "").replace("'", "''")
        started = session.get("created_at", datetime.now().isoformat())
        ended = session.get("ended_at")
        metadata = json.dumps(
            session.get("metadata") or {"source": "episodic-memory-legacy"}
        ).replace("'", "''")

        # Create session in nah_remember with correct column names
        insert_sql = f"""
        INSERT INTO nah_remember.sessions (session_id, application, project_context, created_at, ended_at, metadata)
        VALUES ('{old_session_id}', '{application}', '{project}', '{started}'::timestamptz,
                {f"'{ended}'::timestamptz" if ended else "NULL"}, '{metadata}'::jsonb)
        ON CONFLICT (session_id) DO UPDATE SET
            application = EXCLUDED.application,
            project_context = EXCLUDED.project_context,
            ended_at = EXCLUDED.ended_at,
            metadata = EXCLUDED.metadata
        RETURNING session_id;
        """
        new_session_id = run_sql("notahuman", insert_sql)
        if new_session_id:
            migrated_sessions += 1
            print(f"  Migrated session: {old_session_id[:30]}...")

            # Store mapping for messages migration
            session["migrated_session_id"] = new_session_id

    print(f"  Total sessions migrated: {migrated_sessions}")

    if migrated_sessions == 0:
        return 0

    # Migrate messages
    print("\n=== Migrating Messages from Episodic Memory ===")

    # Use correct column names: token_count instead of tokens_used, timestamp instead of created_at
    sql = """
    SELECT json_agg(row_to_json(m))
    FROM (
        SELECT id, session_id, role, content, token_count, timestamp
        FROM conversation_messages
    ) m;
    """
    result = run_sql("notahuman_episodic_memory", sql)

    if not result or result == "" or result == "null":
        print("  No messages found")
        return migrated_sessions

    try:
        messages = json.loads(result)
        if not messages:
            print("  No messages to migrate")
            return migrated_sessions
    except json.JSONDecodeError as e:
        print(f"  Failed to parse: {e}")
        return migrated_sessions

    migrated_msgs = 0
    # Track message index per session
    session_msg_counts = {}

    for msg in messages:
        old_session_id = msg["session_id"]

        # Check if session exists in nah_remember
        check_session_sql = f"SELECT session_id FROM nah_remember.sessions WHERE session_id = '{old_session_id}' LIMIT 1;"
        session_exists = run_sql("notahuman", check_session_sql)

        if not session_exists:
            print(f"  Skipping message (session not found): {old_session_id[:20]}")
            continue

        # Increment message index for this session
        if old_session_id not in session_msg_counts:
            session_msg_counts[old_session_id] = 0
        session_msg_counts[old_session_id] += 1
        message_index = session_msg_counts[old_session_id]

        role = (msg.get("role") or "user").replace("'", "''")
        content = (msg.get("content") or "").replace("'", "''")[:10000]
        tokens = msg.get("token_count") or 0
        timestamp = msg.get("timestamp", datetime.now().isoformat())

        # nah_remember.messages uses session_id as varchar FK, message_index is required
        insert_sql = f"""
        INSERT INTO nah_remember.messages (session_id, message_index, role, content, token_count, created_at)
        VALUES ('{old_session_id}', {message_index}, '{role}', '{content}', {tokens}, '{timestamp}'::timestamptz)
        ON CONFLICT DO NOTHING;
        """
        run_sql("notahuman", insert_sql)
        migrated_msgs += 1
        print(f"  Migrated message #{message_index}: {role} ({tokens} tokens)")

    print(f"  Total messages migrated: {migrated_msgs}")
    return migrated_sessions


def migrate_memory_snapshots():
    """Migrate memory snapshots to context_snapshots."""
    print("\n=== Migrating Memory Snapshots ===")

    # Check columns in memory_snapshots
    col_sql = """
    SELECT column_name FROM information_schema.columns
    WHERE table_schema = 'public' AND table_name = 'memory_snapshots';
    """
    cols = run_sql("notahuman_episodic_memory", col_sql)
    print(f"  Found columns: {cols.replace(chr(10), ', ')}")

    # Use correct column name: content instead of snapshot_data
    sql = """
    SELECT json_agg(row_to_json(s))
    FROM (
        SELECT id, session_id, snapshot_type, content, created_at, context_summary, key_entities, action_items, decisions_made
        FROM memory_snapshots
    ) s;
    """
    result = run_sql("notahuman_episodic_memory", sql)

    if not result or result == "" or result == "null":
        print("  No snapshots found")
        return 0

    try:
        snapshots = json.loads(result)
        if not snapshots:
            print("  No snapshots to migrate")
            return 0
    except json.JSONDecodeError as e:
        print(f"  Failed to parse: {e}")
        return 0

    migrated = 0
    for snap in snapshots:
        old_session_id = snap.get("session_id", "")

        # Check if session exists in nah_remember (required for FK)
        check_session_sql = f"SELECT session_id FROM nah_remember.sessions WHERE session_id = '{old_session_id}' LIMIT 1;"
        session_exists = run_sql("notahuman", check_session_sql)

        if not session_exists:
            print(f"  Skipping snapshot (session not found): {old_session_id[:20]}")
            continue

        snap_type = (snap.get("snapshot_type") or "context").replace("'", "''")
        content = snap.get("content") or ""
        # Wrap plain text content in JSON object for context_data
        if isinstance(content, str):
            content_obj = {"text": content, "source": "episodic-memory-legacy"}
            content_json = json.dumps(content_obj).replace("'", "''")
        else:
            content_json = json.dumps(content).replace("'", "''")
        created = snap.get("created_at", datetime.now().isoformat())
        context_summary = (snap.get("context_summary") or "").replace("'", "''")

        # Handle array fields
        key_entities = snap.get("key_entities") or []
        action_items = snap.get("action_items") or []
        decisions_made = snap.get("decisions_made") or []

        # Format arrays for PostgreSQL
        key_entities_sql = (
            "ARRAY["
            + ",".join([f"'{e.replace(chr(39), chr(39) + chr(39))}'" for e in key_entities])
            + "]::text[]"
            if key_entities
            else "ARRAY[]::text[]"
        )
        action_items_sql = (
            "ARRAY["
            + ",".join([f"'{a.replace(chr(39), chr(39) + chr(39))}'" for a in action_items])
            + "]::text[]"
            if action_items
            else "ARRAY[]::text[]"
        )
        decisions_sql = (
            "ARRAY["
            + ",".join([f"'{d.replace(chr(39), chr(39) + chr(39))}'" for d in decisions_made])
            + "]::text[]"
            if decisions_made
            else "ARRAY[]::text[]"
        )

        insert_sql = f"""
        INSERT INTO nah_remember.context_snapshots (session_id, snapshot_type, context_data, context_summary, key_entities, action_items, decisions_made, created_at)
        VALUES ('{old_session_id}', '{snap_type}', '{content_json}'::jsonb, '{context_summary}', {key_entities_sql}, {action_items_sql}, {decisions_sql}, '{created}'::timestamptz)
        ON CONFLICT DO NOTHING;
        """
        run_sql("notahuman", insert_sql)
        migrated += 1
        print(f"  Migrated snapshot: {snap_type}")

    print(f"  Total snapshots migrated: {migrated}")
    return migrated


def verify_migration():
    """Verify migration was successful."""
    print("\n=== Verifying Migration ===")

    tables = ["entities", "relationships", "sessions", "messages", "context_snapshots"]

    for table in tables:
        count = run_sql("notahuman", f"SELECT COUNT(*) FROM nah_remember.{table};")
        print(f"  nah_remember.{table}: {count} records")


def main():
    print("=" * 60)
    print("NAH Memory Unified - Data Migration")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Run migrations
    migrate_entities()
    migrate_relationships()
    migrate_episodic_sessions()
    migrate_memory_snapshots()

    # Verify
    verify_migration()

    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print("\nNote: Original data preserved in source tables.")
    print("Old memory systems can be deprecated after verification.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
