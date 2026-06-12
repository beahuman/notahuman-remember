#!/usr/bin/env python3
"""
Integration test for nah-memory-unified
Tests: DB connection, embedding generation, basic operations
"""

import asyncio
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_all():
    results = []

    # Test 1: Database connection
    print("Test 1: Database connection...")
    try:
        import asyncpg

        conn = await asyncpg.connect(
            os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
        )
        count = await conn.fetchval("SELECT COUNT(*) FROM nah_remember.sessions")
        await conn.close()
        print(f"  ✅ Database connected. Sessions count: {count}")
        results.append(("Database", True))
    except Exception as e:
        print(f"  ❌ Database failed: {e}")
        results.append(("Database", False))

    # Test 2: Embedding generation
    print("Test 2: Embedding generation...")
    try:
        import httpx

        response = httpx.post(
            os.getenv("NAH_MODEL_GATEWAY_URL", "http://localhost:4000") + "/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.getenv('NAH_MODEL_GATEWAY_KEY', '')}",
                "Content-Type": "application/json",
            },
            json={"model": "text-embedding-3-small", "input": "integration test"},
            timeout=30.0,
        )
        data = response.json()
        if "data" in data and len(data["data"]) > 0:
            dim = len(data["data"][0].get("embedding", []))
            print(f"  ✅ Embedding generated. Dimensions: {dim}")
            results.append(("Embedding", True))
        else:
            print(f"  ❌ Embedding failed: {data}")
            results.append(("Embedding", False))
    except Exception as e:
        print(f"  ❌ Embedding failed: {e}")
        results.append(("Embedding", False))

    # Test 3: Insert and retrieve session
    print("Test 3: Session CRUD...")
    try:
        import asyncpg

        conn = await asyncpg.connect(
            os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
        )

        # Insert
        await conn.execute(
            """
            INSERT INTO nah_remember.sessions (session_id, application, user_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (session_id) DO NOTHING
        """,
            "integration-test-session",
            "test",
            "test-user",
        )

        # Retrieve
        row = await conn.fetchrow(
            "SELECT * FROM nah_remember.sessions WHERE session_id = $1", "integration-test-session"
        )

        # Delete
        await conn.execute(
            "DELETE FROM nah_remember.sessions WHERE session_id = $1", "integration-test-session"
        )
        await conn.close()

        if row:
            print(f"  ✅ Session CRUD works. Retrieved: {row['session_id']}")
            results.append(("Session CRUD", True))
        else:
            print("  ❌ Session not found after insert")
            results.append(("Session CRUD", False))
    except Exception as e:
        print(f"  ❌ Session CRUD failed: {e}")
        results.append(("Session CRUD", False))

    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for _, v in results if v)
    total = len(results)
    for name, status in results:
        print(f"  {'✅' if status else '❌'} {name}")
    print(f"\nResult: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_all())
    sys.exit(0 if success else 1)
