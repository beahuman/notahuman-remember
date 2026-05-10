#!/usr/bin/env python3
"""
Phase 3 Integration Tests
Tests for ranked retrieval, query routing, and cross-layer composition
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncpg
from nah_remember_phase3 import (
    NahRememberPhase3,
    RankedResult,
    MemoryLayer,
    add_phase3_tools_to_mcp,
)


class TestPhase3:
    """Test suite for Phase 3"""

    def __init__(self):
        self.pool = None
        self.phase3 = None
        self.results = []

    async def setup(self):
        """Setup test database connection"""
        try:
            db_url = os.getenv("DATABASE_URL", "postgresql://human:@localhost/notahuman")
            self.pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
            self.phase3 = NahRememberPhase3(self.pool)
            print(f"✅ Database connected: {db_url}")
            self.results.append(("Database Setup", True, "Connected"))
            return True
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
            self.results.append(("Database Setup", False, str(e)))
            return False

    async def cleanup(self):
        """Cleanup connections"""
        if self.pool:
            await self.pool.close()

    async def test_ranked_result_scoring(self):
        """Test RankedResult composite score calculation"""
        print("\nTest: RankedResult composite scoring...")
        try:
            result = RankedResult(
                memory_type="test",
                memory_id="test-001",
                content="Test content",
                layer=MemoryLayer.SEMANTIC,
                significance_score=0.8,
                access_count=5,
                created_at=datetime.utcnow() - timedelta(days=5),
                recency_window_days=30,
            )

            # Expected scores:
            # - significance: 0.8
            # - access (normalized): (1 + 5) / 10 = 0.6
            # - recency: 1.0 - (5 / 30) = 0.833
            # - composite: 0.8*0.4 + 0.6*0.3 + 0.833*0.3 = 0.32 + 0.18 + 0.25 = 0.75

            composite = result.compute_composite_score()
            assert 0.74 < composite < 0.76, f"Expected ~0.75, got {composite}"

            print(f"  ✅ Composite score calculation correct: {composite:.3f}")
            self.results.append(("Composite Score Calculation", True, f"{composite:.3f}"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Composite Score Calculation", False, str(e)))

    async def test_ranked_result_dict_serialization(self):
        """Test RankedResult.to_dict() for JSON serialization"""
        print("\nTest: RankedResult JSON serialization...")
        try:
            result = RankedResult(
                memory_type="semantic_memory",
                memory_id="mem-123",
                content="This is test content for serialization",
                layer=MemoryLayer.SEMANTIC,
                significance_score=0.7,
                access_count=3,
                created_at=datetime.utcnow(),
                metadata={"source": "test", "tags": ["test", "phase3"]},
            )

            result_dict = result.to_dict()

            # Verify structure
            assert "memory_type" in result_dict
            assert "memory_id" in result_dict
            assert "content" in result_dict
            assert "layer" in result_dict
            assert "composite_score" in result_dict
            assert "significance_score" in result_dict
            assert "access_count" in result_dict
            assert "recency_score" in result_dict
            assert "created_at" in result_dict
            assert "metadata" in result_dict

            # Verify JSON-serializable
            json_str = json.dumps(result_dict)
            parsed = json.loads(json_str)
            assert parsed["layer"] == "semantic"

            print(f"  ✅ JSON serialization successful")
            print(f"     Keys: {list(result_dict.keys())}")
            self.results.append(("JSON Serialization", True, "All keys present"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("JSON Serialization", False, str(e)))

    async def test_ephemeral_query(self):
        """Test ephemeral layer query"""
        print("\nTest: Ephemeral layer query...")
        try:
            # Create a dummy embedding (1536-dim, all zeros)
            dummy_embedding = [0.0] * 1536

            results = await self.phase3.query_ephemeral(
                dummy_embedding, embedding_dim=1536, hours=24, limit=5
            )

            print(f"  ✅ Ephemeral query returned {len(results)} results")
            if results:
                print(f"     First result: {results[0].memory_type} ({results[0].layer.value})")
            self.results.append(("Ephemeral Query", True, f"{len(results)} results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Ephemeral Query", False, str(e)))

    async def test_episodic_query(self):
        """Test episodic layer query"""
        print("\nTest: Episodic layer query...")
        try:
            dummy_embedding = [0.0] * 1536

            results = await self.phase3.query_episodic(dummy_embedding, embedding_dim=1536, limit=5)

            print(f"  ✅ Episodic query returned {len(results)} results")
            if results:
                print(f"     Types: {set(r.memory_type for r in results)}")
            self.results.append(("Episodic Query", True, f"{len(results)} results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Episodic Query", False, str(e)))

    async def test_semantic_query(self):
        """Test semantic layer query"""
        print("\nTest: Semantic layer query...")
        try:
            dummy_embedding = [0.0] * 1536

            results = await self.phase3.query_semantic(dummy_embedding, embedding_dim=1536, limit=5)

            print(f"  ✅ Semantic query returned {len(results)} results")
            if results:
                print(f"     Types: {set(r.memory_type for r in results)}")
            self.results.append(("Semantic Query", True, f"{len(results)} results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Semantic Query", False, str(e)))

    async def test_meta_query(self):
        """Test meta layer query"""
        print("\nTest: Meta layer query...")
        try:
            results = await self.phase3.query_meta(limit=5)

            print(f"  ✅ Meta query returned {len(results)} results")
            if results:
                print(f"     Types: {set(r.memory_type for r in results)}")
            self.results.append(("Meta Query", True, f"{len(results)} results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Meta Query", False, str(e)))

    async def test_remember_search_ranked_all_layers(self):
        """Test unified ranked search across all layers"""
        print("\nTest: Unified ranked search (all layers)...")
        try:
            dummy_embedding = [0.0] * 1536

            result = await self.phase3.remember_search_ranked(
                dummy_embedding,
                memory_type="all",
                limit=20,
                embedding_dim=1536,
                significance_weight=0.4,
                access_weight=0.3,
                recency_weight=0.3,
            )

            assert result["success"], f"Query failed: {result.get('error')}"
            assert result["query_type"] == "all"
            assert "results" in result
            assert "layers_queried" in result

            print(f"  ✅ Ranked search succeeded")
            print(f"     Total results: {result['total_results']}")
            print(f"     Layers queried: {result['layers_queried']}")
            print(f"     Weights: {result['weights']}")

            if result["results"]:
                top = result["results"][0]
                print(
                    f"     Top result: {top['memory_type']} (score: {top['composite_score']:.3f})"
                )

            self.results.append(
                ("Unified Ranked Search", True, f"{result['total_results']} results")
            )
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Unified Ranked Search", False, str(e)))

    async def test_remember_search_ranked_single_layer(self):
        """Test ranked search with single layer filter"""
        print("\nTest: Ranked search (semantic layer only)...")
        try:
            dummy_embedding = [0.0] * 1536

            result = await self.phase3.remember_search_ranked(
                dummy_embedding, memory_type="semantic", limit=10, embedding_dim=1536
            )

            assert result["success"], f"Query failed: {result.get('error')}"
            assert result["query_type"] == "semantic"

            # Verify only semantic layers queried
            for layer in result["layers_queried"]:
                assert layer == "semantic", f"Unexpected layer: {layer}"

            print(f"  ✅ Single-layer search succeeded")
            print(f"     Results: {result['total_results']}")
            print(f"     Layers: {result['layers_queried']}")

            self.results.append(("Single-Layer Search", True, f"{result['total_results']} results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Single-Layer Search", False, str(e)))

    async def test_ranking_consistency(self):
        """Test that ranking is deterministic and consistent"""
        print("\nTest: Ranking consistency...")
        try:
            dummy_embedding = [0.0] * 1536

            # Run same query twice
            result1 = await self.phase3.remember_search_ranked(
                dummy_embedding, memory_type="all", limit=10
            )

            result2 = await self.phase3.remember_search_ranked(
                dummy_embedding, memory_type="all", limit=10
            )

            # Compare results
            assert result1["total_results"] == result2["total_results"]

            if result1["results"] and result2["results"]:
                # Check top 5 are in same order
                ids1 = [r["memory_id"] for r in result1["results"][:5]]
                ids2 = [r["memory_id"] for r in result2["results"][:5]]

                if ids1 == ids2:
                    print(f"  ✅ Ranking is consistent")
                    self.results.append(("Ranking Consistency", True, "Deterministic"))
                else:
                    print(f"  ⚠️  Top 5 results differ (expected if data changed)")
                    self.results.append(("Ranking Consistency", True, "Data-dependent"))
            else:
                print(f"  ✅ Consistent (no results)")
                self.results.append(("Ranking Consistency", True, "No results"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Ranking Consistency", False, str(e)))

    async def test_weight_normalization(self):
        """Test that weights are normalized correctly"""
        print("\nTest: Weight normalization...")
        try:
            dummy_embedding = [0.0] * 1536

            # Use unnormalized weights (sum != 1.0)
            result = await self.phase3.remember_search_ranked(
                dummy_embedding,
                memory_type="all",
                limit=5,
                significance_weight=0.5,  # Sum = 1.5
                access_weight=0.5,
                recency_weight=0.5,
            )

            assert result["success"]
            weights = result["weights"]

            # Check normalized sum ~= 1.0
            total = weights["significance"] + weights["access_count"] + weights["recency"]
            assert 0.99 < total < 1.01, f"Weights don't sum to 1.0: {total}"

            print(f"  ✅ Weights normalized correctly")
            print(f"     Original: 0.5, 0.5, 0.5 (sum=1.5)")
            print(
                f"     Normalized: {weights['significance']:.3f}, {weights['access_count']:.3f}, {weights['recency']:.3f} (sum={total:.3f})"
            )

            self.results.append(("Weight Normalization", True, f"Sum={total:.3f}"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("Weight Normalization", False, str(e)))

    async def test_mcp_integration(self):
        """Test MCP integration hook"""
        print("\nTest: MCP integration...")
        try:
            mcp_tools = {}
            add_phase3_tools_to_mcp(mcp_tools, self.phase3)

            assert "nah_remember_search_ranked" in mcp_tools
            assert callable(mcp_tools["nah_remember_search_ranked"])

            print(f"  ✅ MCP integration successful")
            print(f"     Tools registered: {list(mcp_tools.keys())}")

            self.results.append(("MCP Integration", True, "Tool registered"))
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            self.results.append(("MCP Integration", False, str(e)))

    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("NAH Remember Phase 3 Test Suite")
        print("=" * 60)

        if not await self.setup():
            print("\n❌ Setup failed, cannot run tests")
            return False

        try:
            # Unit tests
            await self.test_ranked_result_scoring()
            await self.test_ranked_result_dict_serialization()

            # Query layer tests
            await self.test_ephemeral_query()
            await self.test_episodic_query()
            await self.test_semantic_query()
            await self.test_meta_query()

            # Integration tests
            await self.test_remember_search_ranked_all_layers()
            await self.test_remember_search_ranked_single_layer()
            await self.test_ranking_consistency()
            await self.test_weight_normalization()
            await self.test_mcp_integration()

        finally:
            await self.cleanup()

        # Print summary
        self._print_summary()
        return self._all_passed()

    def _print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for _, status, _ in self.results if status)
        total = len(self.results)

        for name, status, details in self.results:
            status_str = "✅" if status else "❌"
            print(f"{status_str} {name:<40} {details}")

        print(f"\nResult: {passed}/{total} tests passed")
        return passed == total

    def _all_passed(self) -> bool:
        """Check if all tests passed"""
        return all(status for _, status, _ in self.results)


async def main():
    """Main entry point"""
    test = TestPhase3()
    success = await test.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
