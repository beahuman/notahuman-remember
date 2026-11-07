"""Unit tests for Phase 3 composite ranking logic."""
import pytest
from datetime import datetime, timedelta, timezone
from nah_remember.ranking import (
    RankedResult,
    MemoryLayer,
    NahRememberPhase3,
)


def test_ranked_result_3_factor_scoring():
    result = RankedResult(
        memory_type="test",
        memory_id="test-001",
        content="Test content",
        layer=MemoryLayer.SEMANTIC,
        significance_score=0.8,
        access_count=5,
        created_at=datetime.now(timezone.utc) - timedelta(days=5),
        recency_window_days=30,
    )

    # 4-factor scoring with reinforcement=0.0
    score = result.compute_composite_score(0.4, 0.3, 0.3, 0.0)
    assert 0.74 < score < 0.76


def test_ranked_result_4_factor_scoring():
    result = RankedResult(
        memory_type="test",
        memory_id="test-002",
        content="Test content with reinforcement",
        layer=MemoryLayer.SEMANTIC,
        significance_score=0.9,
        access_count=10,
        reinforcement_score=0.85,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
        recency_window_days=30,
    )

    score = result.compute_composite_score(0.3, 0.2, 0.2, 0.3)
    assert 0.0 <= score <= 1.0


def test_ranked_result_serialization():
    result = RankedResult(
        memory_type="semantic_memory",
        memory_id="mem-123",
        content="Test payload",
        layer=MemoryLayer.SEMANTIC,
        significance_score=0.85,
        access_count=7,
        created_at=datetime.now(timezone.utc),
    )

    d = result.to_dict()
    assert d["memory_id"] == "mem-123"
    assert d["memory_type"] == "semantic_memory"
    assert d["layer"] == "semantic"
    assert "composite_score" in d
