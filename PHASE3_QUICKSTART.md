# Phase 3 Quick Start Guide

## What is Phase 3?

Phase 3 adds **ranked, cross-layer search** to nah-remember. Query across 4 memory layers (recent, episodic, semantic, meta) and get results ranked by a composite score.

## The 4 Memory Layers

```
Ephemeral ──────  Recent messages (< 24h)
Episodic ───────  Project state, entities, snapshots
Semantic ───────  Long-term knowledge, procedures
Meta ────────────  Relationships, evolution history
```

## Quick API

### Search Everything

```python
result = await phase3.remember_search_ranked(
    query_embedding=[0.1, 0.2, ...],  # 1536-dim vector
    memory_type="all",                # or: "ephemeral"|"episodic"|"semantic"|"meta"
    limit=20
)

for item in result["results"]:
    print(f"{item['memory_type']}: {item['composite_score']:.3f}")
    print(f"  {item['content'][:100]}")
```

### Return Value

```python
{
    "success": True,
    "total_results": 8,
    "layers_queried": ["ephemeral", "semantic"],
    "results": [
        {
            "memory_type": "semantic_memory",
            "memory_id": "uuid-123",
            "content": "...",
            "layer": "semantic",
            "composite_score": 0.78,        # Final rank score
            "significance_score": 0.8,      # Importance (40%)
            "access_count": 5,              # Usage (30% → score 0.6)
            "recency_score": 0.83,          # Freshness (30%)
            "created_at": "2025-03-18T..."
        },
        ...
    ]
}
```

## Ranking Formula

```
Score = (Importance × 0.4) + (Usage × 0.3) + (Freshness × 0.3)
```

- **Importance:** Stored when memory created (0.0 - 1.0)
- **Usage:** Normalized access count (log scale, 0.0 - 1.0)
- **Freshness:** Days since created, linear decay over 30 days (0.0 - 1.0)

### Example Calculation

```
Memory created 5 days ago, accessed 3 times, importance = 0.8:
- Recency: 1.0 - (5/30) = 0.833
- Usage: (1+3)/10 = 0.4
- Importance: 0.8
- Score: (0.8 × 0.4) + (0.4 × 0.3) + (0.833 × 0.3)
       = 0.32 + 0.12 + 0.25 = 0.69
```

## Layer-Specific Searches

### Recent Context (Ephemeral)

```python
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="ephemeral",  # Messages from last 24h only
    limit=5
)
```

### Learned Knowledge (Semantic)

```python
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="semantic",    # Long-term memories + procedures
    limit=10
)
```

### Project State (Episodic)

```python
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="episodic",    # Snapshots + entities
    limit=10
)
```

### Memory Evolution (Meta)

```python
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="meta",        # Relationships + history
    limit=10
)
```

## Custom Weights

Adjust importance of each score component:

```python
# Prioritize freshness (for chat context)
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="all",
    significance_weight=0.2,     # Less importance
    access_weight=0.2,           # Less usage
    recency_weight=0.6           # More freshness
    # (auto-normalized)
)

# Prioritize learned knowledge (for task solving)
result = await phase3.remember_search_ranked(
    query_embedding=[...],
    memory_type="semantic",
    significance_weight=0.6,     # More importance
    access_weight=0.3,
    recency_weight=0.1           # Less freshness
)
```

## Integration with MCP

Phase 3 is automatically available as an MCP tool:

```json
{
    "name": "nah_remember_search_ranked",
    "arguments": {
        "query_embedding": [...],
        "memory_type": "all",
        "limit": 20
    }
}
```

## Testing

Run the full test suite:

```bash
cd /Users/human/notahuman/infrastructure/memory/nah-remember-unified
python3 test_phase3.py
```

Expected: ✅ 12/12 tests passed

## Common Patterns

### Pattern 1: Current Context

```python
# Get recent high-priority memories
result = await phase3.remember_search_ranked(
    query_embedding=embed("user question here"),
    memory_type="ephemeral",
    limit=5,
    recency_weight=0.8  # Prioritize fresh
)
```

### Pattern 2: Knowledge Base

```python
# Search your learned knowledge
result = await phase3.remember_search_ranked(
    query_embedding=embed("how do I do X?"),
    memory_type="semantic",
    limit=20,
    significance_weight=0.7  # Prioritize important memories
)
```

### Pattern 3: Full Search

```python
# Comprehensive search across everything
result = await phase3.remember_search_ranked(
    query_embedding=embed("search query"),
    memory_type="all",
    limit=50  # Get all relevant results
)

# Results automatically ranked by composite score
```

### Pattern 4: Debugging

```python
# See what's in memory and how it ranks
result = await phase3.remember_search_ranked(
    query_embedding=[0]*1536,  # Neutral query
    memory_type="all",
    limit=100
)

# Shows age, importance, and usage of all memories
```

## Troubleshooting

**"success": false**
- Check DATABASE_URL is set and PostgreSQL is running
- Verify tables exist: `SELECT COUNT(*) FROM nah_remember.messages`

**No results**
- Verify data exists in tables
- Check EMBEDDING_DIM matches query vector length (default 1536)

**Low scores**
- Normal! Scores are 0.0-1.0. Check relative ranking, not absolute values.

**Unexpected ranking**
- Review weights: are they what you intended?
- Check recency_score: memory might be old (< 30 days)

## What's Next?

- **Use Phase 3 in production** - it's stable and fully tested
- **Monitor query latency** - typical: 200-500ms for all layers
- **Tune weights per use case** - different tasks benefit from different priorities
- **Future:** Hybrid ranking, semantic clustering, feedback loops

---

**Phase 3 is complete and ready to use. Happy searching! 🔍**
