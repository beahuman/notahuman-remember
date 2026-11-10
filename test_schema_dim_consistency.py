#!/usr/bin/env python3
"""
Regression guard: embedding-dimension consistency across nah-remember.

Background (0.1 / Foundation Reset Wave 0): the live pgvector columns are
``vector(1024)`` (mistral-embed natively returns 1024-dim) and the server writes
1024-dim vectors, but ``setup_schema.sql`` and ``nah_remember_phase3.py`` had
drifted to the legacy ``1536`` (OpenAI text-embedding-3-small). A fresh
``setup_schema.sql`` run would have built 1536-dim columns, after which every
mistral write fails, and phase3 queries that relied on the 1536 default would hit
a pgvector dimension mismatch against the 1024 columns.

This test pins the source-of-truth dimension so that drift fails fast in CI,
NO database required — it parses the files directly.

Run standalone:  python3 test_schema_dim_consistency.py
Run via pytest:  pytest test_schema_dim_consistency.py
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "setup_schema.sql"
SERVER = HERE / "server.py"
ENGINE = HERE / "engine.py"
PHASE3 = ENGINE  # Backwards compatibility alias

# The canonical embedding dimension for nah-remember. mistral-embed = 1024.
EXPECTED_DIM = 1024


def _embedding_dim_default(path: Path) -> int:
    """Extract the EMBEDDING_DIM env default from a module: os.getenv("EMBEDDING_DIM", "<N>")."""
    text = path.read_text()
    m = re.search(r'os\.getenv\(\s*["\']EMBEDDING_DIM["\']\s*,\s*["\'](\d+)["\']\s*\)', text)
    assert m, f"{path.name}: could not find EMBEDDING_DIM env default"
    return int(m.group(1))


def test_schema_vector_dims_all_match_expected():
    """Every vector(N) declaration in setup_schema.sql must equal EXPECTED_DIM."""
    text = SCHEMA.read_text()
    dims = [int(n) for n in re.findall(r"vector\((\d+)\)", text)]
    assert dims, "setup_schema.sql declares no vector(N) columns — parse error?"
    bad = sorted({d for d in dims if d != EXPECTED_DIM})
    assert not bad, (
        f"setup_schema.sql has vector dimensions {bad} != expected {EXPECTED_DIM}. "
        f"All {len(dims)} vector() declarations must match the live mistral-embed dim. "
        f"Found: {dims}"
    )


def test_server_embedding_dim_default_matches_expected():
    assert _embedding_dim_default(SERVER) == EXPECTED_DIM


def test_phase3_embedding_dim_default_matches_expected():
    assert _embedding_dim_default(PHASE3) == EXPECTED_DIM


def test_server_and_phase3_agree():
    """server.py and the phase3 module must resolve to the same default dimension."""
    assert _embedding_dim_default(SERVER) == _embedding_dim_default(PHASE3)


def test_no_legacy_1536_in_schema_or_code():
    """Guard against the legacy 1536 reappearing in the schema DDL or module code."""
    for path in (SCHEMA, PHASE3):
        offenders = [
            line.strip()
            for line in path.read_text().splitlines()
            # allow the explanatory comment that names the historical 1536/1024 drift
            if "1536" in line and "drift" not in line.lower()
        ]
        assert not offenders, f"{path.name}: stale 1536 reference(s): {offenders}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS: {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL: {name}\n  {e}")
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
