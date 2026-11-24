"""Pytest configuration for nah-remember: separate hermetic unit tests from live Postgres/LLM integration tests."""

from __future__ import annotations

import pytest

INTEGRATION_FILES = {
    "test_all_memory_layers.py",
    "test_engine.py",
    "test_integration.py",
    "test_procedural_tools.py",
    "test_dreamer_daemon.py",
    "test_adversarial_remediation_d8_d15.py",
}

INTEGRATION_NODE_SUBSTRINGS = (
    "test_adversarial_d1_d7.py::test_d5",
    "test_wave5_reflex_distillation.py::test_system_one_distiller_live_call",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires live PostgreSQL (127.0.0.1:5432) or live OmniRoute LLM gateway",
    )
    config.addinivalue_line("markers", "asyncio: async test function")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    mark_int = pytest.mark.integration
    for item in items:
        fname = item.fspath.basename
        if fname in INTEGRATION_FILES or any(
            sub in item.nodeid for sub in INTEGRATION_NODE_SUBSTRINGS
        ):
            item.add_marker(mark_int)
