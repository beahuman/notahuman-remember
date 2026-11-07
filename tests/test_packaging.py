"""Tests for packaging, versioning, and top-level exports."""
import pytest
import nah_remember


def test_package_version():
    assert nah_remember.__version__ == "0.1.0"


def test_package_exports():
    expected_exports = [
        "__version__",
        "NahRememberMCPServer",
        "NahMemoryMCPServer",
        "MCPHandler",
        "NahRememberPhase3",
        "RankedResult",
        "MemoryLayer",
        "main",
        "cli",
    ]
    for symbol in expected_exports:
        assert hasattr(nah_remember, symbol), f"Missing export: {symbol}"


def test_backward_compat_server_wrapper():
    import server
    assert hasattr(server, "NahRememberMCPServer")
    assert hasattr(server, "cli")
    assert hasattr(server, "main")


def test_backward_compat_phase3_wrapper():
    import nah_remember_phase3
    assert hasattr(nah_remember_phase3, "NahRememberPhase3")
    assert hasattr(nah_remember_phase3, "RankedResult")
