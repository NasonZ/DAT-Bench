"""Pytest configuration and fixtures."""

import pytest


def pytest_addoption(parser):
    """Add command line options for testing."""
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Specific Ollama model to test (e.g., mistral:7b)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "ollama: mark test as Ollama-specific test"
    )