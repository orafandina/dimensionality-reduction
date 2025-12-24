"""Pytest configuration and shared fixtures."""

import logging

import pytest


@pytest.fixture(autouse=True)
def configure_logging() -> None:
    """Configure logging for tests."""
    logging.basicConfig(level=logging.WARNING)


@pytest.fixture
def random_seed() -> None:
    """Set random seed for reproducible tests."""
    import numpy as np
    np.random.seed(42)

