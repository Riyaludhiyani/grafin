"""Smoke and configuration tests for Phase 1."""

import os
from pathlib import Path
from src.config import (
    APP_TITLE,
    DISCLAIMER_TEXT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_HOP_DECAY,
    DEFAULT_MIN_SCORE_THRESHOLD,
    DATA_DIR,
    MARKET_CACHE_DIR,
    CONFIDENCE_TIERS,
)


def test_app_constants():
    """Verify core application identity and disclaimer texts."""
    assert "Autonomous Financial Event Market Impact Graph" in APP_TITLE
    assert "Hypothesized transmission paths, not predictions." in DISCLAIMER_TEXT


def test_propagation_hyperparameters():
    """Verify propagation parameters conform to project requirements."""
    assert DEFAULT_MAX_DEPTH == 4
    assert DEFAULT_HOP_DECAY == 0.8
    assert DEFAULT_MIN_SCORE_THRESHOLD == 0.10
    assert "high" in CONFIDENCE_TIERS.values()
    assert "medium" in CONFIDENCE_TIERS.values()
    assert "low" in CONFIDENCE_TIERS.values()


def test_data_directories_exist():
    """Verify that required cache and data directories are created on initialization."""
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()
    assert MARKET_CACHE_DIR.exists()
    assert MARKET_CACHE_DIR.is_dir()


def test_app_file_exists():
    """Verify the Streamlit entrypoint file exists."""
    app_file = Path("src/ui/app.py")
    assert app_file.exists()
    assert app_file.stat().st_size > 0
