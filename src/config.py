"""Centralized configuration module for Autonomous Financial Event Market Impact Graph."""

import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRAPH_YAML_PATH = DATA_DIR / "graph.yaml"
SAMPLE_EVENTS_PATH = DATA_DIR / "sample_events.json"
MARKET_CACHE_DIR = DATA_DIR / "market_cache"

# Application identity & mandatory disclaimer
APP_TITLE = "Autonomous Financial Event Market Impact Graph"
APP_SUBTITLE = "Explainable Indian Financial Relationship & Event Impact Analysis"
DISCLAIMER_TEXT = "Hypothesized transmission paths, not predictions."

# Graph propagation parameters
DEFAULT_MAX_DEPTH: int = 4
DEFAULT_HOP_DECAY: float = 0.8
DEFAULT_MIN_SCORE_THRESHOLD: float = 0.10

# Valid signs and confidence tiers
RelationshipSign = Literal["positive", "negative", "ambiguous"]
ConfidenceTier = Literal["high", "medium", "low"]

CONFIDENCE_TIERS = {
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
}

# Market data configuration
DEFAULT_MARKET_HORIZON_DAYS: int = 252  # ~1 trading year
MIN_SAMPLE_SIZE_FOR_CORRELATION: int = 30
MAX_LAG_DAYS: int = 5

# LLM provider settings
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "none").lower()
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "10"))
OFFLINE_DEMO_MODE: bool = os.getenv("OFFLINE_DEMO_MODE", "true").lower() in ("true", "1", "yes")

# Ensure required directories exist
def ensure_directories_exist() -> None:
    """Create essential data directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Call on import to guarantee directories
ensure_directories_exist()
