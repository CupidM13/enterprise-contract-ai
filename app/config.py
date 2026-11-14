"""Application configuration and environment management.

Supports seamless switching between Google AI Studio Free Tier ($0, no credit card required)
and Google Cloud Vertex AI enterprise environment via `USE_VERTEX_AI` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Base paths
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
POLICIES_PATH = DATA_DIR / "company_policies.json"
SAMPLE_CONTRACTS_DIR = DATA_DIR / "sample_contracts"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
EVAL_DIR = PROJECT_ROOT / "eval"
EVAL_RESULTS_PATH = EVAL_DIR / "benchmark_results.json"


class Settings:
    """Enterprise Contract AI Settings and Model Routing Configuration."""

    def __init__(self) -> None:
        # Authentication & Provider Selection
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self.use_vertex_ai: bool = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")

        # Vertex AI / GCP Enterprise Configuration
        self.google_cloud_project: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT", None)
        self.google_cloud_location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        # 100% Free-Tier Models (Zero Billing / No Credit Card Required)
        # Stage 1: High-throughput, low-latency clause extraction
        self.model_extraction: str = os.getenv("MODEL_EXTRACTION", "gemini-3.5-flash-lite")
        # Stage 2: Deep Chain-of-Thought reasoning & policy risk audit
        self.model_audit: str = os.getenv("MODEL_AUDIT", "gemini-3.8-flash")
        # Stage 3: Embedding model for corporate policy RAG vector store
        self.model_embedding: str = os.getenv("MODEL_EMBEDDING", "gemini-embedding-001")

        # Rate Limiting & Quota Throttling (Optimized for Free-Tier 15 RPM / 1M TPM quotas)
        self.rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "15"))
        self.rate_limit_min_interval: float = float(os.getenv("RATE_LIMIT_MIN_INTERVAL", "4.0"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "5"))
        self.initial_backoff: float = float(os.getenv("INITIAL_BACKOFF", "2.0"))
        self.max_backoff: float = float(os.getenv("MAX_BACKOFF", "30.0"))

        # FAISS Vector Search Parameters
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
        self.embedding_dimension: int = 768  # gemini-embedding-001 default dim

        # Paths
        self.base_dir = PROJECT_ROOT
        self.data_dir = DATA_DIR
        self.policies_path = POLICIES_PATH
        self.sample_contracts_dir = SAMPLE_CONTRACTS_DIR
        self.faiss_index_dir = FAISS_INDEX_DIR
        self.eval_dir = EVAL_DIR
        self.eval_results_path = EVAL_RESULTS_PATH

        # Uppercase aliases for compatibility
        self.DATA_DIR = DATA_DIR
        self.POLICIES_PATH = POLICIES_PATH
        self.SAMPLE_CONTRACTS_DIR = SAMPLE_CONTRACTS_DIR
        self.FAISS_INDEX_DIR = FAISS_INDEX_DIR
        self.EVAL_RESULTS_PATH = EVAL_RESULTS_PATH

    @property
    def is_configured(self) -> bool:
        """Checks if either a valid Gemini API key or Vertex AI project is supplied."""
        if self.use_vertex_ai:
            return bool(self.google_cloud_project)
        return bool(self.gemini_api_key)

    @property
    def provider_mode(self) -> str:
        """Returns the active AI provider mode string for logging and UI display."""
        if self.use_vertex_ai:
            return f"Vertex AI (Project: {self.google_cloud_project or 'unspecified'}, Region: {self.google_cloud_location})"
        return "Google AI Studio Free Tier ($0 / No Credit Card Required)"

    def update_api_key(self, api_key: str) -> None:
        """Dynamically update API key at runtime (e.g. from UI)."""
        self.gemini_api_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = self.gemini_api_key

    def update_vertex_toggle(self, enabled: bool, project: Optional[str] = None) -> None:
        """Dynamically toggle Vertex AI mode."""
        self.use_vertex_ai = enabled
        os.environ["USE_VERTEX_AI"] = "true" if enabled else "false"
        if project:
            self.google_cloud_project = project
            os.environ["GOOGLE_CLOUD_PROJECT"] = project


# Global singleton settings
settings = Settings()
