"""
Sci-RAG Pipeline — Configuration module.
Loads settings from YAML with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional
import yaml


class Settings:
    """Application settings loaded from config file + env overrides."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self._raw = raw

    # ── LLM ──
    @property
    def llm_provider(self) -> str:
        return os.getenv("LLM_PROVIDER", self._raw.get("llm", {}).get("provider", "openrouter"))

    @property
    def llm_model(self) -> str:
        return os.getenv("LLM_MODEL", self._raw.get("llm", {}).get("model", "deepseek/deepseek-v4-flash"))

    @property
    def llm_temperature(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", str(self._raw.get("llm", {}).get("temperature", 0.1))))

    @property
    def llm_max_tokens(self) -> int:
        return int(os.getenv("LLM_MAX_TOKENS", str(self._raw.get("llm", {}).get("max_tokens", 4096))))

    # ── Embeddings ──
    @property
    def embedding_model(self) -> str:
        return self._raw.get("embeddings", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def embedding_dimension(self) -> int:
        return int(self._raw.get("embeddings", {}).get("dimension", 384))

    # ── Document Manager ──
    @property
    def upload_dir(self) -> str:
        return self._raw.get("document_manager", {}).get("upload_dir", "data/uploads")

    @property
    def allowed_extensions(self) -> list:
        return self._raw.get("document_manager", {}).get("allowed_extensions", [".pdf", ".docx", ".txt", ".md"])

    @property
    def chunk_size(self) -> int:
        return int(self._raw.get("document_manager", {}).get("chunk_size", 1024))

    @property
    def chunk_overlap(self) -> int:
        return int(self._raw.get("document_manager", {}).get("chunk_overlap", 200))

    # ── Semantic Scholar ──
    @property
    def ss_api_base(self) -> str:
        return self._raw.get("semantic_scholar", {}).get("api_base", "https://api.semanticscholar.org/v1")

    @property
    def ss_max_results(self) -> int:
        return int(self._raw.get("semantic_scholar", {}).get("max_results", 10))

    # ── Server ──
    @property
    def host(self) -> str:
        return os.getenv("HOST", self._raw.get("server", {}).get("host", "0.0.0.0"))

    @property
    def port(self) -> int:
        return int(os.getenv("PORT", str(self._raw.get("server", {}).get("port", 8000))))


settings = Settings()
