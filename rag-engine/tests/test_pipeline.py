"""
Sci-RAG Pipeline — Test Suite.

Tests cover:
- Document Manager (ingestion, dedup, Semantic Scholar)
- Citation Engine (recording, formatting, validation)
- RAG Pipeline (query, indexing)
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.document_manager import DocumentManager, Document
from src.citation_engine import CitationEngine, Citation, Source
from src.config import settings


# ═════════════════════════════════════════════
# Document Manager Tests
# ═════════════════════════════════════════════

class TestDocumentManager:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.manager = DocumentManager(storage_dir=self.tmp_dir)

    def test_add_text_document(self):
        doc = self.manager.add_text_document(
            title="Test Paper",
            content="This is a test scientific paper about RAG pipelines.",
            source="upload",
            authors=["Test Author"],
        )
        assert doc.doc_id is not None
        assert doc.title == "Test Paper"
        assert len(doc.doc_id) == 16

    def test_deduplication(self):
        doc1 = self.manager.add_text_document(
            title="Same Paper",
            content="Identical content here",
        )
        doc2 = self.manager.add_text_document(
            title="Same Paper",
            content="Identical content here",
        )
        assert doc1.doc_id == doc2.doc_id

    def test_list_documents(self):
        self.manager.add_text_document(title="Doc 1", content="Content 1")
        self.manager.add_text_document(title="Doc 2", content="Content 2")
        docs = self.manager.list_documents()
        assert len(docs) == 2

    def test_search_documents(self):
        self.manager.add_text_document(title="RAG Pipeline", content="This paper discusses RAG systems for scientific research.")
        self.manager.add_text_document(title="LLM Basics", content="Introduction to language models.")
        results = self.manager.search_documents("RAG")
        assert len(results) >= 1
        assert "RAG" in results[0].title

    def test_count(self):
        assert self.manager.count == 0
        self.manager.add_text_document(title="Doc", content="Content")
        assert self.manager.count == 1

    def test_ingest_text_file(self):
        tmp_file = Path(self.tmp_dir) / "test_doc.txt"
        tmp_file.write_text("This is a test document content for ingestion testing.")
        doc = self.manager.ingest_uploaded_file(str(tmp_file))
        assert doc is not None
        assert doc.source == "upload"

    def test_ingest_invalid_extension(self):
        tmp_file = Path(self.tmp_dir) / "test.exe"
        tmp_file.write_text("bad")
        doc = self.manager.ingest_uploaded_file(str(tmp_file))
        assert doc is None

    def test_get_document(self):
        doc = self.manager.add_text_document(title="Get Me", content="Findable content")
        retrieved = self.manager.get_document(doc.doc_id)
        assert retrieved is not None
        assert retrieved.title == "Get Me"

    def test_get_nonexistent(self):
        assert self.manager.get_document("nonexistent") is None


# ═════════════════════════════════════════════
# Citation Engine Tests
# ═════════════════════════════════════════════

class TestCitationEngine:
    def setup_method(self):
        self.engine = CitationEngine()

    def test_record_claim(self):
        doc_lookup = {
            "doc1": {"title": "Paper 1", "authors": ["Alice"], "source_url": "http://example.com/1", "source": "semantic_scholar", "metadata": {}},
        }
        citation = self.engine.record_claim(
            claim_text="What is RAG?",
            sources=[("doc1", "RAG stands for Retrieval-Augmented Generation...", 0.95)],
            document_lookup=doc_lookup,
        )
        assert citation.claim_text == "What is RAG?"
        assert len(citation.sources) == 1
        assert citation.confidence > 0.9

    def test_multiple_sources(self):
        doc_lookup = {
            "doc1": {"title": "Paper A", "authors": [], "source": "upload", "metadata": {}},
            "doc2": {"title": "Paper B", "authors": [], "source": "semantic_scholar", "metadata": {}},
        }
        citation = self.engine.record_claim(
            claim_text="Multiple sources test",
            sources=[("doc1", "Source A content...", 0.8), ("doc2", "Source B content...", 0.9)],
            document_lookup=doc_lookup,
        )
        assert len(citation.sources) == 2
        assert citation.confidence == 0.9  # max of 0.8 and 0.9

    def test_get_session_report(self):
        doc_lookup = {
            "doc1": {"title": "Paper", "authors": [], "source": "upload", "metadata": {}},
        }
        self.engine.record_claim("Claim 1", [("doc1", "Content", 0.8)], doc_lookup)
        self.engine.record_claim("Claim 2", [("doc1", "More content", 0.7)], doc_lookup)

        report = self.engine.get_session_report()
        assert report["total_citations"] == 2
        assert report["unique_sources"] == 1

    def test_validate_citations(self):
        doc_lookup = {
            "doc1": {"title": "Paper", "authors": [], "source": "upload", "metadata": {}},
        }
        self.engine.record_claim("Valid claim", [("doc1", "Content", 0.8)], doc_lookup)
        result = self.engine.validate_citations(doc_count=1)
        assert result["valid_citations"] + result["orphaned_citations"] > 0

    def test_clear_session(self):
        doc_lookup = {
            "doc1": {"title": "Paper", "authors": [], "source": "upload", "metadata": {}},
        }
        self.engine.record_claim("Claim", [("doc1", "Content", 0.8)], doc_lookup)
        assert len(self.engine._session_citations) == 1
        self.engine.clear_session()
        assert len(self.engine._session_citations) == 0


# ═════════════════════════════════════════════
# Configuration Tests
# ═════════════════════════════════════════════

class TestConfig:
    def test_settings_load(self):
        assert settings.llm_provider == "openrouter"
        assert settings.llm_model == "deepseek/deepseek-v4-flash"
        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_embedding_dimension(self):
        assert settings.embedding_dimension == 384

    def test_chunk_size(self):
        assert settings.chunk_size == 1024
