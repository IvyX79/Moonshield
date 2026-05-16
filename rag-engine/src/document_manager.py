"""
Sci-RAG Pipeline — Document Manager.

Unifies uploaded documents and Semantic Scholar references into a single
cohesive document store with unified indexing, search, and retrieval.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import aiohttp
import arxiv

from .config import settings

logger = logging.getLogger(__name__)


class Document:
    """A unified document — either uploaded or from Semantic Scholar."""

    def __init__(
        self,
        doc_id: str,
        title: str,
        content: str,
        source: str,  # "upload" | "semantic_scholar"
        source_url: Optional[str] = None,
        authors: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        cached_at: Optional[str] = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.source = source
        self.source_url = source_url
        self.authors = authors or []
        self.metadata = metadata or {}
        self.cached_at = cached_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "content_preview": self.content[:200] + ("..." if len(self.content) > 200 else ""),
            "source": self.source,
            "source_url": self.source_url,
            "authors": self.authors,
            "metadata": self.metadata,
            "cached_at": self.cached_at,
        }

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, title={self.title[:50]}, source={self.source})"


class DocumentManager:
    """
    Manages the lifecycle of documents from multiple sources.

    Features:
    - Ingest uploaded PDFs, DOCX, TXT files
    - Search and import from Semantic Scholar via arXiv IDs or search queries
    - Unified storage and indexing
    - Deduplication by content hash
    - Cache Semantic Scholar results to avoid redundant API calls
    """

    def __init__(self, storage_dir: str = "data/documents"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.storage_dir / "manifest.json"
        self._documents: Dict[str, Document] = {}
        self._load_manifest()

    # ── Persistence ──

    def _load_manifest(self):
        """Load document manifest from disk on startup."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    data = json.load(f)
                for d in data:
                    doc = Document(**d)
                    self._documents[doc.doc_id] = doc
                logger.info(f"Loaded {len(self._documents)} documents from manifest")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

    def _save_manifest(self):
        """Persist document manifest to disk."""
        data = [d.to_dict() for d in self._documents.values()]
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Document ID Generation ──

    @staticmethod
    def _make_doc_id(title: str, content_hash: str) -> str:
        """Generate a stable document ID from title and content hash."""
        raw = f"{title}::{content_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _content_hash(text: str) -> str:
        """Hash document content for deduplication."""
        return hashlib.md5(text.encode()).hexdigest()

    # ── Document Ingestion ──

    def add_document(self, doc: Document) -> Document:
        """Add a document, deduplicating by content hash."""
        content_hash = self._content_hash(doc.content)

        # Check for duplicates
        for existing in self._documents.values():
            if self._content_hash(existing.content) == content_hash:
                logger.info(f"Duplicate document skipped: {doc.title}")
                return existing

        # Generate stable ID if not already set
        if not doc.doc_id:
            doc.doc_id = self._make_doc_id(doc.title, content_hash)

        self._documents[doc.doc_id] = doc
        self._save_manifest()
        logger.info(f"Document added: {doc.title} [{doc.doc_id}]")
        return doc

    def add_text_document(
        self,
        title: str,
        content: str,
        source: str = "upload",
        source_url: Optional[str] = None,
        authors: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Document:
        """Add a plain-text document."""
        doc = Document(
            doc_id="",
            title=title,
            content=content,
            source=source,
            source_url=source_url,
            authors=authors,
            metadata=metadata,
        )
        return self.add_document(doc)

    def ingest_uploaded_file(self, file_path: str, title: Optional[str] = None) -> Optional[Document]:
        """
        Ingest an uploaded file (PDF, DOCX, TXT, MD, TEX).
        Returns the created Document or None on failure.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        ext = path.suffix.lower()
        if ext not in settings.allowed_extensions:
            logger.warning(f"Unsupported extension: {ext}")
            return None

        try:
            content = self._extract_text(path)
            doc_title = title or path.stem
            return self.add_text_document(
                title=doc_title,
                content=content,
                source="upload",
                source_url=str(path.absolute()),
                metadata={"file_name": path.name, "file_size": path.stat().st_size, "file_type": ext},
            )
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return None

    @staticmethod
    def _extract_text(path: Path) -> str:
        """Extract text content from a file."""
        ext = path.suffix.lower()

        if ext == ".txt":
            return path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".md":
            return path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".pdf":
            try:
                import pypdf

                reader = pypdf.PdfReader(path)
                text = "\n".join(page.extract_text() for page in reader.pages)
                return text
            except ImportError:
                logger.warning("pypdf not installed — using basic extraction")
                return path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".docx":
            try:
                from docx import Document as DocxDocument

                doc = DocxDocument(path)
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                logger.warning("python-docx not installed — falling back")
                return path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".tex":
            return path.read_text(encoding="utf-8", errors="replace")

        return path.read_text(encoding="utf-8", errors="replace")

    # ── Semantic Scholar Integration ──

    async def search_semantic_scholar(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Semantic Scholar and return paper results."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,externalIds,abstract,url,citationCount",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{settings.ss_api_base}/paper/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Semantic Scholar API returned {resp.status}")
                        return []

                    data = await resp.json()
                    papers = data.get("data", [])
                    return [
                        {
                            "paper_id": p.get("paperId"),
                            "title": p.get("title", "Untitled"),
                            "authors": [a.get("name") for a in p.get("authors", [])],
                            "year": p.get("year"),
                            "abstract": p.get("abstract", ""),
                            "url": p.get("url"),
                            "citation_count": p.get("citationCount", 0),
                            "external_ids": p.get("externalIds", {}),
                        }
                        for p in papers
                    ]
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    async def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search arXiv and return paper results."""
        try:
            search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            results = []
            for paper in search.results():
                results.append({
                    "paper_id": paper.entry_id,
                    "title": paper.title,
                    "authors": [str(a) for a in paper.authors],
                    "year": paper.published.year,
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": str(paper.pdf_url),
                })
            return results
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    async def import_from_semantic_scholar(
        self, paper_id: str, title: str, abstract: str, authors: Optional[List[str]] = None, url: Optional[str] = None
    ) -> Optional[Document]:
        """Import a Semantic Scholar paper as a document."""
        if not abstract:
            logger.warning(f"No abstract available for {title}")
            return None

        return self.add_text_document(
            title=title,
            content=abstract,
            source="semantic_scholar",
            source_url=url or f"https://api.semanticscholar.org/v1/paper/{paper_id}",
            authors=authors or [],
            metadata={"paper_id": paper_id, "imported_via": "semantic_scholar"},
        )

    # ── Query ──

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def list_documents(self, source: Optional[str] = None) -> List[Dict]:
        """List all documents, optionally filtered by source."""
        docs = self._documents.values()
        if source:
            docs = [d for d in docs if d.source == source]
        return [d.to_dict() for d in sorted(docs, key=lambda d: d.cached_at, reverse=True)]

    def search_documents(self, query: str) -> List[Document]:
        """Simple keyword search across documents (basic implementation)."""
        query_lower = query.lower()
        results = []
        for doc in self._documents.values():
            if query_lower in doc.title.lower() or query_lower in doc.content.lower()[:500]:
                results.append(doc)
        return results[:20]

    def all_documents(self) -> List[Document]:
        """Return all documents as a list."""
        return list(self._documents.values())

    @property
    def count(self) -> int:
        return len(self._documents)
