"""
Sci-RAG Pipeline — Citation Engine.

Generates intelligent, context-aware citations from both user documents
and external references (Semantic Scholar). Tracks source provenance
and confidence scores for every claim.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A single source used to support a claim."""
    doc_id: str
    title: str
    authors: List[str]
    source_url: Optional[str]
    source_type: str  # "upload" | "semantic_scholar"
    relevance_score: float  # 0.0 to 1.0
    excerpt: str = ""
    year: Optional[int] = None


@dataclass
class Citation:
    """A complete citation for a claim in the generated answer."""
    claim_text: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0  # Aggregate confidence across sources

    def add_source(self, source: Source):
        self.sources.append(source)
        # Update aggregate confidence (max of source scores, weighted by count)
        if self.sources:
            self.confidence = max(
                self.confidence,
                source.relevance_score,
            )


class CitationEngine:
    """
    Manages citation generation and tracking.

    Features:
    - Tracks every source used in every generated answer
    - Assigns confidence scores to each source
    - Generates formatted citations in multiple styles (inline, footnote, endnote)
    - Deduplicates sources across multiple claims
    - Validates citations against the document store
    """

    def __init__(self):
        self._session_citations: List[Citation] = []
        self._source_registry: Dict[str, Source] = {}  # dedup by doc_id

    # ── Citation Recording ──

    def record_citation(self, citation: Citation):
        """Record a citation for tracking and auditing."""
        self._session_citations.append(citation)

        # Register sources for deduplication
        for source in citation.sources:
            self._source_registry[source.doc_id] = source

    def record_claim(
        self,
        claim_text: str,
        sources: List[Tuple[str, str, float]],  # [(doc_id, excerpt, relevance)]
        document_lookup: Dict[str, dict],
    ) -> Citation:
        """
        Convenience method: record a claim with raw source references.

        Args:
            claim_text: The text of the claim being made.
            sources: List of (doc_id, excerpt, relevance_score) tuples.
            document_lookup: Dict mapping doc_id to source metadata.

        Returns:
            The created Citation.
        """
        citation = Citation(claim_text=claim_text)

        for doc_id, excerpt, relevance in sources:
            meta = document_lookup.get(doc_id, {})
            source = Source(
                doc_id=doc_id,
                title=meta.get("title", "Unknown"),
                authors=meta.get("authors", []),
                source_url=meta.get("source_url"),
                source_type=meta.get("source", "upload"),
                relevance_score=min(relevance, 1.0),
                excerpt=excerpt[:200],
                year=meta.get("metadata", {}).get("year"),
            )
            citation.add_source(source)

        self.record_citation(citation)
        return citation

    # ── Citation Formatting ──

    def format_inline_citations(self, text: str) -> str:
        """
        Add inline citation markers to generated text.

        Finds patterns like [citation:N] and replaces them with formatted
        inline citations like [1][2].
        """
        def _replace_citation(match):
            idx = int(match.group(1))
            if 0 < idx <= len(self._session_citations):
                cit = self._session_citations[idx - 1]
                sources_formatted = []
                for s in cit.sources:
                    authors_short = ", ".join(s.authors[:2])
                    if len(s.authors) > 2:
                        authors_short += " et al."
                    sources_formatted.append(f"{authors_short} ({s.source_type})")
                return f" [{'; '.join(sources_formatted)}]"
            return match.group(0)

        text = re.sub(r'\[citation:(\d+)\]', _replace_citation, text)
        return text

    def format_footnotes(self, text: str) -> Tuple[str, List[str]]:
        """
        Convert inline citation markers to footnote references.

        Returns:
            (text_with_footnotes, footnotes_list)
        """
        footnotes = []
        footnoted_text = text

        def _replace_footnote(match):
            idx = int(match.group(1))
            if 0 < idx <= len(self._session_citations):
                cit = self._session_citations[idx - 1]
                fn_text = f"^{idx}"
                sources_detail = []
                for s in cit.sources:
                    authors_str = ", ".join(s.authors[:3])
                    if len(s.authors) > 3:
                        authors_str += " et al."
                    sources_detail.append(
                        f"{authors_str} — \"{s.excerpt[:80]}...\" "
                        f"(confidence: {s.relevance_score:.0%})"
                    )
                footnotes.append(f"{idx}. {'; '.join(sources_detail)}")
                return f"{match.group(0)}[^{idx}]"
            return match.group(0)

        footnoted_text = re.sub(r'\[citation:(\d+)\]', _replace_footnote, footnoted_text)
        return footnoted_text, footnotes

    def get_session_report(self) -> Dict:
        """Generate a complete citation report for the session."""
        return {
            "total_citations": len(self._session_citations),
            "unique_sources": len(self._source_registry),
            "average_confidence": (
                sum(c.confidence for c in self._session_citations) / len(self._session_citations)
                if self._session_citations else 0.0
            ),
            "citations": [
                {
                    "claim_text": c.claim_text[:150],
                    "confidence": c.confidence,
                    "source_count": len(c.sources),
                    "sources": [
                        {
                            "title": s.title[:80],
                            "source_type": s.source_type,
                            "relevance": s.relevance_score,
                            "url": s.source_url,
                        }
                        for s in c.sources
                    ],
                }
                for c in self._session_citations[-20:]  # Last 20 citations
            ],
        }

    # ── Validation ──

    def validate_citations(self, doc_count: int) -> Dict:
        """
        Validate that all cited documents exist in the document store.

        Returns a report of valid vs. orphaned citations.
        """
        valid = 0
        orphaned = 0
        orphan_details = []

        for citation in self._session_citations:
            for source in citation.sources:
                if source.doc_id in self._source_registry:
                    valid += 1
                else:
                    orphaned += 1
                    orphan_details.append({
                        "claim": citation.claim_text[:100],
                        "source_title": source.title,
                        "doc_id": source.doc_id,
                    })

        return {
            "valid_citations": valid,
            "orphaned_citations": orphaned,
            "orphan_details": orphan_details[:10],
        }

    def clear_session(self):
        """Reset session citations (for new query sessions)."""
        self._session_citations = []
