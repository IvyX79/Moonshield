"""
Sci-RAG Pipeline — Main Entry Point.

FastAPI server providing the REST API for the RAG pipeline,
document management, and AI access layer.

Usage:
    uvicorn src.main:app --host 0.0.0.0 --port 8000

Or directly:
    python src/main.py
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.document_manager import DocumentManager
from src.citation_engine import CitationEngine
from src.rag_pipeline import RAGPipeline
from src.ai_access_layer import AIAccessLayer, PermissionLevel

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sci-rag")


# ── Application State ──

class AppState:
    """Holds application-wide singletons."""
    def __init__(self):
        self.doc_manager = DocumentManager(storage_dir="data/documents")
        self.citation_engine = CitationEngine()
        self.rag_pipeline = RAGPipeline(
            document_manager=self.doc_manager,
            citation_engine=self.citation_engine,
        )
        self.access_layer = AIAccessLayer(
            document_manager=self.doc_manager,
            rag_pipeline=self.rag_pipeline,
        )


state = AppState()


# ── Lifespan ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: initialize on startup, cleanup on shutdown."""
    logger.info("Sci-RAG Pipeline starting up...")

    # Initialize the RAG pipeline (build index)
    success = state.rag_pipeline.initialize()
    if success:
        logger.info("Pipeline initialized — ready to serve queries")
    else:
        logger.warning("Pipeline initialization incomplete — some features may be unavailable")

    # Create a default access token for the primary AI agent
    default_token = state.access_layer.create_token(
        permission=PermissionLevel.FULL_ACCESS,
        expires_in_seconds=None,  # Never expires
        max_queries=None,  # Unlimited
    )
    logger.info(f"Default access token created: {default_token.token_id}")

    yield

    # Shutdown
    logger.info("Sci-RAG Pipeline shutting down")


# ── FastAPI Application ──

app = FastAPI(
    title="Sci-RAG Pipeline",
    description="Enhanced RAG pipeline for Scientific Research Workflows",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    citations: List[dict] = []
    confidence: float = 0.0
    source_count: int = 0


class CitationReport(BaseModel):
    total_citations: int
    unique_sources: int
    average_confidence: float
    citations: List[dict] = []


class AccessTokenResponse(BaseModel):
    token_id: str
    permission: str
    expires_at: str
    queries_limit: Optional[int]


class DocumentListItem(BaseModel):
    doc_id: str
    title: str
    source: str
    authors: List[str] = []
    cached_at: str


# ── Routes ──

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "documents_indexed": state.doc_manager.count,
        "pipeline_initialized": state.rag_pipeline.is_initialized,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG pipeline with a scientific question.

    Returns an answer with citations from all indexed documents.
    """
    result = state.rag_pipeline.query(request.question, top_k=request.top_k)

    return QueryResponse(
        answer=result.get("answer", "No answer generated"),
        citations=result.get("citations", []),
        confidence=result.get("confidence", 0.0),
        source_count=result.get("source_count", 0),
    )


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """
    Upload a document for indexing.

    Supported formats: PDF, DOCX, TXT, MD, LaTeX
    """
    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    # Ingest
    doc_title = title or file.filename
    doc = state.doc_manager.ingest_uploaded_file(str(file_path), title=doc_title)

    if not doc:
        raise HTTPException(status_code=400, detail=f"Failed to ingest file: {file.filename}")

    # Refresh pipeline index
    state.rag_pipeline.refresh_index()

    return {
        "success": True,
        "doc_id": doc.doc_id,
        "title": doc.title,
        "source": doc.source,
    }


@app.get("/documents", response_model=List[DocumentListItem])
async def list_documents(source: Optional[str] = None):
    """List all indexed documents."""
    docs = state.doc_manager.list_documents(source=source)
    return [
        DocumentListItem(
            doc_id=d["doc_id"],
            title=d["title"],
            source=d["source"],
            authors=d.get("authors", []),
            cached_at=d["cached_at"],
        )
        for d in docs
    ]


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    doc = state.doc_manager.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the index."""
    # Note: full deletion requires removing from vector store too
    # This is a simplified implementation
    raise HTTPException(status_code=501, detail="Not implemented yet")


@app.post("/references/search")
async def search_references(query: str = Form(...), max_results: int = Form(10)):
    """
    Search for academic references via Semantic Scholar and arXiv.
    """
    import asyncio

    ss_results, arxiv_results = await asyncio.gather(
        state.doc_manager.search_semantic_scholar(query, max_results=max_results),
        state.doc_manager.search_arxiv(query, max_results=max_results),
    )

    return {
        "query": query,
        "semantic_scholar": ss_results,
        "arxiv": arxiv_results,
        "total": len(ss_results) + len(arxiv_results),
    }


@app.post("/references/import")
async def import_reference(
    paper_id: str = Form(...),
    title: str = Form(...),
    abstract: str = Form(...),
    authors: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
):
    """Import a reference from Semantic Scholar into the document store."""
    import asyncio

    author_list = authors.split(",") if authors else []
    doc = await state.doc_manager.import_from_semantic_scholar(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=author_list,
        url=url,
    )

    if doc:
        state.rag_pipeline.refresh_index()
        return {"success": True, "doc_id": doc.doc_id, "title": doc.title}
    return {"success": False, "reason": "Could not import (empty abstract?)"}


@app.post("/citations/report")
async def get_citation_report():
    """Get a citation report for the current session."""
    report = state.citation_engine.get_session_report()
    return CitationReport(
        total_citations=report["total_citations"],
        unique_sources=report["unique_sources"],
        average_confidence=report["average_confidence"],
        citations=report["citations"],
    )


@app.post("/access/token")
async def create_access_token(
    permission: str = Form("read_query"),
    expires_in: Optional[int] = Form(3600),
    max_queries: Optional[int] = Form(100),
):
    """Create a new access token for AI agents."""
    perm_map = {
        "read_only": PermissionLevel.READ_ONLY,
        "read_query": PermissionLevel.READ_QUERY,
        "full_access": PermissionLevel.FULL_ACCESS,
    }
    perm = perm_map.get(permission, PermissionLevel.READ_QUERY)

    token = state.access_layer.create_token(
        permission=perm,
        expires_in_seconds=expires_in,
        max_queries=max_queries,
    )

    return {
        "token_id": token.token_id,
        "permission": token.permission.value,
        "expires_at": (
            token.expires_at if token.expires_at else "never"
        ),
        "queries_limit": token.max_queries,
    }


# ── Direct Execution ──

def main():
    """Run the server directly."""
    import uvicorn

    logger.info("Starting Sci-RAG Pipeline server...")
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
