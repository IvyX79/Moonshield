# Sci-RAG Pipeline — Architecture Document

## System Overview

The Sci-RAG Pipeline is a production-ready Retrieval-Augmented Generation system designed specifically for scientific and research workflows. It unifies uploaded documents and Semantic Scholar references into a single queryable knowledge base, with proper citation tracking and AI-native access patterns.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (src/main.py)                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │ /query      │  │ /documents/* │  │ /references  │  │ /access │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └────┬────┘  │
└─────────┼────────────────┼─────────────────┼───────────────┼────────┘
          │                │                 │               │
┌─────────▼────────────────▼─────────────────▼───────────────▼────────┐
│                      Orchestration Layer                             │
│  Routes requests to appropriate components, handles errors          │
└─────────────────────────────────────────────────────────────────────┘
          │                │                 │               │
┌─────────▼─────────┐ ┌──▼──────────────────▼──┐ ┌──────────▼────────┐
│  DocumentManager  │ │   RAG Pipeline Core     │ │  AIAccessLayer   │
│                   │ │   (Llama Index)         │ │                  │
│ • File ingestion  │ │ • VectorStoreIndex      │ │ • Token auth     │
│ • SS/arXiv search │ │ • RetrieverQueryEngine  │ │ • Permission     │
│ • Deduplication   │ │ • Node parsing          │ │ • Rate limiting  │
│ • Manifest store  │ │ • Similarity postproc   │ │ • Audit logging  │
└───────────────────┘ └──────────┬──────────────┘ └──────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    CitationEngine        │
                    │                         │
                    │ • Source tracking       │
                    │ • Confidence scoring    │
                    │ • Inline/footnote fmt   │
                    │ • Validation            │
                    └─────────────────────────┘
```

## Data Flow

### Query Flow
```
User/AI → POST /query {"question": "..."}
    → RAGPipeline.query()
        → VectorIndexRetriever (top_k docs)
        → LLM synthesis with source context
        → CitationEngine.record_claim()
    → Response with answer + citations
```

### Document Ingestion Flow
```
Upload → POST /documents/upload
    → File saved to data/uploads/
    → DocumentManager.ingest_uploaded_file()
        → Text extraction (PDF/DOCX/TXT/MD/TEX)
        → Deduplication by content hash
        → Manifest persistence
    → RAGPipeline.refresh_index()
        → Rebuild VectorStoreIndex
```

### Reference Import Flow
```
Search → POST /references/search?query="..."
    → DocumentManager.search_semantic_scholar()
    → DocumentManager.search_arxiv()
    → Returns structured paper list

Import → POST /references/import
    → DocumentManager.import_from_semantic_scholar()
    → Refresh index
```

## Component Details

### DocumentManager
- **Purpose**: Unified document lifecycle management
- **Storage**: JSON manifest + file system for uploaded files
- **Deduplication**: MD5 content hash
- **Sources**: Uploaded files (PDF, DOCX, TXT, MD, TEX), Semantic Scholar API, arXiv API
- **Key methods**: `add_document()`, `ingest_uploaded_file()`, `search_semantic_scholar()`, `search_arxiv()`

### RAG Pipeline (Llama Index)
- **Index**: `VectorStoreIndex` with ChromaDB persistence
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (384-dim)
- **LLM**: OpenRouter (configurable model) via `OpenRouter` LLM class
- **Retrieval**: `VectorIndexRetriever` with configurable `top_k`
- **Post-processing**: `SimilarityPostprocessor` (0.5 cutoff)
- **Query Engine**: `RetrieverQueryEngine` with synthesized responses

### CitationEngine
- **Tracking**: Every query records all sources used
- **Confidence**: Aggregate confidence from max individual source relevance
- **Formats**: Inline (text markers), footnote, session report
- **Validation**: Cross-checks citations against document store
- **Deduplication**: Registry of unique sources across session

### AIAccessLayer
- **Authentication**: Token-based (UUID v4)
- **Permission Levels**: READ_ONLY, READ_QUERY, FULL_ACCESS
- **Rate Limiting**: 30 requests per 60-second window
- **Audit**: Full activity log with timestamps, actions, status
- **Token Controls**: Expiration time, max query count, revocation

## Configuration

See `config/settings.yaml` for all configurable parameters. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | deepseek/deepseek-v4-flash | LLM for answer synthesis |
| `embeddings.model` | all-MiniLM-L6-v2 | Text embedding model |
| `vector_store.type` | chroma | Vector database backend |
| `document_manager.chunk_size` | 1024 | Document chunk size |
| `citation.min_confidence` | 0.6 | Minimum citation confidence |
| `server.port` | 8000 | API server port |

## Deployment

### Production
```bash
# Install dependencies
pip install -r requirements.txt

# Run with uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or directly
python src/main.py
```

### Test
```bash
# Run tests
pytest tests/ -v
```

## Security Considerations

1. **Access tokens** — All AI-agent interactions require tokens with explicit permission levels
2. **Rate limiting** — Prevents abuse of the query endpoint
3. **Audit logging** — All access is logged for review
4. **File validation** — Only allowed extensions are processed; max file size enforced
5. **CORS** — Configured permissive by default; restrict in production

## Extensibility

The pipeline is designed for component swapping:

| Component | Default | Alternatives |
|-----------|---------|--------------|
| LLM | OpenRouter | OpenAI, Anthropic, local (Ollama) |
| Embeddings | HuggingFace MiniLM | OpenAI Embeddings, Cohere |
| Vector Store | ChromaDB | Pinecone, Weaviate, Qdrant, Simple |
| Document Store | JSON manifest | SQLite, PostgreSQL, S3 |
