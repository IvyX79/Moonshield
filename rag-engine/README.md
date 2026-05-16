# Sci-RAG Engine for AimenGPT

A production-ready Llama Index-powered RAG backend that replaces the existing Chroma-only retrieval with a full scientific document understanding pipeline.

## What it adds

- **Llama Index** — Hierarchical document parsing and retrieval (as requested in the bounty)
- **Semantic Scholar + arXiv** — Search and import external references alongside uploaded documents
- **Smart Citations** — Every answer cites sources with confidence scores
- **Document Unification** — Uploaded PDFs and external references in one searchable index
- **Secure AI Access** — Token-based document access for AI agents

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Ask a question, get answer + citations |
| `POST /documents/upload` | Upload a PDF, DOCX, TXT, or MD file |
| `GET /documents` | List all indexed documents |
| `POST /references/search` | Search Semantic Scholar + arXiv |
| `POST /references/import` | Import a paper as a document |
| `GET /health` | Service health |

## How it replaces the existing flow

**Before:** Frontend API routes → ChromaDB directly → raw chunks → LLM

**After:** Frontend API routes → **rag-engine** (Llama Index + citations) → ChromaDB → LLM
