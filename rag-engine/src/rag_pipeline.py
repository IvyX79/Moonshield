"""
Sci-RAG Pipeline — Core RAG Pipeline.

Integrates Llama Index for hierarchical document retrieval and LLM-powered
synthesis, optimized for scientific and research workflows with proper
citation tracking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import settings
from .document_manager import Document, DocumentManager
from .citation_engine import Citation, CitationEngine, Source

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Core RAG pipeline using Llama Index for retrieval and generation.

    Features:
    - Hierarchical node parsing for scientific documents
    - Embedding-based retrieval with hybrid search
    - LLM-powered synthesis with citation tracking
    - Extensible: swap any component (embeddings, LLM, vector store)
    - Performance optimized with caching and async processing
    """

    def __init__(
        self,
        document_manager: DocumentManager,
        citation_engine: CitationEngine,
        llm: Optional[Any] = None,
        embed_model: Optional[Any] = None,
        index: Optional[Any] = None,
    ):
        self.doc_manager = document_manager
        self.citation_engine = citation_engine
        self._llm = llm
        self._embed_model = embed_model
        self._index = index
        self._vector_store = None
        self._initialized = False

    # ── Initialization ──

    def initialize(self) -> bool:
        """
        Initialize the pipeline — set up Llama Index, embeddings, and vector store.

        This is called once at server startup and builds the index from
        all currently managed documents.

        Returns True if initialization succeeded.
        """
        try:
            logger.info("Initializing RAG pipeline...")
            self._setup_llm()
            self._setup_embeddings()
            self._setup_vector_store()
            self._build_index()
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False

    def _setup_llm(self):
        """Configure the LLM (OpenRouter via Llama Index)."""
        try:
            from llama_index.llms.openrouter import OpenRouter

            self._llm = OpenRouter(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            logger.info(f"LLM configured: {settings.llm_model}")
        except ImportError:
            logger.warning("OpenRouter LLM not available — using OpenAI-compatible fallback")
            from llama_index.llms.openai import OpenAI

            self._llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key if hasattr(settings, "openai_api_key") else None,
            )

    def _setup_embeddings(self):
        """Configure the embedding model."""
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self._embed_model = HuggingFaceEmbedding(
                model_name=settings.embedding_model,
                max_length=512,
                embed_batch_size=settings.embedding_dimension,
            )
            logger.info(f"Embeddings configured: {settings.embedding_model}")
        except ImportError:
            logger.warning("HuggingFace embeddings not available — using OpenAI fallback")
            from llama_index.embeddings.openai import OpenAIEmbedding

            self._embed_model = OpenAIEmbedding()

    def _setup_vector_store(self):
        """Configure the vector store (ChromaDB)."""
        persist_dir = Path(self._raw_settings("vector_store", "persist_directory", "data/chroma_db"))
        persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore

            db = chromadb.PersistentClient(path=str(persist_dir))
            collection = db.get_or_create_collection(
                name=self._raw_settings("vector_store", "collection_name", "scientific_docs")
            )
            self._vector_store = ChromaVectorStore(chroma_collection=collection)
            logger.info(f"Vector store configured: {persist_dir}")
        except ImportError:
            logger.warning("ChromaDB not available — using in-memory store")
            from llama_index.vector_stores.simple import SimpleVectorStore

            self._vector_store = SimpleVectorStore()

    def _raw_settings(self, *keys, default=None):
        """Get a nested setting from the config raw dict."""
        val = settings._raw
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default

    def _build_index(self):
        """Build the Llama Index from all managed documents."""
        from llama_index.core import VectorStoreIndex, StorageContext, Document as LlamaDocument

        all_docs = self.doc_manager.all_documents()
        if not all_docs:
            logger.info("No documents to index — creating empty index")
            self._index = VectorStoreIndex.from_documents(
                [],
                embed_model=self._embed_model,
                vector_store=self._vector_store,
            )
            return

        # Convert our Document objects to Llama Index Documents
        llama_docs = []
        for doc in all_docs:
            llama_doc = LlamaDocument(
                text=doc.content,
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "source": doc.source,
                    "source_url": doc.source_url or "",
                    "authors": ", ".join(doc.authors) if doc.authors else "",
                },
            )
            llama_docs.append(llama_doc)

        logger.info(f"Building index from {len(llama_docs)} documents...")
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_documents(
            llama_docs,
            embed_model=self._embed_model,
            storage_context=storage_context,
            show_progress=True,
        )
        logger.info(f"Index built with {len(all_docs)} documents")

    # ── Query ──

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Query the RAG pipeline with a question.

        Args:
            question: The user's question.
            top_k: Number of document chunks to retrieve.

        Returns:
            Dict with 'answer', 'citations', and 'confidence'.
        """
        if not self._initialized:
            success = self.initialize()
            if not success:
                return {
                    "answer": "Pipeline failed to initialize. Please check the logs.",
                    "citations": [],
                    "confidence": 0.0,
                    "error": "initialization_failed",
                }

        if not self._index:
            return {
                "answer": "No documents indexed. Please upload documents or import references first.",
                "citations": [],
                "confidence": 0.0,
                "source_count": 0,
            }

        try:
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.postprocessor import SimilarityPostprocessor

            # Build retriever with the correct top_k
            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
            )

            # Build query engine with citation tracking
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=self._llm,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.5),
                ],
            )

            # Execute the query
            response = query_engine.query(question)

            # Extract sources for citation tracking
            sources = []
            document_lookup = {}

            for node in response.source_nodes:
                doc_id = node.metadata.get("doc_id", "unknown")
                title = node.metadata.get("title", "Untitled")
                source_type = node.metadata.get("source", "upload")

                # Build source record for citation engine
                sources.append((doc_id, node.text[:200], node.score if hasattr(node, 'score') else 0.7))

                # Build lookup for citation engine
                if doc_id not in document_lookup:
                    doc = self.doc_manager.get_document(doc_id)
                    if doc:
                        document_lookup[doc_id] = {
                            "title": doc.title,
                            "authors": doc.authors,
                            "source_url": doc.source_url,
                            "source": doc.source,
                            "metadata": doc.metadata,
                        }
                    else:
                        document_lookup[doc_id] = {
                            "title": title,
                            "authors": [],
                            "source_url": node.metadata.get("source_url"),
                            "source": source_type,
                            "metadata": {},
                        }

            # Record citations
            citation = self.citation_engine.record_claim(
                claim_text=question,
                sources=sources,
                document_lookup=document_lookup,
            )

            return {
                "answer": str(response),
                "citations": [
                    {
                        "doc_id": s.doc_id,
                        "title": s.title[:100],
                        "relevance": s.relevance_score,
                        "source_type": s.source_type,
                    }
                    for s in citation.sources
                ],
                "confidence": citation.confidence,
                "source_count": len(citation.sources),
                "source_nodes": [
                    {
                        "doc_id": n.metadata.get("doc_id", "unknown"),
                        "title": n.metadata.get("title", "Untitled"),
                        "score": n.score if hasattr(n, 'score') else None,
                        "excerpt": n.text[:300],
                    }
                    for n in response.source_nodes
                ],
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Query processing failed: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "error": str(e),
            }

    def query_stream(self, question: str):
        """
        Query with streaming response.

        Yields answer chunks as they're generated.
        """
        if not self._initialized:
            self.initialize()

        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.postprocessor import SimilarityPostprocessor

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=5,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=self._llm,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5),
            ],
            streaming=True,
        )

        response = query_engine.query(question)

        for chunk in response.response_gen:
            yield chunk

    # ── Index Maintenance ──

    def refresh_index(self) -> bool:
        """Rebuild the index from scratch. Call after adding documents."""
        try:
            logger.info("Refreshing index...")
            self._build_index()
            logger.info("Index refreshed successfully")
            return True
        except Exception as e:
            logger.error(f"Index refresh failed: {e}")
            return False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
