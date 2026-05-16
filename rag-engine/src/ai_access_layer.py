"""
Sci-RAG Pipeline — AI Access Layer.

Provides secure, authenticated pathways for AI agents to interact with
user documents. Implements rate limiting, permission scoping, and
activity auditing.

This is the bridge that allows the AI to access documents directly
while maintaining security boundaries.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from .config import settings

logger = logging.getLogger(__name__)


class PermissionLevel(Enum):
    """Access permission levels for AI-document interactions."""
    READ_ONLY = "read_only"
    READ_QUERY = "read_query"       # Can read + query
    FULL_ACCESS = "full_access"     # Read, query, upload, manage


@dataclass
class AccessToken:
    """An access token for AI-document interaction."""
    token_id: str
    permission: PermissionLevel
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    max_queries: Optional[int] = None
    query_count: int = 0
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at and time.time() > self.expires_at:
            return True
        if self.max_queries and self.query_count >= self.max_queries:
            return True
        return False

    def use(self) -> bool:
        """Mark one query use. Returns True if still valid."""
        self.query_count += 1
        if self.is_expired:
            self.is_active = False
            return False
        return True


@dataclass
class AccessLogEntry:
    """A single access log entry for auditing."""
    timestamp: str
    agent_id: str
    action: str
    document_id: Optional[str]
    status: str  # "granted" | "denied" | "error"
    details: str = ""


class AIAccessLayer:
    """
    Manages AI agent access to the document system.

    Features:
    - Token-based authentication for AI agents
    - Permission scoping (read-only, query, full access)
    - Rate limiting per token
    - Full activity audit log
    - Automatic token expiration
    """

    def __init__(self, document_manager=None, rag_pipeline=None):
        self._doc_manager = document_manager
        self._rag_pipeline = rag_pipeline
        self._tokens: Dict[str, AccessToken] = {}
        self._audit_log: List[AccessLogEntry] = []

        # Rate limiting
        self._rate_limit_window = 60  # seconds
        self._rate_limit_max = 30     # max requests per window
        self._request_timestamps: List[float] = []

    # ── Token Management ──

    def create_token(
        self,
        permission: PermissionLevel = PermissionLevel.READ_QUERY,
        expires_in_seconds: Optional[int] = 3600,
        max_queries: Optional[int] = 100,
    ) -> AccessToken:
        """Create a new access token for an AI agent."""
        import uuid

        token = AccessToken(
            token_id=uuid.uuid4().hex[:16],
            permission=permission,
            expires_at=time.time() + expires_in_seconds if expires_in_seconds else None,
            max_queries=max_queries,
        )
        self._tokens[token.token_id] = token
        logger.info(f"Access token created: {token.token_id[:8]}... ({permission.value})")
        return token

    def revoke_token(self, token_id: str) -> bool:
        """Revoke an access token."""
        if token_id in self._tokens:
            self._tokens[token_id].is_active = False
            logger.info(f"Token revoked: {token_id[:8]}...")
            return True
        return False

    def validate_token(self, token_id: str) -> Optional[AccessToken]:
        """Validate and return a token, or None if invalid."""
        token = self._tokens.get(token_id)
        if not token:
            return None
        if not token.is_active or token.is_expired:
            return None
        return token

    # ── Access Control ──

    def _check_rate_limit(self) -> bool:
        """Check if the current request is within rate limits."""
        now = time.time()
        # Clean old timestamps
        self._request_timestamps = [
            t for t in self._request_timestamps
            if now - t < self._rate_limit_window
        ]
        if len(self._request_timestamps) >= self._rate_limit_max:
            return False
        self._request_timestamps.append(now)
        return True

    def _log_access(
        self,
        agent_id: str,
        action: str,
        document_id: Optional[str],
        status: str,
        details: str = "",
    ):
        """Log an access event."""
        entry = AccessLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            agent_id=agent_id,
            action=action,
            document_id=document_id,
            status=status,
            details=details,
        )
        self._audit_log.append(entry)
        # Keep log manageable
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]

    # ── Document Actions ──

    def query_documents(self, token_id: str, question: str) -> Dict:
        """
        Query documents through the RAG pipeline with access control.

        Args:
            token_id: Valid access token.
            question: The query question.

        Returns:
            Query result or error dict.
        """
        # Validate token
        token = self.validate_token(token_id)
        if not token:
            self._log_access(token_id, "query", None, "denied", "Invalid or expired token")
            return {"error": "Access denied: invalid or expired token"}

        # Check permission
        if token.permission == PermissionLevel.READ_ONLY:
            self._log_access(token_id, "query", None, "denied", "Insufficient permissions")
            return {"error": "Access denied: read-only tokens cannot query"}

        # Rate limit
        if not self._check_rate_limit():
            self._log_access(token_id, "query", None, "denied", "Rate limit exceeded")
            return {"error": "Rate limit exceeded. Try again later."}

        # Mark token usage
        if not token.use():
            self._log_access(token_id, "query", None, "denied", "Token exhausted")
            return {"error": "Token usage exhausted"}

        # Execute query
        try:
            if self._rag_pipeline:
                result = self._rag_pipeline.query(question)
                status = "granted" if "error" not in result else "error"
                self._log_access(
                    token_id, "query", None, status,
                    f"Queried: {question[:100]} → {len(result.get('citations', []))} sources"
                )
                return result
            else:
                return {"error": "RAG pipeline not configured"}
        except Exception as e:
            self._log_access(token_id, "query", None, "error", str(e))
            return {"error": f"Query failed: {str(e)}"}

    def list_documents(self, token_id: str, source: Optional[str] = None) -> Dict:
        """List available documents (read-only action)."""
        token = self.validate_token(token_id)
        if not token:
            return {"error": "Access denied: invalid or expired token"}

        if not self._check_rate_limit():
            return {"error": "Rate limit exceeded"}

        try:
            docs = self._doc_manager.list_documents(source) if self._doc_manager else []
            self._log_access(token_id, "list", None, "granted")
            return {"documents": docs, "count": len(docs)}
        except Exception as e:
            return {"error": str(e)}

    def get_document(self, token_id: str, doc_id: str) -> Dict:
        """Get a specific document (read-only action)."""
        token = self.validate_token(token_id)
        if not token:
            return {"error": "Access denied: invalid or expired token"}

        try:
            doc = self._doc_manager.get_document(doc_id) if self._doc_manager else None
            if doc:
                self._log_access(token_id, "read", doc_id, "granted")
                return doc.to_dict()
            else:
                self._log_access(token_id, "read", doc_id, "denied", "Document not found")
                return {"error": "Document not found"}
        except Exception as e:
            return {"error": str(e)}

    def upload_document(self, token_id: str, title: str, content: str, **kwargs) -> Dict:
        """Upload a document (requires full_access permission)."""
        token = self.validate_token(token_id)
        if not token:
            return {"error": "Access denied: invalid or expired token"}

        if token.permission != PermissionLevel.FULL_ACCESS:
            self._log_access(token_id, "upload", None, "denied", "Insufficient permissions")
            return {"error": "Access denied: only full_access tokens can upload"}

        try:
            doc = self._doc_manager.add_text_document(title=title, content=content, **kwargs) if self._doc_manager else None
            if doc:
                self._log_access(token_id, "upload", doc.doc_id, "granted")
                return {"success": True, "doc_id": doc.doc_id, "title": doc.title}
            return {"error": "Failed to add document"}
        except Exception as e:
            return {"error": str(e)}

    # ── Audit ──

    def get_audit_log(self, limit: int = 50) -> List[Dict]:
        """Get the recent access audit log."""
        return [
            {
                "timestamp": e.timestamp,
                "agent": e.agent_id[:8] + "...",
                "action": e.action,
                "status": e.status,
                "details": e.details,
            }
            for e in self._audit_log[-limit:]
        ]

    def get_token_status(self, token_id: str) -> Optional[Dict]:
        """Get the status of a specific token."""
        token = self._tokens.get(token_id)
        if not token:
            return None
        return {
            "token_id": token.token_id[:8] + "...",
            "permission": token.permission.value,
            "created_at": datetime.fromtimestamp(token.created_at).isoformat(),
            "expires_at": datetime.fromtimestamp(token.expires_at).isoformat() if token.expires_at else "never",
            "queries_used": token.query_count,
            "queries_limit": token.max_queries,
            "is_active": token.is_active,
            "is_expired": token.is_expired,
        }
