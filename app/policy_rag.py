"""Local FAISS vector store with persistent disk index for Enterprise Policy RAG.

Zero cloud vector DB cost ($0 Free-Tier Edition), abstracted cleanly behind
a PolicyRetriever interface mirroring Vertex AI Vector Search.
Embeds company_policies.json once on startup via gemini-embedding-001 and caches
the index to disk so API quota is never wasted re-embedding static policies.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np

from app.config import settings
from app.gemini_client import gemini_client
from app.schemas import CompanyPolicy, RetrievedPolicyMatch

logger = logging.getLogger("enterprise_contract_ai.policy_rag")


class PolicyRetriever(abc.ABC):
    """Abstract interface mirroring Vertex AI Vector Search / Feature Store."""

    @abc.abstractmethod
    async def initialize(self, force_reindex: bool = False) -> None:
        """Initializes or loads the persistent vector index."""
        pass

    @abc.abstractmethod
    async def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[RetrievedPolicyMatch]:
        """Retrieves top-k relevant enterprise policies for a given query text."""
        pass

    @abc.abstractmethod
    def get_all_policies(self) -> List[CompanyPolicy]:
        """Returns all loaded corporate policies."""
        pass


class FAISSPolicyRetriever(PolicyRetriever):
    """Production local FAISS implementation with persistent disk index.

    Uses IndexFlatIP (Inner Product over normalized unit vectors) to provide
    exact cosine similarity matching at microsecond latencies.
    """

    def __init__(
        self,
        policies_path: Path = settings.POLICIES_PATH,
        index_dir: Path = settings.FAISS_INDEX_DIR,
        embedding_dim: int = settings.embedding_dimension,
    ) -> None:
        self.policies_path = policies_path
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.index_file = self.index_dir / "policies.index"
        self.metadata_file = self.index_dir / "policies_meta.json"
        self.hash_file = self.index_dir / "policies.hash"

        self.policies: List[CompanyPolicy] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self._is_ready: bool = False

    def _compute_policies_hash(self) -> str:
        """Computes SHA-256 hash of company_policies.json to detect content updates."""
        if not self.policies_path.exists():
            return ""
        content = self.policies_path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _load_policies_from_file(self) -> List[CompanyPolicy]:
        """Loads and parses policies from local JSON file."""
        if not self.policies_path.exists():
            raise FileNotFoundError(f"Policies file not found at {self.policies_path}")
        raw_data = json.loads(self.policies_path.read_text(encoding="utf-8"))
        return [CompanyPolicy.model_validate(item) for item in raw_data]

    async def initialize(self, force_reindex: bool = False) -> None:
        """Loads persistent FAISS index from disk or embeds policies once if missing/stale."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.policies = self._load_policies_from_file()
        current_hash = self._compute_policies_hash()

        # Check if saved index exists and is up to date
        index_is_fresh = (
            not force_reindex
            and self.index_file.exists()
            and self.metadata_file.exists()
            and self.hash_file.exists()
            and self.hash_file.read_text().strip() == current_hash
        )

        if index_is_fresh:
            try:
                logger.info(f"Loading cached FAISS index from disk ({self.index_file})...")
                self.index = faiss.read_index(str(self.index_file))
                self._is_ready = True
                logger.info(f"Successfully loaded persistent FAISS index ({self.index.ntotal} vectors). Zero API quota consumed!")
                return
            except Exception as e:
                logger.warning(f"Failed to read cached index ({e}), re-indexing policies...")

        # If index missing or invalid, generate embeddings once
        await self._build_and_persist_index(current_hash)

    async def _build_and_persist_index(self, current_hash: str) -> None:
        """Builds FAISS index using gemini-embedding-001 and saves to disk."""
        logger.info(f"Embedding {len(self.policies)} enterprise policies via {settings.model_embedding}...")

        # Formulate rich descriptive documents for dense semantic embedding
        policy_documents = [
            f"Policy {p.policy_id} [{p.category}] - {p.title}: {p.rule_summary} "
            f"Mandatory requirement: {p.minimum_requirement}. Severity: {p.severity_if_violated}. "
            f"Guidance: {p.redline_guidance}"
            for p in self.policies
        ]

        # Generate embeddings
        embeddings = await gemini_client.embed_texts(policy_documents, model=settings.model_embedding)
        vectors = np.array(embeddings, dtype=np.float32)

        # Ensure correct dimension
        if vectors.shape[1] != self.embedding_dim:
            self.embedding_dim = vectors.shape[1]

        # L2-normalize vectors for exact cosine similarity with Inner Product index
        faiss.normalize_L2(vectors)

        # Create IndexFlatIP
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)

        # Persist index, metadata, and hash to disk
        faiss.write_index(self.index, str(self.index_file))

        metadata = [p.model_dump() for p in self.policies]
        self.metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.hash_file.write_text(current_hash, encoding="utf-8")

        self._is_ready = True
        logger.info(f"Successfully built and cached FAISS index to {self.index_file} ({self.index.ntotal} vectors).")

    async def retrieve(self, query_text: str, top_k: Optional[int] = None) -> List[RetrievedPolicyMatch]:
        """Retrieves top_k policies most relevant to the provided text query."""
        if not self._is_ready or self.index is None:
            await self.initialize()

        top_k = min(top_k or settings.rag_top_k, len(self.policies))

        # Generate query vector
        query_embeddings = await gemini_client.embed_texts([query_text], model=settings.model_embedding)
        q_vector = np.array(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(q_vector)

        # Perform inner-product similarity search
        scores, indices = self.index.search(q_vector, top_k)  # type: ignore

        results: List[RetrievedPolicyMatch] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.policies):
                # Clamp score to [0.0, 1.0]
                clamped_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score)))
                results.append(
                    RetrievedPolicyMatch(
                        policy=self.policies[idx],
                        relevance_score=round(clamped_score, 4),
                    )
                )

        return results

    def get_all_policies(self) -> List[CompanyPolicy]:
        """Returns all loaded corporate policies."""
        if not self.policies and self.policies_path.exists():
            self.policies = self._load_policies_from_file()
        return self.policies


# Global policy retriever singleton
policy_retriever = FAISSPolicyRetriever()
