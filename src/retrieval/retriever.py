"""Retrieval engine module for ranking and filtering document chunks using FAISS."""

from typing import Any, Dict, List, Optional
import numpy as np

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore


class Retriever:
    """Retrieves and ranks relevant document chunks for natural-language queries."""

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        """Initializes the Retriever.

        Args:
            vector_store: Initialized VectorStore instance containing indexed vectors and metadata.
            embedder: Initialized Embedder instance for generating query embeddings.

        Raises:
            TypeError: If vector_store or embedder is invalid.
        """
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must be a VectorStore instance, got: {type(vector_store).__name__}"
            )
        if not isinstance(embedder, Embedder):
            raise TypeError(
                f"embedder must be an Embedder instance, got: {type(embedder).__name__}"
            )

        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves and ranks chunks matching a query.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of ranked results to return. Must be a positive integer.
            similarity_threshold: Optional minimum cosine similarity score.
                Results with scores strictly less than this threshold are excluded.

        Returns:
            List of ranked chunk dictionaries containing original metadata plus 'score' (float),
            ordered from highest to lowest similarity.

        Raises:
            ValueError: If top_k is not a positive integer, query is not a string, or
                similarity_threshold is invalid.
        """
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got: {type(query).__name__}")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        if similarity_threshold is not None:
            if not isinstance(similarity_threshold, (int, float)) or isinstance(similarity_threshold, bool):
                raise ValueError(
                    f"similarity_threshold must be a float or int, got: {type(similarity_threshold).__name__}"
                )

        # Handle empty queries or empty indexes
        if not query.strip() or self.vector_store.total_vectors == 0:
            return []

        # Generate normalized query embedding
        query_vector = self.embedder.embed_texts([query])

        effective_k = min(top_k, self.vector_store.total_vectors)
        if effective_k == 0:
            return []

        # Search FAISS IndexFlatIP (inner products equal cosine similarity for normalized vectors)
        distances, indices = self.vector_store.index.search(query_vector, effective_k)

        results: List[Dict[str, Any]] = []

        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.vector_store.metadata):
                continue

            similarity_score = float(score)

            if similarity_threshold is not None and similarity_score < similarity_threshold:
                continue

            # Preserve original chunk metadata and append similarity score
            chunk_record = dict(self.vector_store.metadata[idx])
            chunk_record["score"] = similarity_score
            results.append(chunk_record)

        return results


def retrieve_chunks(
    query: str,
    vector_store: VectorStore,
    embedder: Embedder,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Functional convenience wrapper around Retriever.retrieve."""
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    return retriever.retrieve(query=query, top_k=top_k, similarity_threshold=similarity_threshold)

