"""Tests for the retrieval engine module."""

import unittest
from typing import List
import numpy as np

from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever, retrieve_chunks
from src.retrieval.vector_store import VectorStore


class ControlledMockModel:
    """Mock embedding model that maps predefined text keys to known vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vector_map = {}

    def set_vector(self, text: str, vector: np.ndarray):
        self.vector_map[text] = vector.astype(np.float32)

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            if text in self.vector_map:
                vectors.append(self.vector_map[text])
            else:
                # Default orthogonal vector
                vec = np.zeros(self.dim, dtype=np.float32)
                vec[0] = 1.0
                vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class TestRetriever(unittest.TestCase):
    """Test suite for Retriever functionality."""

    def setUp(self):
        self.dim = 384
        self.mock_model = ControlledMockModel(dim=self.dim)
        self.embedder = Embedder(model=self.mock_model, expected_dim=self.dim)
        self.vector_store = VectorStore(dimension=self.dim)

        # Setup 3 known orthogonal/angled vectors
        # v0: [1, 0, 0, ...]
        v0 = np.zeros(self.dim, dtype=np.float32)
        v0[0] = 1.0

        # v1: [0, 1, 0, ...] (orthogonal to v0)
        v1 = np.zeros(self.dim, dtype=np.float32)
        v1[1] = 1.0

        # v2: normalized [1, 1, 0, ...] (cosine ~0.7071 with v0)
        v2 = np.zeros(self.dim, dtype=np.float32)
        v2[0] = 1.0
        v2[1] = 1.0
        v2 = v2 / np.linalg.norm(v2)

        self.mock_model.set_vector("chunk zero text", v0)
        self.mock_model.set_vector("chunk one text", v1)
        self.mock_model.set_vector("chunk two text", v2)

        # Query vector aligns with v0
        self.mock_model.set_vector("query about zero", v0)

        chunks_metadata = [
            {"chunk_id": "doc_chunk_0", "document_name": "doc.pdf", "page_numbers": [1], "text": "chunk zero text"},
            {"chunk_id": "doc_chunk_1", "document_name": "doc.pdf", "page_numbers": [2], "text": "chunk one text"},
            {"chunk_id": "doc_chunk_2", "document_name": "doc.pdf", "page_numbers": [1, 2], "text": "chunk two text"},
        ]

        vectors = np.array([v0, v1, v2], dtype=np.float32)
        self.vector_store.add(vectors, chunks_metadata)
        self.retriever = Retriever(vector_store=self.vector_store, embedder=self.embedder)

    def test_ranking_order(self):
        """Test that returned chunks are ranked by similarity in descending order."""
        results = self.retriever.retrieve(query="query about zero", top_k=3)

        self.assertEqual(len(results), 3)
        # Expected ranking: v0 (1.0), v2 (~0.707), v1 (0.0)
        self.assertEqual(results[0]["chunk_id"], "doc_chunk_0")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)

        self.assertEqual(results[1]["chunk_id"], "doc_chunk_2")
        self.assertAlmostEqual(results[1]["score"], 0.7071, places=3)

        self.assertEqual(results[2]["chunk_id"], "doc_chunk_1")
        self.assertAlmostEqual(results[2]["score"], 0.0, places=4)

    def test_metadata_preservation(self):
        """Test that all original metadata fields are preserved in retrieved results."""
        results = self.retriever.retrieve(query="query about zero", top_k=1)

        self.assertEqual(len(results), 1)
        top_result = results[0]
        self.assertEqual(top_result["chunk_id"], "doc_chunk_0")
        self.assertEqual(top_result["document_name"], "doc.pdf")
        self.assertEqual(top_result["page_numbers"], [1])
        self.assertEqual(top_result["text"], "chunk zero text")
        self.assertIn("score", top_result)

    def test_top_k_parameter(self):
        """Test that top_k restricts the number of returned results."""
        results_top1 = self.retriever.retrieve(query="query about zero", top_k=1)
        self.assertEqual(len(results_top1), 1)
        self.assertEqual(results_top1[0]["chunk_id"], "doc_chunk_0")

        results_top2 = self.retriever.retrieve(query="query about zero", top_k=2)
        self.assertEqual(len(results_top2), 2)
        self.assertEqual(results_top2[0]["chunk_id"], "doc_chunk_0")
        self.assertEqual(results_top2[1]["chunk_id"], "doc_chunk_2")

        # top_k larger than total index items returns all items
        results_top10 = self.retriever.retrieve(query="query about zero", top_k=10)
        self.assertEqual(len(results_top10), 3)

    def test_threshold_filtering(self):
        """Test that chunks with similarity below the threshold are excluded."""
        # Threshold 0.5: should include v0 (1.0) and v2 (~0.707), exclude v1 (0.0)
        results = self.retriever.retrieve(
            query="query about zero",
            top_k=3,
            similarity_threshold=0.5,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual([r["chunk_id"] for r in results], ["doc_chunk_0", "doc_chunk_2"])

        # Threshold 0.9: should include only v0 (1.0)
        results_strict = self.retriever.retrieve(
            query="query about zero",
            top_k=3,
            similarity_threshold=0.9,
        )
        self.assertEqual(len(results_strict), 1)
        self.assertEqual(results_strict[0]["chunk_id"], "doc_chunk_0")

        # Threshold 1.1: excludes everything
        results_none = self.retriever.retrieve(
            query="query about zero",
            top_k=3,
            similarity_threshold=1.1,
        )
        self.assertEqual(results_none, [])

    def test_empty_query_returns_empty_list(self):
        """Test that empty or whitespace-only queries return an empty list."""
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   \t\n  "), [])

    def test_empty_index_returns_empty_list(self):
        """Test that retrieving against an empty vector store returns an empty list."""
        empty_store = VectorStore(dimension=self.dim)
        retriever_empty = Retriever(vector_store=empty_store, embedder=self.embedder)
        results = retriever_empty.retrieve("any query", top_k=5)
        self.assertEqual(results, [])

    def test_convenience_function(self):
        """Test retrieve_chunks functional wrapper."""
        results = retrieve_chunks(
            query="query about zero",
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=2,
            similarity_threshold=0.5,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "doc_chunk_0")

    def test_invalid_parameters_raise_error(self):
        """Test that invalid inputs raise appropriate errors."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("valid query", top_k=0)

        with self.assertRaises(ValueError):
            self.retriever.retrieve("valid query", top_k=-1)

        with self.assertRaises(ValueError):
            self.retriever.retrieve(123)  # non-string query

        with self.assertRaises(ValueError):
            self.retriever.retrieve("valid query", similarity_threshold="invalid")

        with self.assertRaises(TypeError):
            Retriever(vector_store="not a store", embedder=self.embedder)

        with self.assertRaises(TypeError):
            Retriever(vector_store=self.vector_store, embedder="not an embedder")


if __name__ == "__main__":
    unittest.main()

