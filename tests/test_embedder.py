"""Tests for the embedder module."""

import unittest
from typing import List
import numpy as np

from src.embeddings.embedder import Embedder, EXPECTED_EMBEDDING_DIM


class MockModel:
    """Mock sentence-transformer model producing deterministic unnormalized vectors."""

    def __init__(self, dim: int = EXPECTED_EMBEDDING_DIM):
        self.dim = dim

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        # Generate non-unit vectors to explicitly test normalization
        vectors = []
        for i, text in enumerate(texts):
            # Deterministic vector with non-unit norm
            vec = np.full(self.dim, fill_value=(i + 1) * 2.0, dtype=np.float32)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class TestEmbedder(unittest.TestCase):
    """Test suite for Embedder functionality."""

    def setUp(self):
        self.mock_model = MockModel(dim=384)
        self.embedder = Embedder(model=self.mock_model)

    def test_vector_dimensions(self):
        """Test that generated embeddings have expected dimension (384)."""
        texts = ["First chunk text", "Second chunk text"]
        embeddings = self.embedder.embed_texts(texts)

        self.assertEqual(embeddings.shape, (2, 384))
        self.assertEqual(embeddings.dtype, np.float32)

    def test_normalization_for_cosine_similarity(self):
        """Test that all output vectors are normalized to unit L2 norm."""
        texts = ["Text one", "Text two", "A third longer piece of text"]
        embeddings = self.embedder.embed_texts(texts)

        for i, vec in enumerate(embeddings):
            l2_norm = np.linalg.norm(vec)
            self.assertAlmostEqual(float(l2_norm), 1.0, places=5, msg=f"Vector {i} not normalized")

    def test_embed_chunks(self):
        """Test embedding extraction from Phase 2 chunk records."""
        chunks = [
            {"chunk_id": "doc_chunk_0", "document_name": "doc.pdf", "page_numbers": [1], "text": "Chunk text 0"},
            {"chunk_id": "doc_chunk_1", "document_name": "doc.pdf", "page_numbers": [1, 2], "text": "Chunk text 1"},
        ]
        embeddings = self.embedder.embed_chunks(chunks)

        self.assertEqual(embeddings.shape, (2, 384))
        for vec in embeddings:
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=5)

    def test_empty_inputs(self):
        """Test that empty inputs return empty arrays with correct shape and dtype."""
        empty_texts_res = self.embedder.embed_texts([])
        self.assertEqual(empty_texts_res.shape, (0, 384))
        self.assertEqual(empty_texts_res.dtype, np.float32)

        empty_chunks_res = self.embedder.embed_chunks([])
        self.assertEqual(empty_chunks_res.shape, (0, 384))
        self.assertEqual(empty_chunks_res.dtype, np.float32)

    def test_dimension_mismatch_raises_error(self):
        """Test that mismatch in embedding dimension raises ValueError."""
        wrong_dim_model = MockModel(dim=128)
        bad_embedder = Embedder(model=wrong_dim_model, expected_dim=384)

        with self.assertRaises(ValueError) as ctx:
            bad_embedder.embed_texts(["sample text"])
        self.assertIn("Embedding dimension mismatch", str(ctx.exception))

    def test_invalid_input_types(self):
        """Test invalid input types raise ValueError."""
        with self.assertRaises(ValueError):
            self.embedder.embed_texts("not a list")

        with self.assertRaises(ValueError):
            self.embedder.embed_texts([123])

        with self.assertRaises(ValueError):
            self.embedder.embed_chunks("not a list")

        with self.assertRaises(ValueError):
            self.embedder.embed_chunks([{"no_text_key": 1}])


if __name__ == "__main__":
    unittest.main()

