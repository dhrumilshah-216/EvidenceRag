"""Tests for the FAISS vector store module."""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.retrieval.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test suite for VectorStore functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.dim = 384
        self.store = VectorStore(dimension=self.dim)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _generate_vectors(self, n: int) -> np.ndarray:
        """Helper to generate n normalized vectors of dimension self.dim."""
        vecs = np.random.randn(n, self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return (vecs / norms).astype(np.float32)

    def test_indexing_and_total_vectors(self):
        """Test adding vectors and metadata updates total count."""
        vectors = self._generate_vectors(3)
        metadata = [
            {"chunk_id": "doc_chunk_0", "document_name": "doc.pdf", "page_numbers": [1], "text": "text 0"},
            {"chunk_id": "doc_chunk_1", "document_name": "doc.pdf", "page_numbers": [1, 2], "text": "text 1"},
            {"chunk_id": "doc_chunk_2", "document_name": "doc.pdf", "page_numbers": [2], "text": "text 2"},
        ]

        self.store.add(vectors, metadata)

        self.assertEqual(self.store.total_vectors, 3)
        self.assertEqual(len(self.store.metadata), 3)

    def test_metadata_alignment(self):
        """Test that metadata items strictly align with FAISS index vector positions."""
        vectors = self._generate_vectors(4)
        metadata = [
            {"chunk_id": f"chunk_{i}", "document_name": "report.pdf", "page_numbers": [i + 1], "text": f"text {i}"}
            for i in range(4)
        ]

        self.store.add(vectors, metadata)

        for i in range(4):
            self.assertEqual(self.store.metadata[i]["chunk_id"], f"chunk_{i}")
            self.assertEqual(self.store.metadata[i]["page_numbers"], [i + 1])

    def test_persistence_save_and_load(self):
        """Test saving and loading the FAISS index and aligned metadata."""
        vectors = self._generate_vectors(5)
        metadata = [
            {"chunk_id": f"doc_chunk_{i}", "document_name": "manual.pdf", "page_numbers": [1], "text": f"step {i}"}
            for i in range(5)
        ]

        self.store.add(vectors, metadata)
        save_dir = self.temp_path / "index_storage"
        self.store.save(save_dir)

        # Verify files were created
        self.assertTrue((save_dir / "index.faiss").exists())
        self.assertTrue((save_dir / "metadata.json").exists())

        # Load into a new store
        loaded_store = VectorStore.load(save_dir)

        self.assertEqual(loaded_store.dimension, self.dim)
        self.assertEqual(loaded_store.total_vectors, 5)
        self.assertEqual(len(loaded_store.metadata), 5)
        self.assertEqual(loaded_store.metadata, metadata)

    def test_dimension_mismatch_raises_error(self):
        """Test that passing vectors of wrong dimension raises ValueError."""
        wrong_dim_vectors = np.random.randn(2, 128).astype(np.float32)
        metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]

        with self.assertRaises(ValueError) as ctx:
            self.store.add(wrong_dim_vectors, metadata)
        self.assertIn("Vector dimension mismatch", str(ctx.exception))

    def test_count_mismatch_raises_error(self):
        """Test that mismatch between vector count and metadata count raises ValueError."""
        vectors = self._generate_vectors(3)
        metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}]  # only 2 items

        with self.assertRaises(ValueError) as ctx:
            self.store.add(vectors, metadata)
        self.assertIn("Count mismatch", str(ctx.exception))

    def test_empty_inputs_handling(self):
        """Test adding empty vectors and metadata is a no-op."""
        empty_vectors = np.empty((0, self.dim), dtype=np.float32)
        self.store.add(empty_vectors, [])
        self.assertEqual(self.store.total_vectors, 0)
        self.assertEqual(len(self.store.metadata), 0)

    def test_load_misaligned_index_raises_error(self):
        """Test that loading misaligned index and metadata files raises ValueError."""
        vectors = self._generate_vectors(3)
        metadata = [{"chunk_id": "1"}, {"chunk_id": "2"}, {"chunk_id": "3"}]
        self.store.add(vectors, metadata)

        save_dir = self.temp_path / "corrupt_storage"
        self.store.save(save_dir)

        # Corrupt metadata file to have only 1 item
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump([{"chunk_id": "1"}], f)

        with self.assertRaises(ValueError) as ctx:
            VectorStore.load(save_dir)
        self.assertIn("Alignment error", str(ctx.exception))

    def test_load_missing_files_raises_error(self):
        """Test loading from non-existent directory or files raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            VectorStore.load(self.temp_path / "non_existent_dir")


if __name__ == "__main__":
    unittest.main()

