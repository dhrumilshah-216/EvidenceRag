"""FAISS vector store module for indexing normalized embeddings and preserving metadata."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


DEFAULT_INDEX_DIM = 384


class VectorStore:
    """Stores normalized vector embeddings using FAISS IndexFlatIP and tracks aligned chunk metadata."""

    def __init__(self, dimension: int = DEFAULT_INDEX_DIM):
        """Initializes an empty FAISS IndexFlatIP index with aligned metadata.

        Args:
            dimension: Dimensionality of indexed vectors (default: 384).

        Raises:
            ValueError: If dimension is not a positive integer.
            ImportError: If the 'faiss' package is not installed.
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is not installed. Please install faiss-cpu using 'pip install faiss-cpu'."
            )

        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got: {dimension}")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Returns the total number of vectors in the FAISS index."""
        return self.index.ntotal

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Adds normalized vectors and aligned chunk metadata to the index.

        Args:
            vectors: 2D numpy array of shape (N, dimension).
            metadata: List of N metadata dictionaries aligned with vector positions.

        Raises:
            ValueError: If dimensions mismatch, counts mismatch, or inputs are malformed.
        """
        if not isinstance(metadata, list):
            raise ValueError(f"metadata must be a list of dictionaries, got: {type(metadata).__name__}")

        if not isinstance(vectors, np.ndarray):
            raise ValueError(f"vectors must be a numpy ndarray, got: {type(vectors).__name__}")

        # Handle empty input
        if len(metadata) == 0 and vectors.size == 0:
            return

        if len(metadata) == 0 or vectors.size == 0:
            count_vec = vectors.shape[0] if vectors.ndim > 0 else 0
            raise ValueError(
                f"Mismatch: cannot add {count_vec} vectors with {len(metadata)} metadata items."
            )

        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2D numpy array, got shape {vectors.shape}")

        num_vectors, dim = vectors.shape

        if dim != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: index expects {self.dimension}, got {dim}"
            )

        if num_vectors != len(metadata):
            raise ValueError(
                f"Count mismatch: got {num_vectors} vectors but {len(metadata)} metadata items."
            )

        # Validate that each metadata item is a dictionary
        for i, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise ValueError(f"Metadata at index {i} must be a dictionary, got: {type(item).__name__}")

        # FAISS requires contiguous float32 arrays
        contiguous_vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        self.index.add(contiguous_vectors)
        self.metadata.extend(metadata)

    def save(
        self,
        directory_path: Union[str, Path],
        index_filename: str = "index.faiss",
        metadata_filename: str = "metadata.json",
    ) -> None:
        """Persists the FAISS index and metadata to disk.

        Args:
            directory_path: Target directory where files will be saved.
            index_filename: Filename for the FAISS index.
            metadata_filename: Filename for metadata JSON.
        """
        dir_path = Path(directory_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_file = dir_path / index_filename
        metadata_file = dir_path / metadata_filename

        faiss.write_index(self.index, str(index_file))

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        directory_path: Union[str, Path],
        index_filename: str = "index.faiss",
        metadata_filename: str = "metadata.json",
    ) -> "VectorStore":
        """Loads a persisted FAISS index and metadata from disk.

        Args:
            directory_path: Directory containing the saved files.
            index_filename: Filename of the FAISS index.
            metadata_filename: Filename of the metadata JSON.

        Returns:
            Restored VectorStore instance.

        Raises:
            FileNotFoundError: If index or metadata file does not exist.
            ValueError: If index and metadata are misaligned.
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is not installed. Please install faiss-cpu using 'pip install faiss-cpu'."
            )

        dir_path = Path(directory_path)
        index_file = dir_path / index_filename
        metadata_file = dir_path / metadata_filename

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        index = faiss.read_index(str(index_file))

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.ntotal != len(metadata):
            raise ValueError(
                f"Alignment error: loaded FAISS index contains {index.ntotal} vectors "
                f"but metadata contains {len(metadata)} items."
            )

        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata
        return store

