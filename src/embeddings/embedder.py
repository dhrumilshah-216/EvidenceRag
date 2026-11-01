"""Embeddings generation module using sentence-transformers."""

from typing import Any, Dict, List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384


class Embedder:
    """Generates normalized vector embeddings for texts and chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Optional[Any] = None,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
    ):
        """Initializes the Embedder.

        Args:
            model_name: Model identifier on Hugging Face.
            model: Optional pre-loaded model instance (useful for testing or custom models).
            expected_dim: Expected embedding dimensionality (default: 384).
        """
        self.model_name = model_name
        self.expected_dim = expected_dim
        self._model = model

    @property
    def model(self) -> Any:
        """Lazy-loads the SentenceTransformer model if not already provided."""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it using 'pip install sentence-transformers'."
                )
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalizes vectors to unit length (L2 norm) for cosine similarity.

        Args:
            vectors: 2D numpy array of shape (N, dimension).

        Returns:
            Normalized 2D numpy array of float32.
        """
        if vectors.size == 0:
            return vectors.astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = vectors / norms
        return normalized.astype(np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized embeddings for a list of strings.

        Args:
            texts: List of input strings.

        Returns:
            2D numpy array of shape (len(texts), expected_dim) and dtype float32.

        Raises:
            ValueError: If texts input is malformed or generated embeddings have wrong dimension.
        """
        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list of strings, got {type(texts).__name__}")

        if len(texts) == 0:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"Item at index {i} is not a string: {type(text).__name__}")

        raw_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)

        # Validate dimension
        if len(embeddings.shape) != 2 or embeddings.shape[1] != self.expected_dim:
            actual_dim = embeddings.shape[1] if len(embeddings.shape) == 2 else embeddings.shape
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, got {actual_dim}"
            )

        # Normalize vectors for cosine similarity
        return self.normalize(embeddings)

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Generates normalized embeddings for structured chunk records.

        Args:
            chunks: List of chunk dictionaries containing 'text'.

        Returns:
            2D numpy array of shape (len(chunks), expected_dim) and dtype float32.

        Raises:
            ValueError: If chunks input is malformed or missing 'text' field.
        """
        if not isinstance(chunks, list):
            raise ValueError(f"chunks must be a list of dictionaries, got {type(chunks).__name__}")

        if len(chunks) == 0:
            return np.empty((0, self.expected_dim), dtype=np.float32)

        texts: List[str] = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {i} must be a dictionary, got {type(chunk).__name__}")
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {i} is missing required 'text' field.")
            texts.append(str(chunk["text"]))

        return self.embed_texts(texts)


def load_embedder(model_name: str = DEFAULT_MODEL_NAME) -> Embedder:
    """Convenience function to instantiate and return an Embedder.

    Args:
        model_name: Model name identifier.

    Returns:
        Configured Embedder instance.
    """
    return Embedder(model_name=model_name)

