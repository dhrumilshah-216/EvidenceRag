"""Text chunking module for splitting document text into token-based chunks."""

from typing import Any, Dict, List, Tuple


def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Loads and returns the Hugging Face AutoTokenizer for the specified model.

    Args:
        model_name: Hugging Face model identifier.

    Returns:
        Loaded AutoTokenizer instance.

    Raises:
        ImportError: If the 'transformers' package is not installed.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required to load the tokenizer. "
            "Please install it using 'pip install transformers'."
        ) from exc

    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name)


def chunk_document(
    pages: List[Dict[str, Any]],
    tokenizer: Any,
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[Dict[str, Any]]:
    """Splits structured page records into token-based chunks with metadata.

    Args:
        pages: List of structured page records from Phase 1, each containing
            'document_name', 'page_number', and 'text'.
        tokenizer: An explicit tokenizer dependency with encode and decode methods.
        chunk_size: Maximum number of tokens per chunk. Must be a positive integer.
        overlap: Number of overlapping tokens between consecutive chunks.
            Must be a non-negative integer strictly less than chunk_size.

    Returns:
        List of structured chunk records containing:
        - chunk_id (str): Unique deterministic ID in format '{document_name}_chunk_{index}'.
        - document_name (str): Original document filename.
        - page_numbers (List[int]): Contributing page numbers for this chunk.
        - text (str): Decoded chunk text.

    Raises:
        TypeError: If tokenizer does not have encode and decode methods.
        ValueError: If chunk_size or overlap configuration is invalid, or if page records are malformed.
    """
    # Validate tokenizer
    if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer must provide 'encode' and 'decode' methods.")

    # Validate chunk_size
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")

    # Validate overlap
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError(f"overlap must be a non-negative integer, got: {overlap}")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    # Empty pages check
    if not pages:
        return []

    # Group pages by document_name while preserving document order
    docs_pages: Dict[str, List[Dict[str, Any]]] = {}
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(f"Each page record must be a dictionary, got: {type(page).__name__}")
        if "document_name" not in page or "page_number" not in page or "text" not in page:
            raise ValueError(
                "Page record missing required keys ('document_name', 'page_number', 'text')."
            )
        doc_name = str(page["document_name"])
        if doc_name not in docs_pages:
            docs_pages[doc_name] = []
        docs_pages[doc_name].append(page)

    all_chunks: List[Dict[str, Any]] = []
    step = chunk_size - overlap

    for doc_name, doc_records in docs_pages.items():
        # Tokenize each page and map each token to its contributing page number
        doc_tokens: List[Tuple[int, int]] = []
        for record in doc_records:
            raw_text = record.get("text", "")
            if not raw_text or not raw_text.strip():
                continue

            page_num = int(record["page_number"])

            try:
                tokens = tokenizer.encode(raw_text, add_special_tokens=False)
            except TypeError:
                tokens = tokenizer.encode(raw_text)

            for token_id in tokens:
                doc_tokens.append((token_id, page_num))

        if not doc_tokens:
            continue

        # Slide token window across document tokens
        start = 0
        chunk_idx = 0

        while start < len(doc_tokens):
            end = min(start + chunk_size, len(doc_tokens))
            slice_pairs = doc_tokens[start:end]

            token_ids = [pair[0] for pair in slice_pairs]
            contributing_pages = sorted(list({pair[1] for pair in slice_pairs}))

            try:
                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            except TypeError:
                decoded_text = tokenizer.decode(token_ids)

            decoded_text = decoded_text.strip()

            if decoded_text:
                all_chunks.append({
                    "chunk_id": f"{doc_name}_chunk_{chunk_idx}",
                    "document_name": doc_name,
                    "page_numbers": contributing_pages,
                    "text": decoded_text,
                })
                chunk_idx += 1

            if end >= len(doc_tokens):
                break

            start += step

    return all_chunks


# Functional alias
chunk_pages = chunk_document


class Chunker:
    """Configurable text chunker wrapping chunk_document."""

    def __init__(self, tokenizer: Any, chunk_size: int = 256, overlap: int = 32):
        """Initializes the Chunker with tokenizer and chunking parameters.

        Args:
            tokenizer: Tokenizer instance with encode and decode methods.
            chunk_size: Maximum tokens per chunk.
            overlap: Token overlap between consecutive chunks.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunks a list of page records into structured chunk records."""
        return chunk_document(
            pages=pages,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

