"""Evaluation metrics for RAG retrieval and abstention."""

from typing import Any, Dict, List, Set, Tuple


def _extract_page_set(ref_or_chunk: Dict[str, Any]) -> Set[int]:
    """Helper to extract a set of page numbers from either page_number, page_numbers, or relevant_pages."""
    pages: Set[int] = set()
    if "page_numbers" in ref_or_chunk and isinstance(ref_or_chunk["page_numbers"], (list, tuple, set)):
        pages.update(int(p) for p in ref_or_chunk["page_numbers"] if isinstance(p, (int, str)) and str(p).isdigit())
    elif "relevant_pages" in ref_or_chunk and isinstance(ref_or_chunk["relevant_pages"], (list, tuple, set)):
        pages.update(int(p) for p in ref_or_chunk["relevant_pages"] if isinstance(p, (int, str)) and str(p).isdigit())
    elif "page_number" in ref_or_chunk and ref_or_chunk["page_number"] is not None:
        p = ref_or_chunk["page_number"]
        if isinstance(p, (int, str)) and str(p).isdigit():
            pages.add(int(p))
    return pages


def is_chunk_relevant(chunk: Dict[str, Any], reference: Dict[str, Any]) -> bool:
    """Determines whether a retrieved chunk matches a ground-truth relevant reference.

    Matching requires:
    1. Matching document filename.
    2. Page number overlap (if reference specifies pages).

    Args:
        chunk: Retrieved chunk metadata containing 'document_name' and 'page_numbers'.
        reference: Ground truth reference containing 'document_name' and page information.

    Returns:
        True if chunk matches the reference, False otherwise.
    """
    if not isinstance(chunk, dict) or not isinstance(reference, dict):
        return False

    chunk_doc = str(chunk.get("document_name", "")).strip().lower()
    ref_doc = str(
        reference.get("document_name") or reference.get("expected_document", "")
    ).strip().lower()

    if not chunk_doc or not ref_doc or chunk_doc != ref_doc:
        return False

    ref_pages = _extract_page_set(reference)
    chunk_pages = _extract_page_set(chunk)

    # If reference specifies no pages, document match is sufficient
    if not ref_pages:
        return True

    # If chunk specifies pages, there must be at least one overlapping page
    if chunk_pages:
        return bool(chunk_pages & ref_pages)

    # If chunk has no page info but document matches
    return True


def is_chunk_hit(chunk: Dict[str, Any], relevant_references: List[Dict[str, Any]]) -> bool:
    """Checks if a retrieved chunk matches at least one ground-truth reference."""
    if not relevant_references:
        return False
    return any(is_chunk_relevant(chunk, ref) for ref in relevant_references)


def compute_precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_references: List[Dict[str, Any]],
    k: int,
) -> float:
    """Computes Precision@K: proportion of top-K retrieved chunks that are relevant.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries.
        relevant_references: List of ground-truth relevant references.
        k: Cutoff rank. Must be a positive integer.

    Returns:
        Precision@K score between 0.0 and 1.0.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got: {k}")

    top_k = retrieved_chunks[:k]
    if not relevant_references:
        # If no relevant items exist (unanswerable): 1.0 if system abstained (0 retrieved), else 0.0
        return 1.0 if len(top_k) == 0 else 0.0

    if not top_k:
        return 0.0

    hits = sum(1 for chunk in top_k if is_chunk_hit(chunk, relevant_references))
    return float(hits) / float(k)


def compute_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_references: List[Dict[str, Any]],
    k: int,
) -> float:
    """Computes Recall@K: proportion of relevant targets covered by top-K retrieved chunks.

    Ground-truth targets are evaluated as (document_name, page_number) pairs when page
    numbers are specified, or document-level targets otherwise.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries.
        relevant_references: List of ground-truth relevant references.
        k: Cutoff rank. Must be a positive integer.

    Returns:
        Recall@K score between 0.0 and 1.0.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got: {k}")

    top_k = retrieved_chunks[:k]
    if not relevant_references:
        # Unanswerable query: 1.0 if system retrieved nothing, else 0.0
        return 1.0 if len(top_k) == 0 else 0.0

    if not top_k:
        return 0.0

    # Build discrete targets from relevant_references
    targets: Set[Tuple[str, int]] = set()
    doc_only_targets: Set[str] = set()

    for ref in relevant_references:
        doc = str(ref.get("document_name") or ref.get("expected_document", "")).strip().lower()
        pages = _extract_page_set(ref)
        if pages:
            for p in pages:
                targets.add((doc, p))
        elif doc:
            doc_only_targets.add(doc)

    total_targets = len(targets) + len(doc_only_targets)
    if total_targets == 0:
        return 1.0 if len(top_k) == 0 else 0.0

    covered_targets: Set[Tuple[str, int]] = set()
    covered_docs: Set[str] = set()

    for chunk in top_k:
        chunk_doc = str(chunk.get("document_name", "")).strip().lower()
        chunk_pages = _extract_page_set(chunk)

        if chunk_doc in doc_only_targets:
            covered_docs.add(chunk_doc)

        for p in chunk_pages:
            if (chunk_doc, p) in targets:
                covered_targets.add((chunk_doc, p))

    recalled = len(covered_targets) + len(covered_docs)
    return float(recalled) / float(total_targets)


def compute_reciprocal_rank(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_references: List[Dict[str, Any]],
) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of the first relevant retrieved chunk.

    Args:
        retrieved_chunks: Ranked list of retrieved chunks.
        relevant_references: List of ground-truth relevant references.

    Returns:
        Reciprocal rank float (1.0 for rank 1, 0.5 for rank 2, etc., or 0.0 if no hits).
    """
    if not relevant_references:
        # For unanswerable queries: 1.0 if 0 retrieved, else 0.0
        return 1.0 if len(retrieved_chunks) == 0 else 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_chunk_hit(chunk, relevant_references):
            return 1.0 / float(rank)

    return 0.0


def compute_mrr(reciprocal_ranks: List[float]) -> float:
    """Computes Mean Reciprocal Rank across a list of reciprocal rank scores."""
    if not reciprocal_ranks:
        return 0.0
    return float(sum(reciprocal_ranks)) / float(len(reciprocal_ranks))


def evaluate_abstention(
    is_answerable: bool,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates whether the system appropriately abstained or retrieved evidence.

    Args:
        is_answerable: Whether the question is answerable given the corpus.
        retrieved_chunks: List of retrieved chunks.

    Returns:
        Dictionary with 'did_abstain' (bool) and 'abstention_correct' (bool).
    """
    did_abstain = len(retrieved_chunks) == 0

    if is_answerable:
        # Answerable questions should NOT abstain
        abstention_correct = not did_abstain
    else:
        # Unanswerable questions SHOULD abstain
        abstention_correct = did_abstain

    return {
        "did_abstain": did_abstain,
        "abstention_correct": abstention_correct,
    }

