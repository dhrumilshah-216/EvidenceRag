"""Benchmark loading and RAG retrieval evaluation framework."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_abstention,
)


def load_benchmark(file_path_or_data: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Loads and validates benchmark questions from a JSON file path, dict, or list.

    Args:
        file_path_or_data: Path to JSON file, or parsed dict/list representing the benchmark.

    Returns:
        List of validated benchmark question dictionaries with normalized keys:
        - 'question_id' (str)
        - 'question' (str)
        - 'is_answerable' (bool)
        - 'expected_answer' (Optional[str])
        - 'relevant_references' (List[Dict[str, Any]])

    Raises:
        FileNotFoundError: If the benchmark file path does not exist.
        ValueError: If JSON is invalid or required fields are missing/malformed.
    """
    raw_data: Any

    if isinstance(file_path_or_data, (str, Path)):
        path = Path(file_path_or_data)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON format in benchmark file '{path}': {exc}") from exc
    else:
        raw_data = file_path_or_data

    # Extract questions list from dict wrapper if present
    items: List[Any]
    if isinstance(raw_data, dict):
        if "questions" in raw_data and isinstance(raw_data["questions"], list):
            items = raw_data["questions"]
        else:
            raise ValueError("Benchmark dict must contain a 'questions' list.")
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        raise ValueError(f"Expected benchmark to be a list or dict, got: {type(raw_data).__name__}")

    normalized_benchmark: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark item at index {idx} must be a dictionary.")

        # Validate question_id
        qid = item.get("question_id")
        if qid is None or not str(qid).strip():
            raise ValueError(f"Benchmark item at index {idx} missing 'question_id'.")
        question_id = str(qid).strip()

        # Validate question text
        q_text = item.get("question")
        if not isinstance(q_text, str) or not q_text.strip():
            raise ValueError(f"Benchmark item '{question_id}' missing or invalid 'question' text.")
        question = q_text.strip()

        # Validate answerable flag (support 'is_answerable' or 'answerable')
        if "is_answerable" in item:
            ans_val = item["is_answerable"]
        elif "answerable" in item:
            ans_val = item["answerable"]
        else:
            raise ValueError(f"Benchmark item '{question_id}' missing boolean 'is_answerable' field.")

        if not isinstance(ans_val, bool):
            raise ValueError(
                f"Benchmark item '{question_id}' field 'is_answerable' must be a boolean, got: {type(ans_val).__name__}"
            )
        is_answerable = ans_val

        # Normalize relevant references
        relevant_references: List[Dict[str, Any]] = []
        if "relevant_references" in item and isinstance(item["relevant_references"], list):
            relevant_references = item["relevant_references"]
        elif item.get("expected_document") is not None:
            doc_name = str(item["expected_document"]).strip()
            pages = item.get("relevant_pages", [])
            relevant_references = [{"document_name": doc_name, "page_numbers": pages}]

        # Expected answer preserved as-is, never invented
        expected_answer = item.get("expected_answer")
        if expected_answer is not None:
            expected_answer = str(expected_answer)

        normalized_benchmark.append({
            "question_id": question_id,
            "question": question,
            "is_answerable": is_answerable,
            "expected_answer": expected_answer,
            "relevant_references": relevant_references,
        })

    return normalized_benchmark


def evaluate_retrieval(
    retriever: Any,
    benchmark_items: List[Dict[str, Any]],
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    answer_generator: Optional[Any] = None,
) -> Dict[str, Any]:
    """Executes deterministic retrieval evaluation over a benchmark dataset.

    Measures Precision@K, Recall@K, MRR, abstention accuracy, and separates retrieval
    latency from generation latency. Does not use LLM-as-judge or make real LLM calls
    unless an explicit answer_generator is supplied.

    Args:
        retriever: Configured Retriever instance providing a .retrieve() method.
        benchmark_items: List of normalized benchmark question dicts.
        k: Top-K rank cutoff for retrieval metrics.
        similarity_threshold: Optional cosine similarity threshold for retriever.
        answer_generator: Optional answer generator to measure generation latency.

    Returns:
        Structured dictionary containing 'aggregate' and 'per_question' evaluation metrics.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer, got: {k}")

    per_question: List[Dict[str, Any]] = []

    for item in benchmark_items:
        question = item["question"]
        relevant_refs = item.get("relevant_references", [])
        is_answerable = item["is_answerable"]

        # Measure retrieval latency separately
        start_retrieval = time.perf_counter()
        retrieved_chunks = retriever.retrieve(
            query=question,
            top_k=k,
            similarity_threshold=similarity_threshold,
        )
        retrieval_latency_ms = (time.perf_counter() - start_retrieval) * 1000.0

        # Compute retrieval metrics deterministically
        precision = compute_precision_at_k(retrieved_chunks, relevant_refs, k=k)
        recall = compute_recall_at_k(retrieved_chunks, relevant_refs, k=k)
        rr = compute_reciprocal_rank(retrieved_chunks, relevant_refs)

        # Evaluate abstention
        abstention_info = evaluate_abstention(
            is_answerable=is_answerable,
            retrieved_chunks=retrieved_chunks,
        )

        # Optional generation latency measurement
        generation_latency_ms: Optional[float] = None
        if answer_generator is not None:
            start_gen = time.perf_counter()
            _ = answer_generator.generate_answer(question, retrieved_chunks)
            generation_latency_ms = (time.perf_counter() - start_gen) * 1000.0

        per_question.append({
            "question_id": item["question_id"],
            "question": question,
            "is_answerable": is_answerable,
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": rr,
            "did_abstain": abstention_info["did_abstain"],
            "abstention_correct": abstention_info["abstention_correct"],
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "retrieved_chunk_ids": [c.get("chunk_id", "") for c in retrieved_chunks],
            "num_retrieved": len(retrieved_chunks),
            "num_relevant_references": len(relevant_refs),
        })

    # Compute aggregate metrics
    total_q = len(per_question)
    answerable_items = [q for q in per_question if q["is_answerable"]]
    unanswerable_items = [q for q in per_question if not q["is_answerable"]]

    total_ans = len(answerable_items)
    total_unans = len(unanswerable_items)

    mean_prec = (sum(q["precision_at_k"] for q in per_question) / total_q) if total_q > 0 else 0.0
    mean_rec = (sum(q["recall_at_k"] for q in per_question) / total_q) if total_q > 0 else 0.0
    mrr = (sum(q["reciprocal_rank"] for q in per_question) / total_q) if total_q > 0 else 0.0

    ans_prec = (sum(q["precision_at_k"] for q in answerable_items) / total_ans) if total_ans > 0 else 0.0
    ans_rec = (sum(q["recall_at_k"] for q in answerable_items) / total_ans) if total_ans > 0 else 0.0
    ans_mrr = (sum(q["reciprocal_rank"] for q in answerable_items) / total_ans) if total_ans > 0 else 0.0

    unans_abstention_acc = (
        (sum(1 for q in unanswerable_items if q["abstention_correct"]) / total_unans)
        if total_unans > 0 else 1.0
    )
    ans_non_abstention_acc = (
        (sum(1 for q in answerable_items if q["abstention_correct"]) / total_ans)
        if total_ans > 0 else 1.0
    )
    overall_abstention_acc = (
        (sum(1 for q in per_question if q["abstention_correct"]) / total_q)
        if total_q > 0 else 1.0
    )

    mean_ret_latency = (
        (sum(q["retrieval_latency_ms"] for q in per_question) / total_q)
        if total_q > 0 else 0.0
    )

    gen_latencies = [q["generation_latency_ms"] for q in per_question if q["generation_latency_ms"] is not None]
    mean_gen_latency = (sum(gen_latencies) / len(gen_latencies)) if gen_latencies else None

    aggregate = {
        "total_questions": total_q,
        "total_answerable": total_ans,
        "total_unanswerable": total_unans,
        "mean_precision_at_k": mean_prec,
        "mean_recall_at_k": mean_rec,
        "mrr": mrr,
        "answerable_mean_precision_at_k": ans_prec,
        "answerable_mean_recall_at_k": ans_rec,
        "answerable_mrr": ans_mrr,
        "unanswerable_abstention_accuracy": unans_abstention_acc,
        "answerable_non_abstention_accuracy": ans_non_abstention_acc,
        "overall_abstention_accuracy": overall_abstention_acc,
        "mean_retrieval_latency_ms": mean_ret_latency,
        "mean_generation_latency_ms": mean_gen_latency,
    }

    return {
        "aggregate": aggregate,
        "per_question": per_question,
    }

