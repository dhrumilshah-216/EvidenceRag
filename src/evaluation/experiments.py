"""RAG experiment runner for evaluating approved chunking and retrieval configurations."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.embeddings.embedder import Embedder
from src.evaluation.benchmark import evaluate_retrieval, load_benchmark
from src.ingestion.chunker import chunk_document, get_tokenizer
from src.ingestion.pdf_loader import load_pdf
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore

# Approved configurations per Phase 8 specification
APPROVED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "A": {
        "config_name": "A",
        "chunk_size": 256,
        "overlap": 32,
        "top_k": 3,
    },
    "B": {
        "config_name": "B",
        "chunk_size": 512,
        "overlap": 64,
        "top_k": 3,
    },
    "C": {
        "config_name": "C",
        "chunk_size": 512,
        "overlap": 64,
        "top_k": 5,
    },
}


def get_document_mapping(benchmark_path: Path) -> Dict[str, str]:
    """Extracts mapping from original filenames to canonical benchmark document names."""
    mapping: Dict[str, str] = {}
    if benchmark_path.exists():
        try:
            with open(benchmark_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = data.get("documents", [])
            for doc in docs:
                canonical = doc.get("document_name")
                original = doc.get("original_file")
                if canonical and original:
                    mapping[original.lower()] = canonical
                    # Also map without common browser download suffix like '(1)'
                    clean_orig = original.replace("(1)", "").replace(" ", "").lower()
                    mapping[clean_orig] = canonical
                if canonical:
                    mapping[canonical.lower()] = canonical
        except Exception:
            pass
    return mapping


def load_canonical_pages(
    docs_dir: Union[str, Path],
    doc_mapping: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Loads all PDF documents from directory and maps them to canonical names if provided."""
    dir_path = Path(docs_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {dir_path}")

    pdf_files = sorted(list(dir_path.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {dir_path}")

    all_pages: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        canonical_name = pdf_path.name
        if doc_mapping:
            fname_lower = pdf_path.name.lower()
            fname_clean = pdf_path.name.replace("(1)", "").replace(" ", "").lower()
            if fname_lower in doc_mapping:
                canonical_name = doc_mapping[fname_lower]
            elif fname_clean in doc_mapping:
                canonical_name = doc_mapping[fname_clean]

        for p in pages:
            p["document_name"] = canonical_name
            all_pages.append(p)

    return all_pages


def build_index_for_config(
    pages: List[Dict[str, Any]],
    tokenizer: Any,
    embedder: Embedder,
    config: Dict[str, Any],
    index_dir: Path,
) -> VectorStore:
    """Builds and persists a separate FAISS index for a specific chunking configuration."""
    chunks = chunk_document(
        pages=pages,
        tokenizer=tokenizer,
        chunk_size=config["chunk_size"],
        overlap=config["overlap"],
    )

    vector_store = VectorStore(dimension=embedder.expected_dim)
    if chunks:
        embeddings = embedder.embed_chunks(chunks)
        vector_store.add(embeddings, chunks)

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save(index_dir)
    return vector_store


def run_experiments(
    docs_dir: Union[str, Path] = "data/documents",
    benchmark_path: Union[str, Path] = "data/benchmarks/benchmark.json",
    results_dir: Union[str, Path] = "data/benchmarks/results",
    indexes_base_dir: Union[str, Path] = "indexes",
) -> Dict[str, Any]:
    """Runs all three approved experiment configurations against the benchmark.

    Builds separate indexes, records per-question and aggregate results, and saves
    structured CSV outputs.

    Args:
        docs_dir: Directory containing corpus PDFs.
        benchmark_path: Path to benchmark JSON file.
        results_dir: Target directory for results CSVs.
        indexes_base_dir: Base directory for storing configuration FAISS indexes.

    Returns:
        Dictionary containing all experiment results and metadata.
    """
    bench_file = Path(benchmark_path)
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    idx_base = Path(indexes_base_dir)

    # 1. Load benchmark questions
    benchmark_items = load_benchmark(bench_file)

    # 2. Load documents with canonical name mapping
    doc_mapping = get_document_mapping(bench_file)
    pages = load_canonical_pages(docs_dir, doc_mapping)

    # 3. Initialize tokenizer and embedder
    tokenizer = get_tokenizer()
    embedder = Embedder()

    all_experiment_results: Dict[str, Any] = {}
    aggregate_rows: List[Dict[str, Any]] = []
    all_per_question_rows: List[Dict[str, Any]] = []

    # Cache vector stores for shared chunk configurations (e.g. B and C share chunking)
    cached_stores: Dict[str, VectorStore] = {}

    for config_key in ["A", "B", "C"]:
        config = APPROVED_CONFIGS[config_key]
        chunk_sig = f"{config['chunk_size']}_{config['overlap']}"
        cfg_index_dir = idx_base / f"config_{config_key}"

        # Build or reuse index for this chunk signature
        if chunk_sig in cached_stores:
            vector_store = cached_stores[chunk_sig]
            # Still persist into config's specific directory
            vector_store.save(cfg_index_dir)
        else:
            vector_store = build_index_for_config(
                pages=pages,
                tokenizer=tokenizer,
                embedder=embedder,
                config=config,
                index_dir=cfg_index_dir,
            )
            cached_stores[chunk_sig] = vector_store

        retriever = Retriever(vector_store=vector_store, embedder=embedder)

        eval_result = evaluate_retrieval(
            retriever=retriever,
            benchmark_items=benchmark_items,
            k=config["top_k"],
        )

        all_experiment_results[config_key] = {
            "config": config,
            "aggregate": eval_result["aggregate"],
            "per_question": eval_result["per_question"],
            "indexed_chunks": vector_store.total_vectors,
        }

        # Format aggregate CSV row
        agg = eval_result["aggregate"]
        agg_row = {
            "config_name": config["config_name"],
            "chunk_size": config["chunk_size"],
            "overlap": config["overlap"],
            "top_k": config["top_k"],
            "indexed_chunks": vector_store.total_vectors,
            "total_questions": agg["total_questions"],
            "total_answerable": agg["total_answerable"],
            "total_unanswerable": agg["total_unanswerable"],
            "mean_precision_at_k": round(agg["mean_precision_at_k"], 4),
            "mean_recall_at_k": round(agg["mean_recall_at_k"], 4),
            "mrr": round(agg["mrr"], 4),
            "answerable_mean_precision_at_k": round(agg["answerable_mean_precision_at_k"], 4),
            "answerable_mean_recall_at_k": round(agg["answerable_mean_recall_at_k"], 4),
            "answerable_mrr": round(agg["answerable_mrr"], 4),
            "unanswerable_abstention_accuracy": round(agg["unanswerable_abstention_accuracy"], 4),
            "answerable_non_abstention_accuracy": round(agg["answerable_non_abstention_accuracy"], 4),
            "overall_abstention_accuracy": round(agg["overall_abstention_accuracy"], 4),
            "mean_retrieval_latency_ms": round(agg["mean_retrieval_latency_ms"], 2),
        }
        aggregate_rows.append(agg_row)

        # Format per-question CSV rows
        for q_res in eval_result["per_question"]:
            row = {
                "config_name": config["config_name"],
                "question_id": q_res["question_id"],
                "question": q_res["question"],
                "is_answerable": q_res["is_answerable"],
                "precision_at_k": round(q_res["precision_at_k"], 4),
                "recall_at_k": round(q_res["recall_at_k"], 4),
                "reciprocal_rank": round(q_res["reciprocal_rank"], 4),
                "did_abstain": q_res["did_abstain"],
                "abstention_correct": q_res["abstention_correct"],
                "retrieval_latency_ms": round(q_res["retrieval_latency_ms"], 2),
                "num_retrieved": q_res["num_retrieved"],
                "retrieved_chunk_ids": ";".join(q_res["retrieved_chunk_ids"]),
            }
            all_per_question_rows.append(row)

    # 4. Save aggregate results CSV
    agg_csv_path = res_dir / "aggregate_results.csv"
    with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
        if aggregate_rows:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    # 5. Save per-question results CSV
    pq_csv_path = res_dir / "per_question_results.csv"
    with open(pq_csv_path, "w", newline="", encoding="utf-8") as f:
        if all_per_question_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_per_question_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_per_question_rows)

    return all_experiment_results


if __name__ == "__main__":
    print("Running RAG evaluation experiments across configurations A, B, and C...")
    results = run_experiments()
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    for cfg_name, res in results.items():
        agg = res["aggregate"]
        cfg = res["config"]
        print(f"\nConfiguration {cfg_name} (chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']}, top_k={cfg['top_k']}):")
        print(f"  Indexed Chunks:                     {res['indexed_chunks']}")
        print(f"  Precision@{cfg['top_k']}:                      {agg['mean_precision_at_k']:.4f} (Answerable: {agg['answerable_mean_precision_at_k']:.4f})")
        print(f"  Recall@{cfg['top_k']}:                         {agg['mean_recall_at_k']:.4f} (Answerable: {agg['answerable_mean_recall_at_k']:.4f})")
        print(f"  MRR:                                {agg['mrr']:.4f} (Answerable: {agg['answerable_mrr']:.4f})")
        print(f"  Unanswerable Abstention Accuracy:   {agg['unanswerable_abstention_accuracy']:.4f}")
        print(f"  Answerable Non-Abstention Accuracy: {agg['answerable_non_abstention_accuracy']:.4f}")
        print(f"  Overall Abstention Accuracy:        {agg['overall_abstention_accuracy']:.4f}")
        print(f"  Mean Retrieval Latency:             {agg['mean_retrieval_latency_ms']:.2f} ms")
    print("\nResults saved to data/benchmarks/results/")

