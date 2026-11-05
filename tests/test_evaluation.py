"""Unit tests for the RAG evaluation framework and metrics."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.benchmark import evaluate_retrieval, load_benchmark
from src.evaluation.metrics import (
    compute_mrr,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    evaluate_abstention,
    is_chunk_hit,
    is_chunk_relevant,
)


class MockRetriever:
    """Mock retriever returning predetermined chunk results based on query."""

    def __init__(self, query_responses: Dict[str, List[Dict[str, Any]]], delay_sec: float = 0.0):
        self.query_responses = query_responses
        self.delay_sec = delay_sec

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        results = self.query_responses.get(query, [])
        return results[:top_k]


class MockAnswerGenerator:
    """Mock answer generator to test separate latency measurement."""

    def __init__(self, delay_sec: float = 0.01):
        self.delay_sec = delay_sec

    def generate_answer(self, question: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        time.sleep(self.delay_sec)
        return {
            "answer": "Mock answer",
            "cited_chunk_ids": [],
            "source_references": [],
        }


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for core retrieval and abstention metrics."""

    def test_is_chunk_relevant(self):
        """Test matching chunk metadata against ground-truth references."""
        chunk = {"document_name": "doc1.pdf", "page_numbers": [1, 2]}

        # Exact match on document and overlapping page
        ref1 = {"document_name": "doc1.pdf", "page_numbers": [2, 3]}
        self.assertTrue(is_chunk_relevant(chunk, ref1))

        # Disjoint pages
        ref2 = {"document_name": "doc1.pdf", "page_numbers": [3, 4]}
        self.assertFalse(is_chunk_relevant(chunk, ref2))

        # Different document
        ref3 = {"document_name": "doc2.pdf", "page_numbers": [1]}
        self.assertFalse(is_chunk_relevant(chunk, ref3))

        # Document-only reference with no pages specified
        ref4 = {"document_name": "doc1.pdf"}
        self.assertTrue(is_chunk_relevant(chunk, ref4))

    def test_precision_recall_mrr_exact_values(self):
        """Test Precision@K, Recall@K and MRR with known expected values."""
        references = [{"document_name": "report.pdf", "page_numbers": [1, 2]}]

        chunk_hit_p1 = {"document_name": "report.pdf", "page_numbers": [1]}
        chunk_hit_p2 = {"document_name": "report.pdf", "page_numbers": [2]}
        chunk_miss = {"document_name": "other.pdf", "page_numbers": [1]}

        # Case 1: Rank 1 is Hit (p1), Rank 2 is Miss -> for K=2:
        # P@2 = 1/2 = 0.5
        # R@2 = 1/2 = 0.5 (covers 1 of 2 ground-truth pages)
        # RR = 1.0 (first hit at rank 1)
        retrieved_1 = [chunk_hit_p1, chunk_miss]
        self.assertAlmostEqual(compute_precision_at_k(retrieved_1, references, k=2), 0.5)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_1, references, k=2), 0.5)
        self.assertAlmostEqual(compute_reciprocal_rank(retrieved_1, references), 1.0)

        # Case 2: Rank 1 is Miss, Rank 2 is Hit (p1) -> for K=2:
        # P@2 = 1/2 = 0.5
        # R@2 = 1/2 = 0.5
        # RR = 0.5 (first hit at rank 2)
        retrieved_2 = [chunk_miss, chunk_hit_p1]
        self.assertAlmostEqual(compute_precision_at_k(retrieved_2, references, k=2), 0.5)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_2, references, k=2), 0.5)
        self.assertAlmostEqual(compute_reciprocal_rank(retrieved_2, references), 0.5)

        # Case 3: Both Rank 1 and 2 are Hits covering both pages -> for K=2:
        # P@2 = 2/2 = 1.0
        # R@2 = 2/2 = 1.0
        # RR = 1.0
        retrieved_3 = [chunk_hit_p1, chunk_hit_p2]
        self.assertAlmostEqual(compute_precision_at_k(retrieved_3, references, k=2), 1.0)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_3, references, k=2), 1.0)
        self.assertAlmostEqual(compute_reciprocal_rank(retrieved_3, references), 1.0)

        # Case 4: All misses -> for K=2:
        # P@2 = 0.0, R@2 = 0.0, RR = 0.0
        retrieved_4 = [chunk_miss, chunk_miss]
        self.assertAlmostEqual(compute_precision_at_k(retrieved_4, references, k=2), 0.0)
        self.assertAlmostEqual(compute_recall_at_k(retrieved_4, references, k=2), 0.0)
        self.assertAlmostEqual(compute_reciprocal_rank(retrieved_4, references), 0.0)

        # MRR across the 4 queries: (1.0 + 0.5 + 1.0 + 0.0) / 4 = 2.5 / 4 = 0.625
        mrr = compute_mrr([1.0, 0.5, 1.0, 0.0])
        self.assertAlmostEqual(mrr, 0.625)

    def test_abstention_evaluation(self):
        """Test abstention evaluation for answerable and unanswerable questions."""
        chunks = [{"document_name": "doc.pdf", "page_numbers": [1]}]

        # Answerable: retrieval returned chunks -> correct
        res1 = evaluate_abstention(is_answerable=True, retrieved_chunks=chunks)
        self.assertFalse(res1["did_abstain"])
        self.assertTrue(res1["abstention_correct"])

        # Answerable: retrieval returned 0 chunks (abstained falsely) -> incorrect
        res2 = evaluate_abstention(is_answerable=True, retrieved_chunks=[])
        self.assertTrue(res2["did_abstain"])
        self.assertFalse(res2["abstention_correct"])

        # Unanswerable: retrieval returned 0 chunks (abstained correctly) -> correct
        res3 = evaluate_abstention(is_answerable=False, retrieved_chunks=[])
        self.assertTrue(res3["did_abstain"])
        self.assertTrue(res3["abstention_correct"])

        # Unanswerable: retrieval returned chunks (failed to abstain) -> incorrect
        res4 = evaluate_abstention(is_answerable=False, retrieved_chunks=chunks)
        self.assertFalse(res4["did_abstain"])
        self.assertFalse(res4["abstention_correct"])


class TestBenchmarkEvaluation(unittest.TestCase):
    """Test suite for benchmark loading and evaluation workflow."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_benchmark_valid_and_schema_normalization(self):
        """Test loading valid benchmark and normalizing fields."""
        benchmark_data = {
            "questions": [
                {
                    "question_id": "Q1",
                    "question": "What is KNN?",
                    "answerable": True,
                    "expected_document": "knn.pdf",
                    "relevant_pages": [1, 2],
                    "expected_answer": "KNN is a classifier.",
                },
                {
                    "question_id": "Q2",
                    "question": "What is quantum computing?",
                    "is_answerable": False,
                    "relevant_references": [],
                },
            ]
        }
        json_file = self.temp_path / "bench.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f)

        items = load_benchmark(json_file)
        self.assertEqual(len(items), 2)

        self.assertEqual(items[0]["question_id"], "Q1")
        self.assertTrue(items[0]["is_answerable"])
        self.assertEqual(items[0]["relevant_references"], [{"document_name": "knn.pdf", "page_numbers": [1, 2]}])
        self.assertEqual(items[0]["expected_answer"], "KNN is a classifier.")

        self.assertEqual(items[1]["question_id"], "Q2")
        self.assertFalse(items[1]["is_answerable"])
        self.assertEqual(items[1]["relevant_references"], [])
        self.assertIsNone(items[1]["expected_answer"])

    def test_load_benchmark_missing_or_invalid_fields(self):
        """Test that invalid benchmark entries raise ValueError."""
        # Missing question_id
        with self.assertRaises(ValueError):
            load_benchmark([{"question": "Q?", "is_answerable": True}])

        # Missing question text
        with self.assertRaises(ValueError):
            load_benchmark([{"question_id": "1", "is_answerable": True}])

        # Missing is_answerable
        with self.assertRaises(ValueError):
            load_benchmark([{"question_id": "1", "question": "Q?"}])

        # Non-boolean is_answerable
        with self.assertRaises(ValueError):
            load_benchmark([{"question_id": "1", "question": "Q?", "is_answerable": "yes"}])

        # Missing file
        with self.assertRaises(FileNotFoundError):
            load_benchmark(self.temp_path / "non_existent.json")

    def test_load_actual_project_benchmark_file(self):
        """Test loading the project's actual data/benchmarks/benchmark.json file."""
        real_benchmark_path = Path("data/benchmarks/benchmark.json")
        self.assertTrue(real_benchmark_path.exists())

        items = load_benchmark(real_benchmark_path)
        self.assertEqual(len(items), 15)

        # Check first and last questions
        self.assertEqual(items[0]["question_id"], "Q01")
        self.assertTrue(items[0]["is_answerable"])
        self.assertEqual(items[0]["relevant_references"][0]["document_name"], "recommendation_systems_exp5.pdf")

        self.assertEqual(items[-1]["question_id"], "Q15")
        self.assertFalse(items[-1]["is_answerable"])
        self.assertEqual(items[-1]["relevant_references"], [])

    def test_synthetic_benchmark_evaluation(self):
        """Test evaluate_retrieval with small synthetic benchmark and exact known values."""
        synthetic_benchmark = [
            {
                "question_id": "SYN_01",
                "question": "Answerable Q1",
                "is_answerable": True,
                "expected_answer": None,
                "relevant_references": [{"document_name": "doc1.pdf", "page_numbers": [1]}],
            },
            {
                "question_id": "SYN_02",
                "question": "Answerable Q2",
                "is_answerable": True,
                "expected_answer": None,
                "relevant_references": [{"document_name": "doc1.pdf", "page_numbers": [1, 2]}],
            },
            {
                "question_id": "SYN_03",
                "question": "Unanswerable Q3",
                "is_answerable": False,
                "expected_answer": None,
                "relevant_references": [],
            },
        ]

        query_responses = {
            # Q1: Hit at rank 1, Miss at rank 2
            "Answerable Q1": [
                {"chunk_id": "c1", "document_name": "doc1.pdf", "page_numbers": [1]},
                {"chunk_id": "c2", "document_name": "other.pdf", "page_numbers": [1]},
            ],
            # Q2: Miss at rank 1, Hit (p1) at rank 2
            "Answerable Q2": [
                {"chunk_id": "c3", "document_name": "other.pdf", "page_numbers": [3]},
                {"chunk_id": "c4", "document_name": "doc1.pdf", "page_numbers": [1]},
            ],
            # Q3: Empty retrieval (abstained)
            "Unanswerable Q3": [],
        }

        mock_retriever = MockRetriever(query_responses=query_responses, delay_sec=0.001)

        results = evaluate_retrieval(
            retriever=mock_retriever,
            benchmark_items=synthetic_benchmark,
            k=2,
        )

        agg = results["aggregate"]
        per_q = results["per_question"]

        self.assertEqual(len(per_q), 3)

        # Q1 checks
        self.assertEqual(per_q[0]["question_id"], "SYN_01")
        self.assertAlmostEqual(per_q[0]["precision_at_k"], 0.5)
        self.assertAlmostEqual(per_q[0]["recall_at_k"], 1.0)
        self.assertAlmostEqual(per_q[0]["reciprocal_rank"], 1.0)
        self.assertFalse(per_q[0]["did_abstain"])
        self.assertTrue(per_q[0]["abstention_correct"])

        # Q2 checks
        self.assertEqual(per_q[1]["question_id"], "SYN_02")
        self.assertAlmostEqual(per_q[1]["precision_at_k"], 0.5)
        self.assertAlmostEqual(per_q[1]["recall_at_k"], 0.5)
        self.assertAlmostEqual(per_q[1]["reciprocal_rank"], 0.5)
        self.assertFalse(per_q[1]["did_abstain"])
        self.assertTrue(per_q[1]["abstention_correct"])

        # Q3 checks (unanswerable, 0 retrieved)
        self.assertEqual(per_q[2]["question_id"], "SYN_03")
        self.assertAlmostEqual(per_q[2]["precision_at_k"], 1.0)
        self.assertAlmostEqual(per_q[2]["recall_at_k"], 1.0)
        self.assertAlmostEqual(per_q[2]["reciprocal_rank"], 1.0)
        self.assertTrue(per_q[2]["did_abstain"])
        self.assertTrue(per_q[2]["abstention_correct"])

        # Aggregate checks
        self.assertEqual(agg["total_questions"], 3)
        self.assertEqual(agg["total_answerable"], 2)
        self.assertEqual(agg["total_unanswerable"], 1)

        # Expected averages across all 3:
        # P@2: (0.5 + 0.5 + 1.0) / 3 = 2.0 / 3
        self.assertAlmostEqual(agg["mean_precision_at_k"], 2.0 / 3.0)
        # R@2: (1.0 + 0.5 + 1.0) / 3 = 2.5 / 3
        self.assertAlmostEqual(agg["mean_recall_at_k"], 2.5 / 3.0)
        # MRR: (1.0 + 0.5 + 1.0) / 3 = 2.5 / 3
        self.assertAlmostEqual(agg["mrr"], 2.5 / 3.0)

        # Answerable-specific IR metrics
        self.assertAlmostEqual(agg["answerable_mean_precision_at_k"], 0.5)
        self.assertAlmostEqual(agg["answerable_mean_recall_at_k"], 0.75)
        self.assertAlmostEqual(agg["answerable_mrr"], 0.75)

        # Abstention accuracies
        self.assertAlmostEqual(agg["unanswerable_abstention_accuracy"], 1.0)
        self.assertAlmostEqual(agg["answerable_non_abstention_accuracy"], 1.0)
        self.assertAlmostEqual(agg["overall_abstention_accuracy"], 1.0)

        # Latency check
        self.assertGreater(agg["mean_retrieval_latency_ms"], 0.0)
        self.assertIsNone(agg["mean_generation_latency_ms"])

    def test_latency_measurement_separated(self):
        """Test measuring retrieval and generation latencies independently."""
        benchmark_items = [
            {
                "question_id": "LAT_01",
                "question": "Sample Question",
                "is_answerable": True,
                "relevant_references": [],
            }
        ]
        retriever = MockRetriever({"Sample Question": []}, delay_sec=0.01)
        generator = MockAnswerGenerator(delay_sec=0.015)

        results = evaluate_retrieval(
            retriever=retriever,
            benchmark_items=benchmark_items,
            k=3,
            answer_generator=generator,
        )

        item = results["per_question"][0]
        self.assertGreaterEqual(item["retrieval_latency_ms"], 5.0)
        self.assertGreaterEqual(item["generation_latency_ms"], 10.0)

        agg = results["aggregate"]
        self.assertIsNotNone(agg["mean_generation_latency_ms"])
        self.assertGreaterEqual(agg["mean_generation_latency_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()

