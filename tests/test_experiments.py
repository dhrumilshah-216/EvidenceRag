"""Unit tests for the experiment runner module."""

import tempfile
import unittest
from pathlib import Path

from src.evaluation.experiments import (
    APPROVED_CONFIGS,
    get_document_mapping,
)


class TestExperiments(unittest.TestCase):
    """Test suite for experiment configuration and mapping."""

    def test_approved_configurations(self):
        """Verify that all three approved configurations match specifications."""
        self.assertIn("A", APPROVED_CONFIGS)
        self.assertEqual(APPROVED_CONFIGS["A"]["chunk_size"], 256)
        self.assertEqual(APPROVED_CONFIGS["A"]["overlap"], 32)
        self.assertEqual(APPROVED_CONFIGS["A"]["top_k"], 3)

        self.assertIn("B", APPROVED_CONFIGS)
        self.assertEqual(APPROVED_CONFIGS["B"]["chunk_size"], 512)
        self.assertEqual(APPROVED_CONFIGS["B"]["overlap"], 64)
        self.assertEqual(APPROVED_CONFIGS["B"]["top_k"], 3)

        self.assertIn("C", APPROVED_CONFIGS)
        self.assertEqual(APPROVED_CONFIGS["C"]["chunk_size"], 512)
        self.assertEqual(APPROVED_CONFIGS["C"]["overlap"], 64)
        self.assertEqual(APPROVED_CONFIGS["C"]["top_k"], 5)

    def test_document_mapping_extraction(self):
        """Test document name resolution from actual project benchmark."""
        benchmark_path = Path("data/benchmarks/benchmark.json")
        mapping = get_document_mapping(benchmark_path)

        self.assertTrue(len(mapping) > 0)
        self.assertIn("a051_rs_exp_5(1).pdf", mapping)
        self.assertEqual(mapping["a051_rs_exp_5(1).pdf"], "recommendation_systems_exp5.pdf")


if __name__ == "__main__":
    unittest.main()

