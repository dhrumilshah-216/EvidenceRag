"""Unit tests for the grounded answer generator module using mocked API responses."""

import os
import unittest
from unittest.mock import MagicMock, patch

from src.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AnswerGenerator,
)


class MockGenerateContentResponse:
    """Mock Gemini response object."""

    def __init__(self, text: str):
        self.text = text


class TestAnswerGenerator(unittest.TestCase):
    """Test suite for AnswerGenerator with mocked Gemini API."""

    def setUp(self):
        # Create a mock client
        self.mock_client = MagicMock()
        self.mock_models = MagicMock()
        self.mock_client.models = self.mock_models

        self.generator = AnswerGenerator(
            api_key="test_api_key",
            model_name="gemini-2.5-flash",
            client=self.mock_client,
        )

        self.sample_passages = [
            {
                "chunk_id": "doc.pdf_chunk_0",
                "document_name": "doc.pdf",
                "page_numbers": [1],
                "text": "Evidence chunk zero text describing feature A.",
            },
            {
                "chunk_id": "doc.pdf_chunk_1",
                "document_name": "doc.pdf",
                "page_numbers": [1, 2],
                "text": "Evidence chunk one text describing feature B.",
            },
        ]

    def test_grounded_answer_with_valid_citations(self):
        """Test answer generation with valid cited chunk IDs and source references."""
        mock_text = (
            "Feature A is present [doc.pdf_chunk_0] and Feature B is documented [doc.pdf_chunk_1].\n"
            "Citations: doc.pdf_chunk_0, doc.pdf_chunk_1"
        )
        self.mock_models.generate_content.return_value = MockGenerateContentResponse(mock_text)

        result = self.generator.generate_answer(
            question="What features are present?",
            passages=self.sample_passages,
        )

        self.assertEqual(result["answer"], mock_text)
        self.assertEqual(result["cited_chunk_ids"], ["doc.pdf_chunk_0", "doc.pdf_chunk_1"])
        self.assertEqual(len(result["source_references"]), 2)

        self.assertEqual(result["source_references"][0]["chunk_id"], "doc.pdf_chunk_0")
        self.assertEqual(result["source_references"][0]["document_name"], "doc.pdf")
        self.assertEqual(result["source_references"][0]["page_numbers"], [1])

        self.assertEqual(result["source_references"][1]["chunk_id"], "doc.pdf_chunk_1")
        self.assertEqual(result["source_references"][1]["page_numbers"], [1, 2])

    def test_invented_citations_are_rejected(self):
        """Test that hallucinated or non-existent chunk IDs are strictly excluded."""
        # Model cites one valid chunk and one invented chunk
        mock_text = (
            "Feature A is described here [doc.pdf_chunk_0], but extra feature X is from [fake.pdf_chunk_99].\n"
            "Citations: doc.pdf_chunk_0, fake.pdf_chunk_99"
        )
        self.mock_models.generate_content.return_value = MockGenerateContentResponse(mock_text)

        result = self.generator.generate_answer(
            question="Tell me about features",
            passages=self.sample_passages,
        )

        # Only the valid chunk ID must be in cited_chunk_ids
        self.assertEqual(result["cited_chunk_ids"], ["doc.pdf_chunk_0"])
        self.assertNotIn("fake.pdf_chunk_99", result["cited_chunk_ids"])
        self.assertEqual(len(result["source_references"]), 1)
        self.assertEqual(result["source_references"][0]["chunk_id"], "doc.pdf_chunk_0")

    def test_empty_passages_returns_insufficient_evidence(self):
        """Test that empty or None passages returns explicit insufficient-evidence response without API call."""
        result_empty = self.generator.generate_answer(
            question="What is the speed of light?",
            passages=[],
        )
        self.assertEqual(result_empty["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertEqual(result_empty["cited_chunk_ids"], [])
        self.assertEqual(result_empty["source_references"], [])
        self.mock_models.generate_content.assert_not_called()

        result_none = self.generator.generate_answer(
            question="What is the speed of light?",
            passages=None,
        )
        self.assertEqual(result_none["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertEqual(result_none["cited_chunk_ids"], [])
        self.assertEqual(result_none["source_references"], [])
        self.mock_models.generate_content.assert_not_called()

    def test_model_declares_insufficient_evidence(self):
        """Test when model explicitly reports insufficient evidence from the provided passages."""
        mock_text = INSUFFICIENT_EVIDENCE_ANSWER
        self.mock_models.generate_content.return_value = MockGenerateContentResponse(mock_text)

        result = self.generator.generate_answer(
            question="Can this do feature Z?",
            passages=self.sample_passages,
        )

        self.assertEqual(result["answer"], INSUFFICIENT_EVIDENCE_ANSWER)
        self.assertEqual(result["cited_chunk_ids"], [])
        self.assertEqual(result["source_references"], [])

    def test_missing_environment_variables_raises_error(self):
        """Test that missing GEMINI_API_KEY or GEMINI_MODEL raises ValueError and no model is assumed."""
        with patch.dict(os.environ, {}, clear=True):
            # Missing API key
            with self.assertRaises(ValueError) as ctx_key:
                AnswerGenerator(api_key=None, model_name=None, client=self.mock_client)
            self.assertIn("GEMINI_API_KEY is not set", str(ctx_key.exception))

            # Has API key but missing Model name
            with self.assertRaises(ValueError) as ctx_model:
                AnswerGenerator(api_key="valid_key", model_name=None, client=self.mock_client)
            self.assertIn("GEMINI_MODEL is not set", str(ctx_model.exception))

    def test_api_error_handling(self):
        """Test that API generation failures raise clear RuntimeError."""
        self.mock_models.generate_content.side_effect = Exception("Resource exhausted / Quota exceeded")

        with self.assertRaises(RuntimeError) as ctx:
            self.generator.generate_answer(
                question="What is this?",
                passages=self.sample_passages,
            )
        self.assertIn("Gemini API generation error", str(ctx.exception))

    def test_empty_api_response_handling(self):
        """Test that empty API response raises RuntimeError."""
        self.mock_models.generate_content.return_value = MockGenerateContentResponse("")

        with self.assertRaises(RuntimeError) as ctx:
            self.generator.generate_answer(
                question="What is this?",
                passages=self.sample_passages,
            )
        self.assertIn("Gemini API returned an empty response", str(ctx.exception))

    def test_invalid_question_raises_value_error(self):
        """Test that empty or non-string question raises ValueError."""
        with self.assertRaises(ValueError):
            self.generator.generate_answer(question="", passages=self.sample_passages)

        with self.assertRaises(ValueError):
            self.generator.generate_answer(question="   \n  ", passages=self.sample_passages)

        with self.assertRaises(ValueError):
            self.generator.generate_answer(question=12345, passages=self.sample_passages)

    def test_prompt_construction(self):
        """Test that build_prompt includes evidence details, chunk IDs, and the question."""
        prompt = self.generator.build_prompt("How does feature A work?", self.sample_passages)

        self.assertIn("doc.pdf_chunk_0", prompt)
        self.assertIn("doc.pdf_chunk_1", prompt)
        self.assertIn("Evidence chunk zero text describing feature A.", prompt)
        self.assertIn("How does feature A work?", prompt)
        self.assertIn("Citations:", prompt)


if __name__ == "__main__":
    unittest.main()

