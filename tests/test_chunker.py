"""Tests for the token-based chunking module."""

import unittest
from typing import List

from src.ingestion.chunker import Chunker, chunk_document, chunk_pages, get_tokenizer

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MockTokenizer:
    """Deterministic word-level mock tokenizer for unit testing."""

    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.counter = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self.counter
                self.inv_vocab[self.counter] = word
                self.counter += 1
            tokens.append(self.vocab[word])
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.inv_vocab.get(tid, "") for tid in token_ids)


class TestChunker(unittest.TestCase):
    """Test suite for token-based chunker functionality."""

    def setUp(self):
        self.tokenizer = MockTokenizer()

    def test_single_page_boundary_and_overlap(self):
        """Test that chunks adhere to chunk_size and have correct overlap."""
        # 10 words = 10 tokens: word0 word1 ... word9
        text = "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9"
        pages = [{"document_name": "test.pdf", "page_number": 1, "text": text}]

        # chunk_size = 4, overlap = 2 -> step = 2
        # Chunk 0: word0 word1 word2 word3 (tokens 0..3)
        # Chunk 1: word2 word3 word4 word5 (tokens 2..5)
        # Chunk 2: word4 word5 word6 word7 (tokens 4..7)
        # Chunk 3: word6 word7 word8 word9 (tokens 6..9)
        chunks = chunk_document(pages, self.tokenizer, chunk_size=4, overlap=2)

        self.assertEqual(len(chunks), 4)

        self.assertEqual(chunks[0]["chunk_id"], "test.pdf_chunk_0")
        self.assertEqual(chunks[0]["text"], "word0 word1 word2 word3")
        self.assertEqual(chunks[0]["page_numbers"], [1])

        self.assertEqual(chunks[1]["chunk_id"], "test.pdf_chunk_1")
        self.assertEqual(chunks[1]["text"], "word2 word3 word4 word5")
        self.assertEqual(chunks[1]["page_numbers"], [1])

        self.assertEqual(chunks[2]["chunk_id"], "test.pdf_chunk_2")
        self.assertEqual(chunks[2]["text"], "word4 word5 word6 word7")

        self.assertEqual(chunks[3]["chunk_id"], "test.pdf_chunk_3")
        self.assertEqual(chunks[3]["text"], "word6 word7 word8 word9")

    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        text = "word0 word1 word2 word3 word4 word5"
        pages = [{"document_name": "doc.pdf", "page_number": 1, "text": text}]

        chunks = chunk_document(pages, self.tokenizer, chunk_size=3, overlap=0)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "word0 word1 word2")
        self.assertEqual(chunks[1]["text"], "word3 word4 word5")

    def test_multi_page_spanning_preserves_page_references(self):
        """Test that chunks spanning multiple pages preserve all contributing page numbers."""
        pages = [
            {"document_name": "report.pdf", "page_number": 1, "text": "p1_w0 p1_w1 p1_w2"},
            {"document_name": "report.pdf", "page_number": 2, "text": "p2_w0 p2_w1 p2_w2"},
        ]
        # Total tokens = 6: [p1_w0, p1_w1, p1_w2, p2_w0, p2_w1, p2_w2]
        # chunk_size = 4, overlap = 2 -> step = 2
        # Chunk 0: p1_w0 p1_w1 p1_w2 p2_w0 -> spans pages 1 and 2
        # Chunk 1: p1_w2 p2_w0 p2_w1 p2_w2 -> spans pages 1 and 2
        chunks = chunk_document(pages, self.tokenizer, chunk_size=4, overlap=2)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["page_numbers"], [1, 2])
        self.assertEqual(chunks[0]["chunk_id"], "report.pdf_chunk_0")
        self.assertEqual(chunks[0]["text"], "p1_w0 p1_w1 p1_w2 p2_w0")

        self.assertEqual(chunks[1]["page_numbers"], [1, 2])
        self.assertEqual(chunks[1]["chunk_id"], "report.pdf_chunk_1")
        self.assertEqual(chunks[1]["text"], "p1_w2 p2_w0 p2_w1 p2_w2")

    def test_single_and_multi_page_in_same_document(self):
        """Test chunks where some are single-page and some are multi-page."""
        pages = [
            {"document_name": "doc.pdf", "page_number": 1, "text": "a b c d e"},
            {"document_name": "doc.pdf", "page_number": 2, "text": "f g"},
        ]
        # Total tokens = 7: [a, b, c, d, e, f, g]
        # chunk_size = 4, overlap = 2 -> step = 2
        # Chunk 0 (0..4): a b c d -> page [1]
        # Chunk 1 (2..6): c d e f -> page [1, 2]
        # Chunk 2 (4..7): e f g -> page [1, 2]
        chunks = chunk_document(pages, self.tokenizer, chunk_size=4, overlap=2)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["page_numbers"], [1])
        self.assertEqual(chunks[1]["page_numbers"], [1, 2])
        self.assertEqual(chunks[2]["page_numbers"], [1, 2])

    def test_short_text_single_chunk(self):
        """Test that text with fewer tokens than chunk_size produces a single chunk."""
        pages = [{"document_name": "short.pdf", "page_number": 1, "text": "only three words"}]

        chunks = chunk_document(pages, self.tokenizer, chunk_size=100, overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "short.pdf_chunk_0")
        self.assertEqual(chunks[0]["page_numbers"], [1])
        self.assertEqual(chunks[0]["text"], "only three words")

    def test_empty_and_whitespace_input(self):
        """Test handling of empty page lists and whitespace-only pages."""
        self.assertEqual(chunk_document([], self.tokenizer, 100, 10), [])

        whitespace_pages = [
            {"document_name": "blank.pdf", "page_number": 1, "text": "   "},
            {"document_name": "blank.pdf", "page_number": 2, "text": "\n\t  \n"},
        ]
        self.assertEqual(chunk_document(whitespace_pages, self.tokenizer, 100, 10), [])

    def test_multiple_documents_restarts_indexing(self):
        """Test that chunk indexing resets to 0 for each document."""
        pages = [
            {"document_name": "docA.pdf", "page_number": 1, "text": "docA content here"},
            {"document_name": "docB.pdf", "page_number": 1, "text": "docB content here"},
        ]
        chunks = chunk_document(pages, self.tokenizer, chunk_size=100, overlap=10)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "docA.pdf_chunk_0")
        self.assertEqual(chunks[0]["document_name"], "docA.pdf")
        self.assertEqual(chunks[1]["chunk_id"], "docB.pdf_chunk_0")
        self.assertEqual(chunks[1]["document_name"], "docB.pdf")

    def test_chunker_class_wrapper(self):
        """Test Chunker class interface."""
        chunker = Chunker(tokenizer=self.tokenizer, chunk_size=3, overlap=1)
        pages = [{"document_name": "wrap.pdf", "page_number": 1, "text": "w1 w2 w3 w4"}]
        chunks = chunker.chunk(pages)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "wrap.pdf_chunk_0")

    def test_invalid_chunk_size(self):
        """Test that invalid chunk_size values raise ValueError."""
        pages = [{"document_name": "test.pdf", "page_number": 1, "text": "sample"}]
        for invalid_size in [0, -1, -50, "100", True, False, 2.5]:
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaises(ValueError):
                    chunk_document(pages, self.tokenizer, chunk_size=invalid_size, overlap=0)

    def test_invalid_overlap(self):
        """Test that invalid overlap values raise ValueError."""
        pages = [{"document_name": "test.pdf", "page_number": 1, "text": "sample"}]
        # Negative overlap
        with self.assertRaises(ValueError):
            chunk_document(pages, self.tokenizer, chunk_size=10, overlap=-1)

        # Overlap equal to chunk_size
        with self.assertRaises(ValueError):
            chunk_document(pages, self.tokenizer, chunk_size=10, overlap=10)

        # Overlap greater than chunk_size
        with self.assertRaises(ValueError):
            chunk_document(pages, self.tokenizer, chunk_size=10, overlap=15)

    def test_invalid_tokenizer(self):
        """Test that a tokenizer missing encode or decode raises TypeError."""
        pages = [{"document_name": "test.pdf", "page_number": 1, "text": "sample"}]
        with self.assertRaises(TypeError):
            chunk_document(pages, None, chunk_size=10, overlap=2)

        with self.assertRaises(TypeError):
            chunk_document(pages, object(), chunk_size=10, overlap=2)

    def test_malformed_page_record(self):
        """Test that malformed page records raise ValueError."""
        with self.assertRaises(ValueError):
            chunk_document(["not a dict"], self.tokenizer, chunk_size=10, overlap=2)

        with self.assertRaises(ValueError):
            chunk_document([{"document_name": "doc.pdf"}], self.tokenizer, chunk_size=10, overlap=2)

    @unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers package not installed")
    def test_with_real_autotokenizer(self):
        """Test with real AutoTokenizer if transformers is available."""
        real_tokenizer = get_tokenizer("sentence-transformers/all-MiniLM-L6-v2")
        pages = [
            {"document_name": "real.pdf", "page_number": 1, "text": "This is a real document text."},
            {"document_name": "real.pdf", "page_number": 2, "text": "And this is the second page of text."},
        ]
        chunks = chunk_document(pages, real_tokenizer, chunk_size=8, overlap=2)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["document_name"], "real.pdf")
        self.assertIn("chunk_id", chunks[0])
        self.assertIn("page_numbers", chunks[0])


if __name__ == "__main__":
    unittest.main()

