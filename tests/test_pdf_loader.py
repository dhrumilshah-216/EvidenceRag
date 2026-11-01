"""Tests for the PDF loader module."""

import os
import tempfile
import unittest
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.ingestion.pdf_loader import load_pdf


class TestPDFLoader(unittest.TestCase):
    """Test suite for load_pdf functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_pdf(self, filename: str, pages_text: list) -> Path:
        """Helper to create a temporary PDF with specified text per page."""
        file_path = self.temp_path / filename
        doc = fitz.open()
        for text in pages_text:
            page = doc.new_page()
            if text:
                page.insert_text((50, 50), text)
        doc.save(str(file_path))
        doc.close()
        return file_path

    def test_load_valid_pdf(self):
        """Test loading a valid multi-page PDF."""
        pdf_path = self._create_pdf(
            "sample.pdf",
            ["Page one text content.", "Page two text content."]
        )

        records = load_pdf(pdf_path)

        self.assertEqual(len(records), 2)

        self.assertEqual(records[0]["document_name"], "sample.pdf")
        self.assertEqual(records[0]["page_number"], 1)
        self.assertIn("Page one text content.", records[0]["text"])

        self.assertEqual(records[1]["document_name"], "sample.pdf")
        self.assertEqual(records[1]["page_number"], 2)
        self.assertIn("Page two text content.", records[1]["text"])

    def test_skip_empty_pages_preserves_numbering(self):
        """Test that empty or whitespace-only pages are skipped while preserving 1-based page numbers."""
        pdf_path = self._create_pdf(
            "with_empty_pages.pdf",
            [
                "Active page one.",
                "",                  # Empty page
                "Active page three.",
                "   \n\t  \n  ",     # Whitespace-only page
                "Active page five.",
            ]
        )

        records = load_pdf(pdf_path)

        self.assertEqual(len(records), 3)

        self.assertEqual(records[0]["page_number"], 1)
        self.assertEqual(records[0]["text"], "Active page one.")

        self.assertEqual(records[1]["page_number"], 3)
        self.assertEqual(records[1]["text"], "Active page three.")

        self.assertEqual(records[2]["page_number"], 5)
        self.assertEqual(records[2]["text"], "Active page five.")

    def test_missing_file_raises_error(self):
        """Test that a non-existent file path raises FileNotFoundError."""
        missing_path = self.temp_path / "does_not_exist.pdf"
        with self.assertRaises(FileNotFoundError):
            load_pdf(missing_path)

    def test_directory_path_raises_error(self):
        """Test that passing a directory path raises ValueError."""
        with self.assertRaises(ValueError):
            load_pdf(self.temp_path)

    def test_invalid_extension_raises_error(self):
        """Test that a non-PDF file raises ValueError."""
        txt_path = self.temp_path / "document.txt"
        txt_path.write_text("Not a PDF file.")
        with self.assertRaises(ValueError):
            load_pdf(txt_path)

    def test_empty_path_raises_error(self):
        """Test that an empty path string raises ValueError."""
        with self.assertRaises(ValueError):
            load_pdf("")

    def test_corrupt_pdf_raises_runtime_error(self):
        """Test that a corrupt or unreadable PDF raises RuntimeError."""
        corrupt_path = self.temp_path / "corrupt.pdf"
        corrupt_path.write_bytes(b"This is not a valid PDF binary content.")
        with self.assertRaises(RuntimeError):
            load_pdf(corrupt_path)


if __name__ == "__main__":
    unittest.main()

