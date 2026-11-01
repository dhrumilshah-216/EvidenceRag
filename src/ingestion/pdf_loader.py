"""PDF loader module for extracting text page-by-page from PDF documents."""

from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def load_pdf(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads a PDF file and extracts text page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of structured records containing:
        - document_name (str): Filename of the document.
        - page_number (int): 1-based page number.
        - text (str): Extracted text content for that page.

    Raises:
        ValueError: If file_path is empty, not a PDF, or points to a directory.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the PDF is corrupted, encrypted, or unreadable.
    """
    if not file_path:
        raise ValueError("File path must be provided.")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid file extension. Expected a PDF file: {file_path}")

    document_name = path.name

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{document_name}': {exc}") from exc

    try:
        if doc.is_encrypted:
            raise RuntimeError(f"PDF '{document_name}' is encrypted or password-protected.")

        records: List[Dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page_number = page_idx + 1
            page = doc[page_idx]
            text = page.get_text()

            # Skip empty or whitespace-only pages without losing page numbering
            if not text or not text.strip():
                continue

            records.append({
                "document_name": document_name,
                "page_number": page_number,
                "text": text.strip(),
            })

        return records
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Error reading PDF '{document_name}': {exc}") from exc
    finally:
        doc.close()

