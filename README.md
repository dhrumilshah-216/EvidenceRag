# EvidenceRAG

An evidence-grounded document question-answering system built using Retrieval-Augmented Generation (RAG), with an evaluation framework to measure how chunking and retrieval configurations affect retrieval quality and answer reliability.

---

## 1. Architecture

EvidenceRAG is designed as a modular, beginner-understandable pipeline without high-level wrapper frameworks (no LangChain or LangGraph):

```
PDF Documents
     │
     ▼
[ src/ingestion/pdf_loader.py ]  --> Text extraction page-by-page (PyMuPDF)
     │
     ▼
[ src/ingestion/chunker.py ]     --> Token-based sliding window with source metadata
     │
     ▼
[ src/embeddings/embedder.py ]   --> sentence-transformers/all-MiniLM-L6-v2 (L2 normalized)
     │
     ▼
[ src/retrieval/vector_store.py]--> FAISS IndexFlatIP vector store & aligned metadata
     │
     ▼
[ src/retrieval/retriever.py ]   --> Cosine similarity search, top-k ranking & thresholding
     │
     ▼
[ src/generation/answer_generator.py ] --> Grounded Q&A using official google-genai SDK
     │
     ├──> [ app.py ]                  --> Interactive Streamlit web interface
     └──> [ src/evaluation/ ]         --> Deterministic evaluation framework & experiments
```

- **Ingestion (`src/ingestion/pdf_loader.py`)**: Extracts text page-by-page from PDFs using PyMuPDF, preserving document filenames and 1-based page numbers while skipping blank pages.
- **Chunking (`src/ingestion/chunker.py`)**: Splits document text into token-bounded chunks with sliding window overlap using `all-MiniLM-L6-v2` tokenizer. Preserves multi-page references (`page_numbers: List[int]`) and deterministic chunk IDs (`{doc}_chunk_{i}`).
- **Embeddings (`src/embeddings/embedder.py`)**: Encodes text chunks into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` and normalizes them to unit L2 norm.
- **Vector Store (`src/retrieval/vector_store.py`)**: Stores vectors in a FAISS `IndexFlatIP` index with 1-to-1 aligned metadata. Supports saving and loading from disk.
- **Retriever (`src/retrieval/retriever.py`)**: Encodes natural-language queries, performs cosine similarity search, applies optional thresholds, and returns ranked chunks.
- **Answer Generator (`src/generation/answer_generator.py`)**: Synthesizes grounded answers using Gemini (`google-genai`), enforces chunk citations (`[chunk_id]`), rejects hallucinated citations, and falls back to explicit insufficient-evidence notices.
- **Evaluation Framework (`src/evaluation/`)**: Deterministic evaluation measuring Precision@K, Recall@K, MRR, abstention accuracy, and independent retrieval/generation latency without LLM-as-judge calls.
- **Interface (`app.py`)**: Clean Streamlit UI for uploading PDFs, indexing documents, asking questions, and inspecting cited passages with similarity scores.

---

## 2. Setup

### Prerequisites
- Python 3.10+
- Virtual environment (recommended: `.venv`)

### Installation
1. Clone or navigate to the repository directory.
2. Activate your virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

### Configuration
Create a `.env` file from `.env.example`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
*Note: Credentials and model names are never hardcoded or silently defaulted.*

---

## 3. Usage

### Interactive Streamlit Application
Launch the web interface:
```powershell
streamlit run app.py
```
- **1. Documents**: Upload one or more PDF files and click **Process Documents**. The app indexes them into FAISS and persists them to `indexes/`.
- **2. Ask a Question**: Input a natural-language question, configure `top_k` and similarity threshold, and click **Ask**. View the grounded answer, source citations, and retrieved passages with exact cosine similarity scores.

### Running Automated Tests
Run all 61 automated unit tests across all project phases:
```powershell
pytest tests
```

---

## 4. Evaluation Methodology

Evaluation is executed deterministically against manually curated ground-truth benchmarks (`data/benchmarks/benchmark.json`):
- **Benchmark Structure**: Contains answerable and unanswerable questions with ground-truth document and page references (`expected_document`, `relevant_pages`).
- **Precision@K**: The fraction of retrieved chunks in the top K that match ground-truth source references.
- **Recall@K**: The fraction of ground-truth relevant document/page targets covered by top-K retrieved chunks.
- **MRR (Mean Reciprocal Rank)**: The reciprocal rank (1/rank) of the first relevant chunk retrieved.
- **Abstention Accuracy**:
  - *Unanswerable questions*: Correct if the system abstains (0 chunks retrieved above threshold).
  - *Answerable questions*: Correct if the system retrieves relevant evidence.
- **Latency Separation**: `retrieval_latency_ms` and `generation_latency_ms` are measured independently with high-precision timers.
- **Deterministic**: No LLM calls are made during retrieval metric calculations.

---

## 5. Actual Experiment Results

The three approved configurations were evaluated on the benchmark dataset (`data/benchmarks/benchmark.json`) over the same document corpus (`data/documents/`):

| Configuration | Chunk Size | Overlap | Top K | Indexed Chunks | Precision@K (All / Answerable) | Recall@K (All / Answerable) | MRR (All / Answerable) | Unanswerable Abstention | Answerable Non-Abstention | Overall Abstention | Mean Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config A** | 256 | 32 | 3 | 20 | **0.5333 / 0.6667** | **0.8000 / 1.0000** | **0.6778 / 0.8472** | 0.0000 | 1.0000 | 0.8000 | 13.57 ms |
| **Config B** | 512 | 64 | 3 | 10 | 0.4222 / 0.5278 | 0.7333 / 0.9167 | 0.6556 / 0.8194 | 0.0000 | 1.0000 | 0.8000 | 13.19 ms |
| **Config C** | 512 | 64 | 5 | 10 | 0.3067 / 0.3833 | **0.8000 / 1.0000** | 0.6689 / 0.8361 | 0.0000 | 1.0000 | 0.8000 | **12.47 ms** |

### Key Findings
1. **Chunk Granularity (Config A vs. Config B)**:
   - Smaller chunks (`chunk_size=256`, Config A) achieved the highest retrieval precision (`0.6667` on answerable questions) and higher MRR (`0.8472`) compared to larger chunks (`0.5278` and `0.8194` for Config B).
   - Smaller chunks isolate specific factual propositions with less surrounding non-relevant text, making top-ranked chunks more focused.
2. **Top-K Expansion (Config B vs. Config C)**:
   - Expanding from `top_k=3` to `top_k=5` with 512-token chunks increased answerable recall from `0.9167` to `1.0000` (matching Config A's 100% recall), but lowered precision from `0.5278` to `0.3833` due to including lower-ranked non-relevant passages.
3. **Abstention Behavior**:
   - All configurations achieved 100% non-abstention accuracy on answerable questions.
   - When evaluating pure top-K retrieval without a similarity threshold cutoff, top-K candidates are always returned even for unanswerable questions (`0.0` unanswerable abstention accuracy). Setting a calibrated cosine similarity threshold in the retriever allows appropriate abstention.

---

## 6. Commands to Reproduce Experiments

To re-run the benchmark experiment across all three configurations and regenerate the results CSV files:
```powershell
python -m src.evaluation.experiments
```
Results are saved to:
- `data/benchmarks/results/aggregate_results.csv`
- `data/benchmarks/results/per_question_results.csv`
- `indexes/config_A/`
- `indexes/config_B/`
- `indexes/config_C/`

---

## 7. Limitations

1. **Embedding-Only Retrieval**: Current retrieval relies solely on dense cosine similarity (`all-MiniLM-L6-v2`). Lexical keywords or exact identifiers may benefit from hybrid search (e.g. BM25 + dense).
2. **Threshold Tuning for Abstention**: Top-K retrieval returns the nearest vectors regardless of absolute semantic match. Effective retrieval-level abstention requires calibrating similarity thresholds based on corpus density.
3. **Document Extraction**: Text extraction uses PyMuPDF layout parsing and does not perform OCR on embedded raster images or complex diagrams.
