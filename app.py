"""Streamlit interface for EvidenceRAG."""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_document, get_tokenizer
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.answer_generator import (
    AnswerGenerator,
    INSUFFICIENT_EVIDENCE_ANSWER,
)

DOCS_DIR = Path("data/documents")
INDEX_DIR = Path("indexes")


def init_session_state() -> None:
    """Initializes Streamlit session state objects if not already present."""
    if "vector_store" not in st.session_state:
        # Check if saved index exists on disk
        if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "metadata.json").exists():
            try:
                st.session_state.vector_store = VectorStore.load(INDEX_DIR)
            except Exception:
                st.session_state.vector_store = VectorStore()
        else:
            st.session_state.vector_store = VectorStore()

    if "embedder" not in st.session_state:
        st.session_state.embedder = Embedder()

    if "retriever" not in st.session_state:
        st.session_state.retriever = Retriever(
            vector_store=st.session_state.vector_store,
            embedder=st.session_state.embedder,
        )

    if "processed_documents" not in st.session_state:
        existing_docs = {
            m.get("document_name")
            for m in st.session_state.vector_store.metadata
            if isinstance(m, dict) and "document_name" in m
        }
        st.session_state.processed_documents = sorted(list(existing_docs))


def main() -> None:
    st.set_page_config(page_title="EvidenceRAG", layout="wide")
    st.title("EvidenceRAG")
    st.caption("Evidence-grounded question answering with citations")

    init_session_state()

    # Section 1: Documents
    st.header("1. Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        else:
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            with st.spinner("Processing uploaded documents..."):
                try:
                    tokenizer = get_tokenizer()
                    total_chunks_added = 0
                    newly_processed = []

                    for uploaded_file in uploaded_files:
                        file_path = DOCS_DIR / uploaded_file.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # 1. Ingestion
                        pages = load_pdf(file_path)
                        if not pages:
                            st.warning(f"No readable text found in {uploaded_file.name}.")
                            continue

                        # 2. Chunking
                        chunks = chunk_document(
                            pages=pages,
                            tokenizer=tokenizer,
                            chunk_size=256,
                            overlap=32,
                        )
                        if not chunks:
                            st.warning(f"No chunks generated for {uploaded_file.name}.")
                            continue

                        # 3. Embedding
                        embeddings = st.session_state.embedder.embed_chunks(chunks)

                        # 4. Vector store indexing
                        st.session_state.vector_store.add(embeddings, chunks)
                        total_chunks_added += len(chunks)
                        newly_processed.append(uploaded_file.name)

                    # Persist index to disk
                    st.session_state.vector_store.save(INDEX_DIR)

                    # Update processed documents list
                    for doc in newly_processed:
                        if doc not in st.session_state.processed_documents:
                            st.session_state.processed_documents.append(doc)

                    st.success(
                        f"Successfully processed {len(newly_processed)} document(s) "
                        f"({total_chunks_added} chunks added). Index updated."
                    )
                except Exception as exc:
                    st.error(f"Error processing documents: {exc}")

    # Display processed document names and status
    if st.session_state.processed_documents:
        st.write("**Processed Documents in Index:**")
        for doc_name in st.session_state.processed_documents:
            st.write(f"- {doc_name}")
        st.write(f"*Total indexed chunks: {st.session_state.vector_store.total_vectors}*")
    else:
        st.info("No documents indexed yet. Upload and process PDFs to begin.")

    st.divider()

    # Section 2: Ask Questions
    st.header("2. Ask a Question")
    question = st.text_input("Enter your question based on the indexed documents:")

    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.number_input("Top K Passages", min_value=1, max_value=20, value=4, step=1)
    with col2:
        threshold = st.slider(
            "Similarity Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )

    if st.button("Ask"):
        if not question or not question.strip():
            st.warning("Please enter a question.")
        elif st.session_state.vector_store.total_vectors == 0:
            st.warning("No documents have been indexed yet. Please upload and process documents first.")
        else:
            with st.spinner("Retrieving evidence and generating answer..."):
                try:
                    # 1. Retrieval
                    passages = st.session_state.retriever.retrieve(
                        query=question,
                        top_k=int(top_k),
                        similarity_threshold=float(threshold),
                    )

                    # 2. Answer generation
                    generator = AnswerGenerator()
                    result = generator.generate_answer(question=question, passages=passages)

                    # 3. Display answer
                    st.subheader("Answer")
                    answer_text = result.get("answer", "")
                    if INSUFFICIENT_EVIDENCE_ANSWER.lower() in answer_text.lower():
                        st.info(answer_text)
                    else:
                        st.write(answer_text)

                    # 4. Display citations
                    cited_ids = result.get("cited_chunk_ids", [])
                    source_refs = result.get("source_references", [])

                    if cited_ids:
                        st.subheader("Citations")
                        for ref in source_refs:
                            doc_name = ref.get("document_name", "unknown")
                            pages = ref.get("page_numbers", [])
                            pages_str = ", ".join(map(str, pages))
                            cid = ref.get("chunk_id", "")
                            st.write(f"- **[{cid}]** {doc_name} (Page(s): {pages_str})")

                    # 5. Display retrieved passages with similarity scores
                    with st.expander("View Retrieved Evidence Passages & Similarity Scores"):
                        if not passages:
                            st.write("No passages retrieved above the similarity threshold.")
                        else:
                            for idx, passage in enumerate(passages, start=1):
                                score = passage.get("score", 0.0)
                                cid = passage.get("chunk_id", "unknown")
                                doc = passage.get("document_name", "unknown")
                                pages = passage.get("page_numbers", [])
                                pages_str = ", ".join(map(str, pages))
                                text = passage.get("text", "")

                                st.markdown(
                                    f"**Passage {idx}** | Score: `{score:.4f}` | "
                                    f"Chunk: `{cid}` | File: `{doc}` (Pages: {pages_str})"
                                )
                                st.text(text)
                                st.write("---")
                except Exception as exc:
                    st.error(f"Error during retrieval or answer generation: {exc}")


if __name__ == "__main__":
    main()

