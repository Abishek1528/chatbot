"""PDF RAG backend: PDF extraction, chunking, embeddings, retrieval and Gemini answers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import uuid

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.errors import ServerError


class PDFRAGAssistant:
    """A small, single-document RAG pipeline for a Streamlit application."""

    def __init__(
        self,
        google_api_key: str,
        db_path: str = "chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gemini-4-hydrogen",
        top_k: int = 5,
    ) -> None:
        if not google_api_key:
            raise ValueError("Google API key is missing. Add GOOGLE_API_KEY to your .env file.")

        self.embedder = SentenceTransformer(embedding_model)
        primary_llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            max_retries=2,
        )
        fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=google_api_key,
            max_retries=2,
        )
        self.llm = primary_llm.with_fallbacks(
            [fallback_llm],
            exceptions_to_handle=(ServerError,),
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = None
        self.document_name: str | None = None
        self.chunk_count = 0
        self.top_k = top_k

    @staticmethod
    def _extract_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
        """Return non-empty page text with 1-based page numbers."""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}") from exc

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"page": page_number, "text": text})

        if not pages:
            raise ValueError(
                "No selectable text was found in this PDF. It may be a scanned document; "
                "run OCR on it and upload the searchable PDF."
            )
        return pages

    @staticmethod
    def _make_chunks(pages: list[dict[str, Any]], chunk_size: int = 1200, overlap: int = 200):
        chunks = []
        for page_data in pages:
            text = page_data["text"]
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append({"text": chunk, "page": page_data["page"]})
                if end >= len(text):
                    break
                start = end - overlap
        return chunks

    def process_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Extract, chunk, embed and index the uploaded PDF. Returns chunk count."""
        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty.")

        pages = self._extract_pages(pdf_bytes)
        chunks = self._make_chunks(pages)
        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")

        # Use a fresh collection so a newly uploaded PDF cannot mix with old content.
        collection_name = f"pdf_{uuid.uuid4().hex[:20]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        documents = [item["text"] for item in chunks]
        embeddings = self.embedder.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"page": int(item["page"]), "source": filename}
            for item in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        self.document_name = filename
        self.chunk_count = len(chunks)
        return self.chunk_count

    def ask_question(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and ask Gemini to answer from the PDF only."""
        if self.collection is None or self.chunk_count == 0:
            raise RuntimeError("Upload and process a PDF before asking a question.")
        if not question or not question.strip():
            raise ValueError("Please enter a question.")

        query_embedding = self.embedder.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].tolist()
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, self.chunk_count),
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        evidence = []
        for doc, metadata, distance in zip(docs, metadatas, distances):
            evidence.append(
                {
                    "page": metadata.get("page", "?"),
                    "source": metadata.get("source", self.document_name or "PDF"),
                    "text": doc,
                    "distance": float(distance),
                }
            )

        context = "\n\n".join(
            f"[Page {item['page']}]\n{item['text']}" for item in evidence
        )
        history_text = ""
        if chat_history:
            recent = chat_history[-6:]
            history_text = "\n".join(
                f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
                for item in recent
            )

        prompt = f"""You are a helpful PDF question-answering assistant.
Answer the user's question using ONLY the PDF context below.
- If the answer is not present in the supplied context, clearly say that the PDF does not provide enough information.
- Do not invent facts or use outside knowledge.
- Explain the answer in clear, simple language.
- Cite supporting page numbers like (Page 2). Only cite pages present in the context.
- If useful, answer in short bullet points.

PREVIOUS CONVERSATION (for follow-up context only):
{history_text or 'No previous conversation.'}

PDF CONTEXT:
{context}

USER QUESTION:
{question.strip()}

ANSWER:"""

        response = self.llm.invoke(prompt)
        answer = response.content
        if isinstance(answer, list):
            answer = "\n".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in answer
            )

        return {"answer": str(answer), "evidence": evidence}
