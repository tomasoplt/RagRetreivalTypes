"""
Four RAG retrieval strategies using Ollama + ChromaDB:
1) Vector Store-backed Retriever
2) Multi-Query Retriever
3) Self-Query (LLM-generated metadata filters; implemented explicitly for reliability)
4) Parent Document Retriever

Requirements:
- Ollama running locally
- ollama pull qwen3-vl:8b-instruct
- ollama pull nomic-embed-text
- pip install -r requirements.txt

Run:
    python rag_retrievers.py
Then choose a retrieval strategy from the menu.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain's "classic" retriever implementations are maintained in this package.
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.storage import InMemoryStore


OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3-vl:8b-instruct"
EMBED_MODEL = "nomic-embed-text"

# None means a temporary/in-memory-style Chroma collection for this run.
# Set to e.g. "./chroma_data" to persist the vector database between runs.
CHROMA_PERSIST_DIRECTORY = None

SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "The Custody application stores information about securities, "
            "including shares and bonds. The application is maintained by the IT team. "
            "Production deployments require approval from the application owner "
            "and must be recorded in the deployment log."
        ),
        metadata={
            "title": "Custody application overview",
            "department": "IT",
            "year": 2025,
            "category": "architecture",
        },
    ),
    Document(
        page_content=(
            "To deploy the Custody application, run the automated release pipeline "
            "in the production environment. Confirm that database migrations have "
            "completed successfully, run the smoke tests, and record the release "
            "version and approver in the deployment log."
        ),
        metadata={
            "title": "Custody production deployment",
            "department": "IT",
            "year": 2025,
            "category": "operations",
        },
    ),
    Document(
        page_content=(
            "The database backup procedure runs every night. The operations team "
            "checks backup status each morning and performs a restore test monthly. "
            "Backup failures must be escalated to the on-call database administrator."
        ),
        metadata={
            "title": "Database backup procedure",
            "department": "IT",
            "year": 2024,
            "category": "operations",
        },
    ),
    Document(
        page_content=(
            "Invoices are reviewed by the Finance department. An invoice must contain "
            "a supplier, invoice number, issue date, due date, currency, and total amount. "
            "Invoices above the approval threshold require a manager's approval."
        ),
        metadata={
            "title": "Invoice processing rules",
            "department": "Finance",
            "year": 2025,
            "category": "business",
        },
    ),
    Document(
        page_content=(
            "Application security reviews include dependency scanning, static analysis, "
            "secret detection, and review of authentication and authorization. Critical "
            "findings must be fixed or explicitly accepted by the designated risk owner."
        ),
        metadata={
            "title": "Application security review",
            "department": "Security",
            "year": 2026,
            "category": "security",
        },
    ),
    Document(
        page_content=(
            "The API uses OAuth 2.0 and OpenID Connect for authentication. Access tokens "
            "must be validated by the API, including issuer, audience, signature, and expiry. "
            "Authorization policies determine which operations a user may perform."
        ),
        metadata={
            "title": "API authentication and authorization",
            "department": "IT",
            "year": 2026,
            "category": "security",
        },
    ),
]


def content_of(response: Any) -> str:
    """Normalize an Ollama/LangChain response to plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def print_documents(docs: list[Document]) -> None:
    if not docs:
        print("\nNo matching documents found.")
        return

    print(f"\nReturned {len(docs)} document(s):")
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        title = metadata.get("title", "(untitled)")
        print(f"\n--- {index}. {title} ---")
        print(
            "Metadata:",
            {
                key: metadata.get(key)
                for key in ("department", "year", "category")
                if key in metadata
            },
        )
        print(doc.page_content)


class LocalRAG:
    def __init__(self) -> None:
        self.llm = ChatOllama(
            model=CHAT_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        self.embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Main vector store: each original document is split into smaller chunks.
        self.vectorstore = Chroma(
            collection_name="rag_chunks",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )

        # Parent retriever gets its own vector store because it indexes child chunks
        # and returns the larger parent documents from InMemoryStore.
        self.parent_vectorstore = Chroma(
            collection_name="rag_parent_children",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
        )
        self.parent_store = InMemoryStore()

        self.parent_retriever = ParentDocumentRetriever(
            vectorstore=self.parent_vectorstore,
            docstore=self.parent_store,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=180,
                chunk_overlap=30,
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=60,
            ),
            search_kwargs={"k": 5},
        )

        self._ingest_documents()

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            ),
            llm=self.llm,
            include_original=True,
        )

    def _ingest_documents(self) -> None:
        # Avoid duplicate insertion if the script is changed to use persistent storage.
        # For this demo, clear collections before loading the sample data.
        for store in (self.vectorstore, self.parent_vectorstore):
            existing = store.get()
            ids = existing.get("ids", [])
            if ids:
                store.delete(ids=ids)

        # Main store contains smaller chunks with original metadata.
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
        )
        chunks = child_splitter.split_documents(SAMPLE_DOCUMENTS)
        if chunks:
            self.vectorstore.add_documents(chunks)

        # ParentDocumentRetriever creates and stores parent/child relationships.
        self.parent_retriever.add_documents(SAMPLE_DOCUMENTS, ids=None)

    # 1. Vector Store-backed Retriever
    def vector_search(self, question: str) -> list[Document]:
        """Semantic similarity search using the vector-store-backed retriever."""
        return self.vector_retriever.invoke(question)

    # 2. Multi-Query Retriever
    def multi_query_search(self, question: str) -> list[Document]:
        """Ask the LLM to generate alternate queries and combine retrieved results."""
        return self.multi_query_retriever.invoke(question)

    # 3. Self-Querying Retriever
    def self_query_search(self, question: str) -> list[Document]:
        """
        Ask the LLM to separate the semantic query from metadata filters.

        This is an explicit, defensive implementation of the Self-Query pattern.
        It translates the generated filters to Chroma's `where` format rather than
        relying on a provider-specific query-constructor translator.
        """
        allowed_fields = {
            "department": "string",
            "year": "integer",
            "category": "string",
        }
        prompt = f"""
You convert a user's search request into JSON for a document search system.
Available metadata fields:
- department (string): IT, Finance, Security
- year (integer): document year
- category (string): architecture, operations, business, security

Return ONLY valid JSON with this exact shape:
{{
  "query": "semantic search text with metadata constraints removed",
  "filters": [
    {{"field": "department", "operator": "eq", "value": "IT"}},
    {{"field": "year", "operator": "gte", "value": 2025}}
  ]
}}

Rules:
- Use only the listed metadata fields.
- Supported operators: eq, ne, gt, gte, lt, lte.
- Use an empty filters list if the user did not specify a metadata constraint.
- Do not invent filters. If unsure, omit the filter.
- The query should preserve the user's main subject.
User request: {question}
"""
        raw = content_of(self.llm.invoke(prompt)).strip()
        parsed = self._parse_json_object(raw)

        semantic_query = parsed.get("query")
        if not isinstance(semantic_query, str) or not semantic_query.strip():
            semantic_query = question

        filters = parsed.get("filters", [])
        where = self._filters_to_chroma_where(filters, allowed_fields)

        # If the model returns no valid filters, do ordinary similarity search.
        if where is None:
            return self.vectorstore.similarity_search(semantic_query, k=5)

        try:
            return self.vectorstore.similarity_search(
                semantic_query,
                k=5,
                filter=where,
            )
        except Exception as exc:
            print(f"[Self-Query warning] Chroma filter failed ({exc}); retrying without filters.")
            return self.vectorstore.similarity_search(semantic_query, k=5)

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        # Tolerate code fences or a little explanatory text around the JSON.
        raw = re.sub(r"^\s*```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            print("[Self-Query warning] Model did not return JSON; using original query.")
            return {}
        try:
            result = json.loads(raw[start : end + 1])
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            print("[Self-Query warning] Invalid JSON; using original query.")
            return {}

    @staticmethod
    def _filters_to_chroma_where(
        filters: Any,
        allowed_fields: dict[str, str],
    ) -> dict[str, Any] | None:
        if not isinstance(filters, list):
            return None

        operator_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
        }
        conditions: list[dict[str, Any]] = []

        for item in filters:
            if not isinstance(item, dict):
                continue
            field = item.get("field")
            operator = item.get("operator")
            value = item.get("value")

            if field not in allowed_fields or operator not in operator_map:
                continue

            expected_type = allowed_fields[field]
            if expected_type == "integer":
                # Do not silently accept booleans as integers.
                if isinstance(value, bool):
                    continue
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            elif not isinstance(value, str):
                continue

            conditions.append({field: {operator_map[operator]: value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # 4. Parent Document Retriever
    def parent_document_search(self, question: str) -> list[Document]:
        """Search child chunks but return their larger parent documents."""
        return self.parent_retriever.invoke(question)


def main() -> None:
    print("Initializing local RAG demo...")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embedding model: {EMBED_MODEL}")
    print("Make sure Ollama is running and both models have been pulled.")

    try:
        rag = LocalRAG()
    except Exception as exc:
        print("\nCould not initialize RAG.")
        print(f"Error: {exc}")
        print("\nCheck these commands:")
        print(f"  ollama pull {CHAT_MODEL}")
        print(f"  ollama pull {EMBED_MODEL}")
        print("  ollama list")
        print("  pip install -r requirements.txt")
        raise

    strategies = {
        "1": ("Vector Store-backed Retriever", rag.vector_search),
        "2": ("Multi-Query Retriever", rag.multi_query_search),
        "3": ("Self-Querying Retriever", rag.self_query_search),
        "4": ("Parent Document Retriever", rag.parent_document_search),
    }

    print("\nReady. Sample question: Find IT deployment documentation from 2025.")
    while True:
        print("\nChoose strategy:")
        for key, (label, _) in strategies.items():
            print(f"  {key}. {label}")
        print("  q. Quit")

        choice = input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            break
        if choice not in strategies:
            print("Please enter 1, 2, 3, 4, or q.")
            continue

        question = input("Question: ").strip()
        if not question:
            print("Question cannot be empty.")
            continue

        label, search_fn = strategies[choice]
        print(f"\n[{label}] Searching...")
        try:
            docs = search_fn(question)
            print_documents(docs)
        except Exception as exc:
            print(f"\nSearch failed: {exc}")
            print("Check the Ollama service, model names, and installed package versions.")


if __name__ == "__main__":
    main()
