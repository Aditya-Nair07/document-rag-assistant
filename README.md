# RAG Retrieval-Augmented Generation Pipeline

This repository contains a Python-based retrieval-augmented generation system for indexing and querying a document corpus. It loads mixed document formats, splits them into overlapping chunks, creates vector embeddings, stores them in a FAISS index, and uses a Groq-hosted language model to summarize retrieved context.

The project uses modular coding rather than a monolithic architecture: ingestion, embedding, vector storage, and retrieval each live in dedicated modules under [src](src). That makes the codebase easy to extend for new document types, different embedding models, or alternative vector stores.

## What this project does

The core workflow is implemented across [src/data_loader.py](src/data_loader.py), [src/embedding.py](src/embedding.py), [src/vectorstore.py](src/vectorstore.py), and [src/search.py](src/search.py). It is designed for document-heavy knowledge retrieval tasks where a user asks a question and the system finds the most relevant passages before generating a concise answer.

## Interesting techniques used in the code

- Document ingestion with LangChain community loaders in [src/data_loader.py](src/data_loader.py). The implementation supports PDF, text, CSV, Excel, Word, and JSON files through the LangChain community document loaders, which keeps the ingestion layer flexible without forcing a custom parser for each format. See [LangChain document loaders](https://python.langchain.com/docs/concepts/document_loaders).
- Recursive chunking with overlap in [src/embedding.py](src/embedding.py). The project uses LangChain's recursive text splitter to break large documents into smaller passages while preserving local context. This is a standard but important technique for retrieval quality because it avoids splitting semantic units too aggressively. See [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter).
- Dense embeddings with sentence-transformers in [src/embedding.py](src/embedding.py) and [src/vectorstore.py](src/vectorstore.py). The system converts text chunks into dense vector representations using the sentence-transformers library, which is a common choice for semantic search and retrieval tasks. See [sentence-transformers](https://www.sbert.net/).
- Approximate and exact vector search with FAISS in [src/vectorstore.py](src/vectorstore.py). The repository builds a FAISS index from embeddings and then queries it for nearest-neighbor retrieval, which is a core piece of modern semantic search systems. See [FAISS](https://faiss.ai/).
- Retrieval-augmented summarization in [src/search.py](src/search.py). The retrieval step is followed by an LLM call that summarizes the most relevant passages for a user query, which is the key pattern behind many RAG applications. See [LangChain](https://www.langchain.com/) and [Groq](https://console.groq.com/docs/).
- Environment-based configuration with [src/search.py](src/search.py) and [.env.example](.env.example). The project reads API credentials from environment variables, which is a practical pattern for keeping secrets out of source control. See [python-dotenv](https://github.com/theskumar/python-dotenv).

## Notable libraries and technologies

- [LangChain](https://www.langchain.com/) and [LangChain Community](https://python.langchain.com/docs/community): orchestration for document loading, chunking, and LLM integration.
- [sentence-transformers](https://www.sbert.net/): embedding generation for semantic similarity search.
- [FAISS](https://faiss.ai/): fast vector indexing and similarity search.
- [Chroma](https://www.trychroma.com/): present in the repository as a persisted vector-store artifact under [data/vector_store](data/vector_store).
- [Groq](https://console.groq.com/docs/) and [langchain-groq](https://python.langchain.com/docs/integrations/llms/groq): model serving and LLM access for summarization.
- [PyPDF](https://pypi.org/project/pypdf/) and [PyMuPDF](https://pymupdf.readthedocs.io/): PDF parsing support.
- [NumPy](https://numpy.org/): numerical array handling for embeddings and vector operations.
- [python-dotenv](https://github.com/theskumar/python-dotenv): environment variable loading.

## Project structure

```text
.
├── data/
│   ├── pdf/
│   ├── text_files/
│   └── vector_store/
├── faiss_store/
├── notebook/
├── src/
├── README.md
├── requirements.txt
└── .env.example
```

- [data](data) contains the source documents and persisted vector-store artifacts. The [data/text_files](data/text_files) folder holds plain-text samples, while [data/pdf](data/pdf) is reserved for PDF-based corpora.
- [faiss_store](faiss_store) is the runtime directory used by the FAISS index and metadata files created by [src/vectorstore.py](src/vectorstore.py).
- [notebook](notebook) contains exploratory notebooks that help document the pipeline and its experiments.
- [src](src) contains the application modules for ingestion, embedding, indexing, and retrieval.
