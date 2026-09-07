import os
from typing import Generator, List, Dict, Any, Tuple
from dotenv import load_dotenv
try:
    from src.vectorstore import FaissVectorStore
except ImportError:
    from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b",
        groq_api_key: str = None
    ):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            if docs:
                self.vectorstore.build_from_documents(docs)
        else:
            try:
                self.vectorstore.load()
            except Exception as e:
                print(f"[WARN] Could not load existing vector store: {e}")

    def get_llm(self, model_name: str = None, temperature: float = 0.2) -> ChatGroq:
        api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured. Please add it to your .env file or deployment secrets.")
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name or self.llm_model,
            temperature=temperature,
            streaming=True
        )

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks with metadata."""
        if self.vectorstore.index is None or self.vectorstore.index.ntotal == 0:
            return []
        raw_results = self.vectorstore.query(query, top_k=top_k)
        sources = []
        for r in raw_results:
            meta = r.get("metadata") or {}
            sources.append({
                "source": meta.get("source", "Unknown Document"),
                "page": meta.get("page", 1),
                "text": meta.get("text", ""),
                "score": float(r.get("distance", 0.0))
            })
        return sources

    def build_prompt(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Construct a structured context prompt for the LLM."""
        context_parts = []
        for i, s in enumerate(sources, 1):
            src_name = s.get("source", f"Doc {i}")
            page_info = f" (Page {s.get('page')})" if s.get("page") else ""
            context_parts.append(f"--- Document [{i}]: {src_name}{page_info} ---\n{s.get('text', '').strip()}")
        
        context_text = "\n\n".join(context_parts)
        return (
            "You are DocuMind AI, an expert research assistant. Answer the user's question accurately and concisely "
            "based strictly on the provided document context. If the answer cannot be found in the context, "
            "clearly state that the documents do not contain that information.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            "Helpful and Detailed Answer:"
        )

    def search_and_stream(
        self,
        query: str,
        top_k: int = 4,
        model_name: str = "openai/gpt-oss-20b",
        temperature: float = 0.2
    ) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """Stream the generated response token-by-token and return the retrieved sources."""
        sources = self.retrieve(query, top_k=top_k)
        if not sources:
            def empty_gen():
                yield "No relevant documents found in the current knowledge base. Please upload or index documents first."
            return empty_gen(), []

        prompt = self.build_prompt(query, sources)
        llm = self.get_llm(model_name=model_name, temperature=temperature)

        def token_generator():
            for chunk in llm.stream([prompt]):
                if chunk.content:
                    yield chunk.content

        return token_generator(), sources

    def search_and_summarize(self, query: str, top_k: int = 4) -> str:
        """Synchronous generation for backwards compatibility."""
        sources = self.retrieve(query, top_k=top_k)
        if not sources:
            return "No relevant documents found."
        prompt = self.build_prompt(query, sources)
        llm = self.get_llm()
        response = llm.invoke([prompt])
        return response.content

    def rebuild_index(self, data_dir: str = "data"):
        """Re-scan data directory and rebuild the FAISS vector index."""
        try:
            from src.data_loader import load_all_documents
        except ImportError:
            from data_loader import load_all_documents
        docs = load_all_documents(data_dir)
        self.vectorstore.build_from_documents(docs)
        self.vectorstore.load()
        return len(docs)

if __name__ == "__main__":
    rag = RAGSearch()
    query = "What is machine learning?"
    print("Testing synchronous query:")
    print(rag.search_and_summarize(query, top_k=2))