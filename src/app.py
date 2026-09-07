import os
import shutil
import sys
from pathlib import Path
import streamlit as st

# Add src to python path if running directly
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from search import RAGSearch

FAISS_DIR = ROOT_DIR / "faiss_store"
DATA_DIR = ROOT_DIR / "data"

# Streamlit Page Config
st.set_page_config(
    page_title="DocuMind AI — Intelligent Document RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #3B82F6;
        font-size: 0.9rem;
    }
    .source-title {
        font-weight: 600;
        color: #1E293B;
    }
    .source-meta {
        color: #64748B;
        font-size: 0.8rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="⚡ Initializing DocuMind Vector Engine...")
def load_rag_engine(api_key: str = None) -> RAGSearch:
    # Resolve secret on server-side without exposing to browser UI
    resolved_key = api_key
    if not resolved_key:
        try:
            if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                resolved_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not resolved_key:
        resolved_key = os.getenv("GROQ_API_KEY")
    return RAGSearch(persist_dir=str(FAISS_DIR), groq_api_key=resolved_key)


# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    # API Key Handling (Blank by default so server secret is never leaked to public visitors)
    user_api_key = st.text_input(
        "Groq API Key (Optional override)",
        value="",
        type="password",
        placeholder="Using server key by default",
        help="Leave blank to use the secure server key. Enter a custom key only if you want to override."
    )
    
    # Model Selection
    model_choice = st.selectbox(
        "LLM Model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "qwen/qwen3.6-27b",
            "groq/compound-mini",
            "groq/compound",
            "allam-2-7b"
        ],
        index=1,
        help="Current active Groq-hosted high-speed models"
    )

    # Retrieval Parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=4)
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.divider()

    # Document Uploader
    st.markdown("### 📁 Document Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
        help="Upload custom documents to expand your RAG index"
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Documents", type="primary", use_container_width=True):
            with st.spinner("Saving documents and updating FAISS vector store..."):
                target_dir = DATA_DIR / "uploads"
                target_dir.mkdir(parents=True, exist_ok=True)
                for file in uploaded_files:
                    save_path = target_dir / file.name
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Re-index
                try:
                    engine = load_rag_engine(api_key=user_api_key or None)
                    num_docs = engine.rebuild_index(data_dir=str(DATA_DIR))
                    st.success(f"Indexed {len(uploaded_files)} new file(s)! Total corpus: {num_docs} documents.")
                    st.cache_resource.clear()
                except Exception as err:
                    st.error(f"Error rebuilding index: {err}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by **LangChain** + **FAISS** + **Groq LPU**")


# --- Main Application Area ---
st.markdown('<div class="main-header">⚡ DocuMind AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Modular Document RAG Engine — Instant Semantic Search & AI Answers</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am **DocuMind AI**. Ask me anything about your uploaded documents, or select one of the suggested prompts below to get started!",
            "sources": []
        }
    ]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Retrieved Sources & Citations", expanded=False):
                for i, src in enumerate(message["sources"], 1):
                    source_name = src.get("source", "Document")
                    page_num = src.get("page", 1)
                    score = src.get("score", 0.0)
                    text_snippet = src.get("text", "")
                    st.markdown(f"**[{i}] {source_name}** `(Page {page_num} | Distance: {score:.4f})`")
                    st.caption(text_snippet[:350] + ("..." if len(text_snippet) > 350 else ""))

# Starter Suggestions (Quick Prompts)
if len(st.session_state.messages) <= 1:
    st.markdown("##### 💡 Suggested Questions:")
    s_col1, s_col2, s_col3 = st.columns(3)
    prompt_to_run = None
    with s_col1:
        if st.button("🤖 What is Machine Learning?"):
            prompt_to_run = "What is Machine Learning?"
    with s_col2:
        if st.button("🐍 What are Python's key features?"):
            prompt_to_run = "What are the key features and characteristics of Python?"
    with s_col3:
        if st.button("📄 Summarize the corpus"):
            prompt_to_run = "Summarize the key topics covered in the loaded documents."
else:
    prompt_to_run = None

# User Input Box
user_prompt = st.chat_input("Ask a question about your documents...") or prompt_to_run

if user_prompt:
    # Append User Message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process and Stream Assistant Response
    with st.chat_message("assistant"):
        try:
            engine = load_rag_engine(api_key=user_api_key or None)
            stream_gen, retrieved_sources = engine.search_and_stream(
                query=user_prompt,
                top_k=top_k,
                model_name=model_choice,
                temperature=temperature
            )

            # Stream response to UI
            response_text = st.write_stream(stream_gen)

            # Display citations accordion
            if retrieved_sources:
                with st.expander("📚 Retrieved Sources & Citations", expanded=False):
                    for i, src in enumerate(retrieved_sources, 1):
                        source_name = src.get("source", "Document")
                        page_num = src.get("page", 1)
                        score = src.get("score", 0.0)
                        text_snippet = src.get("text", "")
                        st.markdown(f"**[{i}] {source_name}** `(Page {page_num} | Distance: {score:.4f})`")
                        st.caption(text_snippet[:350] + ("..." if len(text_snippet) > 350 else ""))

            # Save to conversation history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": retrieved_sources
            })

        except Exception as err:
            error_msg = str(err)
            if "GROQ_API_KEY" in error_msg or "api_key" in error_msg.lower():
                st.error("⚠️ **Groq API Key Required**: Please provide a valid Groq API key in `.env` or in the sidebar setting. You can get one for free at [console.groq.com](https://console.groq.com/keys).")
            else:
                st.error(f"❌ An error occurred during retrieval: {error_msg}")
