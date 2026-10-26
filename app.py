import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openrouter import ChatOpenRouter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="centered")

# ---------- CSS + ANIMATIONS ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #1a1a3e, #2b1b4d, #14243e);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        color: #eaeaf5;
    }
    .stApp::before, .stApp::after {
        content: ""; position: fixed; border-radius: 50%;
        filter: blur(80px); opacity: 0.22; pointer-events: none; z-index: 0;
    }
    .stApp::before {
        width: 320px; height: 320px; background: #8e7cff;
        top: 5%; left: 8%; animation: floatA 14s ease-in-out infinite;
    }
    .stApp::after {
        width: 380px; height: 380px; background: #ff6ec7;
        bottom: 5%; right: 8%; animation: floatB 17s ease-in-out infinite;
    }
    #MainMenu, footer {visibility: hidden;}

    .hero-title {
        text-align: center; font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #8e7cff, #ff6ec7, #4fd1ff, #8e7cff);
        background-size: 300% 100%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: shimmer 6s linear infinite, floatTitle 4s ease-in-out infinite;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center; color: #a9a9c8; margin-bottom: 1.5rem;
        animation: fadeUp 1s ease both;
    }

    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
        animation: slideInLeft 0.7s ease both;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
        animation: slideInLeft 0.5s cubic-bezier(0.2,0.8,0.2,1) both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    [data-testid="stChatMessage"]:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 28px rgba(142,124,255,0.3);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(142,124,255,0.35), rgba(255,110,199,0.25));
        animation: slideInRight 0.5s cubic-bezier(0.2,0.8,0.2,1) both;
    }
    [data-testid="stChatMessage"] img,
    [data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] {
        animation: popIn 0.5s ease both; transition: transform 0.3s ease;
    }
    [data-testid="stChatMessage"]:hover [data-testid^="stChatMessageAvatar"] {
        transform: rotate(-10deg) scale(1.15);
    }

    .typing { display: flex; gap: 6px; padding: 6px 2px; }
    .typing span {
        width: 10px; height: 10px; border-radius: 50%;
        background: linear-gradient(135deg, #8e7cff, #ff6ec7);
        animation: bounce 1.2s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }

    .doc-badge {
        padding: 0.6rem 0.9rem; border-radius: 12px; margin: 0.5rem 0;
        background: rgba(79,209,255,0.12); border: 1px solid rgba(79,209,255,0.4);
        animation: fadeUp 0.5s ease both;
    }
    .mode-pill {
        text-align: center; margin: 0 auto 1rem auto; width: fit-content;
        padding: 0.35rem 1rem; border-radius: 999px; font-size: 0.85rem;
        background: rgba(79,209,255,0.15); border: 1px solid rgba(79,209,255,0.5);
        animation: fadeUp 0.5s ease both, glowPulse 3s ease-in-out infinite;
    }

    [data-testid="stChatInput"] {
        border-radius: 16px; border: 1px solid rgba(142,124,255,0.5);
        transition: box-shadow 0.3s ease, border-color 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #ff6ec7; animation: glowPulse 2s ease-in-out infinite;
    }

    .stButton > button {
        position: relative; overflow: hidden; width: 100%;
        border-radius: 14px; border: 1px solid rgba(142,124,255,0.5);
        background: rgba(142,124,255,0.12); color: #eaeaf5;
        padding: 0.7rem 1rem; transition: all 0.25s ease;
        animation: fadeUp 0.6s ease both;
    }
    .stButton > button::after {
        content: ""; position: absolute; top: 0; left: -100%;
        width: 60%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.35), transparent);
        transition: left 0.6s ease;
    }
    .stButton > button:hover::after { left: 130%; }
    .stButton > button:hover {
        background: linear-gradient(90deg, #8e7cff, #ff6ec7);
        border-color: transparent; transform: translateY(-3px) scale(1.04);
        box-shadow: 0 8px 20px rgba(255,110,199,0.35); color: white;
    }
    .stButton > button:active { transform: scale(0.97); }

    @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
    @keyframes floatA { 0%,100%{transform:translate(0,0)} 50%{transform:translate(60px,80px)} }
    @keyframes floatB { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-70px,-60px)} }
    @keyframes shimmer { 0%{background-position:0% 50%} 100%{background-position:300% 50%} }
    @keyframes floatTitle { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
    @keyframes fadeUp { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:translateY(0)} }
    @keyframes slideInLeft { from{opacity:0; transform:translateX(-40px)} to{opacity:1; transform:translateX(0)} }
    @keyframes slideInRight { from{opacity:0; transform:translateX(40px)} to{opacity:1; transform:translateX(0)} }
    @keyframes popIn { 0%{opacity:0; transform:scale(0.3)} 70%{transform:scale(1.15)} 100%{opacity:1; transform:scale(1)} }
    @keyframes bounce { 0%,80%,100%{transform:translateY(0); opacity:0.5} 40%{transform:translateY(-10px); opacity:1} }
    @keyframes glowPulse { 0%,100%{box-shadow:0 0 6px rgba(255,110,199,0.4)} 50%{box-shadow:0 0 22px rgba(255,110,199,0.8)} }

    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
    /* ===== Mobile layout ===== */
    @media (max-width: 640px) {
        .stApp { overflow-x: hidden; }
        .stApp::before, .stApp::after { display: none; }

        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 3.5rem !important;
        }
        .hero-title { font-size: 1.7rem; }
        .hero-sub { font-size: 0.9rem; padding: 0 0.5rem; }

        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
            animation-name: fadeUp;
        }
        .stButton > button { padding: 0.6rem 0.7rem; font-size: 0.9rem; }
        .mode-pill { font-size: 0.75rem; }
    }

    /* Stop hover effects from sticking on touch screens */
    @media (hover: none) {
        [data-testid="stChatMessage"]:hover,
        .stButton > button:hover { transform: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown('<h1 class="hero-title">🤖 AI Chat Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Ask anything, or upload a document for answers grounded in your own content</p>', unsafe_allow_html=True)


# ---------- RAG helpers ----------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    # Free, runs on your PC. Downloads a small model (~130 MB) the first time only.
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def build_index(file_bytes):
    """PDF bytes -> (vector store, number of pages, number of chunks)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.remove(path)

    pages = [p for p in pages if p.page_content.strip()]
    if not pages:
        return None, 0, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    store = InMemoryVectorStore(get_embeddings())
    store.add_documents(chunks)
    return store, len(pages), len(chunks)


# ---------- Chat helpers ----------
def to_langchain(messages):
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]


def stream_response(question, history, model, temperature, max_tokens, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder("history"),
            ("user", "{question}"),
        ]
    )
    llm = ChatOpenRouter(
        model=model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.stream({"system_prompt": system_prompt, "history": history, "question": question})


def show_sources(sources):
    if sources:
        with st.expander("📚 Sources"):
            for s in sources:
                st.markdown(f"**Page {s['page']}**")
                st.caption(s["text"])


# ---------- State ----------
defaults = {"messages": [], "pending": None, "store": None, "doc_key": None, "doc_info": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Sidebar: normal chatbot settings ----------
st.sidebar.title("⚙️ Settings")

model = st.sidebar.selectbox("Model", ["openrouter/free"])  # add exact IDs from openrouter.ai/models

personas = {
    "Helpful assistant": "You are a helpful Q&A assistant. Answer clearly and concisely.",
    "Friendly tutor": "You are a friendly tutor. Explain things simply with small examples.",
    "Funny buddy": "You are a witty, funny assistant. Keep answers correct but playful.",
    "Short & direct": "You are a concise assistant. Answer in 1-3 sentences.",
}
persona = st.sidebar.selectbox("Personality", list(personas.keys()))
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max tokens", 100, 1500, 600)

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()

# ---------- Sidebar: optional RAG panel ----------
st.sidebar.markdown("---")
use_doc = False
top_k = 4
with st.sidebar.expander("📄 Document Assistant", expanded=False):
    st.caption("Upload a PDF to get answers grounded in its content, with page references.")
    uploaded = st.file_uploader("Upload a PDF document", type=["pdf"])
    top_k = st.slider("Chunks to retrieve", 2, 8, 4)

    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if st.session_state.doc_key != key:  # only rebuild when a new file is uploaded
            with st.spinner("Reading and indexing your PDF..."):
                try:
                    store, n_pages, n_chunks = build_index(uploaded.getvalue())
                except Exception as e:
                    store, n_pages, n_chunks = None, 0, 0
                    st.error(f"Could not read the PDF: {e}")
            if store is None:
                st.error("No text found. This may be a scanned PDF (images only).")
            else:
                st.session_state.store = store
                st.session_state.doc_key = key
                st.session_state.doc_info = (uploaded.name, n_pages, n_chunks)
                st.session_state.messages = []  # fresh chat for the new document
    else:
        st.session_state.store = None
        st.session_state.doc_key = None
        st.session_state.doc_info = None

    if st.session_state.store is not None:
        name, n_pages, n_chunks = st.session_state.doc_info
        st.markdown(
            f'<div class="doc-badge">✅ <b>{name}</b><br>{n_pages} pages · {n_chunks} chunks indexed</div>',
            unsafe_allow_html=True,
        )
        use_doc = st.toggle("Answer from the uploaded document", value=True)

doc_mode = use_doc and st.session_state.store is not None

# ---------- Mode indicator ----------
if doc_mode:
    st.markdown(
        f'<div class="mode-pill">📄 Document mode: answering from <b>{st.session_state.doc_info[0]}</b></div>',
        unsafe_allow_html=True,
    )

# ---------- Welcome screen with suggestion buttons ----------
if not st.session_state.messages:
    if doc_mode:
        st.markdown("#### 📄 Your document is ready. Try one of these")
        suggestions = [
            "Summarize this document",
            "What are the key points?",
            "List the important terms",
            "What questions could be asked from this?",
        ]
    else:
        st.markdown("#### 👋 Try one of these to get started")
        suggestions = [
            "Explain AI in simple words",
            "Give me a fun fact",
            "Help me plan my study day",
            "Write a short motivational quote",
        ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}"):
            st.session_state.pending = s
            st.rerun()

# ---------- Show past messages ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

# ---------- Handle new input ----------
user_input = st.chat_input("Type your question here...")
if not user_input and st.session_state.pending:
    user_input = st.session_state.pending
st.session_state.pending = None

if user_input:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY is missing. Add it to your .env file and restart the app.")
        st.stop()

    history = to_langchain(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        typing = st.empty()
        typing.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
        try:
            sources = []
            if doc_mode:
                # RETRIEVE: find the chunks closest in meaning to the question
                docs = st.session_state.store.similarity_search(user_input, k=top_k)
                context = "\n\n".join(
                    f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
                )
                system_prompt = (
                    "You are a document Q&A assistant. Answer using ONLY the context below. "
                    "If the answer is not in the context, say you could not find it in the document. "
                    "Mention page numbers like (p. 3) when you use them.\n\n"
                    f"CONTEXT:\n{context}"
                )
                seen = set()
                for d in docs:
                    page = d.metadata.get("page", 0) + 1
                    if page not in seen:
                        seen.add(page)
                        sources.append({"page": page, "text": d.page_content[:300] + "..."})
                sources.sort(key=lambda s: s["page"])
            else:
                system_prompt = personas[persona]

            stream = stream_response(user_input, history, model, temperature, max_tokens, system_prompt)
            first_chunk = next(stream, "")
            typing.empty()

            def full_stream():
                yield first_chunk
                yield from stream

            reply = st.write_stream(full_stream())
            show_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
        except Exception as e:
            typing.empty()
            st.error(f"Something went wrong: {e}")