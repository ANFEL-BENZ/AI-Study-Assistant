"""
StudyMate AI - Streamlit interface (matches the reference design).

Only the interface lives in this file. Your existing RAG code (PDF extraction,
chunking, embeddings, FAISS, Ollama llama3.2:3b) stays exactly as it is.

HOW TO CONNECT YOUR BACKEND
---------------------------
Go to the section "1. BACKEND ADAPTER" below. It has three small functions.
Each has a TODO showing where to call your existing code. Nothing is faked:
until you connect them, the app shows a clear message instead of an invented answer.

Requirements: Streamlit 1.43 or newer (for the paperclip in the chat box and icons).
Run with:  streamlit run app.py
"""

import html
import json
import urllib.request

import streamlit as st

from pathlib import Path
import io
import json
import urllib.request
import urllib.error

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(r"D:\AI_Study_Assistant")
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"

MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# =============================================================================
# 0. SETTINGS
# =============================================================================
MODEL_NAME = "llama3.2:3b"

OLLAMA_URL = "http://127.0.0.1:11434"

MAX_FILE_MB = 50
PLACEHOLDER = "Ask anything about your documents…"
TOP_K = 4
MIN_SIMILARITY = 0.20

EXAMPLE_QUESTIONS = [
    "Summarize my course",
    "Explain a difficult concept",
    "Find key information in my documents",
]

# =============================================================================
# 1. BACKEND ADAPTER  (connect your existing code here)
# =============================================================================
# Example of importing your existing modules (rename to match your project):
#
#   from rag_pipeline import ingest_pdf, delete_document, ask


def extract_pdf_chunks(pdf_path: Path) -> list[dict]:
    """Extract PDF text and split it into overlapping chunks."""

    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def rebuild_index() -> None:
    """Rebuild the FAISS index from all PDFs in the documents folder."""

    all_chunks = []

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        all_chunks.extend(extract_pdf_chunks(pdf_path))

    if not all_chunks:
        INDEX_PATH.unlink(missing_ok=True)
        CHUNKS_PATH.unlink(missing_ok=True)
        return

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Write temporary files first
    temp_index = DATA_DIR / "faiss.tmp"
    temp_chunks = DATA_DIR / "chunks.tmp"

    faiss.write_index(index, str(temp_index))
    temp_chunks.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    temp_index.replace(INDEX_PATH)
    temp_chunks.replace(CHUNKS_PATH)


def process_pdf(file_name: str, file_bytes: bytes) -> None:
    """Save and index one PDF. Raise an exception if processing fails."""

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF.")

    pdf_path = DOCUMENTS_DIR / Path(file_name).name
    old_bytes = pdf_path.read_bytes() if pdf_path.exists() else None

    try:
        # Save the PDF
        pdf_path.write_bytes(file_bytes)

        # Extract text and create chunks
        chunks = extract_pdf_chunks(pdf_path)

        if not chunks:
            raise ValueError(
                "No readable text was found in this PDF. "
                "It may be a scanned document."
            )

        # Rebuild the index using all stored PDFs
        rebuild_index()

    except Exception:
        # Restore the previous version if processing fails
        if old_bytes is None:
            pdf_path.unlink(missing_ok=True)
        else:
            pdf_path.write_bytes(old_bytes)

        # Restore the index after rollback
        rebuild_index()
        raise




def remove_document(file_name: str) -> None:
    """Remove a PDF and rebuild the index without its passages."""

    pdf_path = DOCUMENTS_DIR / Path(file_name).name

    if not pdf_path.exists():
        return

    # Keep a backup in case rebuilding fails
    backup_bytes = pdf_path.read_bytes()

    try:
        pdf_path.unlink()
        rebuild_index()

    except Exception:
        # Restore the PDF if removal fails
        pdf_path.write_bytes(backup_bytes)

        # Restore the index from the remaining documents
        rebuild_index()
        raise




def answer_question(
    question: str,
    history: list[dict]
) -> tuple[str, list[dict]]:
    """Retrieve relevant PDF passages and generate an answer with Ollama."""

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise ValueError("The knowledge base is empty. Upload a PDF first.")

    index = faiss.read_index(str(INDEX_PATH))

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if index.ntotal == 0 or not chunks:
        raise ValueError("No searchable passages are available.")

    model = get_embedding_model()
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    k = min(TOP_K, index.ntotal)
    scores, indices = index.search(question_embedding, k)

    retrieved = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        score = float(score)
        chunk = chunks[idx]

        print(
            f"[RETRIEVAL] score={score:.3f} "
            f"source={chunk.get('source', chunk.get('file', 'Unknown PDF'))} "
            f"page={chunk.get('page')}"
        )

        if score >= MIN_SIMILARITY:
            retrieved.append(chunk)

    if not retrieved:
        raise ValueError(
            "I couldn't find sufficiently relevant passages in your PDFs. "
            "Try rephrasing your question or adding a relevant document."
        )

    context_parts = []
    sources = []

    for chunk in retrieved:
        source = chunk.get("source", chunk.get("file", "Unknown PDF"))
        page = chunk.get("page")

        context_parts.append(
            f"[Source: {source}, page: {page}]\n{chunk.get('text', '')}"
        )

        sources.append({
            "source": source,
            "page": page
        })

    context = "\n\n".join(context_parts)

    history_text = ""
    for message in history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if role in ("user", "assistant"):
            history_text += f"{role.capitalize()}: {content}\n"

    prompt = f"""
You are StudyMate AI, a helpful study assistant.

Answer the student's question using the provided document context.
Use the context as your evidence. If it does not contain enough
information, say so clearly instead of guessing.
Explain concepts step by step and use the student's language when possible.
Do not invent facts or page references.

Previous conversation:
{history_text}

Document context:
{context}

Student's question:
{question}
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="replace")
        print(f"[OLLAMA HTTP ERROR] Status: {e.code}")
        print(f"[OLLAMA HTTP ERROR] Details: {details}")
        raise RuntimeError(
            f"Ollama returned HTTP {e.code}: {details}"
        ) from e

    
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        print(f"[OLLAMA CONNECTION ERROR] {reason!r}")
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}: {reason!r}"
    ) from e


    answer = result.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    unique = []
    seen = set()

    for source in sources:
        key = (source["source"], source["page"])
        if key not in seen:
            seen.add(key)
            unique.append(source)

    return answer, unique




# =============================================================================
# 2. PAGE SETUP AND STYLING
# =============================================================================
st.set_page_config(page_title="StudyMate AI", page_icon="✨", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #F8F9FC;
    --text: #172033;
    --muted: #7A8398;
    --accent: #7565E8;
    --accent-dark: #6455D6;
    --accent-soft: #EFEDFD;
    --card: #FFFFFF;
    --border: #E6E9F0;
    --green: #1F8A5B;
    --green-soft: #DFF5EA;
    --shadow: 0 1px 2px rgba(23, 32, 51, 0.04), 0 6px 20px rgba(23, 32, 51, 0.06);
}

html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
    color: var(--text);
}
.stApp { background: var(--bg); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- Main column ---------- */
.block-container { max-width: 1040px; padding: 1.2rem 2rem 8rem; }
[data-testid="stBottomBlockContainer"] { max-width: 1040px; padding: 0 2rem 0.5rem; }
[data-testid="stBottom"] > div { background: var(--bg); }
[data-testid="stBottom"]::after {
    content: "StudyMate can make mistakes. Check important information against your course materials.";
    display: block; text-align: center; font-size: 0.78rem; color: var(--muted);
    padding: 0 1rem 0.9rem; background: var(--bg);
}



/* ---------- Heading ---------- */
.eyebrow { display: flex; align-items: center; gap: 0.7rem; color: var(--muted);
           font-size: 0.8rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; }
.eyebrow::before { content: ""; width: 30px; height: 2px; background: var(--accent); }
.page-title { font-size: 3rem; font-weight: 700; letter-spacing: -0.035em; line-height: 1.1;
              margin: 0.7rem 0 0.4rem; padding: 0; }
.page-subtitle { color: var(--muted); font-size: 1.15rem; margin: 0 0 2.2rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: -3rem;
    padding-bottom: 0;
}

.brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: -40px;
    margin-bottom: 4rem;
}
.brand-icon { width: 46px; height: 46px; border-radius: 14px; background: var(--accent);
              display: flex; align-items: center; justify-content: center;
              box-shadow: 0 6px 16px rgba(117, 101, 232, 0.35); }
.brand-name { font-size: 1.35rem; font-weight: 500; letter-spacing: -0.02em; color: #000000; }
.brand-name b { color: #000000; font-weight: 700; }

/* The upload control becomes one big violet button */
[data-testid="stFileUploader"] label { display: none; }
[data-testid="stFileUploaderDropzone"] { background: transparent; border: none; padding: 0; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"] button {
    width: 100%; height: 62px; border: none; border-radius: 14px;
    background: var(--accent); box-shadow: 0 8px 18px rgba(117, 101, 232, 0.3);
    font-size: 0 !important; line-height: 0;
}
[data-testid="stFileUploaderDropzone"] button * { font-size: 0 !important; }
[data-testid="stFileUploaderDropzone"] button::before {
    content: "+   Add PDF documents"; font-size: 1.05rem; font-weight: 600; color: #fff; line-height: 1;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--accent-dark); }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderPagination"] { display: none; }
[data-testid="stFileUploader"] small { display: none; }

.side-title { display: flex; align-items: center; gap: 0.6rem; margin: 1.8rem 0 0.8rem;
              font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em;
              text-transform: uppercase; color: var(--muted); }
.badge { background: var(--accent-soft); color: var(--accent); border-radius: 999px;
         padding: 0.1rem 0.6rem; font-size: 0.75rem; letter-spacing: 0; }

.doc-row { display: flex; align-items: center; gap: 0.8rem; padding: 0.35rem 0; }
.doc-icon { flex: none; width: 44px; height: 44px; border-radius: 12px; background: #FDECEE;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 2px 6px rgba(209, 67, 67, 0.12); }
.doc-info { min-width: 0; flex: 1; }
.doc-name { font-size: 0.95rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-meta { font-size: 0.8rem; color: var(--muted); margin-top: 2px; }
.doc-meta.failed { color: #C0392B; white-space: normal; }

/* Trash button: quiet icon */
section[data-testid="stSidebar"] .stButton > button {
    border: none; background: transparent; box-shadow: none; color: var(--muted);
    min-height: 2.2rem; padding: 0.2rem;
}
section[data-testid="stSidebar"] .stButton > button:hover { color: #C0392B; background: #FDECEE; }

.kb-card { display: flex; align-items: center; gap: 0.8rem; margin-top: 1.4rem; padding: 0.9rem 1rem;
           border: 1px solid var(--border); border-radius: 14px; background: #FBFBFE; }
.kb-icon { flex: none; width: 24px; height: 24px; border-radius: 50%;
           border: 2px solid #34C38F; color: #34C38F; display: flex;
           align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; }
.kb-icon.idle { border-color: #C5CBD8; color: #C5CBD8; }
.kb-title { font-weight: 600; font-size: 0.95rem; }
.kb-sub { font-size: 0.82rem; color: var(--muted); }

.side-footer { display: flex; align-items: center; gap: 0.7rem; margin-top: 2.5rem;
               padding-top: 1rem; border-top: 1px solid var(--border); }
.side-avatar { width: 38px; height: 38px; border-radius: 50%; background: var(--accent-soft);
               color: var(--accent); display: flex; align-items: center; justify-content: center; }
.side-footer b { font-size: 0.92rem; display: block; }
.side-footer span { font-size: 0.78rem; color: var(--muted); }

/* ---------- Empty state ---------- */
.empty { text-align: center; padding: 2rem 0 1.2rem; }
.empty-icon { width: 58px; height: 58px; margin: 0 auto 1rem; border-radius: 18px;
              background: var(--accent-soft); display: flex; align-items: center; justify-content: center; }
.empty h3 { font-size: 1.4rem; font-weight: 600; margin: 0 0 0.35rem; }
.empty p { color: var(--muted); margin: 0; }
.main-buttons .stButton > button, .stMain .stButton > button {
    border-radius: 12px; border: 1px solid var(--border); background: var(--card);
    box-shadow: var(--shadow); font-weight: 500; color: var(--text);
}
.stMain .stButton > button:hover { border-color: var(--accent); color: var(--accent); }

/* ---------- Chat messages (open layout, no cards) ---------- */
[data-testid="stChatMessage"] { background: transparent; border: none; box-shadow: none;
                                padding: 0.9rem 0; gap: 1.1rem; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    width: 44px; height: 44px; border-radius: 13px; flex: none; }
[data-testid="stChatMessageAvatarUser"] { background: var(--green-soft); color: var(--green); }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent-soft); color: var(--accent); }
.msg-label { font-size: 0.85rem; font-weight: 600; color: #4A5368; margin-bottom: 0.45rem; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
    font-size: 1.06rem; line-height: 1.75; color: #3A4358; }
[data-testid="stChatMessage"] code { background: #EEF0F6; border-radius: 6px;
                                     padding: 0.1rem 0.4rem; font-size: 0.88em; }
[data-testid="stChatMessage"] pre { border-radius: 12px; }

/* Sources: quiet line under the answer */
[data-testid="stExpander"] { margin-top: 0.9rem; max-width: 90%; }
[data-testid="stExpander"] details { border: none; border-top: 1px solid var(--border);
                                     border-radius: 0; background: transparent; }
[data-testid="stExpander"] summary { padding: 0.9rem 0; font-size: 0.95rem; }

.source-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    padding: 0.55rem 0;
    font-size: 0.9rem;
}

.source-document {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    overflow-wrap: anywhere;
}

.source-pages {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding-left: 24px;
}

.source-page {
    display: inline-block;
    font-size: 0.75rem;
    color: var(--accent);
    background: var(--accent-soft);
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
}


/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    border: 1px solid var(--border);
    border-radius: 20px;
    background: #FFFFFF !important;
    box-shadow: var(--shadow);
    padding: 0.25rem;
}

[data-testid="stChatInput"] > div {
    background: #FFFFFF !important;
}

[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #000000 !important;
    font-size: 1.02rem;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #7A8398 !important;
    opacity: 1;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent);
}

[data-testid="stChatInput"] button {
    border-radius: 12px;
}

[data-testid="stChatInputSubmitButton"] {
    background: #DDD8FA;
    color: var(--accent);
}

[data-testid="stChatInputSubmitButton"]:hover {
    background: var(--accent);
    color: #FFFFFF;
}

/* ---------- Tablet / mobile ---------- */
@media (max-width: 900px) {
    .block-container { padding: 1rem 1.2rem 8rem; }
    .page-title { font-size: 2.2rem; }
}
@media (max-width: 600px) {
    .page-title { font-size: 1.8rem; }
    .page-subtitle { font-size: 1rem; }
    [data-testid="stExpander"] { max-width: 100%; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Small inline icons (SVG) so the page needs no image files
ICON_SPARKLE = (
    '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.7 4.7 1.8-4.7 1.8L12 16l-1.8-4.7'
    '-4.7-1.8 4.7-1.8z"/><path d="M19 15l.8 2.2 2.2.8-2.2.8L19 21l-.8-2.2-2.2-.8 2.2-.8z"/></svg>'
)
ICON_FILE = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#D14343" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 '
    '0 0 2-2V8z"/><path d="M14 3v5h5"/><path d="M9 13h6M9 17h6"/></svg>'
)
ICON_HOME = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>'
)
ICON_CAP = (
    '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#7565E8" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M22 9L12 4 2 9l10 5z"/><path d="M6 11.5V16c0 1.5 '
    '2.7 3 6 3s6-1.5 6-3v-4.5"/></svg>'
)

# =============================================================================
# 3. SESSION STATE
# =============================================================================
# documents: {file_name: {"status": "ready" | "failed", "message": str, "size": int}}
st.session_state.setdefault("documents", {})
st.session_state.setdefault("messages", [])  # chat history for this session
st.session_state.setdefault("pending_question", None)  # set by example buttons
st.session_state.setdefault("uploader_key", 0)  # changing it resets the uploader


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================
@st.cache_data(ttl=15, show_spinner=False)
def check_local_model() -> str:
    """Ask Ollama (really) whether it is running and has the model installed."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as response:
            data = json.loads(response.read().decode())
    except Exception:
        return "off"
    names = [m.get("name", "") for m in data.get("models", [])]
    return "ok" if any(n.startswith(MODEL_NAME) for n in names) else "missing"


STATUS_TEXT = {
    "ok": "Local model connected",
    "missing": f"Model {MODEL_NAME} not installed",
    "off": "Local model offline",
}


def format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    return f"{max(size // 1024, 1)} KB"


def ready_documents() -> list[str]:
    """Names of documents the backend has confirmed as processed."""
    return [n for n, d in st.session_state.documents.items() if d["status"] == "ready"]


def validate_pdf(uploaded_file) -> str | None:
    """Return an error message if the file is not acceptable, otherwise None."""
    if not uploaded_file.name.lower().endswith(".pdf"):
        return "Only PDF files are supported."
    if uploaded_file.size == 0:
        return "This file is empty."
    if uploaded_file.size > MAX_FILE_MB * 1024 * 1024:
        return f"File is larger than {MAX_FILE_MB} MB."
    if not uploaded_file.getvalue().startswith(b"%PDF"):
        return "This file does not look like a valid PDF."
    return None


def handle_new_uploads(files) -> bool:
    """Validate and process files we have not seen yet. Returns True if any were handled."""
    new_files = [f for f in files if f.name not in st.session_state.documents]
    for f in new_files:
        error = validate_pdf(f)
        if error:
            st.session_state.documents[f.name] = {"status": "failed", "message": error, "size": f.size}
            continue
        # A file is only marked "ready" after the backend finishes without an error.
        with st.sidebar.status(f"Processing {f.name}…", expanded=False) as status:
            try:
                process_pdf(f.name, f.getvalue())
                st.session_state.documents[f.name] = {"status": "ready", "message": "", "size": f.size}
                status.update(label=f"{f.name} is ready", state="complete")
            except Exception as e:  # show a friendly message instead of a traceback
                st.session_state.documents[f.name] = {
                    "status": "failed",
                    "message": f"Could not process this file. {e}",
                    "size": f.size,
                }
                status.update(label=f"Could not process {f.name}", state="error")
    return bool(new_files)


def unique_sources(sources: list[dict]) -> list[dict]:
    """Remove duplicate (document, page) pairs, keep order."""
    seen, result = set(), []
    for s in sources:
        key = (s.get("source"), s.get("page"))
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"



def render_sources(sources: list[dict]) -> None:
    """Display retrieved documents and their page numbers neatly."""
    if not sources:
        return

    doc_pages: dict[str, set[int]] = {}

    for source in sources:
        doc = str(source.get("source") or "Unknown document")
        page = source.get("page")

        doc_pages.setdefault(doc, set())

        if page is not None:
            try:
                doc_pages[doc].add(int(page))
            except (TypeError, ValueError):
                pass

    summary = plural(len(doc_pages), "document")
    page_count = sum(len(pages) for pages in doc_pages.values())

    if page_count:
        summary += f" · {plural(page_count, 'page')}"

    with st.expander(f"**Sources**  :gray[{summary}]"):
        st.caption("Passages retrieved from your documents")

        for doc in sorted(doc_pages):
            pages = sorted(doc_pages[doc])

            chips = "".join(
                f'<span class="source-page">Page {page}</span>'
                for page in pages
            )

            if not chips:
                chips = '<span class="source-page">Page unknown</span>'

            st.markdown(
                f'<div class="source-item">'
                f'<div class="source-document">📄 '
                f'<span>{html.escape(doc)}</span></div>'
                f'<div class="source-pages">{chips}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


def render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    """One chat message: avatar, small label, Markdown text, optional sources."""
    is_ai = role == "assistant"
    avatar = ":material/auto_awesome:" if is_ai else ":material/person:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f'<div class="msg-label">{"StudyMate AI" if is_ai else "You"}</div>',
                    unsafe_allow_html=True)
        st.markdown(content)  # keeps Markdown, lists and code blocks
        if is_ai:
            render_sources(sources or [])


def read_chat_input(value) -> tuple[str, list]:
    """The chat box returns text (and optionally attached files)."""
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []
    return (getattr(value, "text", "") or ""), list(getattr(value, "files", []) or [])


# =============================================================================
# 5. SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-icon">{ICON_SPARKLE}</div>'
        '<div class="brand-name">StudyMate <b>AI</b></div></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if uploaded and handle_new_uploads(uploaded):
        st.session_state.uploader_key += 1  # clear the uploader so it is ready for more files
        st.rerun()

    n_docs = len(st.session_state.documents)
    st.markdown(
        f'<div class="side-title">Your documents <span class="badge">{n_docs}</span></div>',
        unsafe_allow_html=True,
    )
    if not n_docs:
        st.caption("No documents yet. Add a PDF to get started.")

    for name in list(st.session_state.documents):
        doc = st.session_state.documents[name]
        failed = doc["status"] == "failed"
        meta = html.escape(doc["message"]) if failed else format_size(doc["size"])
        col_info, col_btn = st.columns([6, 1.2], vertical_alignment="center")
        with col_info:
            st.markdown(
                f'<div class="doc-row"><div class="doc-icon">{ICON_FILE}</div>'
                f'<div class="doc-info"><div class="doc-name" title="{html.escape(name)}">'
                f'{html.escape(name)}</div>'
                f'<div class="doc-meta{" failed" if failed else ""}">{meta}</div></div></div>',
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("", key=f"remove_{name}", icon=":material/delete:", help=f"Remove {name}"):
                try:
                    if not failed:
                        remove_document(name)
                except Exception as e:
                    st.error(f"Could not remove {name}: {e}")
                else:
                    del st.session_state.documents[name]
                    st.rerun()

    # Show the knowledge-base status only after at least one document is ready.
    n_ready = len(ready_documents())
    if n_ready:
        st.markdown(
            '<div class="kb-card"><div class="kb-icon">✓</div><div>'
            '<div class="kb-title">Knowledge base ready</div>'
            f'<div class="kb-sub">{plural(n_ready, "document")} indexed</div></div></div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# 6. MAIN WORKSPACE
# =============================================================================
# ---- Top bar: menu only -------------------------------------------------------------

st.markdown('<div class="topbar-line"></div>', unsafe_allow_html=True)

# ---- Heading ---------------------------------------------------------------------
st.markdown(
    '<div class="eyebrow">Study assistant</div>'
    '<h1 class="page-title">Your personal study space</h1>'
    '<p class="page-subtitle">Ask questions and learn from your course materials.</p>',
    unsafe_allow_html=True,
)

# ---- Empty state -----------------------------------------------------------------
if not st.session_state.messages and not st.session_state.pending_question:
    st.markdown(
        f'<div class="empty"><div class="empty-icon">{ICON_CAP}</div>'
        "<h3>What would you like to learn today?</h3>"
        "<p>Upload your PDFs and ask your first question.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.pending_question = example  # handled like a typed question below
            st.rerun()

# ---- Chat history ------------------------------------------------------------------
for message in st.session_state.messages:
    render_message(message["role"], message["content"], message.get("sources"))

# ---- Chat box (the paperclip attaches PDFs) ----------------------------------------
try:
    chat_value = st.chat_input(PLACEHOLDER, accept_file="multiple", file_type=["pdf"])
except TypeError:  # older Streamlit without file attachments in the chat box
    chat_value = st.chat_input(PLACEHOLDER)

typed_text, attached_files = read_chat_input(chat_value)
question = typed_text.strip() or st.session_state.pending_question
st.session_state.pending_question = None

if attached_files and handle_new_uploads(attached_files):
    st.session_state.pending_question = question  # keep the typed question, ask after processing
    st.rerun()

# ---- Answer a new question -----------------------------------------------------------
if question:
    history = list(st.session_state.messages)  # conversation before this question
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    if not ready_documents():
        reply = (
            "I don't have any processed documents yet. "
            "Add a PDF with the **Add PDF documents** button, wait until it appears "
            "in the knowledge base, then ask again."
        )
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
        render_message("assistant", reply)
    else:
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            st.markdown('<div class="msg-label">StudyMate AI</div>', unsafe_allow_html=True)
            try:
                with st.spinner("Searching your documents and writing an answer…"):
                    answer, sources = answer_question(question, history)
                sources = unique_sources(sources)
                st.markdown(answer)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except NotImplementedError as e:
                msg = f"The study assistant isn't connected to your backend yet. {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {msg}"})
            
            except ConnectionError as e:
                msg = f"Could not connect to Ollama: {e}"
                st.error(msg)
                st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ {msg}"
            })

            except Exception as e:
                msg = f"Something went wrong while answering. Please try again. ({e})"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {msg}"})