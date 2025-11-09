# app.py — ADHD Assistant (improved)
import os
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import textwrap
import re
from typing import List, Tuple


# ---------------- Config ----------------
st.set_page_config(page_title='ADHD Assistant', layout='centered')

# Paths & constants
DB_PATH = "data/chat.db"
KB_DIR = Path("kb")
EMB_FILE = KB_DIR / "embeddings.npy"
DOCS_FILE = KB_DIR / "docs.pkl"
CLEAN_FILE = KB_DIR / "cleaned_docs.pkl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_RETURN_DOCS = 4
SIMILARITY_THRESHOLD = 0.25  # suppress weak references
CHUNK_CHAR_SIZE = 1200       # ~800–900 tokens rough proxy
CHUNK_OVERLAP = 150

# Ensure directories exist
KB_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ---------------- Database ----------------
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
metadata = MetaData()
conversations = Table(
    'conversations', metadata,
    Column('id', Integer, primary_key=True),
    Column('user', String(128)),
    Column('role', String(16)),
    Column('message', Text),
    Column('created_at', DateTime, default=datetime.utcnow)
)
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# ---------------- Session State ----------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None


# ---------------- Crisis detection ----------------
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'i want to die', 'self harm', 'self-harm', 'hurt myself',
    'hang myself', 'cant go on', "can't go on", 'i am going to kill myself', 'i’m going to kill myself',
    'im going to kill myself'
]

def has_crisis(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


# ---------------- ADHD Knowledge Framework ----------------
ADHD_PRINCIPLES = {
    "evidence_based": [
        "ADHD involves differences in executive functioning that affect attention, working memory, and task initiation",
        "Medication combined with behavioral strategies is often most effective",
        "Structure and routine can reduce decision fatigue and improve follow-through"
    ],
    "strengths_focused": [
        "ADHD brains often excel at creativity and rapid problem-solving",
        "Hyperfocus can be channeled with clear start cues and time boxes",
        "Spontaneity and energy can be assets in dynamic environments"
    ],
    "practical_strategies": [
        "Break tasks into smaller, manageable steps",
        "Use external reminders to offload working memory",
        "Create consistent routines to reduce friction at start"
    ]
}

RESPONSE_FRAMEWORKS = {
    "understanding": [
        "What you’re describing lines up with how ADHD impacts executive functions. ",
        "Your experience is consistent with ADHD patterns around attention, memory, and initiation. ",
        "Many people with ADHD report similar challenges in day-to-day planning and follow-through. "
    ],
    "validation": [
        "It’s understandable this feels frustrating; these are real, common challenges. ",
        "You’re not alone in this—these difficulties are frequently reported with ADHD. ",
        "This is a recognized pattern; it’s not a personal failing. "
    ],
    "strategy": [
        "Here’s a small, structured plan you can try today: ",
        "Evidence-based tweaks that help in practice: ",
        "A practical, low-friction approach to test now: "
    ],
    "hope": [
        "Small repeats of what works add up quickly. ",
        "With a few consistent supports, day-to-day effort drops a lot. ",
        "These tweaks compound; many people see steady gains within weeks. "
    ]
}


# ---------------- Utility: text cleaning & chunking ----------------
def strip_kb_headers(text: str) -> str:
    """Remove KB metadata headers if present."""
    lines = text.split('\n')
    content_lines = []
    for line in lines:
        if line.startswith(('Source-URL:', 'Title:', 'Fetched-At:', 'Publish-Date:')):
            continue
        if line.strip() == '-' * 10:
            continue
        if line.strip():
            content_lines.append(line)
    cleaned = ' '.join(content_lines).strip()
    return cleaned if cleaned else text.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_CHAR_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = strip_kb_headers(text)
    if not t:
        return []
    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + chunk_size)
        # try not to split mid-sentence
        if end < n:
            period = t.rfind('.', start, end)
            if period != -1 and period > start + 200:
                end = period + 1
        chunks.append(t[start:end].strip())
        if end == n:
            break
        start = max(end - overlap, 0)
    return [c for c in chunks if c]


# ---------------- Knowledge Base Integration ----------------
@st.cache_resource(show_spinner=False)
def init_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def build_local_kb(_embed_model):
    """
    Load .txt files from kb/, chunk, embed, and cache to disk.
    Returns:
      docs: list of dicts: {"id", "text", "source"}
      embeddings: np.ndarray [N, D] L2-normalized
      cleaned_cache: dict id->cleaned_text (for previews)
    """
    docs = []
    cleaned_cache = {}

    # Gather and chunk all kb files
    for f in sorted(KB_DIR.glob("*.txt")):
        try:
            raw = f.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        chunks = chunk_text(raw)
        for i, ch in enumerate(chunks):
            doc_id = f"{f.name}::chunk_{i:04d}"
            docs.append({"id": doc_id, "text": ch, "source": f.name})
            cleaned_cache[doc_id] = ch

    # No docs? return empty arrays
    if len(docs) == 0:
        embeddings = np.zeros((0, embed_model.get_sentence_embedding_dimension()), dtype=np.float32)
        return docs, embeddings, cleaned_cache

    # Reuse on-disk cache only if ids match
    if EMB_FILE.exists() and DOCS_FILE.exists():
        try:
            with open(DOCS_FILE, "rb") as fh:
                saved_ids = pickle.load(fh)
            if len(saved_ids) == len(docs) and all(saved_ids[i] == docs[i]["id"] for i in range(len(docs))):
                embeddings = np.load(EMB_FILE)
                # load cleaned cache if present (non-fatal if missing)
                if CLEAN_FILE.exists():
                    with open(CLEAN_FILE, "rb") as fh:
                        saved_clean = pickle.load(fh)
                        cleaned_cache.update(saved_clean)
                return docs, embeddings, cleaned_cache
        except Exception:
            pass

    # Fresh encode
    texts = [d["text"] for d in docs]
    embeddings = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    # Persist
    np.save(EMB_FILE, embeddings)
    with open(DOCS_FILE, "wb") as fh:
        pickle.dump([d["id"] for d in docs], fh)
    with open(CLEAN_FILE, "wb") as fh:
        pickle.dump(cleaned_cache, fh)

    return docs, embeddings, cleaned_cache

embed_model = init_embed_model()
docs, doc_embeddings, cleaned_cache = build_local_kb(embed_model)

def retrieve_context_local(query: str, top_k: int = MAX_RETURN_DOCS) -> Tuple[List[str], List[str], List[float]]:
    if len(docs) == 0:
        return [], [], []
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    sims = doc_embeddings @ q_emb
    top_idx = np.argsort(-sims)[:top_k]
    top_texts = [docs[i]["text"] for i in top_idx]
    top_sources = [docs[i]["source"] for i in top_idx]
    top_scores = [float(sims[i]) for i in top_idx]
    return top_texts, top_sources, top_scores


# ---------------- Generative Response System ----------------
def extract_pull_quotes(retrieved_texts: List[str], max_quotes: int = 2) -> List[str]:
    """Short, readable evidence nuggets."""
    quotes = []
    for t in retrieved_texts:
        cleaned = strip_kb_headers(t)
        # split approx by sentence
        parts = re.split(r'(?<=[.!?])\s+', cleaned)
        for p in parts:
            p = p.strip()
            if 60 <= len(p) <= 220:   # keep tight and readable
                quotes.append(p)
                if len(quotes) >= max_quotes:
                    return quotes
    return quotes[:max_quotes]

def detect_themes(user_lower: str) -> List[str]:
    themes = []
    if any(w in user_lower for w in ['focus', 'concentrat', 'distract']):
        themes.append("attention and focus challenges")
    if any(w in user_lower for w in ['motivat', 'procrastinat', 'start', 'initiat']):
        themes.append("motivation and initiation difficulties")
    if any(w in user_lower for w in ['organiz', 'forget', 'lose', 'memory']):
        themes.append("organization and memory challenges")
    if any(w in user_lower for w in ['overwhelm', 'anxious', 'anxiety', 'stress', 'emotion']):
        themes.append("emotional regulation and overwhelm")
    if any(w in user_lower for w in ['time', 'late', 'schedule', 'plan']):
        themes.append("time management challenges")
    if not themes:
        themes.append("executive function challenges")
    return themes

def strategies_for(themes: List[str]) -> List[str]:
    s = []
    for t in themes:
        if "attention and focus" in t:
            s.extend([
                "Use 25-minute focus blocks with a 2-minute transition ritual (clear desk, open only one tab)",
                "Block obvious distractors (phone outside reach; desktop DND for 30 min)",
                "Start with a 60-second *warm-up*: write the task name and the first two bullet points"
            ])
        if "motivation and initiation" in t:
            s.extend([
                "Apply the 2-minute rule: do *only* the first micro-step, then reassess",
                "Pair a dull task with a small treat (music you like, a coffee) to reduce start friction",
                "Write a ‘why now’ sticky—1 sentence on the immediate benefit of doing the task"
            ])
        if "organization and memory" in t:
            s.extend([
                "Create a single capture bucket (one notes app or one notebook) and review it at a set time",
                "Use visual anchors: dedicated tray for keys/wallet; single download folder + daily triage",
                "Externalize reminders: two timed alarms for the same checkpoint (e.g., 9:00 plan / 9:05 start)"
            ])
        if "emotional regulation" in t or "overwhelm" in t:
            s.extend([
                "When overwhelmed: list only the next *one* subtask; everything else goes to a ‘later list’",
                "Box the work: 10 minutes max; if it still feels heavy, schedule a second 10-minute box later",
                "Physiological reset: 3 slow breaths (4-in, 6-out) before you start"
            ])
        if "time management" in t:
            s.extend([
                "Plan with two alarms: 09:00 five-line plan, 09:05 start first 25-minute block",
                "Estimate by halves: if unsure, assume twice as long and schedule buffers",
                "Calendar everything that exceeds 15 minutes (with a verb-first title)"
            ])
        if "executive function challenges" in t:
            s.extend([
                "Default morning routine: wake, hydrate, 5-line plan, first 25-minute block",
                "One-screen rule: only the doc you’re editing and one reference tab",
                "Close each session by queuing tomorrow’s first micro-step (one sentence)"
            ])
    return s

def pick_strategies_dedup(strategies: List[str], max_n: int = 3) -> List[str]:
    seen = set()
    ordered = []
    for s in strategies:
        k = s.lower().strip()
        if k not in seen:
            ordered.append(s)
            seen.add(k)
        if len(ordered) >= max_n:
            break
    return ordered

def render_actionables(items: List[str], detail: str) -> str:
    if not items:
        return ""
    if detail == "Short":
        return " • " + " | ".join(items)
    if detail == "Detailed":
        bullets = "\n".join([f"- {it}" for it in items])
        return "\n" + bullets
    # Medium
    bullets = "\n".join([f"- {it}" for it in items])
    return "\n" + bullets

def generate_conservative_response(
    user_text: str,
    retrieved_texts: List[str],
    tone_style: str = "Balanced",
    detail_level: str = "Medium"
) -> str:
    user_lower = user_text.lower()
    themes = detect_themes(user_lower)
    strategies = strategies_for(themes)
    selected = pick_strategies_dedup(strategies, 3)

    # Tone controls
    if tone_style == "Direct":
        intro = " ".join([RESPONSE_FRAMEWORKS["understanding"][0]])
        validate = ""
    elif tone_style == "Empathetic":
        intro = RESPONSE_FRAMEWORKS["understanding"][1]
        validate = RESPONSE_FRAMEWORKS["validation"][0]
    else:
        intro = RESPONSE_FRAMEWORKS["understanding"][2]
        validate = RESPONSE_FRAMEWORKS["validation"][1]

    plan_intro = RESPONSE_FRAMEWORKS["strategy"][0]
    hope = RESPONSE_FRAMEWORKS["hope"][0]

    # Evidence pull quotes (if any retrieved)
    quotes = extract_pull_quotes(retrieved_texts, max_quotes=2)

    parts = []
    parts.append(intro)
    parts.append(validate)
    parts.append(f"This often relates to {', '.join(themes)}. ")
    if quotes:
        parts.append("Here are brief evidence notes: ")
        for q in quotes:
            parts.append(f"“{q}” ")

    parts.append(plan_intro)
    parts.append(render_actionables(selected, detail_level))
    parts.append("\n")
    parts.append(hope)
    return "".join(parts).strip()


# ---------------- Persistence helpers ----------------
def save_message(user_name: str, role: str, message: str):
    try:
        stmt = conversations.insert().values(
            user=user_name,
            role=role,
            message=message,
            created_at=datetime.utcnow()
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        st.warning(f"Note: failed to persist a message to DB ({e}).")

    st.session_state.messages.append({
        "role": role,
        "content": message,
        "timestamp": datetime.utcnow()
    })

def load_conversation_history(limit: int = 20):
    try:
        rows = session.query(conversations).order_by(conversations.c.created_at.desc()).limit(limit).all()
        return [{"role": r.role, "content": r.message, "timestamp": r.created_at} for r in reversed(rows)]
    except Exception:
        return []


# ---------------- UI: Sidebar ----------------
with st.sidebar:
    st.header("💬 Chat Controls")

    user_name = st.text_input("Your name", value="User", key="user_name")

    tone_style = st.selectbox("Tone", ["Balanced", "Direct", "Empathetic"], index=0)
    detail_level = st.selectbox("Detail", ["Short", "Medium", "Detailed"], index=1)

    st.markdown("---")
    st.header("📚 Knowledge Base")
    st.info(f"Indexed {len(docs)} chunks")

    if st.button("🔄 Re-index Knowledge Base"):
        with st.spinner("Re-indexing..."):
            try:
                if EMB_FILE.exists():
                    EMB_FILE.unlink()
                if DOCS_FILE.exists():
                    DOCS_FILE.unlink()
                if CLEAN_FILE.exists():
                    CLEAN_FILE.unlink()
                # clear caches
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
            # rebuild
            try:
                _model = init_embed_model()
                _docs, _embs, _cleaned = build_local_kb(_model)
                st.success(f"✅ Re-indexed {len(_docs)} chunks")
            except Exception as e:
                st.error(f"Re-index failed: {e}")
            st.rerun()

    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------- UI: Conversation history ----------------
st.title("🧠 ADHD Assistant")
st.markdown("*Evidence-based ADHD support with a practical plan*")

st.subheader("💭 Conversation")
if not st.session_state.messages:
    st.session_state.messages = load_conversation_history()

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(f"**{user_name}:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.write(f"**Assistant:** {message['content']}")


# ---------------- Chat handling ----------------
def handle_prompt(prompt_text: str, *, render_references: bool = True):
    # Save + show user message
    save_message(user_name, "user", prompt_text)
    with st.chat_message("user"):
        st.write(f"**{user_name}:** {prompt_text}")

    # Crisis branch
    if has_crisis(prompt_text):
        with st.chat_message("assistant"):
            st.error("""
🚨 **Immediate Support Needed**

If you're in crisis or having thoughts of self-harm, please contact:
- **Emergency services:** 112 (EU) or your local emergency number
- **Crisis Text Line (US/CA/UK/IE):** Text HOME to 741741 / 85258 / 50808
- **Suicide & Crisis Lifeline (US):** 988

*This app provides educational support only and cannot provide emergency care.*
            """)
        save_message(user_name, "assistant", "Provided emergency contacts and guidance.")
        return

    # Normal branch
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consulting research..."):
            retrieved_texts, sources, scores = retrieve_context_local(prompt_text)
            response = generate_conservative_response(
                prompt_text,
                retrieved_texts,
                tone_style=tone_style,
                detail_level=detail_level
            )
        st.write(f"**Assistant:** {response}")

        # References block (only if strong enough matches)
        if render_references and retrieved_texts and any(s >= SIMILARITY_THRESHOLD for s in scores):
            with st.expander("📖 Research References"):
                for i, (text, source, score) in enumerate(zip(retrieved_texts, sources, scores), 1):
                    if score >= SIMILARITY_THRESHOLD:
                        preview = textwrap.shorten(strip_kb_headers(text), width=320, placeholder="…")
                        st.markdown(f"**Reference {i}** • Source: `{source}` • Relevance: {score:.2f}")
                        st.write(preview)
                        st.markdown("---")

    save_message(user_name, "assistant", response)


# Chat input (no prefill; Starters use direct send)
prompt = st.chat_input("Type your message here...")
if prompt:
    handle_prompt(prompt)

# Conversation starters (fixed: send immediately)
st.markdown("---")
st.subheader("🚀 Conversation Starters")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Focus issues", use_container_width=True):
        handle_prompt("I'm having trouble focusing on my work today")
with col2:
    if st.button("Feeling overwhelmed", use_container_width=True):
        handle_prompt("I feel completely overwhelmed by all my tasks")
with col3:
    if st.button("Organization help", use_container_width=True):
        handle_prompt("I need help getting organized with ADHD")

# Professional disclaimer
with st.expander("ℹ️ Important Disclaimer"):
    st.markdown("""
**Professional Medical Disclaimer**

This assistant provides educational information based on current ADHD research and clinical understanding.
It is not a substitute for professional medical advice, diagnosis, or treatment.

**Please note:**
- Responses are generated from available research literature and practical frameworks
- Always consult qualified healthcare professionals for personal medical advice
- Individual experiences with ADHD vary significantly
- Treatment should be personalized under professional guidance
    """)

st.caption("💡 Tip: Ask for a 5-line plan for your next task if you’re stuck.")
