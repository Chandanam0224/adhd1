# app.py — ADHD Study & Support Assistant (with time-based chats + Difficulty Coach)

import os
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime, date
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, MetaData, Table
)
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import textwrap
import re
from typing import List, Tuple, Optional


# ---------------- Config ----------------
st.set_page_config(
    page_title='ADHD Study & Support Assistant',
    layout='wide',
    page_icon='🧠'
)

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
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}
)
metadata = MetaData()

conversations = Table(
    'conversations', metadata,
    Column('id', Integer, primary_key=True),
    Column('user', String(128)),
    Column('role', String(16)),
    Column('message', Text),
    Column('created_at', DateTime, default=datetime.utcnow)
)

tasks = Table(
    'tasks', metadata,
    Column('id', Integer, primary_key=True),
    Column('task_text', Text, nullable=False),
    Column('due_date', String(32)),
    Column('is_done', Integer, default=0)
)

checkins = Table(
    'checkins', metadata,
    Column('id', Integer, primary_key=True),
    Column('day', String(32), nullable=False),
    Column('mood', Integer),
    Column('focus', Integer),
    Column('note', Text),
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

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "role": "Student",
        "main_struggle": "Procrastination",
        "focus_hours": "Evening",
    }

for key in ["messages_morning", "messages_study", "messages_night", "messages_difficulty"]:
    if key not in st.session_state:
        st.session_state[key] = []


def get_user_profile() -> dict:
    return st.session_state.user_profile


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

# ---------------- Difficulty Coach Templates ----------------
DIFFICULTY_TEMPLATES = {
    "inattention": """It makes sense that this is exhausting for you. Difficulties like losing focus, forgetting tasks, losing items, or zoning out in conversations are commonly linked to how ADHD affects attention and working memory, not to laziness or lack of care.

**Small experiments you can try:**
- Pick *only one* task and define the first 2–3 minutes very clearly (for example: “Open notebook, write the heading, list 3 sub-points”).
- Use a **single capture system** (one notes app or one notebook) for all tasks, and review it once at a fixed time every day.
- Keep your workspace “just clear enough”: only what belongs to the current task on the main surface.

These difficulties say more about how your brain is wired than about your character. If inattention has been affecting your studies, work, or relationships for a long time, it can help to talk to a mental health professional who understands ADHD. I’m not a doctor and I can’t diagnose you, but I can help you break things into manageable steps.""",

    "hyperactivity": """Feeling restless, fidgety, or “always on” can be really uncomfortable, especially when you’re expected to sit still or be calm. In ADHD, this can show up as mental restlessness as much as physical movement.

**Things you can test:**
- Give your body **a job**: a fidget object, stress ball, ring, or doodling while listening so some of the extra energy has a place to go.
- Schedule **movement breaks** on purpose: 3–5 minutes of walking, stretching, or pacing between focused blocks.
- When your mind races at rest, try a **“brain dump”**: write down everything in your head for 3–5 minutes, then pick just one small thing to act on or consciously park everything for tomorrow.

None of this makes you “too much” as a person—it’s your nervous system trying to regulate. If the restlessness is very intense or affects your sleep, work, or health, a professional can help you explore options. I’m here for practical strategies, not to replace medical care.""",

    "impulsivity": """Acting quickly, interrupting, or reacting strongly before thinking doesn’t mean you’re a bad person—it often reflects how ADHD affects impulse control and emotional brakes.

**Skills you can experiment with:**
- Use a **2-second pause rule** in conversations: silently count “1–2” before jumping in. If you still want to speak, start with “Can I add something?”.
- For non-urgent decisions, create a **“24-hour rule”**: write the idea down, and promise yourself you’ll revisit it tomorrow before acting.
- When emotions spike, practice **“name it to tame it”**: “Right now I feel … because …”. Putting it into words engages the thinking part of your brain.

If impulsivity is creating problems in relationships, money, or safety, it’s worth discussing with a therapist or doctor. I can’t tell you what diagnosis you have, but we can practice safer habits around your impulses.""",

    "emotional_ef": """Mood swings, overthinking, time-blindness, procrastination, hyperfocus, and feeling “mentally tired from being normal” are common with ADHD and other executive-function differences. They’re not proof that you’re weak—they’re signs that your brain is doing a lot of invisible work.

**Possible supports:**
- For **time-blindness**, rely on *external time*: timers, alarms, or visual clocks. Assume “I’ll remember” is not reliable, and that’s okay.
- For **procrastination**, ask: “What is the *smallest visible version* of this task?” and do just that tiny version.
- For **hyperfocus**, set alarms before you start and agree that when it rings twice you will stand up once, even for 30 seconds.

If these patterns are long-term and painful, an ADHD-informed professional can help you design supports around your brain, not against it. I’m here to break things down with you, but not to diagnose or treat.""",

    "social": """Many people with ADHD describe feeling “too much”, misunderstood, or out of sync in conversations. That doesn’t mean you’re a bad friend or awkward on purpose.

**Social strategies to try:**
- Use **check-in questions**: “Am I making sense?” or “Am I talking too much?” so people can guide you instead of you guessing.
- Before changing topics, **tag it**: “This reminds me of something similar…” instead of jumping suddenly.
- If you worry about oversharing, first write the full version in your notes, then say a shorter, safer version out loud.

Everyone has different social energy and style. If you often leave conversations feeling ashamed or confused, social-skills or ADHD-informed counselling can help. I can help you plan and rehearse conversations, but not formally evaluate you.""",

    "academic": """Struggling with deadlines, organization, or consistency despite being intelligent is a very classic ADHD pattern. It’s usually about planning and execution, not about ability.

**Supports you can put in place:**
- Turn “finish project” into very small calendar blocks: “30 min – outline intro”, “20 min – clean figures”, “15 min – check references”.
- Use one **master task list** and one calendar instead of many scattered lists; review them briefly at the same time every day.
- For verbal instructions, immediately write down key steps and read them back: “So I’ll do A, then B, then C—is that right?”.

If feedback says “capable but inconsistent”, that can be a sign of executive-function challenges. A professional can help with accommodations or treatment options. I can’t decide if you have ADHD, but we can build systems around how you actually work."""
}


def generate_difficulty_coach_reply(difficulty_key: str, user_text: str) -> str:
    base = DIFFICULTY_TEMPLATES.get(difficulty_key)
    if not base:
        return "I’m not sure which area this falls under, but we can still look for small, ADHD-friendly steps together."

    user_part = user_text.strip()
    if user_part:
        intro = f"You said: “{user_part}”. Thanks for putting that into words.\n\n"
    else:
        intro = ""
    return intro + base


# ---------------- Utility: text cleaning & chunking ----------------
def strip_kb_headers(text: str) -> str:
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
    docs = []
    cleaned_cache = {}

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

    if len(docs) == 0:
        dim = _embed_model.get_sentence_embedding_dimension()
        embeddings = np.zeros((0, dim), dtype=np.float32)
        return docs, embeddings, cleaned_cache

    if EMB_FILE.exists() and DOCS_FILE.exists():
        try:
            with open(DOCS_FILE, "rb") as fh:
                saved_ids = pickle.load(fh)
            if len(saved_ids) == len(docs) and all(saved_ids[i] == docs[i]["id"] for i in range(len(docs))):
                embeddings = np.load(EMB_FILE)
                if CLEAN_FILE.exists():
                    with open(CLEAN_FILE, "rb") as fh:
                        saved_clean = pickle.load(fh)
                        cleaned_cache.update(saved_clean)
                return docs, embeddings, cleaned_cache
        except Exception:
            pass

    texts = [d["text"] for d in docs]
    embeddings = _embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

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
    quotes = []
    for t in retrieved_texts:
        cleaned = strip_kb_headers(t)
        parts = re.split(r'(?<=[.!?])\s+', cleaned)
        for p in parts:
            p = p.strip()
            if 60 <= len(p) <= 220:
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
                "Use 25-minute focus blocks with a 2-minute transition ritual (clear desk, open only one tab).",
                "Block obvious distractors (phone outside reach; desktop DND for 30 min).",
                "Start with a 60-second warm-up: write the task name and the first two bullet points."
            ])
        if "motivation and initiation" in t:
            s.extend([
                "Apply the 2-minute rule: do only the first micro-step, then reassess.",
                "Pair a dull task with a small treat (music you like, a coffee) to reduce start friction.",
                "Write a “why now” sticky—one sentence on the immediate benefit of doing the task."
            ])
        if "organization and memory" in t:
            s.extend([
                "Create a single capture bucket (one notes app or one notebook) and review it at a set time.",
                "Use visual anchors: a dedicated tray for keys/wallet; a single download folder + daily triage.",
                "Externalize reminders: two timed alarms for the same checkpoint (for example, 9:00 plan / 9:05 start)."
            ])
        if "emotional regulation" in t or "overwhelm" in t:
            s.extend([
                "When overwhelmed: list only the next one subtask; everything else goes to a “later list”.",
                "Box the work: 10 minutes max; if it still feels heavy, schedule a second 10-minute box later.",
                "Try a short physiological reset: 3 slow breaths (4-in, 6-out) before you start."
            ])
        if "time management" in t:
            s.extend([
                "Plan with two alarms: 09:00 five-line plan, 09:05 start the first 25-minute block.",
                "Estimate by halves: if unsure, assume twice as long and schedule buffers.",
                "Calendar everything that exceeds 15 minutes with a verb-first title."
            ])
        if "executive function challenges" in t:
            s.extend([
                "Default morning routine: wake, hydrate, 5-line plan, first 25-minute block.",
                "One-screen rule: only the doc you’re editing and one reference tab.",
                "Close each session by queuing tomorrow’s first micro-step in one sentence."
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
    bullets = "\n".join([f"- {it}" for it in items])
    return "\n" + bullets


def generate_conservative_response(
    user_text: str,
    retrieved_texts: List[str],
    tone_style: str = "Balanced",
    detail_level: str = "Medium",
    user_profile: Optional[dict] = None,
    session_type: str = "general",
) -> str:
    user_lower = user_text.lower()
    themes = detect_themes(user_lower)
    strategies = strategies_for(themes)
    selected = pick_strategies_dedup(strategies, 3)

    if tone_style == "Direct":
        understanding = RESPONSE_FRAMEWORKS["understanding"][0]
        validation = ""
    elif tone_style == "Empathetic":
        understanding = RESPONSE_FRAMEWORKS["understanding"][1]
        validation = RESPONSE_FRAMEWORKS["validation"][0]
    else:
        understanding = RESPONSE_FRAMEWORKS["understanding"][2]
        validation = RESPONSE_FRAMEWORKS["validation"][1]

    plan_intro = RESPONSE_FRAMEWORKS["strategy"][0]
    hope = RESPONSE_FRAMEWORKS["hope"][0]

    profile_snippet = ""
    if user_profile:
        role = user_profile.get("role") or "person"
        struggle = user_profile.get("main_struggle") or "executive function challenges"
        focus_time = user_profile.get("focus_hours") or "whenever energy is steady"
        profile_snippet = (
            f"As a {role.lower()} mainly struggling with {struggle.lower()}, "
            f"it can help to schedule experiments around your usual focus window "
            f"({focus_time.lower()}). "
        )

    if session_type == "morning":
        session_opening = (
            "Because this is a *morning focus* check-in, "
            "let's choose one gentle, low-friction task that gets your brain moving "
            "instead of expecting full productivity immediately."
        )
    elif session_type == "study":
        session_opening = (
            "For this *study block*, we'll aim for one clearly defined task and a short time box "
            "so that getting started feels lighter."
        )
    elif session_type == "night":
        session_opening = (
            "Since you're in a *night wind-down* space, let's focus on closing loops and being kind "
            "to your future self rather than pushing heavy work."
        )
    else:
        session_opening = (
            "We'll keep things small and concrete so changes feel realistic, not overwhelming."
        )

    quotes = extract_pull_quotes(retrieved_texts, max_quotes=2)

    parts = []
    themes_text = ", ".join(themes)
    parts.append(
        "### 1. What I'm hearing\n"
        f"{understanding}{validation}"
        f"This often relates to **{themes_text}**.\n"
    )

    action_text = render_actionables(selected, detail_level)
    parts.append(
        "### 2. A tiny next step for now\n"
        f"{session_opening}\n\n"
        f"{plan_intro}{action_text}\n"
    )

    parts.append(
        "### 3. Longer-term support\n"
        f"{profile_snippet}"
        f"{hope}"
        " This is not a diagnosis or medical advice—it's ADHD-friendly support based on patterns that many people report.\n"
    )

    if quotes:
        parts.append("### Evidence snapshot (from my knowledge base)\n")
        for q in quotes:
            parts.append(f"- “{q}”\n")

    return "\n".join(parts).strip()


# ---------------- Persistence helpers ----------------
def save_message(user_name: str, role: str, message: str, messages_key: str):
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

    if messages_key not in st.session_state:
        st.session_state[messages_key] = []
    st.session_state[messages_key].append({
        "role": role,
        "content": message,
        "timestamp": datetime.utcnow()
    })


def load_conversation_history(limit: int = 20):
    try:
        rows = session.query(conversations).order_by(
            conversations.c.created_at.desc()
        ).limit(limit).all()
        return [
            {"role": r.role, "content": r.message, "timestamp": r.created_at}
            for r in reversed(rows)
        ]
    except Exception:
        return []


# ---------------- Tasks & Reflections helpers ----------------
def add_task(task_text: str, due_date: date):
    try:
        stmt = tasks.insert().values(
            task_text=task_text,
            due_date=due_date.isoformat(),
            is_done=0
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        st.warning(f"Could not save task: {e}")


def get_tasks():
    try:
        rows = session.query(tasks).order_by(
            tasks.c.is_done.asc(),
            tasks.c.due_date.asc()
        ).all()
        return rows
    except Exception:
        return []


def set_task_done(task_id: int, done: bool):
    try:
        stmt = (
            tasks.update()
            .where(tasks.c.id == task_id)
            .values(is_done=int(done))
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        st.warning(f"Could not update task: {e}")


def add_checkin(day_str: str, mood: int, focus: int, note: str):
    try:
        stmt = checkins.insert().values(
            day=day_str,
            mood=mood,
            focus=focus,
            note=note,
            created_at=datetime.utcnow()
        )
        session.execute(stmt)
        session.commit()
    except Exception as e:
        st.warning(f"Could not save reflection: {e}")


def get_recent_checkins(limit: int = 7):
    try:
        rows = session.query(checkins).order_by(
            checkins.c.created_at.desc()
        ).limit(limit).all()
        return rows
    except Exception:
        return []


# ---------------- Chat handling ----------------
def handle_prompt(
    prompt_text: str,
    user_name: str,
    tone_style: str,
    detail_level: str,
    session_type: str,
    messages_key: str,
    *,
    render_references: bool = True
):
    save_message(user_name, "user", prompt_text, messages_key)
    with st.chat_message("user"):
        st.markdown(f"**{user_name}:** {prompt_text}")

    if has_crisis(prompt_text):
        with st.chat_message("assistant"):
            st.error(
                """🚨 **Immediate Support Needed**

If you're in crisis or having thoughts of self-harm, please contact:
- **Emergency services:** your local emergency number
- **Trusted people:** a close friend, family member, or mentor
- **Mental health professional:** as soon as you can

This app provides educational support only and cannot provide emergency care.
"""
            )
        save_message(user_name, "assistant", "Provided emergency guidance.", messages_key)
        return

    user_profile = get_user_profile()

    with st.chat_message("assistant"):
        with st.spinner("🔍 Consulting ADHD-friendly strategies..."):
            retrieved_texts, sources, scores = retrieve_context_local(prompt_text)
            response = generate_conservative_response(
                prompt_text,
                retrieved_texts,
                tone_style=tone_style,
                detail_level=detail_level,
                user_profile=user_profile,
                session_type=session_type,
            )
        st.markdown(f"**Assistant:**\n\n{response}")

        if render_references and retrieved_texts and any(s >= SIMILARITY_THRESHOLD for s in scores):
            with st.expander("📖 Research references used in this reply"):
                for i, (text, source, score) in enumerate(zip(retrieved_texts, sources, scores), 1):
                    if score >= SIMILARITY_THRESHOLD:
                        preview = textwrap.shorten(
                            strip_kb_headers(text),
                            width=320,
                            placeholder="…"
                        )
                        st.markdown(f"**Reference {i}** • Source: `{source}` • Relevance: {score:.2f}")
                        st.write(preview)
                        st.markdown("---")

    save_message(user_name, "assistant", response, messages_key)


def handle_difficulty_chat(
    difficulty_key: str,
    user_desc: str,
    user_name: str,
    messages_key: str,
):
    user_text = user_desc or f"I want help with {difficulty_key} difficulties."
    save_message(user_name, "user", user_text, messages_key)
    with st.chat_message("user"):
        st.markdown(f"**{user_name}:** {user_text}")

    reply = generate_difficulty_coach_reply(difficulty_key, user_desc)
    with st.chat_message("assistant"):
        st.markdown(f"**Assistant:**\n\n{reply}")
    save_message(user_name, "assistant", reply, messages_key)


# ---------------- UI: Sidebar ----------------
with st.sidebar:
    st.header("🧠 ADHD Study & Support")

    st.subheader("🧩 Your ADHD Profile")
    profile = get_user_profile()
    profile["name"] = st.text_input("Name (optional)", value=profile["name"])
    profile["role"] = st.selectbox(
        "You are mainly a...",
        ["Student", "Working professional", "Both", "Other"],
        index=["Student", "Working professional", "Both", "Other"].index(profile["role"])
        if profile["role"] in ["Student", "Working professional", "Both", "Other"] else 0,
    )
    profile["main_struggle"] = st.selectbox(
        "Your main struggle",
        ["Procrastination", "Time blindness", "Distraction", "Emotional overwhelm", "All of the above"],
        index=0 if profile["main_struggle"] not in [
            "Procrastination", "Time blindness", "Distraction",
            "Emotional overwhelm", "All of the above"
        ] else ["Procrastination", "Time blindness", "Distraction",
                "Emotional overwhelm", "All of the above"].index(profile["main_struggle"])
    )
    profile["focus_hours"] = st.selectbox(
        "You usually focus better in the...",
        ["Morning", "Afternoon", "Evening", "Late night"],
        index=["Morning", "Afternoon", "Evening", "Late night"].index(profile["focus_hours"])
        if profile["focus_hours"] in ["Morning", "Afternoon", "Evening", "Late night"] else 2,
    )

    st.markdown("---")
    st.header("💬 Chat Controls")

    user_name = st.text_input("Display name", value=profile["name"] or "User", key="user_name_input")
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
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Failed to clear cache: {e}")
            try:
                _model = init_embed_model()
                _docs, _embs, _cleaned = build_local_kb(_model)
                st.success(f"✅ Re-indexed {len(_docs)} chunks")
            except Exception as e:
                st.error(f"Re-index failed: {e}")
            st.rerun()

    if st.button("🗑️ Clear All Chats"):
        for key in ["messages_morning", "messages_study", "messages_night", "messages_difficulty"]:
            st.session_state[key] = []
        st.rerun()


# ---------------- UI: Main Layout (Tabs) ----------------
st.title("🧠 ADHD Study & Support Assistant")
st.caption(
    "Helps with focus, planning, routines and ADHD-friendly strategies. "
    "Not a doctor or therapist; for support and information only."
)

tab_morning, tab_study, tab_night, tab_difficulty, tab_planner, tab_reflection = st.tabs(
    ["🌅 Morning Focus", "📚 Study Session", "🌙 Night Wind-down", "🧩 Difficulty Coach", "📅 Planner", "📝 Reflection"]
)

# ----- Morning Focus Chat -----
with tab_morning:
    messages_key = "messages_morning"
    st.subheader("🌅 Morning Focus Chat")
    st.markdown("Use this space to gently start your day and decide the *first* tiny step.")

    for message in st.session_state[messages_key]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{user_name or 'User'}:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:**\n\n{message['content']}")

    prompt = st.chat_input("What’s the first thing on your mind this morning?")
    if prompt:
        handle_prompt(
            prompt,
            user_name=user_name or "User",
            tone_style=tone_style,
            detail_level=detail_level,
            session_type="morning",
            messages_key=messages_key,
        )

    st.markdown("---")
    st.subheader("🚀 Morning Starters")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Plan my morning", use_container_width=True, key="morning_plan"):
            handle_prompt(
                "Help me plan a simple ADHD-friendly morning routine for today.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="morning",
                messages_key=messages_key,
            )
    with col2:
        if st.button("I woke up late", use_container_width=True, key="morning_late"):
            handle_prompt(
                "I woke up late and feel guilty and behind. Help me reset realistically.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="morning",
                messages_key=messages_key,
            )
    with col3:
        if st.button("Low energy", use_container_width=True, key="morning_energy"):
            handle_prompt(
                "My energy is low this morning but I still have work. Help me pick one tiny step.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="morning",
                messages_key=messages_key,
            )

# ----- Study Session Chat -----
with tab_study:
    messages_key = "messages_study"
    st.subheader("📚 Study Session Chat")
    st.markdown("Use this for focused blocks: assignments, test prep, project work, or reading.")

    for message in st.session_state[messages_key]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{user_name or 'User'}:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:**\n\n{message['content']}")

    prompt = st.chat_input("What are you trying to study or work on right now?")
    if prompt:
        handle_prompt(
            prompt,
            user_name=user_name or "User",
            tone_style=tone_style,
            detail_level=detail_level,
            session_type="study",
            messages_key=messages_key,
        )

    st.markdown("---")
    st.subheader("🎯 Study Starters")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Can't start studying", use_container_width=True, key="study_start"):
            handle_prompt(
                "I know I need to study but I cannot get myself to start.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="study",
                messages_key=messages_key,
            )
    with col2:
        if st.button("Too many subjects", use_container_width=True, key="study_subjects"):
            handle_prompt(
                "I have too many subjects and feel overwhelmed. Help me prioritize and plan one block.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="study",
                messages_key=messages_key,
            )
    with col3:
        if st.button("Phone distractions", use_container_width=True, key="study_phone"):
            handle_prompt(
                "I keep picking up my phone while studying. Help me create an ADHD-friendly study setup.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="study",
                messages_key=messages_key,
            )

# ----- Night Wind-down Chat -----
with tab_night:
    messages_key = "messages_night"
    st.subheader("🌙 Night Wind-down Chat")
    st.markdown("Use this to close the day, lower the pressure, and be kind to your future self.")

    for message in st.session_state[messages_key]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{user_name or 'User'}:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:**\n\n{message['content']}")

    prompt = st.chat_input("How did today go, honestly?")
    if prompt:
        handle_prompt(
            prompt,
            user_name=user_name or "User",
            tone_style=tone_style,
            detail_level=detail_level,
            session_type="night",
            messages_key=messages_key,
        )

    st.markdown("---")
    st.subheader("🌙 Night Starters")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("I feel unproductive", use_container_width=True, key="night_unproductive"):
            handle_prompt(
                "The day is ending and I feel like I did nothing useful.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="night",
                messages_key=messages_key,
            )
    with col2:
        if st.button("Guilty about procrastination", use_container_width=True, key="night_guilt"):
            handle_prompt(
                "I procrastinated a lot today and feel guilty and anxious about tomorrow.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="night",
                messages_key=messages_key,
            )
    with col3:
        if st.button("Plan tomorrow gently", use_container_width=True, key="night_tomorrow"):
            handle_prompt(
                "Help me close today and plan a gentle, realistic start for tomorrow.",
                user_name=user_name or "User",
                tone_style=tone_style,
                detail_level=detail_level,
                session_type="night",
                messages_key=messages_key,
            )

# ----- Difficulty Coach -----
with tab_difficulty:
    messages_key = "messages_difficulty"
    st.subheader("🧩 Difficulty Coach")
    st.markdown("Pick an area you’re struggling with and tell me what it looks like for you.")

    for message in st.session_state[messages_key]:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**{user_name or 'User'}:** {message['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:**\n\n{message['content']}")

    difficulty = st.selectbox(
        "Which area feels most relevant right now?",
        [
            "Inattention-related difficulties",
            "Hyperactivity-related difficulties",
            "Impulsivity-related difficulties",
            "Emotional & executive-function challenges",
            "Social difficulties",
            "Academic/work difficulties",
        ]
    )

    user_desc = st.text_area(
        "Describe what this looks like for you (optional but helpful):",
        placeholder="Example: I keep forgetting assignments and zoning out in class..."
    )

    if st.button("Get support", use_container_width=True):
        key_map = {
            "Inattention-related difficulties": "inattention",
            "Hyperactivity-related difficulties": "hyperactivity",
            "Impulsivity-related difficulties": "impulsivity",
            "Emotional & executive-function challenges": "emotional_ef",
            "Social difficulties": "social",
            "Academic/work difficulties": "academic",
        }
        d_key = key_map.get(difficulty, "inattention")
        handle_difficulty_chat(
            difficulty_key=d_key,
            user_desc=user_desc,
            user_name=user_name or "User",
            messages_key=messages_key,
        )

# ----- Planner tab -----
with tab_planner:
    st.subheader("📅 ADHD-Friendly Task Planner")
    st.markdown("Focus on *tiny, concrete* tasks. Let this be your external brain for today.")

    with st.form("add_task_form", clear_on_submit=True):
        task_text = st.text_input(
            "Task (keep it small and specific)",
            placeholder="e.g., Read 3 pages of OS unit-2"
        )
        due = st.date_input("Target date", value=date.today())
        submitted = st.form_submit_button("Add Task")

        if submitted and task_text.strip():
            add_task(task_text.strip(), due)
            st.success("Task added 🎯")

    rows = get_tasks()
    if not rows:
        st.info("No tasks yet. Add one above! Keep them *tiny* and ADHD-friendly.")
    else:
        st.markdown("### Your tasks")
        for row in rows:
            cols = st.columns([0.1, 0.6, 0.3])
            row_id = row.id
            is_done = bool(row.is_done)
            with cols[0]:
                done = st.checkbox(
                    "",
                    value=is_done,
                    key=f"task_done_{row_id}"
                )
            with cols[1]:
                st.write(row.task_text)
            with cols[2]:
                st.write(f"Due: {row.due_date or '-'}")

            if done != is_done:
                set_task_done(row_id, done)

# ----- Reflection tab -----
with tab_reflection:
    st.subheader("📝 Daily Reflection")

    today_str = date.today().isoformat()
    st.markdown("Quick check-in to notice patterns over time.")

    mood = st.slider("How is your mood today?", min_value=1, max_value=5, value=3)
    focus_score = st.slider("How was your focus today?", min_value=1, max_value=5, value=3)
    note = st.text_area(
        "Anything you want to reflect on?",
        placeholder="What went well? What was hard? Any small win you want to remember?"
    )

    if st.button("Save today's reflection"):
        add_checkin(today_str, int(mood), int(focus_score), note.strip())
        st.success("Saved. Nice job checking in with yourself 💚")

    st.markdown("### Last 7 days")
    checkins_rows = get_recent_checkins(limit=7)
    if not checkins_rows:
        st.info("No reflections yet.")
    else:
        for c in checkins_rows:
            st.markdown(
                f"**{c.day}** – Mood: {c.mood or '-'} /5, Focus: {c.focus or '-'} /5  \n"
                f"{(c.note or '_No note_')}"
            )

# ----- Disclaimer -----
with st.expander("ℹ️ Important Disclaimer"):
    st.markdown(
        """**Professional Medical Disclaimer**

This assistant provides educational information based on current ADHD research and clinical understanding.
It is not a substitute for professional medical advice, diagnosis, or treatment.

**Please note:**
- Responses are generated from available research literature and practical frameworks
- Always consult qualified healthcare professionals for personal medical advice
- Individual experiences with ADHD vary significantly
- Treatment should be personalized under professional guidance
"""
    )

st.caption("💡 Tip: Ask for a 5-line plan for your next task if you’re stuck.")
