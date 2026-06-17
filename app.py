
import streamlit as st
 
from rag_chain import build_chain
 
# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="CSE Graduate Advisor",
    page_icon="🎓",
    layout="centered",
)
 
# ----------------------------------------------------------------------
# Styling: CSUSB blue banner header + chat bubbles
# (student = right/blue, advisor = left/grey)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root { --csusb-blue: #00509e; --csusb-dark: #0a2240; }
      .main .block-container { max-width: 820px; padding-top: 1.5rem; }
 
      /* Blue banner header with WHITE text */
      .advisor-header {
        background: linear-gradient(135deg, var(--csusb-blue), var(--csusb-dark));
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(10, 34, 64, 0.18);
      }
      .advisor-header h1 { font-size: 1.7rem; margin: 0; color: #ffffff; }
      .advisor-header p  { margin: 0.4rem 0 0; color: #dce8f6; font-size: 0.95rem; }
 
      /* --- Chat bubble alignment --- */
      /* A row wraps each turn so we can push it left or right */
      .chat-row { display: flex; margin: 0.35rem 0; }
      .chat-row.user      { justify-content: flex-end; }
      .chat-row.assistant { justify-content: flex-start; }
 
      .bubble {
        max-width: 78%;
        padding: 0.7rem 1rem;
        border-radius: 16px;
        line-height: 1.5;
        font-size: 0.97rem;
        word-wrap: break-word;
      }
      .bubble.user {
        background: var(--csusb-blue);
        color: #ffffff;
        border-bottom-right-radius: 4px;
      }
      .bubble.assistant {
        background: #f0f2f6;
        color: #1a1a1a;
        border-bottom-left-radius: 4px;
      }
 
      /* Source pills sit under the advisor bubble, left aligned */
      .sources-row { display: flex; justify-content: flex-start; margin: 0 0 0.5rem; }
      .sources-inner { max-width: 78%; }
      .source-pill {
        display: inline-block; font-size: 0.72rem; color: #000000; font-weight: 700;
        background: #eef3f9; border: 1px solid #d6e2f0;
        border-radius: 999px; padding: 2px 10px; margin: 4px 4px 0 0;
        text-decoration: none;
      }
      .source-pill:hover { background: #dce8f6; }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ----------------------------------------------------------------------
# Load the RAG chain once and cache it across reruns
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the advisor's knowledge base...")
def load_chain():
    return build_chain()
 
 
def short_source(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1].replace("-", " ")
    return tail[:40] if tail else url
 
 
def render_turn(role: str, content: str, sources=None):
    """Render one chat turn as an aligned bubble (+ optional source pills)."""
    safe = content.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div class="chat-row {role}"><div class="bubble {role}">{safe}</div></div>',
        unsafe_allow_html=True,
    )
    if sources:
        pills = "".join(
            f'<a class="source-pill" href="{u}" target="_blank">{short_source(u)}</a>'
            for u in sources
        )
        st.markdown(
            f'<div class="sources-row"><div class="sources-inner">{pills}</div></div>',
            unsafe_allow_html=True,
        )
 
 
# ----------------------------------------------------------------------
# Header (blue banner, white text)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="advisor-header">
      <h1>🎓 CSE Graduate Advisor</h1>
      <p>Ask about CSUSB Computer Science master's admissions, requirements,
      deadlines, the thesis process, and advising contacts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
chain, retriever = load_chain()
 
# ----------------------------------------------------------------------
# Conversation state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm the CSE Graduate Advisor. Ask me about the MS in "
                       "Computer Science — admissions, deadlines, requirements, or "
                       "the thesis process.",
            "sources": [],
        }
    ]
 
# ----------------------------------------------------------------------
# Sidebar: starter questions + reset
# ----------------------------------------------------------------------
STARTERS = [
    "What are the admission requirements for the MS in Computer Science?",
    "What are the application deadlines?",
    "Who do I contact for graduate advising?",
    "What is the thesis review process?",
]
 
with st.sidebar:
    st.subheader("Try asking")
    clicked = None
    for q in STARTERS:
        if st.button(q, use_container_width=True):
            clicked = q
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = st.session_state.messages[:1]
        st.rerun()
    st.caption("Answers come only from official CSUSB pages. Always confirm "
               "important decisions with your advisor.")
 
# ----------------------------------------------------------------------
# Render conversation so far
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    render_turn(msg["role"], msg["content"], msg.get("sources"))
 
# ----------------------------------------------------------------------
# Handle input (typed question or clicked starter)
# ----------------------------------------------------------------------
typed = st.chat_input("Ask a question about CSE graduate studies...")
question = typed or clicked
 
if question:
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    render_turn("user", question)
 
    with st.spinner("Looking through the CSE materials..."):
        try:
            answer = chain.invoke(question)
            sources = sorted({
                doc.metadata.get("source", "")
                for doc in retriever.invoke(question)
                if doc.metadata.get("source")
            })
        except Exception as e:
            answer = (
                "Something went wrong reaching the model. This is often a "
                "temporary rate limit — wait a moment and try again."
            )
            sources = []
            st.error(f"Details: {e}")
 
    render_turn("assistant", answer, sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
 