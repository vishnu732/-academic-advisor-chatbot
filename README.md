# 🎓 CSE Graduate Advisor Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers Computer Science &
Engineering graduate-program questions — admissions, requirements, deadlines,
the thesis process, and advising contacts — grounded entirely in official
CSUSB department web pages.

Built with Python, LangChain, FAISS, Google Gemini, and Streamlit.

---

## Why this project

University department websites scatter the answers students actually need
across dozens of pages. This chatbot ingests those pages once, indexes them
semantically, and lets a student ask a plain-English question and get a
**grounded, source-cited answer** — without hunting through navigation menus.

Crucially, it is built to **not hallucinate**: every answer is constrained to
retrieved source material, and out-of-scope questions are politely declined.

---

## Features

- **Retrieval-Augmented Generation** — answers are grounded in real CSUSB CSE
  pages, not the model's training data.
- **Source citations** — every answer shows clickable pills linking to the exact
  pages it drew from, so users can verify.
- **Guardrails** — refuses to answer questions outside CSE graduate academics,
  and says "I don't have that information" rather than guessing.
- **Clean chat UI** — student/advisor message alignment, starter questions, and
  a one-click conversation reset.
- **Robust ingestion** — strips navigation boilerplate (SoupStrainer),
  batches embeddings to respect API rate limits, and skips unreachable pages
  instead of crashing.

---

## Tech Stack

| Layer            | Technology                          |
|------------------|-------------------------------------|
| Language         | Python 3.10+                        |
| LLM              | Google Gemini (`gemini-2.5-flash-lite`) |
| Embeddings       | Google Gemini (`gemini-embedding-001`)  |
| RAG framework    | LangChain                           |
| Vector store     | FAISS (local, on-disk)              |
| Web scraping     | WebBaseLoader + BeautifulSoup       |
| Frontend         | Streamlit                           |

---

## Architecture

The system has two phases: an **offline ingestion pipeline** that builds the
knowledge base once, and a **runtime loop** that answers questions against it.
See [`docs/architecture.md`](docs/architecture.md) for the full diagram.

```
urls.txt / data ──► load + clean ──► chunk ──► embed ──► FAISS vectorstore
                                                              │
student question ──► retrieve top-k ──► prompt + guardrails ──► Gemini ──► answer + sources
```

---

## Getting Started

### Prerequisites
- Python 3.10 or newer
- A free [Google Gemini API key](https://aistudio.google.com/apikey)

### 1. Clone and set up
```bash
git clone https://github.com/vishnu732/academic-advisor-chatbot.git
cd academic-advisor-chatbot
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API key
Create a file named `.env` in the project root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Build the knowledge base
```bash
python ingest.py
```
This fetches the pages in `urls.txt`, chunks and embeds them, and saves a FAISS
index to `vectorstore/`. (Already built? You can skip this step.)

### 4. Run the app
```bash
streamlit run app.py
```
Open the local URL Streamlit prints (usually http://localhost:8501).

> Prefer the terminal? Run `python rag_chain.py` for a command-line chat.

---

## Project Structure

```
academic-advisor-chatbot/
├── app.py               # Streamlit chat interface
├── rag_chain.py         # RAG chain: retriever + prompt + Gemini (reused by app)
├── ingest.py            # Ingestion pipeline: load → clean → chunk → embed → FAISS
├── urls.txt             # Source pages that form the knowledge base
├── requirements.txt     # Python dependencies
├── data/                # Optional local .txt / .pdf sources
├── vectorstore/         # Generated FAISS index (index.faiss + index.pkl)
├── docs/
│   ├── architecture.md  # System architecture diagram
│   └── technical-report.md
├── .env                 # Your API key (never committed)
└── .gitignore
```

---

## Customizing for a Different Department

This is built for CSUSB CSE graduate advising, but it generalizes:
1. Replace the URLs in `urls.txt` with your department's pages.
2. Adjust the prompt persona in `rag_chain.py` (`SYSTEM_PROMPT`).
3. Re-run `python ingest.py` to rebuild the knowledge base.

---

## Engineering Notes

A few real problems solved along the way (detailed in the
[technical report](docs/technical-report.md)):

- **Rate limits** — embeddings are batched with pauses and exponential-backoff
  retries to stay within the Gemini free tier.
- **Navigation noise** — early versions retrieved page menus instead of content;
  fixed by extracting only the `<main>` region with `SoupStrainer`, which
  dramatically improved answer quality.
- **Retrieval recall** — tuned `top-k` from 4 to 6 so relevant-but-lower-ranked
  chunks (like advising contacts) reliably make the context window.

---

## Limitations & Future Work

- **Stateless** — each question is answered independently; no multi-turn memory
  yet. ("Tell me more about that" won't resolve the previous topic.)
- **Snapshot knowledge** — the index reflects the pages at ingest time; re-run
  `ingest.py` to refresh after the website changes.
- **Possible extensions** — conversation memory, automatic periodic re-ingestion,
  hybrid keyword+semantic retrieval, and cloud deployment for public access.

---

## License

MIT — see `LICENSE`.

## Acknowledgements

Knowledge base sourced from publicly available
[CSUSB School of Computer Science & Engineering](https://www.csusb.edu/cse) and
[Graduate Studies](https://www.csusb.edu/graduate-studies) pages.
