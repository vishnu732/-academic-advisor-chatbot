# Technical Report: CSE Graduate Advisor Chatbot

## 1. Problem Statement

Prospective and current graduate students in Computer Science routinely need
answers to procedural questions — admission requirements, application deadlines,
degree requirements, the thesis process, and who to contact for advising. This
information exists on the university website but is spread across many pages and
buried under navigation, making it tedious to find.

The goal of this project was to build a chatbot that answers these questions in
natural language, grounded strictly in official department sources, while
refusing to answer questions it has no basis for — i.e., a system that is
**helpful but does not hallucinate**.

## 2. Approach: Retrieval-Augmented Generation

A plain large language model cannot reliably answer institution-specific
questions: the information may post-date its training, and it has no way to cite
sources. Retrieval-Augmented Generation (RAG) addresses both issues by
retrieving relevant source passages at query time and instructing the model to
answer only from them.

The pipeline has two phases:

**Ingestion (offline, run once):** source pages are fetched, cleaned, split into
overlapping chunks, embedded into vectors, and stored in a FAISS index on disk.

**Runtime (per question):** the question is embedded, the most similar chunks are
retrieved from FAISS, and those chunks plus the question are passed to the LLM
inside a prompt that enforces grounding and scope guardrails.

## 3. Implementation

### 3.1 Knowledge Base
The knowledge base is defined declaratively in `urls.txt` — a list of CSUSB CSE
and Graduate Studies pages covering admissions, requirements, deadlines, forms,
the thesis process, and advising contacts. Local `.txt`/`.pdf` files in `data/`
are also supported as supplementary sources.

### 3.2 Document Processing
- **Loading:** `WebBaseLoader` fetches each URL individually with browser-like
  headers (some university servers reject default bot user-agents).
- **Content extraction:** a `SoupStrainer` restricts parsing to the `<main>` and
  `<article>` regions, discarding navigation, header, and footer markup. A
  line-level cleaner removes any residual menu artifacts. This step proved
  essential to answer quality (see Section 4).
- **Chunking:** `RecursiveCharacterTextSplitter` with a 1000-character chunk size
  and 150-character overlap. The overlap prevents facts from being severed at
  chunk boundaries.

### 3.3 Embeddings and Vector Store
Chunks are embedded with Google's `gemini-embedding-001` model and indexed in
FAISS, which is saved to `vectorstore/` as `index.faiss` and `index.pkl`. FAISS
was chosen for being local, fast, dependency-light, and free — appropriate for a
read-mostly knowledge base of this size.

### 3.4 RAG Chain
`rag_chain.py` composes the retriever, a prompt template, and the chat model
(`gemini-2.5-flash-lite`) using LangChain's expression language. The prompt:
- instructs the model to answer only from retrieved context,
- supplies an explicit fallback ("I don't have that information...") for missing
  answers, and
- restricts scope to CSE graduate academics.

A low temperature (0.2) keeps answers factual and consistent. The chain is
exposed through a single `build_chain()` function reused by both the terminal
interface and the web UI, so retrieval logic is never duplicated.

### 3.5 Interface
`app.py` is a Streamlit chat application that imports `build_chain()`. It renders
aligned chat bubbles (student right, advisor left), shows clickable source pills
beneath each answer, offers starter questions, and caches the loaded chain across
reruns with `@st.cache_resource`.

## 4. Engineering Challenges and Solutions

### 4.1 API Rate Limiting
The Gemini free tier limits embedding requests per minute and per day. Naively
embedding the full corpus at once triggered `429 RESOURCE_EXHAUSTED` errors. The
ingestion script was restructured to embed in batches of 80 with 60-second
pauses and exponential-backoff retries, allowing large corpora to be processed
within free-tier limits.

### 4.2 Navigation Boilerplate Polluting Retrieval
The most impactful problem: early versions returned generic or "no information"
answers even when the information was present. Investigation showed each fetched
page contained roughly 200 lines of navigation menus before any real content, so
chunks were dominated by menu links and the relevant text ranked poorly. Scoping
extraction to the `<main>` content region with `SoupStrainer` removed this noise,
shrank documents substantially, and immediately improved retrieval — the advising
query went from a failed lookup to a precise, correctly sourced answer.

### 4.3 Retrieval Recall
With `top-k = 4`, some relevant chunks (e.g., advising contacts) were occasionally
crowded out. Increasing `top-k` to 6 improved recall of these
relevant-but-lower-ranked passages without introducing noticeable noise.

### 4.4 Model Deprecation
An initial chat model (`gemini-2.0-flash`) returned a `limit: 0` quota error after
being retired from the free tier. Migrating to `gemini-2.5-flash-lite` — the
current, generously-provisioned free-tier model — resolved it. This reinforced a
practical lesson: model availability is a moving target and should be treated as
a configuration value, not a hard-coded assumption.

## 5. Results

The final system reliably answers in-scope questions (admissions, requirements,
deadlines, advising pathways) with accurate, source-cited responses, and
correctly declines out-of-scope questions. Each answer surfaces the exact source
pages, making responses verifiable.

## 6. Limitations and Future Work

- **No conversation memory.** Each query is independent; follow-up questions that
  depend on prior turns ("who is that?") are not resolved. Adding a
  history-aware retriever would address this.
- **Static snapshot.** The index reflects the source pages at ingest time;
  refreshing requires re-running ingestion. A scheduled re-ingestion job would
  keep it current.
- **Retrieval strategy.** Pure semantic retrieval could be strengthened with a
  hybrid keyword+semantic approach or a reranking step for higher precision.
- **Access.** The app currently runs locally. Deploying to a hosting platform
  would make it publicly accessible without local setup.

## 7. Conclusion

This project demonstrates an end-to-end RAG system: sourcing and cleaning real
web data, building a semantic index within practical API constraints, grounding
an LLM to prevent hallucination, and presenting results through a verifiable,
user-friendly interface. The most valuable engineering insight was that
**retrieval quality is dominated by input quality** — cleaning the source data
mattered more than any model or parameter change.
