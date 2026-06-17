# System Architecture

## High-Level Overview

```mermaid
flowchart TB
    subgraph ingestion["INGESTION PIPELINE (offline, run once)"]
        direction TB
        URLS["urls.txt<br/>(CSUSB CSE pages)"] --> LOADER["WebBaseLoader<br/>+ SoupStrainer<br/>(main content only)"]
        DATA["data/<br/>(.txt / .pdf)"] --> LOADER
        LOADER --> CLEAN["Clean &amp; strip<br/>nav boilerplate"]
        CLEAN --> SPLIT["RecursiveCharacter<br/>TextSplitter<br/>(1000 / 150 overlap)"]
        SPLIT --> EMBED["Gemini Embeddings<br/>(gemini-embedding-001)<br/>batched, rate-limited"]
        EMBED --> FAISS[("FAISS<br/>vectorstore/")]
    end

    subgraph runtime["RUNTIME (per question)"]
        direction TB
        USER(["Student question"]) --> RETRIEVER["FAISS retriever<br/>(top-k = 6)"]
        FAISS -.loads.-> RETRIEVER
        RETRIEVER --> CONTEXT["Relevant chunks<br/>+ source URLs"]
        CONTEXT --> PROMPT["Prompt template<br/>(grounding + guardrails)"]
        PROMPT --> LLM["Gemini Chat<br/>(gemini-2.5-flash-lite)"]
        LLM --> ANSWER(["Grounded answer<br/>+ source pills"])
    end

    subgraph ui["INTERFACE"]
        STREAMLIT["Streamlit chat UI<br/>(app.py)"]
    end

    STREAMLIT --> USER
    ANSWER --> STREAMLIT
