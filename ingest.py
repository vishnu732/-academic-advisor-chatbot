
import os
import re
import time
 
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/124.0 Safari/537.36")
 
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
DATA_DIR = "data"
URLS_FILE = "urls.txt"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "models/gemini-embedding-001"
 
BATCH_SIZE = 80
PAUSE_BETWEEN_BATCHES = 60
MAX_RETRIES = 5
 
# Browser-like headers so servers (like the CSUSB catalog) don't reject us
BROWSER_HEADERS = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
 
SMOKE_TEST_QUESTION = "What are the requirements for the MS in Computer Science?"
 
 
def load_local_files():
    txt_docs = DirectoryLoader(
        DATA_DIR, glob="**/*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    ).load()
    pdf_docs = DirectoryLoader(
        DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader,
    ).load()
    return txt_docs + pdf_docs
 
 
def load_web_pages():
    if not os.path.exists(URLS_FILE):
        return []
    with open(URLS_FILE, encoding="utf-8") as f:
        urls = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    if not urls:
        return []
 
    print(f"  Fetching {len(urls)} web page(s) from {URLS_FILE} ...")
    docs = []
    for url in urls:
        loaded = None
        for attempt in range(1, 4):  # up to 3 tries per URL
            try:
                loader = WebBaseLoader(
                    url,
                    header_template=BROWSER_HEADERS,
                    requests_kwargs={"timeout": 30},
                )
                loaded = loader.load()
                break
            except Exception as e:
                if attempt < 3:
                    time.sleep(3 * attempt)
                else:
                    print(f"    SKIPPED {url}  ({type(e).__name__})")
        if not loaded:
            continue
        for doc in loaded:
            doc.page_content = re.sub(r"\n{3,}", "\n\n", doc.page_content)
            doc.page_content = re.sub(r"[ \t]{2,}", " ", doc.page_content).strip()
            size = len(doc.page_content)
            flag = "   <-- nearly empty: page likely needs JavaScript" if size < 500 else ""
            print(f"    {url}: {size} characters{flag}")
            docs.append(doc)
    return docs
 
 
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
 
 
def embed_in_batches(chunks, embeddings):
    vectorstore = None
    total = len(chunks)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{num_batches} ({len(batch)} chunks) ...", flush=True)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait = 30 * attempt
                    print(f"    Rate limited. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES}) ...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise SystemExit("Still rate limited after retries. Try again in a few minutes.")
        if batch_num < num_batches:
            print(f"  Pausing {PAUSE_BETWEEN_BATCHES}s to respect the rate limit ...")
            time.sleep(PAUSE_BETWEEN_BATCHES)
    return vectorstore
 
 
def main():
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not found. Add it to your .env file.")
 
    print("Loading knowledge base sources ...")
    documents = load_local_files() + load_web_pages()
    if not documents:
        raise SystemExit("No sources loaded. Check urls.txt / data/ and your connection.")
    print(f"  Loaded {len(documents)} document(s) in total.")
 
    print("Splitting documents into chunks ...")
    chunks = split_documents(documents)
    print(f"  Created {len(chunks)} chunks.")
 
    est = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Embedding in batches of {BATCH_SIZE} (roughly {est} minute(s)) ...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = embed_in_batches(chunks, embeddings)
 
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to {VECTORSTORE_DIR}/")
 
    print("\n--- Smoke test ---")
    results = vectorstore.similarity_search(SMOKE_TEST_QUESTION, k=2)
    print(f"Question: {SMOKE_TEST_QUESTION}")
    print(f"Top matching chunk (from {results[0].metadata.get('source', 'unknown')}):\n")
    print(results[0].page_content[:400])
    print("\nIngestion complete. The knowledge base is ready for the RAG chain.")
 
 
if __name__ == "__main__":
    main()