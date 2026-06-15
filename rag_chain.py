
import os
 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
 
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"   # current free-tier chat model   # fast, free-tier friendly chat model
RETRIEVE_K = 4                    # how many chunks to pull per question
 
SYSTEM_PROMPT = """You are a helpful Academic Advisor assistant for the \
Computer Science and Engineering (CSE) graduate programs at CSUSB. Answer the \
student's question using ONLY the context provided below.
 
Rules:
- Base every answer strictly on the context. Do not invent courses, dates, \
policies, deadlines, or contact details.
- If the answer is not in the context, say: "I don't have that information in \
the CSE graduate materials I was given. Please check with the department \
office or your graduate advisor." Do not guess.
- If the question is unrelated to CSE graduate studies or academics, politely \
say that you can only help with CSE graduate academic questions.
- Be concise, friendly, and specific. Quote exact requirements, deadlines, and \
contact details when they appear in the context.
 
Context:
{context}
 
Student question: {question}
 
Answer:"""
 
 
def format_docs(docs):
    """Join retrieved chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
 
 
def build_chain():
    """Build and return (chain, retriever). Reused by the Streamlit app."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY not found. Add it to your .env file.")
 
    # 1. Load the saved vector store
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this file ourselves
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})
 
    # 2. The chat model
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2)
 
    # 3. The prompt
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
 
    # 4. Wire it together: retrieve -> stuff into prompt -> LLM -> text
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever
 
 
def main():
    print("Loading the Academic Advisor ... ", end="", flush=True)
    chain, retriever = build_chain()
    print("ready!\n")
    print("Ask me about CSUSB CSE graduate admissions, requirements,")
    print("deadlines, thesis process, or advising contacts.")
    print("Type 'quit' or 'exit' to stop.\n")
 
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
 
        answer = chain.invoke(question)
        print(f"\nAdvisor: {answer}\n")
 
        sources = {
            doc.metadata.get("source", "unknown")
            for doc in retriever.invoke(question)
        }
        print(f"  [sources: {', '.join(sources)}]\n")
 
 
if __name__ == "__main__":
    main()
 