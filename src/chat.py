import os
from transformers import pipeline
from src.chunking import load_and_chunk_documents  # optional if you rebuild vectorstore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")


# --- Load vectorstore ---
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return db


# --- Load small LLM using transformers pipeline directly ---
def load_llm():
    print("Loading LLM (this may take a minute)...")

    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device=-1  # CPU
    )

    return pipe  # return the raw pipeline object


# --- Chat loop ---
def main():
    print("Loading vector database...")
    db = load_vectorstore()

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = load_llm()

    print("\nRAG Chatbot ready. Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Use the correct method for your current LangChain version
        docs = retriever._get_relevant_documents(query, run_manager=None)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question: {query}

Answer:
"""

        # Directly call the transformers pipeline
        result = llm(prompt)
        response = result[0]["generated_text"]

        print("\nBot:", response, "\n")


if __name__ == "__main__":
    main()
