import os

# Robust imports for langchain
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from src.chunking import load_and_chunk_documents

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma")


def build_vectorstore():
    print("Loading and chunking documents...")
    documents = load_and_chunk_documents()

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building Chroma vector database...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    db.persist()
    print(f"Vector DB saved to: {CHROMA_DIR}")
    print(f"Total vectors: {db._collection.count()}")


if __name__ == "__main__":
    build_vectorstore()
