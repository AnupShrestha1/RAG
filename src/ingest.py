from src.chunking import load_and_chunk_documents
from src.vectorstore import build_vectorstore
from src.loader import load_dataset

def ingest_data():
    # Load dataset
    docs = load_dataset()  # from loader.py
    # Chunk documents
    chunks = load_and_chunk_documents(docs)
    # Build/update vectorstore
    build_vectorstore(chunks)
    print("Data ingested and vectorstore updated.")
