import os

# Try all known import paths
try:
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise RuntimeError("LangChain text splitter not installed correctly")

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def load_and_chunk_documents(chunk_size=300, chunk_overlap=50):
    documents = []

    files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".txt")]
    print(f"Found {len(files)} articles")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for file in files:
        path = os.path.join(RAW_DATA_PATH, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)

        for chunk in chunks:
            documents.append(
                Document(page_content=chunk, metadata={"source": file})
            )

    print(f"Total chunks: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_and_chunk_documents()
    print("Sample chunk:\n", docs[0].page_content[:500])
