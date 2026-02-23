from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

emb = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

vs = Chroma(
    collection_name=os.getenv("CHROMA_COLLECTION", "isaw_articles"),
    embedding_function=emb,
    persist_directory=os.getenv("CHROMA_DIR", "chroma"),
)

print("count:", vs._collection.count())

# IDs are returned by default; don't request "ids" in include
sample = vs._collection.get(limit=1, include=["metadatas", "documents"])
print("sample id:", sample["ids"][0])
print("sample metadata keys:", sorted(sample["metadatas"][0].keys()))
print("sample text head:", sample["documents"][0][:200])
