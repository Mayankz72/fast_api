"""Retrieval over the Qdrant collection."""
import os

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")


class Retriever:
    def __init__(self) -> None:
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.qdrant = QdrantClient(url=QDRANT_URL)

    def embed_query(self, query: str) -> list[float]:
        resp = self.openai.embeddings.create(model=EMBED_MODEL, input=[query])
        return resp.data[0].embedding

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=COLLECTION, query=vector, limit=top_k
        ).points
        return [
            {
                "text": h.payload["text"],
                "source": h.payload["source"],
                "kind": h.payload["kind"],
                "heading_path": h.payload["heading_path"],
                "score": h.score,
            }
            for h in hits
        ]
