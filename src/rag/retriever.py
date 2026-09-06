"""Retrieval over the Qdrant collection.

Supports two embedding backends, selected by the EMBED_BACKEND env var:
- "gemini" (default): Gemini API embeddings, collection QDRANT_COLLECTION
- "local": offline sentence-transformers embeddings, collection QDRANT_COLLECTION_LOCAL

Use "local" while the Gemini free-tier indexing job (slow, rate-limited) is still
catching up, or to avoid API calls entirely.
"""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "gemini")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")

LOCAL_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_COLLECTION = os.environ.get("QDRANT_COLLECTION_LOCAL", "fastapi_corpus_local")
# BGE models are trained to embed queries with this instruction prefix; documents
# get no prefix (see embed_and_index_local.py).
LOCAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self) -> None:
        self.qdrant = QdrantClient(url=QDRANT_URL)
        if EMBED_BACKEND == "local":
            from sentence_transformers import SentenceTransformer

            self.collection = LOCAL_COLLECTION
            self.local_model = SentenceTransformer(LOCAL_MODEL_NAME)
        else:
            from google import genai
            from google.genai import types

            self.collection = GEMINI_COLLECTION
            self.genai = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY_EMBED", os.environ["GEMINI_API_KEY"])
            )
            self._embed_config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")

    def embed_query(self, query: str) -> list[float]:
        if EMBED_BACKEND == "local":
            vec = self.local_model.encode(LOCAL_QUERY_PREFIX + query, normalize_embeddings=True)
            return vec.tolist()
        resp = self.genai.models.embed_content(
            model=GEMINI_MODEL, contents=query, config=self._embed_config
        )
        return resp.embeddings[0].values

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.embed_query(query)
        hits = self.qdrant.query_points(
            collection_name=self.collection, query=vector, limit=top_k
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
