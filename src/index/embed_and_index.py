"""Embed all chunks and load them into Qdrant."""
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import chunk_corpus  # noqa: E402

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")


def get_clients() -> tuple[OpenAI, QdrantClient]:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return openai_client, qdrant_client


def ensure_collection(qdrant: QdrantClient) -> None:
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def embed_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    chunks = chunk_corpus(root / "docs", root / "code")
    print(f"Loaded {len(chunks)} chunks to index")

    openai_client, qdrant_client = get_clients()
    ensure_collection(qdrant_client)

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding + upserting"):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = embed_batch(openai_client, [c.text for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": c.text,
                    "source": c.source,
                    "kind": c.kind,
                    "heading_path": c.heading_path,
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        qdrant_client.upsert(collection_name=COLLECTION, points=points)

    count = qdrant_client.count(COLLECTION).count
    print(f"Indexed {count} points into Qdrant collection '{COLLECTION}'")


if __name__ == "__main__":
    main()
