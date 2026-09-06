"""Embed all chunks and load them into Qdrant.

Resumable: progress is checkpointed to CHECKPOINT_PATH after every successful batch,
so re-running after a rate-limit/quota error picks up where it left off instead of
re-embedding (and re-spending free-tier quota on) chunks already indexed.

Batches are sized by character budget, not a fixed chunk count: the free tier's real
constraint (confirmed empirically) is ~1000 *tokens* per minute for this model, not
request count, and chunk lengths vary a lot (some are 20 chars, some 2000+), so a
fixed-count batch can randomly land on several long chunks and blow the budget. At
this rate, indexing the full corpus takes several hours - that's the accepted
tradeoff for a $0 pipeline (see PROGRESS.md).
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.genai.types import HttpOptions, HttpRetryOptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.chunk import chunk_corpus  # noqa: E402

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
CHAR_BUDGET_PER_BATCH = 2400  # ~600 tokens, extra margin under the ~1000 tokens/min cap
REQUEST_PACING_SECONDS = 75.0
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 70

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "fastapi_corpus")
CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "embed_checkpoint.json"


def get_clients() -> tuple[genai.Client, QdrantClient]:
    # attempts=1 disables the SDK's own silent internal retry-on-429, so every
    # HTTP request we make is visible and accounted for in our own pacing/backoff.
    genai_client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY_EMBED", os.environ["GEMINI_API_KEY"]),
        http_options=HttpOptions(retry_options=HttpRetryOptions(attempts=1)),
    )
    qdrant_client = QdrantClient(url=QDRANT_URL)
    return genai_client, qdrant_client


def load_checkpoint() -> int:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))["next_index"]
    return 0


def save_checkpoint(next_index: int) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"next_index": next_index}), encoding="utf-8")


def ensure_collection(qdrant: QdrantClient, fresh_start: bool) -> None:
    exists = qdrant.collection_exists(COLLECTION)
    if fresh_start and exists:
        qdrant.delete_collection(COLLECTION)
        exists = False
    if not exists:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def make_batches(chunks: list, start_index: int) -> list[tuple[int, list]]:
    """Group chunks[start_index:] into (start_offset, batch) pairs, each batch
    capped at CHAR_BUDGET_PER_BATCH total characters (at least one chunk per batch,
    even if that single chunk alone exceeds the budget)."""
    batches = []
    i = start_index
    n = len(chunks)
    while i < n:
        batch = [chunks[i]]
        total_chars = len(chunks[i].text)
        j = i + 1
        while j < n and total_chars + len(chunks[j].text) <= CHAR_BUDGET_PER_BATCH:
            batch.append(chunks[j])
            total_chars += len(chunks[j].text)
            j += 1
        batches.append((i, batch))
        i = j
    return batches


def embed_batch(genai_client: genai.Client, texts: list[str]) -> list[list[float]]:
    """One request = one batchEmbedContents call. On 429, sleep past the quota
    window and retry a few times rather than hammering it with exponential backoff
    (which just burns more of the same per-minute quota)."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            resp = genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in resp.embeddings]
        except ClientError as e:
            if e.code == 429 and attempt < RATE_LIMIT_RETRIES:
                print(f"\nRate limited, sleeping {RATE_LIMIT_BACKOFF_SECONDS}s before retry...")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    chunks = chunk_corpus(root / "docs", root / "code")
    print(f"Loaded {len(chunks)} chunks to index")

    start_index = load_checkpoint()
    genai_client, qdrant_client = get_clients()
    ensure_collection(qdrant_client, fresh_start=(start_index == 0))

    if start_index:
        print(f"Resuming from chunk {start_index} (checkpoint found)")

    batches = make_batches(chunks, start_index)

    try:
        for offset, batch in tqdm(batches, desc="Embedding + upserting"):
            vectors = embed_batch(genai_client, [c.text for c in batch])
            points = [
                PointStruct(
                    id=offset + j,
                    vector=vec,
                    payload={
                        "text": c.text,
                        "source": c.source,
                        "kind": c.kind,
                        "heading_path": c.heading_path,
                    },
                )
                for j, (c, vec) in enumerate(zip(batch, vectors))
            ]
            qdrant_client.upsert(collection_name=COLLECTION, points=points)
            save_checkpoint(offset + len(batch))
            time.sleep(REQUEST_PACING_SECONDS)
    except ClientError as e:
        if e.code == 429:
            print(
                f"\nStill rate-limited after {RATE_LIMIT_RETRIES} retries. "
                f"Progress is checkpointed — just re-run this script later to resume."
            )
            sys.exit(1)
        raise

    count = qdrant_client.count(COLLECTION).count
    print(f"Indexed {count} points into Qdrant collection '{COLLECTION}'")
    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
