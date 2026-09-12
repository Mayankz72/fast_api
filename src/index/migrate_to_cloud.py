"""One-off migration: copy an existing local Qdrant collection's points to a
Qdrant Cloud cluster, for deploying the app somewhere the local Docker Qdrant
isn't reachable from. Scrolls + upserts the already-embedded points directly -
no re-embedding, so this costs no Gemini API quota.

Usage:
    python -m src.index.migrate_to_cloud --collection fastapi_corpus_contextual

Reads the destination cluster's URL/API key from QDRANT_CLOUD_URL and
QDRANT_CLOUD_API_KEY (kept separate from QDRANT_URL/QDRANT_API_KEY, which the
app itself reads at request time, so this script's source is always "whatever
local Qdrant docker-compose is running" regardless of what the app is
currently pointed at).
"""
import argparse
import os
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

LOCAL_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
CLOUD_URL = os.environ.get("QDRANT_CLOUD_URL")
CLOUD_API_KEY = os.environ.get("QDRANT_CLOUD_API_KEY")

BATCH_SIZE = 100
RETRIEVE_RETRIES = 8
BATCH_PACING_SECONDS = 0.5


def retrieve_with_backoff(source: QdrantClient, collection: str, ids: list[int]):
    """Both scroll()'s offset pagination and retrieve() hit the same server-side
    panic ("OffsetZero" unwrap - the same bug class as the RERANK_FETCH_K=20
    retrieval panics in run_deepeval.py, see PROGRESS.md/RESOURCES.md). Manual
    probing showed it's not tied to a specific corrupt point - the identical ID
    range succeeds or fails depending on batch size and which flags
    (with_payload/with_vectors) are combined, and got worse the more rapid-fire
    requests were fired at the container - consistent with local Docker Qdrant
    getting resource-starved (each point here carries a 3072-dim float vector,
    ~12KB) under back-to-back large-batch reads rather than one corrupt point.
    Retries with real backoff (not just a flat 2s) give the container room to
    recover; if a batch still won't go through, halve it and retry each half -
    this must terminate successfully at size 1 (a single point's data isn't
    inherently corrupt, confirmed by manual spot checks), so never silently
    drop points - just keep retrying smaller/slower."""
    for attempt in range(RETRIEVE_RETRIES + 1):
        try:
            return source.retrieve(
                collection_name=collection, ids=ids, with_payload=True, with_vectors=True
            )
        except UnexpectedResponse as e:
            if e.status_code != 500:
                raise
            if attempt < RETRIEVE_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            if len(ids) > 1:
                mid = len(ids) // 2
                print(f"  batch of {len(ids)} ({ids[0]}-{ids[-1]}) still failing after retries, splitting in half...")
                time.sleep(3)
                left = retrieve_with_backoff(source, collection, ids[:mid])
                time.sleep(3)
                right = retrieve_with_backoff(source, collection, ids[mid:])
                return left + right
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Collection name, same on both source and destination")
    args = parser.parse_args()

    if not CLOUD_URL or not CLOUD_API_KEY:
        raise SystemExit("Set QDRANT_CLOUD_URL and QDRANT_CLOUD_API_KEY in .env first.")

    source = QdrantClient(url=LOCAL_URL)
    dest = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY)

    if not source.collection_exists(args.collection):
        raise SystemExit(f"Source collection '{args.collection}' not found at {LOCAL_URL}")

    info = source.get_collection(args.collection)
    total = source.count(args.collection).count
    print(f"Source: {args.collection} ({total} points, dim={info.config.params.vectors.size})")

    if not dest.collection_exists(args.collection):
        dest.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(
                size=info.config.params.vectors.size,
                distance=info.config.params.vectors.distance,
            ),
        )
        print(f"Created destination collection '{args.collection}' on {CLOUD_URL}")

    already = dest.count(args.collection).count
    if already >= total:
        print(f"Destination already has {already}/{total} points - nothing to do.")
        return

    # Point IDs are the corpus chunk's absolute index (0..total-1) - see
    # embed_and_index.py. Batching over this known range (rather than
    # scroll()'s offset pagination) sidesteps the OffsetZero panic entirely.
    # upsert() is idempotent by ID, so re-sending already-migrated batches on a
    # rerun is harmless - not worth the extra complexity of tracking a resume
    # point for a collection this size.
    migrated = 0
    for start in range(0, total, BATCH_SIZE):
        batch_ids = list(range(start, min(start + BATCH_SIZE, total)))
        points = retrieve_with_backoff(source, args.collection, batch_ids)
        if not points:
            continue
        upsert_points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        dest.upsert(collection_name=args.collection, points=upsert_points)
        migrated += len(points)
        print(f"  migrated {migrated}/{total}")
        time.sleep(BATCH_PACING_SECONDS)

    final = dest.count(args.collection).count
    print(f"Done. Destination now has {final} points.")


if __name__ == "__main__":
    main()
