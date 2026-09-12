# FastAPI Docs RAG — with an evaluation harness

A RAG system that answers questions about FastAPI (docs + source code), with an
automated evaluation harness measuring retrieval and generation quality against
a golden dataset built from real, answered GitHub Discussions.

**Stack:** Qdrant (vector store) · Google Gemini (`gemini-embedding-001` embeddings +
`gemini-3.5-flash-lite` generation, free tier) · DeepEval (metrics/CI) ·
Arize Phoenix (tracing) · FastAPI (serving)

See [`PROGRESS.md`](PROGRESS.md) for the build log and [`RESOURCES.md`](RESOURCES.md)
for the research papers and techniques this design is based on.

## Try the live demo

**https://fastapi-docs-rag.onrender.com** redirects to an interactive Swagger UI -
expand **POST /query**, click **"Try it out"**, edit the request body, click
**"Execute"**. Or from a terminal:

```bash
curl -X POST https://fastapi-docs-rag.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add a query parameter with a default value?"}'
```

Deployed on Render's free tier (spins down after inactivity - the first request
after a lull takes ~30-60s to wake up) against a Qdrant Cloud free-tier cluster
holding the Contextual-Retrieval-indexed corpus (`fastapi_corpus_contextual`,
see the ablation in `RESOURCES.md`). `GET /health` for a quick liveness check
that doesn't spend any Gemini API quota.

## Why this project

Most portfolio RAG projects are "chatbot over a PDF" with no way to know if it
actually works. This one adds the piece that's usually missing: a golden
evaluation set and automated metrics (faithfulness, answer relevancy, contextual
precision/recall) so retrieval and generation quality can be measured and improved
with actual numbers, not vibes.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # fill in GEMINI_API_KEY and (optionally) GITHUB_TOKEN
docker compose up -d          # starts Qdrant on localhost:6333
```

## Pipeline

```bash
# 1. Pull the corpus (docs + source) and the Q&A golden-dataset source
python src/ingest/fetch_docs.py
python src/ingest/fetch_discussions.py      # needs GITHUB_TOKEN in .env

# 2. Chunk + embed + index into Qdrant
python src/index/embed_and_index.py

# 3. Build the golden eval set from fetched discussions
python src/eval/build_golden_dataset.py

# 4. Ask a question
python src/rag/pipeline.py "How do I define a path parameter with a type?"

# 5. Run the eval harness (scores 30 golden examples by default)
python src/eval/run_deepeval.py 30

# 6. (optional) Trace calls in Phoenix while querying
python src/eval/phoenix_tracing.py    # UI at http://localhost:6006

# 7. Serve the API
uvicorn src.api.main:app --reload
```

## Deployment

The live demo above runs as a Docker container on Render, reading from a Qdrant
Cloud cluster instead of local Docker Qdrant. To redeploy elsewhere:

```bash
# 1. Copy an already-indexed local collection to Qdrant Cloud (no re-embedding)
#    - set QDRANT_CLOUD_URL / QDRANT_CLOUD_API_KEY in .env first
python -m src.index.migrate_to_cloud --collection fastapi_corpus_contextual

# 2. Build and smoke-test the image locally
docker build -t fastapi-rag .
docker run -p 7860:7860 --env-file .env \
  -e QDRANT_URL=$QDRANT_CLOUD_URL -e QDRANT_API_KEY=$QDRANT_CLOUD_API_KEY \
  fastapi-rag

# 3. Push to a host that builds from the Dockerfile (Render, Fly.io, etc.),
#    setting GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION,
#    and EMBED_BACKEND=gemini as that host's environment variables/secrets.
```

The Dockerfile binds to `$PORT` if set (Render's convention), falling back to
7860 (Hugging Face Spaces' convention) otherwise - adjust for other hosts.

## Design decisions worth calling out (for the writeup)

- **Chunking is content-aware, not fixed-size.** Markdown docs are split on
  headers (keeping the heading path as context); Python source is split on
  `ast` function/class boundaries so every code chunk is syntactically complete.
- **Golden dataset is real, not synthetic.** FastAPI routes support questions
  through GitHub Discussions (Q&A category) with a marked accepted answer —
  those (question, accepted-answer) pairs are the ground truth, avoiding the
  bias of LLM-generated eval questions.
- **Eval is automated, not a one-off notebook run.** `run_deepeval.py` can be
  wired into CI to catch retrieval/generation regressions when the chunking
  strategy, prompt, or model changes.
