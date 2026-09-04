# Build Log — FastAPI Docs RAG + Eval Harness

Running log of decisions, steps, and issues for this project. Updated as we go.

## Goal

A resume-worthy RAG project that stands out by including a real evaluation
harness (metrics, golden dataset, tracing) instead of just a "chatbot over docs"
demo. See `README.md` for the pitch and setup instructions.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Domain/corpus | FastAPI docs + source (`tiangolo/fastapi`) | Technically differentiated (code-aware chunking, multi-file context); meta angle of a FastAPI app about FastAPI |
| Golden dataset source | GitHub Discussions (Q&A category, answered only) | Real (question, ground-truth-answer) pairs for free, instead of hand-written/synthetic Q&A |
| LLM stack | OpenAI `gpt-4o-mini` + `text-embedding-3-small` | Cheapest fast path to v1; best-supported by eval tooling |
| Vector DB | Qdrant, via Docker | Production-grade over an in-memory store; one step from being deployable |
| Eval framework | DeepEval | pytest/CI-friendly — "tested like software," not a one-off notebook |
| Tracing | Arize Phoenix, run via Docker (not pip) | See Issues below — the pip package conflicts with Python 3.11 |

## Steps completed

1. Scaffolded project structure (`src/ingest`, `src/index`, `src/rag`, `src/eval`, `src/api`)
2. `src/ingest/fetch_docs.py` — shallow-clones the FastAPI repo, stages `docs/en/docs/*.md` and `fastapi/*.py`
   - Ran successfully: 155 markdown docs, 48 python source files staged
3. `src/ingest/chunk.py` — markdown-aware (header-path-preserving) chunking for docs, `ast`-based function/class chunking for source
   - Ran successfully: 2,280 doc chunks + 552 code chunks = 2,832 total
4. `src/ingest/fetch_discussions.py` — GraphQL pull of answered GitHub Discussions as golden Q&A pairs (needs `GITHUB_TOKEN`) — **not yet run**
5. `src/index/embed_and_index.py` — embeds all chunks (`text-embedding-3-small`) and upserts into Qdrant — **not yet run** (needs `OPENAI_API_KEY`)
6. `src/eval/build_golden_dataset.py` — filters/formats fetched discussions into `data/processed/golden_dataset.json` — **not yet run**
7. `src/rag/{retriever,generator,pipeline}.py` — retrieve-then-generate pipeline — **not yet run end-to-end**
8. `src/eval/run_deepeval.py` — scores the pipeline against the golden set on Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall — **not yet run**
9. `src/eval/phoenix_tracing.py` — OTel + OpenInference instrumentation exporting traces to Phoenix — **not yet run**
10. `src/api/main.py` — FastAPI `/query` endpoint wrapping the pipeline
11. Docker Compose brings up Qdrant (`:6333`) and Phoenix (`:6006`) — both verified healthy
12. Git repo initialized, first commit made with the full scaffold

## Issues hit + resolutions

- **`arize-phoenix` pip package fails to import on Python 3.11.** `phoenix/__init__.py` (v20.8.0) pulls in code with a `dataclass` mutable-default pattern that only Python 3.13 allows; older versions (<8) require compiling `sqlean-py` from source (needs MSVC Build Tools, not installed); v12.35.0 installs but has a version-skew bug between core `arize-phoenix` and the separately-versioned `arize-phoenix-evals` package (`phoenix.evals.models` module missing).
  - **Fix:** dropped the `arize-phoenix` pip dependency entirely. Phoenix now runs as its own Docker container (`arizephoenix/phoenix:latest`, confirmed running Python 3.13 internally — validates the diagnosis). The app only needs the lightweight `opentelemetry-sdk` + `openinference-instrumentation-openai` packages to export traces to it over OTLP — no heavy/fragile phoenix package in the app's own environment.

## Current state

- Qdrant + Phoenix containers running and healthy
- All Python dependencies installed in `.venv`
- All modules import cleanly (verified via a standalone import smoke test)
- Corpus cloned and chunked
- **Blocked on:** `OPENAI_API_KEY` and `GITHUB_TOKEN` in `D:\project\.env` — nothing downstream of embedding/generation has been run yet

## Next steps

1. User fills in `.env` (`OPENAI_API_KEY`, `GITHUB_TOKEN`)
2. Fetch golden dataset from GitHub Discussions
3. Embed + index corpus into Qdrant
4. Build golden dataset file
5. Smoke-test the RAG pipeline on a sample question
6. Run the DeepEval harness, review scores
7. Iterate on chunking/retrieval based on what the metrics show
8. Wire up Phoenix tracing for a debugging demo
9. Stretch: CI workflow running `run_deepeval.py` as a regression gate; deploy the FastAPI app
