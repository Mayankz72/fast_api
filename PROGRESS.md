# Build Log — FastAPI Docs RAG + Eval Harness

Running log of decisions, steps, and issues for this project. Updated as we go.
See `RESOURCES.md` for the research papers and engineering write-ups backing these decisions.

## Goal

A resume-worthy RAG project that stands out by including a real evaluation
harness (metrics, golden dataset, tracing) instead of just a "chatbot over docs"
demo. See `README.md` for the pitch and setup instructions.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Domain/corpus | FastAPI docs + source (`tiangolo/fastapi`) | Technically differentiated (code-aware chunking, multi-file context); meta angle of a FastAPI app about FastAPI |
| Golden dataset source | GitHub Discussions (Q&A category, answered only) | Real (question, ground-truth-answer) pairs for free, instead of hand-written/synthetic Q&A |
| LLM stack | ~~OpenAI `gpt-4o-mini` + `text-embedding-3-small`~~ → Gemini `gemini-3.5-flash-lite` (generation) + `gemini-embedding-001` (embeddings), judged by `gemini-3.1-flash-lite` in DeepEval | User has no budget for OpenAI billing; Google AI Studio's free tier covers this at $0. See Issues below for the free tier's real throughput constraints. |
| API keys | Two separate Google accounts' keys: `GEMINI_API_KEY` (generation + eval judge) and `GEMINI_API_KEY_EMBED` (embeddings only) | Each free-tier account/project gets its own independent quota buckets. Splitting embedding and generation traffic across two accounts means the (very tight) embedding throughput doesn't compete with the (separately tight) generation daily cap - see Issues. User's own accounts, disposed of after the project. |
| Embedding backend (dev) | Local `BAAI/bge-small-en-v1.5` via `sentence-transformers`, separate Qdrant collection `fastapi_corpus_local` | Gemini's free embedding tier is too slow to iterate against (see Issues) — local embeddings give an immediately-usable pipeline for development/demo, while the Gemini-embedded collection remains the "official" $0-API-stack version once it finishes indexing. Switch via `EMBED_BACKEND=local\|gemini` in `.env`. |
| Vector DB | Qdrant, via Docker | Production-grade over an in-memory store; one step from being deployable |
| Eval framework | DeepEval | pytest/CI-friendly — "tested like software," not a one-off notebook |
| Tracing | Arize Phoenix, run via Docker (not pip) | See Issues below — the pip package conflicts with Python 3.11 |

## Steps completed

1. Scaffolded project structure (`src/ingest`, `src/index`, `src/rag`, `src/eval`, `src/api`)
2. `src/ingest/fetch_docs.py` — shallow-clones the FastAPI repo, stages `docs/en/docs/*.md` and `fastapi/*.py`
   - Ran successfully: 155 markdown docs, 48 python source files staged
3. `src/ingest/chunk.py` — markdown-aware (header-path-preserving) chunking for docs, `ast`-based function/class chunking for source
   - Ran successfully: 2,280 doc chunks + 552 code chunks = 2,832 total
4. `src/ingest/fetch_discussions.py` — GraphQL pull of answered GitHub Discussions as golden Q&A pairs (needs `GITHUB_TOKEN`) — **not yet run** (GitHub PATs are free — create one at github.com/settings/tokens with `public_repo`/read scope, no billing involved)
5. `src/index/embed_and_index.py` — embeds all chunks via Gemini (`gemini-embedding-001`, 3072-dim) and upserts into Qdrant collection `fastapi_corpus` — **running in the background, will take hours** (see Issues)
   - `src/index/embed_and_index_local.py` — same corpus, embedded locally via `BAAI/bge-small-en-v1.5` (384-dim) into collection `fastapi_corpus_local` — **ran successfully, all 2,832 chunks indexed in ~6 min**
6. `src/eval/build_golden_dataset.py` — filters/formats fetched discussions into `data/processed/golden_dataset.json` — **not yet run** (blocked on step 4)
7. `src/rag/{retriever,generator,pipeline}.py` — retrieve-then-generate pipeline, `EMBED_BACKEND` env var picks local vs. Gemini retrieval — **smoke-tested successfully end-to-end against the local backend** (correct, grounded, sourced answer for "How do I define a path parameter with a type?")
8. `src/eval/run_deepeval.py` — scores the pipeline against the golden set on Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, judged by `gemini-3.1-flash-lite` via DeepEval's native `GeminiModel` — **ran successfully, first real scores in.** Runs in small resumable batches (default 2/call) that append to `data/processed/eval_report.json` rather than overwrite, skipping already-scored questions — see Issues below for why.
   - **First 2/437 results:** Faithfulness and Answer Relevancy both 1.00 on both examples (generation stays grounded, no hallucination beyond retrieved context). Contextual Precision/Recall lower (0.375-1.00) — retrieval ranking/coverage is the current weak point, not generation. This is exactly the gap the planned Contextual Retrieval ablation (step 6 below) targets.
9. `src/eval/phoenix_tracing.py` — OTel + OpenInference instrumentation (`openinference-instrumentation-google-genai`) exporting traces to Phoenix — **not yet run**
10. `src/api/main.py` — FastAPI `/query` endpoint wrapping the pipeline
11. Docker Compose brings up Qdrant (`:6333`) and Phoenix (`:6006`) — both verified healthy
12. Git repo initialized, first commit made with the full scaffold

## Issues hit + resolutions

- **`arize-phoenix` pip package fails to import on Python 3.11.** `phoenix/__init__.py` (v20.8.0) pulls in code with a `dataclass` mutable-default pattern that only Python 3.13 allows; older versions (<8) require compiling `sqlean-py` from source (needs MSVC Build Tools, not installed); v12.35.0 installs but has a version-skew bug between core `arize-phoenix` and the separately-versioned `arize-phoenix-evals` package (`phoenix.evals.models` module missing).
  - **Fix:** dropped the `arize-phoenix` pip dependency entirely. Phoenix now runs as its own Docker container (`arizephoenix/phoenix:latest`, confirmed running Python 3.13 internally — validates the diagnosis). The app only needs the lightweight `opentelemetry-sdk` + `openinference-instrumentation-google-genai` packages to export traces to it over OTLP — no heavy/fragile phoenix package in the app's own environment.

- **Gemini AI Studio free tier for `gemini-embedding-001` throttles at ~1000 *tokens*/minute, not a clean requests-per-minute count.** Discovered empirically: isolated 1-5-chunk batches (a few hundred tokens) succeed reliably; 20-50-chunk batches (2,000+ tokens) 429 almost every time regardless of pacing, and the SDK's own hidden internal retry-on-429 was silently multiplying our real request count, compounding the problem. At this real throughput, embedding the full 2,832-chunk corpus takes multiple hours.
  - **Fix:** `embed_and_index.py` now (a) disables the SDK's internal retry (`HttpRetryOptions(attempts=1)`) so every HTTP call is visible and accounted for, (b) batches by a character budget (~2,400 chars/batch, capped for margin) instead of a fixed chunk count, since chunk length varies wildly (20–2,000+ chars) and a fixed count can randomly land on several long chunks, (c) paces requests ~75s apart, and (d) checkpoints progress to `data/processed/embed_checkpoint.json` after every successful batch so a rerun after hitting quota resumes instead of re-embedding (and re-spending quota on) already-indexed chunks.
  - **Workaround for iteration speed:** added `src/index/embed_and_index_local.py` using `BAAI/bge-small-en-v1.5` (free, offline, no rate limits) into a separate Qdrant collection (`fastapi_corpus_local`). `retriever.py` picks the backend via `EMBED_BACKEND=local|gemini`. Use `local` for day-to-day dev/demo; the Gemini-embedded collection is the "everything free, all-Gemini" version for the record, and finishes indexing unattended in the background.
  - Also hit: `gemini-2.5-flash-lite` and `gemini-2.5-flash` both returned 404 ("no longer available to new users") despite appearing in `models.list()` - generation now uses `gemini-3.5-flash-lite`.

- **Every free-tier Gemini *chat* model on this account caps at ~20 `generate_content` calls/day** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), confirmed identically on both `gemini-3.5-flash` and `gemini-3.6-flash` - looks like a blanket account-level restriction, not something specific to one model, since a brand-new model showed the exact same "20" limit on its first use. There's also a separate, tighter per-minute cap (`gemini-3.5-flash`: 5/min) that DeepEval's default concurrent metric execution (`asyncio.gather` over all 4 metrics per test case, immune to the `async_config` throttle which only paces *between* test cases) blows through instantly regardless of the daily cap.
  - **Fix:** `run_deepeval.py` no longer calls DeepEval's `evaluate()`. It calls `metric.measure(test_case)` one metric at a time in a plain loop with explicit pacing (20s) and its own retry/backoff on both 429 (quota) and 503 (transient model overload - also observed once). Given the ~20/day ceiling and 4 metrics x 1-2 calls/metric, only ~2 golden examples fit per model per day - so the harness now processes a small batch (default 2) per invocation and **appends** to `eval_report.json`, skipping already-scored questions, so the eval set builds up over several days instead of failing to finish a big run in one. Settled on `gemini-3.1-flash-lite` as the long-term judge model (for consistent scoring across days) after `gemini-3.5-flash` and `gemini-3.6-flash` were both already spent on same-day testing.
  - Ran a second free Google account's key through the same accounting: also capped at 20/day on first use, confirming the account-wide (not per-model) theory. Now split across two accounts by *role* rather than trying to round-robin models: `GEMINI_API_KEY_EMBED` (fresh account) handles all embedding traffic, `GEMINI_API_KEY` (original account) handles all generation/judge traffic - see the API keys decision row above.
  - **Further hardening after running the eval loop unattended ("run until quota expires"):** hit two more failure modes that aren't quota-related and shouldn't kill the whole run: (a) DeepEval's per-attempt timeout (~88s) was too short for some larger-context judge calls - raised to 180s via `DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE`; (b) the judge model occasionally returns malformed JSON for its structured verdict output (`gemini-3.1-flash-lite` is a small/cheap model, this happens more than it would with a bigger judge) - DeepEval raises a plain `ValueError` for this, uncaught by the 429/503/timeout-specific handling. Now the per-test-case loop catches *any* exception other than a 429/503 quota error: quota errors stop the whole run (that example isn't marked scored, so it's retried fresh next time), anything else gets recorded as an `"error"` entry for that question (so it isn't retried forever) and the loop moves on to the next example. This is what makes an unattended "run until quota's actually gone" loop viable.

## Current state

- Qdrant + Phoenix containers running and healthy
- All Python dependencies installed in `.venv` (incl. `google-genai`, `sentence-transformers`)
- `GEMINI_API_KEY`, `GEMINI_API_KEY_EMBED`, and `GITHUB_TOKEN` all set in `.env` and verified live
- Corpus cloned and chunked (2,832 chunks)
- `fastapi_corpus_local` (Qdrant, local `bge-small` embeddings): **fully indexed, ready to use**
- `fastapi_corpus` (Qdrant, Gemini embeddings, on `GEMINI_API_KEY_EMBED`): indexing in the background, resumable, running unattended until its account's quota is exhausted for the day (checkpoint index climbing steadily, no errors as of this writing - ~900/2832)
- `EMBED_BACKEND=local` currently set in `.env` — RAG pipeline smoke-tested end-to-end successfully on this backend
- Golden dataset built: 437 examples in `data/processed/golden_dataset.json`
- DeepEval harness (on `GEMINI_API_KEY`) also running unattended in a loop until its account's daily quota is exhausted, appending new scores to `data/processed/eval_report.json` as it goes
- Nothing currently blocked — everything running on $0 API cost, just rate-limited by free-tier daily quotas across two accounts

## Next steps

1. Keep running `python -m src.eval.run_deepeval` periodically (2 examples/day by default against the free-tier judge quota) to grow the scored eval set
2. Once the Gemini embedding job finishes indexing `fastapi_corpus`, optionally re-run the eval with `EMBED_BACKEND=gemini` and compare local-vs-Gemini embedding quality as a bonus ablation
3. Implement Anthropic's Contextual Retrieval in `chunk.py` (prepend LLM-generated context to each chunk before embedding) — re-run eval, compare scores as a v1→v2 ablation ([RESOURCES.md](RESOURCES.md)). Early signal already points here: Faithfulness/Answer Relevancy are perfect so far, Contextual Precision/Recall are the weak spot.
4. Fix context ordering in `generator.py` per the Lost-in-the-Middle finding (most relevant chunks first *and* last, not buried in the middle)
5. Wire up Phoenix tracing for a debugging demo
6. Stretch: AutoRAG-style sweep over chunk size/top_k picked by eval score; CI workflow running `run_deepeval.py` as a regression gate; deploy the FastAPI app
