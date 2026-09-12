---
title: FastAPI Docs RAG
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# FastAPI Docs RAG — with an evaluation harness

A RAG system that answers questions about FastAPI (docs + source code), with an
automated evaluation harness measuring retrieval and generation quality against
a golden dataset built from real, answered GitHub Discussions.

**Stack:** Qdrant (vector store) · Google Gemini (`gemini-embedding-001` embeddings +
`gemini-3.5-flash-lite` generation, free tier) · DeepEval (metrics/CI) ·
Arize Phoenix (tracing) · FastAPI (serving)

See [`PROGRESS.md`](PROGRESS.md) for the build log and [`RESOURCES.md`](RESOURCES.md)
for the research papers and techniques this design is based on.

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
