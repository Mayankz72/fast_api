# Research & Resources

Papers and engineering write-ups backing the design decisions in this project.
Useful both for grounding build choices and for citing in the resume
writeup/README ("built on X technique from Y paper" reads a lot better than
"used LangChain").

## Foundational papers

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** (Lewis et al., 2020) — the original RAG paper. Combines a parametric LM with a non-parametric retrieval index; established the retrieve-then-generate paradigm this whole project is built on.
- **[Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)** (Karpukhin et al., 2020) — the dense bi-encoder retrieval approach underlying every embedding-based vector search (including ours via `text-embedding-3-small` + Qdrant).
- **[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)** (Khattab & Zaharia, 2020) — late-interaction retrieval; relevant if we add a reranking stage (see "possible upgrades" below).

## Evaluation — the part that differentiates this project

- **[RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)** (Es et al., 2023/2024 EACL) — reference-free RAG metrics (the four-metric framework — faithfulness, answer relevancy, context precision/recall — that DeepEval's RAG metrics and our `run_deepeval.py` are built on).
- **[ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)** (Saad-Falcon et al., 2023) — uses lightweight LM judges trained on synthetic data to score RAG systems; a good "prior art" citation for why an automated eval harness is a real research problem, not just tooling.
- **[RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/abs/2408.08067)** (2024) — finer-grained diagnosis than RAGAS (claim-level checking); good reference if we want to extend `run_deepeval.py` with more granular failure analysis later.
- **[AutoRAG: Automated Framework for Optimization of RAG Pipeline](https://arxiv.org/abs/2410.20878)** (2024) — automates trying different chunking/retrieval/reranking configs and picking the best by eval score. Directly relevant stretch goal: turn our eval harness into a small grid search over chunk size / top-k.

## Survey (for the "prior art" section of a writeup)

- **[Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)** (Gao et al., 2023, updated 2024) — the most-cited RAG survey; maps "Naive RAG → Advanced RAG → Modular RAG" which is a clean way to frame where this project's architecture sits (we're Advanced RAG: pre-retrieval chunking optimization + post-retrieval — could push toward Modular by adding a reranker/router).

## Practical techniques worth adopting

- **[Anthropic — Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)** — prepends a short LLM-generated summary of "where this chunk sits in the document" to each chunk before embedding/BM25 indexing. Reported **49% reduction in failed retrievals**, 67% when combined with reranking. **This is a concrete, cheap upgrade for `src/ingest/chunk.py`** — worth doing as a v2 experiment and comparing DeepEval scores before/after (turns it into an actual ablation study for the resume writeup).
- **[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)** (Liu et al., 2023) — LLMs attend more to the start/end of context than the middle. Implication for us: when `top_k` context chunks are stuffed into the prompt in `generator.py`, ordering matters — most-relevant chunks should be placed at the start *and* end, not buried in the middle. Another concrete, citable improvement.

## How this maps to our build

| Paper/technique | Where it shows up in this repo |
|---|---|
| Lewis et al. (RAG) | `src/rag/pipeline.py` — the whole retrieve→generate structure |
| Karpukhin et al. (DPR) | `src/rag/retriever.py` — dense embedding retrieval via Qdrant |
| RAGAS metrics | `src/eval/run_deepeval.py` — Faithfulness/AnswerRelevancy/ContextualPrecision/ContextualRecall |
| Anthropic Contextual Retrieval | **Not yet implemented** — planned v2 improvement to `src/ingest/chunk.py` |
| Lost in the Middle | **Not yet implemented** — planned reordering fix in `src/rag/generator.py`'s `build_context()` |
| AutoRAG-style config search | **Stretch goal** — sweep chunk size / top_k and pick best by eval score |

Sources: [Lewis et al., RAG](https://arxiv.org/abs/2005.11401) · [Karpukhin et al., DPR](https://arxiv.org/abs/2004.04906) · [Khattab & Zaharia, ColBERT](https://arxiv.org/abs/2004.12832) · [Es et al., RAGAS](https://arxiv.org/abs/2309.15217) · [Saad-Falcon et al., ARES](https://arxiv.org/abs/2311.09476) · [RAGChecker](https://arxiv.org/abs/2408.08067) · [AutoRAG](https://arxiv.org/abs/2410.20878) · [Gao et al., RAG Survey](https://arxiv.org/abs/2312.10997) · [Anthropic, Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) · [Liu et al., Lost in the Middle](https://arxiv.org/abs/2307.03172)
