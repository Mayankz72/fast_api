from fastapi import FastAPI
from pydantic import BaseModel

from ..eval.phoenix_tracing import start_tracing
from ..rag.pipeline import RAGPipeline

start_tracing()

app = FastAPI(title="FastAPI Docs RAG")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline.top_k = req.top_k
    result = pipeline.run(req.question)
    return QueryResponse(answer=result.answer, sources=result.contexts)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
