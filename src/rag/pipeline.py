"""End-to-end RAG pipeline: retrieve then generate."""
from dataclasses import dataclass

from .generator import Generator
from .retriever import Retriever


@dataclass
class RAGResult:
    answer: str
    contexts: list[dict]


class RAGPipeline:
    def __init__(self, top_k: int = 5) -> None:
        self.retriever = Retriever()
        self.generator = Generator()
        self.top_k = top_k

    def run(self, question: str) -> RAGResult:
        contexts = self.retriever.retrieve(question, top_k=self.top_k)
        answer = self.generator.generate(question, contexts)
        return RAGResult(answer=answer, contexts=contexts)


if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()
    question = " ".join(sys.argv[1:]) or "How do I define a path parameter with a type?"
    result = pipeline.run(question)
    print("ANSWER:\n", result.answer)
    print("\nSOURCES:")
    for c in result.contexts:
        print(f"  [{c['score']:.3f}] {c['source']} — {c['heading_path']}")
