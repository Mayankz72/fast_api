"""Run the RAG pipeline over the golden dataset and score it with DeepEval.

Metrics:
- Faithfulness: does the answer avoid claims unsupported by retrieved context?
- Answer Relevancy: does the answer address the question?
- Contextual Precision: are the most relevant chunks ranked first?
- Contextual Recall: does retrieved context cover what the ground-truth answer needs?
"""
import json
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import RAGPipeline  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "golden_dataset.json"
REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_report.json"


def load_golden(limit: int | None = None) -> list[dict]:
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data[:limit] if limit else data


def main(limit: int = 30) -> None:
    golden = load_golden(limit)
    pipeline = RAGPipeline()

    test_cases = []
    for item in golden:
        result = pipeline.run(item["question"])
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result.answer,
                expected_output=item["expected_answer"],
                retrieval_context=[c["text"] for c in result.contexts],
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=0.7),
        AnswerRelevancyMetric(threshold=0.7),
        ContextualPrecisionMetric(threshold=0.7),
        ContextualRecallMetric(threshold=0.7),
    ]

    eval_result = evaluate(test_cases=test_cases, metrics=metrics)

    summary = []
    for test_result in eval_result.test_results:
        summary.append(
            {
                "input": test_result.input,
                "success": test_result.success,
                "metrics": [
                    {"name": m.name, "score": m.score, "success": m.success, "reason": m.reason}
                    for m in test_result.metrics_data
                ],
            }
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved eval report -> {REPORT_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(limit=n)
