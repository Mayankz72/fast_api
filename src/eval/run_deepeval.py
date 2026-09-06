"""Run the RAG pipeline over the golden dataset and score it with DeepEval.

Metrics:
- Faithfulness: does the answer avoid claims unsupported by retrieved context?
- Answer Relevancy: does the answer address the question?
- Contextual Precision: are the most relevant chunks ranked first?
- Contextual Recall: does retrieved context cover what the ground-truth answer needs?

Every free-tier Gemini chat model on this account caps at ~20 generate_content
calls/day (confirmed on gemini-3.5-flash and gemini-3.6-flash alike - looks like an
account-wide limit, not model-specific). With 4 metrics x ~1-2 judge calls each,
that's only ~2 golden examples/day against a single model. So this runs a small
daily batch and *appends* to the existing report rather than overwriting, picking
up the next unscored golden examples each time - the eval set builds up slowly
across days instead of trying (and failing) to score everything in one run.

DeepEval's evaluate() also fires all metrics for a test case concurrently via
asyncio.gather regardless of async_config (that setting only paces *between* test
cases), which alone blows through the also-tight per-minute cap. So metrics are
measured one at a time here, synchronously, with explicit pacing.
"""
import json
import os
import sys
import time
from pathlib import Path

# Some judge responses (larger retrieval contexts) exceed DeepEval's default ~88s
# per-attempt timeout - must be set before deepeval reads its settings.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from google.genai.errors import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import RAGPipeline  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "golden_dataset.json"
REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_report.json"

METRIC_PACING_SECONDS = 20.0
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 65
# gemini-3.5-flash and gemini-3.6-flash both already exhausted for today (see
# module docstring) - gemini-3.1-flash-lite is untouched so far; stick with one
# model long-term so results are judged consistently across days.
JUDGE_MODEL = "gemini-3.1-flash-lite"


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def load_existing_report() -> list[dict]:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    return []


def next_unscored_batch(golden: list[dict], already_scored: set[str], batch_size: int) -> list[dict]:
    return [item for item in golden if item["question"] not in already_scored][:batch_size]


def measure_with_backoff(metric, test_case: LLMTestCase) -> None:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            metric.measure(test_case, _show_indicator=False)
            return
        except APIError as e:
            # 429 = quota/rate limit, 503 = transient model overload - both worth
            # backing off and retrying rather than crashing the whole run.
            if e.code in (429, 503) and attempt < RATE_LIMIT_RETRIES:
                print(f"  {e.code} error, sleeping {RATE_LIMIT_BACKOFF_SECONDS}s before retry...")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            raise
        except TimeoutError:
            # DeepEval's per-attempt timeout (~88s) on a slow judge response - not
            # a quota issue, just retry.
            if attempt < RATE_LIMIT_RETRIES:
                print("  request timed out, retrying...")
                continue
            raise


def main(batch_size: int = 2) -> None:
    golden = load_golden()
    existing_report = load_existing_report()
    already_scored = {entry["input"] for entry in existing_report}

    batch = next_unscored_batch(golden, already_scored, batch_size)
    if not batch:
        print(f"All {len(golden)} golden examples already scored - nothing to do.")
        return

    pipeline = RAGPipeline()
    print(f"Running pipeline over {len(batch)} new golden examples "
          f"({len(already_scored)}/{len(golden)} already scored so far)...")
    test_cases = []
    for item in batch:
        result = pipeline.run(item["question"])
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result.answer,
                expected_output=item["expected_answer"],
                retrieval_context=[c["text"] for c in result.contexts],
            )
        )

    judge = GeminiModel(model=JUDGE_MODEL, api_key=os.environ["GEMINI_API_KEY"])
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        ContextualPrecisionMetric(threshold=0.7, model=judge),
        ContextualRecallMetric(threshold=0.7, model=judge),
    ]

    report = existing_report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, test_case in enumerate(test_cases):
        print(f"Scoring test case {i + 1}/{len(test_cases)}: {test_case.input[:60]!r}")
        metrics_data = []
        try:
            for metric in metrics:
                measure_with_backoff(metric, test_case)
                metrics_data.append(
                    {
                        "name": metric.__name__,
                        "score": metric.score,
                        "success": metric.is_successful(),
                        "reason": metric.reason,
                    }
                )
                time.sleep(METRIC_PACING_SECONDS)
            report.append(
                {
                    "input": test_case.input,
                    "success": all(m["success"] for m in metrics_data),
                    "metrics": metrics_data,
                }
            )
        except APIError as e:
            if e.code in (429, 503):
                # Quota/overload exhausted for the day - stop the whole run, don't
                # mark this example as scored so it's retried fresh next time.
                REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
                raise
            # Some other API error - record it as an error entry so this example
            # doesn't get retried forever, and move on to the next one.
            report.append({"input": test_case.input, "success": False, "error": str(e)})
        except Exception as e:
            # Anything else non-quota (timeouts, the judge model returning
            # malformed JSON for a tricky example, etc.) - one bad example
            # shouldn't block the whole batch. Record and move on.
            print(f"  error scoring this example, skipping: {e}")
            report.append({"input": test_case.input, "success": False, "error": str(e)})
        # Save after every test case, not just at the end - a later test case
        # hitting the daily quota wall shouldn't lose already-scored results.
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved eval report -> {REPORT_PATH} ({len(report)}/{len(golden)} total scored)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(batch_size=n)
