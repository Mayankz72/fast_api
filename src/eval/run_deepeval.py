"""Run the RAG pipeline over the golden dataset and score it with DeepEval.

Metrics:
- Faithfulness: does the answer avoid claims unsupported by retrieved context?
- Answer Relevancy: does the answer address the question?
- Contextual Precision: are the most relevant chunks ranked first?
- Contextual Recall: does retrieved context cover what the ground-truth answer needs?

Every free-tier Gemini chat model has its own daily generate_content quota, varying
by model (as low as 20/day, as high as 500/day - discovered empirically, no
documented pattern). With 4 metrics x ~1-2 judge calls each, a single model's quota
only covers a handful to a few dozen golden examples/day. Rather than stopping for
the day once one model runs dry, JUDGE_MODEL_CANDIDATES is tried in order and
whichever one still has quota is used - so the loop keeps making progress on a
single day using multiple models' quotas back to back. This runs a small batch per
invocation and *appends* to the existing report rather than overwriting, picking up
the next unscored golden examples each time - the eval set builds up over multiple
runs instead of trying (and failing) to score everything in one run.

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
from google import genai
from google.genai.errors import APIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import RAGPipeline  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "golden_dataset.json"
REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_report.json"

METRIC_PACING_SECONDS = 20.0
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SECONDS = 65
# Each model has its own separate daily quota (some as low as 20/day, some as high
# as 500/day - varies by model, discovered empirically, see PROGRESS.md). Rather
# than manually swapping JUDGE_MODEL every time the current one runs dry, try this
# list in order and use whichever one still has quota today - keeps the eval loop
# going instead of stopping for the day just because one specific model is tapped.
JUDGE_MODEL_CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
]


def build_metrics_with_available_judge(probe_client: genai.Client, remaining_candidates: list[str]):
    """Pop exhausted/unavailable models off the front of remaining_candidates until
    one responds, and build the metrics list against it. Returns None if every
    candidate is exhausted."""
    while remaining_candidates:
        model = remaining_candidates[0]
        try:
            probe_client.models.generate_content(model=model, contents="OK")
        except APIError as e:
            if e.code in (429, 404, 503):
                print(f"  judge model {model} unavailable ({e.code}), trying next candidate...")
                remaining_candidates.pop(0)
                continue
            raise
        print(f"Using judge model: {model}")
        judge = GeminiModel(model=model, api_key=os.environ["GEMINI_API_KEY"])
        return [
            FaithfulnessMetric(threshold=0.7, model=judge),
            AnswerRelevancyMetric(threshold=0.7, model=judge),
            ContextualPrecisionMetric(threshold=0.7, model=judge),
            ContextualRecallMetric(threshold=0.7, model=judge),
        ]
    return None


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

    probe_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    remaining_candidates = list(JUDGE_MODEL_CANDIDATES)
    metrics = build_metrics_with_available_judge(probe_client, remaining_candidates)
    if metrics is None:
        print("All candidate judge models exhausted or unavailable for today.")
        sys.exit(1)

    report = existing_report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for i, test_case in enumerate(test_cases):
        print(f"Scoring test case {i + 1}/{len(test_cases)}: {test_case.input[:60]!r}")
        while True:
            try:
                metrics_data = []
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
                break
            except APIError as e:
                if e.code in (429, 503):
                    # This judge model is done for today - swap to the next
                    # candidate and retry this same example, rather than
                    # stopping the whole run over one exhausted model.
                    print(f"  judge model exhausted ({e.code}), switching candidates...")
                    remaining_candidates.pop(0)
                    metrics = build_metrics_with_available_judge(probe_client, remaining_candidates)
                    if metrics is None:
                        print("All candidate judge models exhausted or unavailable for today.")
                        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
                        sys.exit(1)
                    continue
                # Some other API error - record it as an error entry so this
                # example doesn't get retried forever, and move on.
                report.append({"input": test_case.input, "success": False, "error": str(e)})
                break
            except Exception as e:
                # Anything else non-quota (timeouts, the judge model returning
                # malformed JSON for a tricky example, etc.) - one bad example
                # shouldn't block the whole batch. Record and move on.
                print(f"  error scoring this example, skipping: {e}")
                report.append({"input": test_case.input, "success": False, "error": str(e)})
                break
        # Save after every test case, not just at the end - a later test case
        # hitting the daily quota wall shouldn't lose already-scored results.
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved eval report -> {REPORT_PATH} ({len(report)}/{len(golden)} total scored)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    main(batch_size=n)
