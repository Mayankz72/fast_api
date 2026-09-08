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
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

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


def build_metrics_with_available_judge(probe_client: genai.Client, remaining_candidates: list[str], api_key: str):
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
        judge = GeminiModel(model=model, api_key=api_key)
        return [
            FaithfulnessMetric(threshold=0.7, model=judge),
            AnswerRelevancyMetric(threshold=0.7, model=judge),
            ContextualPrecisionMetric(threshold=0.7, model=judge),
            ContextualRecallMetric(threshold=0.7, model=judge),
        ]
    return None


def load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def load_report(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def load_all_scored_questions(report_path: Path) -> set[str]:
    """Union of what's already scored in the main report and (if different)
    this shard's own report - so parallel shards don't waste quota rescoring
    an example another shard (or a previous unsharded run) already did."""
    scored = {entry["input"] for entry in load_report(REPORT_PATH)}
    if report_path != REPORT_PATH:
        scored |= {entry["input"] for entry in load_report(report_path)}
    return scored


def next_unscored_batch(golden: list[dict], already_scored: set[str], batch_size: int) -> list[dict]:
    return [item for item in golden if item["question"] not in already_scored][:batch_size]


def run_pipeline_with_backoff(pipeline: RAGPipeline, question: str):
    """A brief internet blip shouldn't kill an unattended run - retry a few times
    on transient network failures before giving up."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return pipeline.run(question)
        except httpx.TransportError as e:
            if attempt < RATE_LIMIT_RETRIES:
                print(f"  network error ({e}), sleeping 30s before retry...")
                time.sleep(30)
                continue
            raise


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
        except httpx.TransportError as e:
            if attempt < RATE_LIMIT_RETRIES:
                print(f"  network error ({e}), sleeping 30s before retry...")
                time.sleep(30)
                continue
            raise


def main(
    batch_size: int = 2,
    api_key_env: str = "GEMINI_API_KEY",
    report_path: Path = REPORT_PATH,
    start: int = 0,
    end: int | None = None,
) -> None:
    golden = load_golden()[start:end]
    already_scored = load_all_scored_questions(report_path)
    existing_report = load_report(report_path)

    batch = next_unscored_batch(golden, already_scored, batch_size)
    if not batch:
        print(f"All {len(golden)} golden examples in this shard's range already scored - nothing to do.")
        return

    pipeline = RAGPipeline(generator_api_key_env=api_key_env)
    print(f"Running pipeline over {len(batch)} new golden examples "
          f"({len(already_scored)} already scored across all shards so far, "
          f"{len(golden)} in this shard's range)...")
    test_cases = []
    for item in batch:
        try:
            result = run_pipeline_with_backoff(pipeline, item["question"])
        except RuntimeError:
            # All candidate generation models exhausted for this account today
            # (see Generator._pick_model) - nothing left to do this run.
            print("All candidate generation models exhausted or unavailable for today.")
            sys.exit(1)
        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=result.answer,
                expected_output=item["expected_answer"],
                retrieval_context=[c["text"] for c in result.contexts],
            )
        )

    api_key = os.environ.get(api_key_env, os.environ["GEMINI_API_KEY"])
    probe_client = genai.Client(api_key=api_key)
    remaining_candidates = list(JUDGE_MODEL_CANDIDATES)
    metrics = build_metrics_with_available_judge(probe_client, remaining_candidates, api_key)
    if metrics is None:
        print("All candidate judge models exhausted or unavailable for today.")
        sys.exit(1)

    report = existing_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
                    metrics = build_metrics_with_available_judge(probe_client, remaining_candidates, api_key)
                    if metrics is None:
                        print("All candidate judge models exhausted or unavailable for today.")
                        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved eval report -> {report_path} ({len(report)} total scored in this shard's file)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int, nargs="?", default=2)
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY", help="Env var name holding the generation/judge API key to use")
    parser.add_argument("--report-path", type=str, default=str(REPORT_PATH), help="Where this shard writes its scored results (default: the shared eval_report.json)")
    parser.add_argument("--start", type=int, default=0, help="First golden-example index this shard handles (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="Last golden-example index this shard handles (exclusive); default = end of dataset")
    parsed = parser.parse_args()
    main(
        batch_size=parsed.batch_size,
        api_key_env=parsed.api_key_env,
        report_path=Path(parsed.report_path),
        start=parsed.start,
        end=parsed.end,
    )
