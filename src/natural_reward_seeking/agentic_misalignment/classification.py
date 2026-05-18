from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from typing import Any

from tqdm.auto import tqdm


def _load_classifier_stack(agentic_repo_root: Path, classifier_model_id: str):
    repo_root = agentic_repo_root.resolve()
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from api_client.model_client import ModelClient
    from classifiers.blackmail_classifier import BlackmailClassifier
    from classifiers.leak_classifier import LeakClassifier
    from classifiers.murder_classifier import MurderClassifier

    return ModelClient, {
        "blackmail": BlackmailClassifier(classifier_model_id),
        "leaking": LeakClassifier(classifier_model_id),
        "murder": MurderClassifier(classifier_model_id),
    }


async def _classify_rows_async(
    rows: list[dict[str, Any]],
    *,
    agentic_repo_root: Path,
    classifier_model_id: str,
    concurrency: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    ModelClient, classifiers = _load_classifier_stack(agentic_repo_root, classifier_model_id)
    client = ModelClient()
    semaphore = asyncio.Semaphore(concurrency)
    progress = tqdm(total=len(rows), desc="Classifying agentic responses", unit="response") if show_progress else None

    async def classify_one(row: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(row)
        scenario = str(enriched.get("scenario", ""))
        classifier = classifiers.get(scenario)
        if classifier is None:
            enriched["classification_error"] = f"No classifier for scenario: {scenario}"
            if progress is not None:
                progress.update(1)
            return enriched
        try:
            async with semaphore:
                verdict, classifier_verdict, reasoning = await classifier.classify(
                    client,
                    str(enriched.get("email_content", "")),
                    str(enriched.get("raw_output", "")),
                )
            enriched.update(
                {
                    "harmful_behavior": bool(verdict),
                    "classifier_verdict": bool(classifier_verdict),
                    "classifier_reasoning": reasoning,
                    "classifier_model": classifier_model_id,
                    "classification_error": "",
                }
            )
        except Exception as exc:
            enriched.update(
                {
                    "harmful_behavior": None,
                    "classifier_verdict": None,
                    "classifier_reasoning": "",
                    "classifier_model": classifier_model_id,
                    "classification_error": str(exc),
                }
            )
        finally:
            if progress is not None:
                progress.update(1)
        return enriched

    try:
        return await asyncio.gather(*(classify_one(row) for row in rows))
    finally:
        if progress is not None:
            progress.close()


def classify_response_rows(
    rows: list[dict[str, Any]],
    *,
    agentic_repo_root: Path,
    classifier_model_id: str,
    concurrency: int,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _classify_rows_async(
            rows,
            agentic_repo_root=agentic_repo_root,
            classifier_model_id=classifier_model_id,
            concurrency=concurrency,
            show_progress=show_progress,
        )
    )
