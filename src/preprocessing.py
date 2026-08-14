from __future__ import annotations

import re
from typing import Any


_CONTRACTIONS = re.compile(r"\b(\w+)\s*'\s+(s|t|re|ve|ll|d|m)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Repair tokenization artefacts such as ``doesn' t`` without rewriting text."""
    text = _CONTRACTIONS.sub(r"\1'\2", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    required = (
        "question_text",
        "answer_choice_texts",
        "rationale_choice_texts",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if len(record["answer_choice_texts"]) != 4:
        raise ValueError("Exactly four answer candidates are required")
    if len(record["rationale_choice_texts"]) != 4:
        raise ValueError("Exactly four rationale candidates are required")

    output = dict(record)
    output["question_text"] = clean_text(record["question_text"])
    output["answer_choice_texts"] = [clean_text(x) for x in record["answer_choice_texts"]]
    output["rationale_choice_texts"] = [clean_text(x) for x in record["rationale_choice_texts"]]
    return output
