from __future__ import annotations

from collections.abc import Sequence

from .model import ConstrainedVLM


def _format_choices(choices: Sequence[str]) -> str:
    labels = ("A", "B", "C", "D")
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def answer_prompt(question: str, answers: Sequence[str]) -> str:
    return (
        "Study the labelled image carefully. Select the most plausible answer.\n\n"
        f"Question: {question}\n\nAnswer candidates:\n{_format_choices(answers)}\n\n"
        "Reply with exactly one letter: A, B, C, or D."
    )


def rationale_prompt(
    question: str,
    selected_answer: str,
    rationales: Sequence[str],
) -> str:
    return (
        "Study the labelled image. Select the rationale that best supports the given answer.\n\n"
        f"Question: {question}\nGiven answer: {selected_answer}\n\n"
        f"Rationale candidates:\n{_format_choices(rationales)}\n\n"
        "Reply with exactly one letter: A, B, C, or D."
    )


def score_answer(
    model: ConstrainedVLM, image, question: str, answers: Sequence[str], temperature: float
):
    return model.choice_probabilities(image, answer_prompt(question, answers), temperature)


def score_rationales_for_every_answer(
    model: ConstrainedVLM,
    image,
    question: str,
    answers: Sequence[str],
    rationales: Sequence[str],
    temperature: float,
):
    return [
        model.choice_probabilities(
            image,
            rationale_prompt(question, answer, rationales),
            temperature,
        )
        for answer in answers
    ]
