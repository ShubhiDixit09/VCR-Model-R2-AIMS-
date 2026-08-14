from __future__ import annotations

import numpy as np


def joint_distribution(answer_probabilities, rationale_probabilities) -> np.ndarray:
    answer = np.asarray(answer_probabilities, dtype=np.float64)
    rationale = np.asarray(rationale_probabilities, dtype=np.float64)
    if answer.shape != (4,) or rationale.shape != (4, 4):
        raise ValueError("Expected answer shape (4,) and rationale shape (4, 4)")
    joint = answer[:, None] * rationale
    return joint / joint.sum()


def select_joint_pair(answer_probabilities, rationale_probabilities) -> tuple[int, int, float]:
    joint = joint_distribution(answer_probabilities, rationale_probabilities)
    answer_index, rationale_index = np.unravel_index(np.argmax(joint), joint.shape)
    return int(answer_index), int(rationale_index), float(joint[answer_index, rationale_index])
