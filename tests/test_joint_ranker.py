import numpy as np

from src.joint_ranker import joint_distribution, select_joint_pair


def test_joint_distribution_is_normalised():
    answer = [0.1, 0.6, 0.2, 0.1]
    rationale = np.full((4, 4), 0.25)
    assert np.isclose(joint_distribution(answer, rationale).sum(), 1.0)


def test_select_joint_pair():
    answer = [0.1, 0.6, 0.2, 0.1]
    rationale = [
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.1, 0.7, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
    answer_index, rationale_index, _ = select_joint_pair(answer, rationale)
    assert (answer_index, rationale_index) == (1, 2)
