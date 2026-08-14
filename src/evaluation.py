from __future__ import annotations

import pandas as pd


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No prediction rows supplied")
    return {
        "Q_to_A_accuracy": 100.0 * frame["q_to_a_correct"].mean(),
        "QA_to_R_accuracy": 100.0 * frame["qa_to_r_correct"].mean(),
        "Q_to_AR_accuracy": 100.0 * frame["joint_ar_correct"].mean(),
        "samples": int(len(frame)),
    }
