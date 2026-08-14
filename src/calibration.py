from __future__ import annotations

import numpy as np


def apply_temperature(probabilities, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    values = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    converted = np.exp(logits)
    return converted / converted.sum(axis=-1, keepdims=True)
