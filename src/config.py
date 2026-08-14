from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    model_id: str
    load_in_4bit: bool
    min_pixels: int
    max_pixels: int
    answer_temperature: float
    rationale_temperature: float
    view: str
    seed: int


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    return Settings(
        model_id=raw["model"]["id"],
        load_in_4bit=bool(raw["model"].get("load_in_4bit", True)),
        min_pixels=int(raw["image"]["min_pixels"]),
        max_pixels=int(raw["image"]["max_pixels"]),
        answer_temperature=float(raw["inference"]["answer_temperature"]),
        rationale_temperature=float(raw["inference"]["rationale_temperature"]),
        view=str(raw["image"].get("view", "grounded")),
        seed=int(raw["inference"].get("seed", 42)),
    )
