from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import load_settings
from .data_loader import read_jsonl
from .grounding import create_grounded_view
from .joint_ranker import select_joint_pair
from .model import ConstrainedVLM
from .preprocessing import normalise_record
from .scoring import score_answer, score_rationales_for_every_answer


class VCRInferencePipeline:
    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        self.settings = load_settings(config_path)
        self.vlm = ConstrainedVLM(self.settings)

    def predict(self, record: dict, image_root: str | Path = ".") -> dict:
        record = normalise_record(record)
        image_path = Path(image_root) / record["image_path"]
        image = Image.open(image_path).convert("RGB")

        if self.settings.view == "grounded":
            if "boxes" not in record or "objects" not in record:
                raise ValueError("Grounded inference requires boxes and objects")
            image = create_grounded_view(image, record["boxes"], record["objects"])

        answer_probabilities = score_answer(
            self.vlm,
            image,
            record["question_text"],
            record["answer_choice_texts"],
            self.settings.answer_temperature,
        )
        rationale_probabilities = score_rationales_for_every_answer(
            self.vlm,
            image,
            record["question_text"],
            record["answer_choice_texts"],
            record["rationale_choice_texts"],
            self.settings.rationale_temperature,
        )
        answer_index, rationale_index, confidence = select_joint_pair(
            answer_probabilities, rationale_probabilities
        )

        return {
            "annot_id": record.get("annot_id"),
            "answer_prediction": answer_index,
            "rationale_prediction": rationale_index,
            "answer_label": f"A{answer_index}",
            "rationale_label": f"R{rationale_index}",
            "joint_confidence": confidence,
            "answer_probabilities": np.asarray(answer_probabilities).tolist(),
            "rationale_probabilities": np.asarray(rationale_probabilities).tolist(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grounded constrained VCR inference")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    pipeline = VCRInferencePipeline(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for record in read_jsonl(args.input):
            prediction = pipeline.predict(record, args.image_root)
            output.write(json.dumps(prediction) + "\n")

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
