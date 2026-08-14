# Grounded Joint Reasoning for VCR

An inference-only system for the two-stage **Visual Commonsense Reasoning (VCR)** task:

1. select one answer from four candidates, and
2. select one of four supplied rationales conditioned on the answer.

The system never generates a free-text rationale and requires no human intervention during inference.

## What is different?

- **Explicit object grounding:** VCR object references such as `[person0]` are aligned with labelled bounding boxes rendered on the image.
- **Constrained discrimination:** the VLM scores only `A/B/C/D`; it cannot evade the task through open-ended generation.
- **Full 4 × 4 joint ranking:** all 16 answer-rationale combinations are considered using
  `P(A | I,Q) × P(R | I,Q,A)`.
- **Evidence-driven choices:** original, grounded and dual views; 3B versus 7B backbones; temperature calibration; and permutation averaging were tested rather than assumed useful.
- **Leakage control:** labels are isolated from prompts, image-disjoint samples are used, and final settings are evaluated on a fresh locked holdout.

## Final configuration

| Component | Choice |
|---|---|
| Backbone | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit NF4 |
| Visual input | Grounded/boxed image |
| Answer scoring | Constrained next-token A–D probability |
| Rationale scoring | Constrained A–D probability conditioned on each answer |
| Joint decision | Product over all 16 pairs |
| Temperatures | 1.0 / 1.0 (calibration was rejected after a negative ablation) |
| Fine-tuning | None |

## Locked holdout results

The final settings were locked before evaluating 20 previously untouched validation examples.

| Metric | Accuracy |
|---|---:|
| Q→A | 60% |
| QA→R | 40% |
| Q→AR (joint pair) | 15% |

This is a small diagnostic holdout, so the values are reported as evidence of the implemented pipeline—not as a claim of benchmark-level performance.

## Repository layout

```text
configs/       final inference configuration
demo/          Gradio demonstration
notebooks/     experiment notebook
outputs/       predictions, metrics and ablations
report/        short technical report
src/           reusable pipeline modules
tests/         lightweight unit tests
```

## Installation

Python 3.10+ and an NVIDIA GPU are recommended.

```bash
pip install -r requirements.txt
```

## Input format

Each JSONL record must contain:

```json
{
  "annot_id": "example-0",
  "image_path": "images/example.jpg",
  "objects": ["person", "person"],
  "boxes": [[12, 20, 200, 400], [240, 30, 430, 410]],
  "question_text": "Why is [person0] looking at [person1]?",
  "answer_choice_texts": ["...", "...", "...", "..."],
  "rationale_choice_texts": ["...", "...", "...", "..."]
}
```

## Inference

```bash
python -m src.inference \
  --input data/examples.jsonl \
  --image-root data \
  --output outputs/predictions.jsonl
```

## Demo

```bash
python demo/app.py
```

The lightweight demo accepts a clean uploaded image. The official evaluation path uses VCR bounding boxes and the grounded view.

## Tests

```bash
pytest -q
```

## Reproducibility and limitations

- VCR is loaded from `Rowan/vcr` on Hugging Face; dataset images are not committed.
- The model is frozen, so no trained weights are included. The complete inference script is the submitted alternative permitted by the task.
- The current rationale stage is weaker than the answer stage. Larger-scale evaluation or parameter-efficient fine-tuning is future work.
- Results depend on GPU, library versions, and the exact VCR split.

## Author

Shubhi Dixit — AIMS DTU Recruitment Round 2
