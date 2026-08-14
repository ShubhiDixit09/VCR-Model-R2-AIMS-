from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import VCRInferencePipeline


pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = VCRInferencePipeline(str(ROOT / "configs/default.yaml"))
    return pipeline


def predict(image, question, a0, a1, a2, a3, r0, r1, r2, r3):
    if image is None:
        raise gr.Error("Please upload an image")
    # Demo accepts an unboxed image, so use a temporary original-view override.
    model = get_pipeline()
    old_view = model.settings.view
    object.__setattr__(model.settings, "view", "original")
    temp_path = ROOT / "demo_samples" / "uploaded_demo.png"
    temp_path.parent.mkdir(exist_ok=True)
    image.save(temp_path)
    record = {
        "annot_id": "demo",
        "image_path": str(temp_path),
        "question_text": question,
        "answer_choice_texts": [a0, a1, a2, a3],
        "rationale_choice_texts": [r0, r1, r2, r3],
    }
    try:
        result = model.predict(record)
    finally:
        object.__setattr__(model.settings, "view", old_view)
    return result["answer_label"], result["rationale_label"], result["joint_confidence"]


with gr.Blocks(title="Grounded Joint VCR") as demo:
    gr.Markdown("# Grounded Joint Visual Commonsense Reasoning")
    gr.Markdown("Upload an image and provide exactly four answers and four rationales.")
    image = gr.Image(type="pil", label="Image")
    question = gr.Textbox(label="Question")
    answers = [gr.Textbox(label=f"Answer A{i}") for i in range(4)]
    rationales = [gr.Textbox(label=f"Rationale R{i}") for i in range(4)]
    run = gr.Button("Select answer-rationale pair", variant="primary")
    answer_output = gr.Textbox(label="Selected answer")
    rationale_output = gr.Textbox(label="Selected rationale")
    confidence_output = gr.Number(label="Joint confidence")
    run.click(
        predict,
        inputs=[image, question, *answers, *rationales],
        outputs=[answer_output, rationale_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch(share=os.getenv("GRADIO_SHARE", "false").lower() == "true")
