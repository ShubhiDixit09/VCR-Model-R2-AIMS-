from __future__ import annotations

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

from .config import Settings


class ConstrainedVLM:
    """Qwen-VL wrapper that scores only the supplied A/B/C/D choices."""

    LABELS = ("A", "B", "C", "D")

    def __init__(self, settings: Settings) -> None:
        quantization = None
        if settings.load_in_4bit:
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(
            settings.model_id,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(
            settings.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        ).eval()
        self.label_token_ids = [
            self.processor.tokenizer.encode(label, add_special_tokens=False)[-1]
            for label in self.LABELS
        ]

    @torch.inference_mode()
    def choice_probabilities(self, image, prompt: str, temperature: float = 1.0):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1, self.label_token_ids].float()
        probabilities = torch.softmax(logits / temperature, dim=-1)
        return probabilities.cpu().numpy()
