from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont


COLORS = ("#00A6D6", "#F28E2B", "#2CA02C", "#D62728", "#9467BD", "#8C564B")


def create_grounded_view(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    objects: Sequence[str],
) -> Image.Image:
    """Overlay stable VCR object IDs such as person0 on a copy of the image."""
    grounded = image.convert("RGB").copy()
    draw = ImageDraw.Draw(grounded)
    font = ImageFont.load_default()

    for index, (box, name) in enumerate(zip(boxes, objects)):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = (int(float(v)) for v in box[:4])
        color = COLORS[index % len(COLORS)]
        label = f"{name}{index}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        text_height = bottom - top
        y_text = max(0, y1 - text_height - 4)
        draw.rectangle((x1, y_text, x1 + right - left + 6, y1), fill=color)
        draw.text((x1 + 3, y_text + 1), label, fill="white", font=font)

    return grounded
