from __future__ import annotations

import logging
from pathlib import Path
from random import choices
from string import ascii_lowercase, digits

import clip
import torch
from PIL import Image

LABELS = ["screenshot", "photograph", "painting"]
SCREENSHOT_THRESHOLD = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger: logging.Logger = logging.getLogger("prntscraper")
model = None
preprocess = None
text_tokens = None


def load_model() -> None:
    global model, preprocess, text_tokens

    if model is None:
        logger.debug("[cyan]Loading CLIP model...[/cyan]")
        model, preprocess = clip.load("ViT-B/32", device=DEVICE)
        text_tokens = clip.tokenize(LABELS).to(DEVICE)


def gen_code() -> str:
    chars = ascii_lowercase + digits
    length = choices(list(range(6, 9)))[0]
    return "".join(choices(chars, k=length))


def is_image_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def classify_image(path: Path) -> dict[str, float]:
    load_model()
    assert model is not None
    assert preprocess is not None
    assert text_tokens is not None

    with Image.open(path) as image:
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits_per_image, _ = model(image_tensor, text_tokens)
        probs = logits_per_image.softmax(dim=-1)

    return dict(zip(LABELS, probs[0].cpu().numpy()))
