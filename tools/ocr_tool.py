from functools import lru_cache
from pathlib import Path

import easyocr
from agents import function_tool

from .context import get_current_image


@lru_cache(maxsize=1)
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=True)


@function_tool
def read_text(image_path: str | None = None) -> dict:
    """Read text from the current image or an exact crop path."""
    path = get_current_image() if not image_path or not image_path.strip() else Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OCR image not found: {path}")
    items = []
    for bbox, text, confidence in get_ocr_reader().readtext(str(path), detail=1):
        items.append({"text": text, "confidence": float(confidence), "bbox": [[float(x), float(y)] for x, y in bbox]})
    return {"found": bool(items), "image_path": str(path), "full_text": " ".join(item["text"] for item in items), "items": items}