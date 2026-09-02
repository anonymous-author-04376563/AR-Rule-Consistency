from pathlib import Path
import shutil

from PIL import Image

from .context import get_current_image


CROP_DIR = Path(__file__).resolve().parent.parent / "debug_crops"


def clear_debug_crops() -> None:
    """Remove temporary crop images from the previous sample."""
    if CROP_DIR.exists():
        shutil.rmtree(CROP_DIR)


def _safe_name(region_name: str) -> str:
    return region_name.replace(" ", "_").replace("/", "_")


def _save_crop(image: Image.Image, bounds: tuple[int, int, int, int], name: str) -> str:
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    crop_path = CROP_DIR / name
    image.crop(bounds).save(crop_path)
    return str(crop_path.resolve())


def crop_bbox(bbox: list[float], region_name: str, index: int) -> str:
    """Save a tight crop with the original 50% bbox padding."""
    image = Image.open(get_current_image()).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * 0.5
    pad_y = (y2 - y1) * 0.5
    expanded = (
        max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y)),
    )
    return _save_crop(
        image,
        expanded,
        f"{_safe_name(region_name)}_{index}_tight.png",
    )


def crop_context_bbox(bbox: list[float], region_name: str, index: int) -> str:
    """Save an adaptive context crop centered on a grounding bbox."""
    image = Image.open(get_current_image()).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = bbox
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    bbox_area_ratio = (bbox_width * bbox_height) / (width * height)

    # Small detections need proportionally more surrounding context than
    # large detections. The scale is derived from the bbox's image coverage.
    context_scale = max(2.0, min(12.0, 0.5 / (bbox_area_ratio ** 0.5)))
    crop_width = min(width, bbox_width * context_scale)
    crop_height = min(height, bbox_height * context_scale)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, min(width - crop_width, center_x - crop_width / 2.0))
    top = max(0, min(height - crop_height, center_y - crop_height / 2.0))
    bounds = (
        int(left),
        int(top),
        int(left + crop_width),
        int(top + crop_height),
    )
    return _save_crop(
        image,
        bounds,
        f"{_safe_name(region_name)}_{index}_context.png",
    )