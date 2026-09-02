from pathlib import Path


_CURRENT_IMAGE_PATH: Path | None = None


def set_current_image(image_path: str | Path) -> Path:
    global _CURRENT_IMAGE_PATH
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    _CURRENT_IMAGE_PATH = path
    return path


def get_current_image() -> Path:
    if _CURRENT_IMAGE_PATH is None:
        raise RuntimeError("Call set_current_image before running the agent.")
    return _CURRENT_IMAGE_PATH