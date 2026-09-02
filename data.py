import json
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
METADATA_PATH = DATASET_DIR / "all_metadata.json"


def load_metadata(path: Path = METADATA_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as metadata_file:
        records = json.load(metadata_file)
    if not isinstance(records, list):
        raise ValueError(f"Metadata must be a JSON list: {path}")
    return records


def find_record(scene: int, rule: int, inconsistency_type: str) -> dict[str, Any]:
    matches = [
        record for record in load_metadata()
        if int(record["SceneID"]) == scene
        and int(record["Rule ID"]) == rule
        and record["Inconsistency Type"].casefold() == inconsistency_type.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one record for Scene {scene}, Rule {rule}, "
            f"{inconsistency_type!r}; found {len(matches)}."
        )
    return matches[0]


def find_image(image_name: str) -> Path:
    matches = [path for path in DATASET_DIR.rglob(image_name) if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one image named {image_name!r} under {DATASET_DIR}; "
            f"found {len(matches)}. Copy the dataset images into the experiment "
            "dataset directory before running the notebook."
        )
    return matches[0].resolve()