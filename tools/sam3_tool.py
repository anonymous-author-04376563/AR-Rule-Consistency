from functools import lru_cache
from pathlib import Path
import re

import torch
from PIL import Image
from agents import function_tool
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from .context import get_current_image
from .crop_tool import crop_context_bbox, crop_bbox
from .region_grouping import group_bboxes


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.5
BPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "sam3"
    / "sam3"
    / "assets"
    / "bpe_simple_vocab_16e6.txt.gz"
)
_STATE_CACHE: dict[str, object] = {}


@lru_cache(maxsize=1)
def get_sam3_processor() -> Sam3Processor:
    if not BPE_PATH.is_file():
        raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {BPE_PATH}")
    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        device=DEVICE,
        eval_mode=True,
    )
    return Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)


def _get_image_state(image_path: Path):
    key = str(image_path.resolve())
    if key not in _STATE_CACHE:
        processor = get_sam3_processor()
        image = Image.open(key).convert("RGB")
        if DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _STATE_CACHE[key] = processor.set_image(image)
            torch.cuda.synchronize()
        else:
            _STATE_CACHE[key] = processor.set_image(image)
    return _STATE_CACHE[key]


@function_tool
def ground_object(object_name: str) -> dict:
    """Ground one short, single-entity noun phrase with SAM3.

    Queries containing multiple targets are rejected so the model can retry
    with separate grounding calls. Each region includes a tight crop for
    appearance inspection and an adaptive context crop for spatial relations.
    """
    normalized_name = object_name.strip()
    if (
        not normalized_name
        or re.search(r"\band\b", normalized_name, flags=re.IGNORECASE)
        or "," in normalized_name
        or ";" in normalized_name
    ):
        raise ValueError(
            "ground_object requires one short noun phrase for one entity. "
            "Do not use 'and', commas, semicolons, lists, or multiple targets. "
            "Retry with one query such as 'banana' or 'green arrow'."
        )
    image_path = get_current_image()
    processor = get_sam3_processor()
    state = _get_image_state(image_path)
    processor.reset_all_prompts(state)
    if DEVICE == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = processor.set_text_prompt(state=state, prompt=object_name)
        torch.cuda.synchronize()
    else:
        output = processor.set_text_prompt(state=state, prompt=object_name)

    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    raw_instances = [
        {"bbox": [float(value) for value in box], "score": float(score)}
        for box, score in zip(boxes, scores)
    ]
    raw_bboxes = [instance["bbox"] for instance in raw_instances]
    grouped = group_bboxes(raw_bboxes)
    regions = []
    for index, bbox in enumerate(grouped):
        tight_crop_path = crop_bbox(bbox, object_name, index)
        regions.append(
            {
                "bbox": bbox,
                "crop_path": tight_crop_path,
                "tight_crop_path": tight_crop_path,
                "context_crop_path": crop_context_bbox(
                    bbox,
                    object_name,
                    index,
                ),
            }
        )
    return {
        "found": bool(raw_instances), "query": object_name,
        "query_is_candidate_only": True,
        "raw_instance_count": len(raw_instances), "region_count": len(regions),
        "regions": regions, "raw_instances": raw_instances,
    }