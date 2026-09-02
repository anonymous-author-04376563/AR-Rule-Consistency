import base64
from pathlib import Path

from agents import function_tool
from openai import OpenAI


REGION_VLM_MODEL = "gpt-5.6-luna"
client: OpenAI | None = None


def _image_data_url(image_path: str) -> str:
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Region image not found: {path}")
    media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


@function_tool
def inspect_region(crop_path: str, question: str) -> dict:
    """Report visual evidence from a crop returned by ground_object."""
    global client
    if client is None:
        client = OpenAI()
    response = client.responses.create(
        model=REGION_VLM_MODEL,
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "Inspect only this AR crop. Report observable evidence for the question; do not decide rule consistency or guess."}]},
            {"role": "user", "content": [{"type": "input_image", "image_url": _image_data_url(crop_path)}, {"type": "input_text", "text": question}]},
        ],
    )
    return {"crop_path": crop_path, "question": question, "observation": response.output_text}