import base64
import json
from pathlib import Path
from typing import Any

from agents import Runner
from agents.items import ToolCallItem, ToolCallOutputItem

from ..agents import create_agent
from ..configs import ExperimentConfig
from ..data import find_image, find_record
from ..tools import get_sam3_processor, set_current_image


def _data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image format: {suffix}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def _tool_trace(result: Any) -> list[dict[str, Any]]:
    traces = []
    by_call_id = {}
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            call_id = getattr(raw, "call_id", None) or getattr(raw, "id", None)
            entry = {"call_id": call_id, "tool_name": getattr(raw, "name", "unknown"), "arguments": getattr(raw, "arguments", None), "output": None}
            traces.append(entry)
            if call_id is not None:
                by_call_id[call_id] = entry
        elif isinstance(item, ToolCallOutputItem):
            entry = by_call_id.get(item.call_id)
            if entry is not None:
                output = item.output
                try:
                    output = json.loads(output) if isinstance(output, str) else output
                except json.JSONDecodeError:
                    pass
                entry["output"] = output
    return traces


async def run_experiment(
    config: ExperimentConfig,
    agent: Any | None = None,
    max_turns: int = 30,
) -> dict[str, Any]:
    record = find_record(config.scene, config.rule, config.inconsistency_type)
    image_path = find_image(record["Image Name"])
    set_current_image(image_path)
    get_sam3_processor()
    if agent is None:
        agent = create_agent(config.model, config.thinking)
    user_text = (
        "Inspect this AR image and evaluate the application rule below. "
        "Use visual tools when useful and account for every relevant entity. "
        "Carefully distinguish physical markings, printed borders, and object "
        "edges from virtual AR overlays. Use scene geometry, occlusion, "
        "alignment, and overlay style as evidence; contrast or edge sharpness "
        "is supporting evidence only, not a decisive test. "
        "When associating an arrow with an object, use its direction and tip, "
        "relative placement, alignment, and occlusion; proximity alone is not "
        "sufficient. "
        "When the rule mainly concerns an AR cue's color, shape, size, style, "
        "or presence, consider grounding the virtual cue first; if it is unclear "
        "or its target relation is difficult to establish, ground relevant "
        "physical objects as a complementary route. This is a preference, not a "
        "fixed order. "
        "Treat every ground_object category as a candidate location rather than "
        "a verified identity. Similar-looking signs or icons may be confused by "
        "grounding; verify identity from the crop and full image before using it. "
        "For each evaluated entity, reason in this order: observed object, "
        "whether it is a target under the rule's positive condition, required "
        "AR content for that target status, observed AR content, binary "
        "verdict, and supporting reason. Distinguish target membership from "
        "the fact that an object is being evaluated. If an object is a target, "
        "apply the rule's required positive marking to it; do not apply the "
        "alternative marking allowed for non-target objects. Keep the "
        "observation separate from the rule verdict. Before finalizing, check that every "
        "reason supports its verdict: a reason that says the rule is violated "
        "cannot have a CONSISTENT verdict, and a reason that says the rule is "
        "satisfied cannot have an INCONSISTENT verdict.\n\n"
        f"Application: {record['Application']}\nRule: {record['Rule']}"
    )
    result = await Runner.run(
        agent,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": _data_url(image_path)},
            ],
        }],
        max_turns=max_turns,
    )
    return {
        "config": config.__dict__,
        "metadata": record,
        "image_path": str(image_path),
        "agent_output": result.final_output,
        "tool_trace": _tool_trace(result),
    }