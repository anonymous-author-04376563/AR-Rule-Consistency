from math import hypot


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _iou(first: list[float], second: list[float]) -> float:
    intersection = _area([
        max(first[0], second[0]), max(first[1], second[1]),
        min(first[2], second[2]), min(first[3], second[3]),
    ])
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _close(first: list[float], second: list[float]) -> bool:
    first_scale = max(first[2] - first[0], first[3] - first[1])
    second_scale = max(second[2] - second[0], second[3] - second[1])
    distance = hypot(
        (first[0] + first[2] - second[0] - second[2]) / 2,
        (first[1] + first[3] - second[1] - second[3]) / 2,
    )
    return distance <= (first_scale + second_scale) * 0.75 / 2


def group_bboxes(bboxes: list[list[float]]) -> list[list[float]]:
    groups: list[list[float]] = []
    for raw_bbox in bboxes:
        bbox = [float(value) for value in raw_bbox]
        for index, group in enumerate(groups):
            if _iou(group, bbox) >= 0.3 or _close(group, bbox):
                groups[index] = [
                    min(group[0], bbox[0]), min(group[1], bbox[1]),
                    max(group[2], bbox[2]), max(group[3], bbox[3]),
                ]
                break
        else:
            groups.append(bbox)
    return groups