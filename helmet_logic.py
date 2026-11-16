from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal, TypeAlias

Box: TypeAlias = tuple[float, float, float, float]
HelmetStatusMode: TypeAlias = Literal["explicit", "paired"] | None


def helmet_label(class_name: str) -> str:
    normalized = class_name.casefold().replace("_", " ").replace("-", " ")
    words = set(normalized.split())
    compact = normalized.replace(" ", "")

    if "without" in words or "nohelmet" in compact or (
        "no" in words and "helmet" in words
    ):
        return "No helmet"
    if "helmet" in words and words.intersection({"with", "wearing", "worn", "on"}):
        return "With helmet"
    if words.intersection({"motorcycle", "motorcycles", "motorbike", "motorbikes"}):
        return "Motorcycle"
    if "helmet" in words:
        return "Helmet"
    return " ".join(word.capitalize() for word in normalized.split())


def helmet_status_mode(class_names: Iterable[str]) -> HelmetStatusMode:
    labels = {helmet_label(class_name) for class_name in class_names}
    if "No helmet" in labels and labels.intersection({"With helmet", "Helmet"}):
        return "explicit"
    if {"Helmet", "Motorcycle"}.issubset(labels):
        return "paired"
    return None


def associate_helmets(
    motorcycle_boxes: Sequence[Box],
    helmet_boxes: Sequence[Box],
) -> list[bool]:
    statuses = [False] * len(motorcycle_boxes)

    for helmet_box in helmet_boxes:
        helmet_x = (helmet_box[0] + helmet_box[2]) / 2
        helmet_y = (helmet_box[1] + helmet_box[3]) / 2
        candidates: list[tuple[float, int]] = []

        for index, (x1, y1, x2, y2) in enumerate(motorcycle_boxes):
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            if not x1 - width * 0.25 <= helmet_x <= x2 + width * 0.25:
                continue
            if not y1 - height * 0.35 <= helmet_y <= y1 + height * 0.35:
                continue

            distance = abs(helmet_y - y1) / height
            distance += 0.35 * abs(helmet_x - (x1 + x2) / 2) / width
            candidates.append((distance, index))

        if candidates:
            _, closest_motorcycle = min(candidates)
            statuses[closest_motorcycle] = True

    return statuses
