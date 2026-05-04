"""
Inspection rule mapping -- Chinese keywords to COCO class names.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# (chinese_keywords, target_coco_classes, person_required)
RULE_KEYWORD_MAP: List[Tuple[List[str], List[str], bool]] = [
    # Smoking -- person + handheld objects as heuristic proxy
    (["抽烟", "吸烟", "香烟", "smoking"], ["person", "cell phone", "bottle", "cup"], True),
    # Trash / litter on ground
    (["垃圾", "杂物", "废弃", "trash", "litter", "garbage", "rubbish"],
     ["bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange",
      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
      "suitcase", "backpack", "handbag", "book", "umbrella",
      "sports ball", "teddy bear", "vase", "potted plant"], False),
    # Person in restricted area
    (["人员", "行人", "陌生人", "无关人员", "intruder", "trespasser"],
     ["person"], True),
    # Fire hazard objects
    (["火源", "火灾", "易燃", "fire"],
     ["bottle", "laptop", "cell phone", "book"], False),
    # Suspicious items
    (["可疑", "异常物品", "不明物体", "suspicious"],
     ["backpack", "suitcase", "handbag"], False),
    # Animals
    (["动物", "猫", "狗", "鸟", "animal", "cat", "dog", "bird"],
     ["cat", "dog", "bird", "horse", "sheep", "cow", "bear", "zebra", "giraffe"], False),
    # Vehicles
    (["车辆", "违规停车", "vehicle", "car"],
     ["car", "motorcycle", "truck", "bus", "bicycle"], False),
]


def match_rule_to_classes(rule_text: str) -> Tuple[List[str], bool]:
    """Given rule text, return (target_classes, person_required)."""
    text_lower = rule_text.lower()
    for keywords, classes, person_req in RULE_KEYWORD_MAP:
        for kw in keywords:
            if kw in text_lower:
                return list(classes), person_req
    return ["person"], True


def create_rule(text: str) -> dict:
    """Create a rule dict from user text input."""
    classes, person_req = match_rule_to_classes(text)
    return {
        "id": "rule_" + uuid.uuid4().hex[:6],
        "text": text.strip(),
        "target_classes": classes,
        "person_related": person_req,
        "enabled": True,
        "created_at": time.time(),
    }


def evaluate_rule(rule: dict, detected_objects: List[dict]) -> Optional[dict]:
    """Check detected objects against a rule. Returns match dict or None."""
    target_classes = rule.get("target_classes", [])
    person_required = rule.get("person_related", False)

    matched = []
    has_person = False
    for obj in detected_objects:
        label = (obj.get("label") or "").lower()
        if label not in target_classes:
            continue
        matched.append(obj)
        if label == "person":
            has_person = True

    if person_required and not has_person:
        return None
    if not matched:
        return None

    confidences = [o.get("confidence", 0) for o in matched]
    max_conf = max(confidences)
    labels = [o.get("label", "?") for o in matched]
    bboxes = [o.get("bbox", []) for o in matched]

    return {
        "detected_objects": labels,
        "confidence": round(max_conf, 3),
        "bounding_boxes": bboxes,
    }
