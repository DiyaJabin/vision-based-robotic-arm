from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd


FAILURE_CATEGORY_PATTERNS = {
    "no detection": (
        "no detection",
        "no_valid_targets",
        "no valid targets",
        "no_objects_detected",
        "empty_detection",
        "not detected",
        "detection_missing",
        "no_detection",
    ),
    "low confidence": (
        "low confidence",
        "low_confidence",
        "invalid_detector_confidence",
        "confidence_too_low",
        "confidence low",
    ),
    "failed validation": (
        "failed validation",
        "validation_failed",
        "opencv_fail",
        "no_colour_contour",
        "no color contour",
        "invalid_bbox",
        "unsupported_class",
    ),
    "unreachable": (
        "unreachable",
        "ik_residual",
        "arm_collision",
        "joint_limits",
        "collision",
        "reachability_failed",
    ),
    "insufficient clearance": (
        "insufficient clearance",
        "clearance",
        "table_edge_clearance",
        "gripper_clearance",
        "clearance_failure",
    ),
    "IK failure": (
        "ik failure",
        "ik_failed",
        "ik_failure",
        "inverse_kinematics_failed",
    ),
    "grasp failure": (
        "grasp failure",
        "grasp_failed",
        "grasp_failure",
        "close_gripper_failure",
        "motion_timeout",
        "timeout",
    ),
    "placement failure": (
        "placement failure",
        "placement_failed",
        "placement_failure",
        "placement_not_supported_in_destination",
        "destination_contact_verified",
    ),
    "timeout": (
        "timeout",
        "timed_out",
        "observation_limit",
        "pick_attempt_limit",
    ),
    "runtime unavailable": (
        "runtime unavailable",
        "runtime_unavailable",
        "model unavailable",
        "model_unavailable",
        "weights_missing",
        "not available",
        "skipped",
        "not_available",
    ),
}

SUCCESS_MARKERS = ("destination_contact_verified", "all_objects_placed", "verified")


def classify_failure(reason: object) -> str:
    """Map a raw failure reason to a stable evaluation category."""
    if reason is None or (isinstance(reason, float) and pd.isna(reason)):
        return "unknown"
    text = str(reason).strip().lower()
    if not text:
        return "unknown"
    if any(marker in text for marker in SUCCESS_MARKERS):
        return "success"
    for label, patterns in FAILURE_CATEGORY_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return label
    return "unknown"


def group_failure_reasons(data) -> dict[str, int]:
    """Summarize failure categories from a DataFrame or a list of trial records."""
    if isinstance(data, pd.DataFrame):
        rows = data.to_dict(orient="records")
    elif isinstance(data, list):
        rows = data
    else:
        rows = list(data or [])

    counts: Counter[str] = Counter()
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = row.get("failure_reason")
        if row.get("grasp_success") in (1, True, "1", "true"):
            continue
        if row.get("placement_success") in (1, True, "1", "true"):
            continue
        category = classify_failure(reason)
        if category == "success":
            continue
        counts[category] += 1
    return dict(sorted(counts.items()))
