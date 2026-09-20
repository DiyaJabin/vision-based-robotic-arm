from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results"
TRIAL_COLUMNS = [
    "trial_id",
    "method",
    "num_objects",
    "objects_detected",
    "valid_detections",
    "selected_object",
    "selected_class",
    "detection_confidence",
    "validation_score",
    "graspability_score",
    "perception_latency_ms",
    "ik_feasible",
    "grasp_attempted",
    "grasp_success",
    "placement_success",
    "retry_count",
    "failure_reason",
    "total_execution_time_s",
]


def ensure_result_dir(output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir is not None else DEFAULT_RESULT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _coerce_numeric(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_trial_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {name: record.get(name) for name in TRIAL_COLUMNS}
    row["trial_id"] = record.get("trial_id", 0)
    row["method"] = str(record.get("method", "unknown"))
    row["num_objects"] = int(record.get("num_objects", 0) or 0)
    row["objects_detected"] = int(record.get("objects_detected", 0) or 0)
    row["valid_detections"] = int(record.get("valid_detections", 0) or 0)
    row["selected_object"] = record.get("selected_object")
    row["selected_class"] = record.get("selected_class")
    row["detection_confidence"] = _coerce_numeric(record.get("detection_confidence"))
    row["validation_score"] = _coerce_numeric(record.get("validation_score"))
    row["graspability_score"] = _coerce_numeric(record.get("graspability_score"))
    row["perception_latency_ms"] = _coerce_numeric(record.get("perception_latency_ms"))
    row["ik_feasible"] = bool(record.get("ik_feasible", False))
    row["grasp_attempted"] = bool(record.get("grasp_attempted", False))
    row["grasp_success"] = bool(record.get("grasp_success", False))
    row["placement_success"] = bool(record.get("placement_success", False))
    row["retry_count"] = int(record.get("retry_count", 0) or 0)
    row["failure_reason"] = str(record.get("failure_reason") or "unknown")
    row["total_execution_time_s"] = _coerce_numeric(record.get("total_execution_time_s"))
    return row


class TrialLogger:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = ensure_result_dir(output_dir)
        self.csv_path = self.output_dir / "trial_results.csv"
        self.json_path = self.output_dir / "trial_results.json"

    def log_trial(self, record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _normalize_trial_record(record)
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
        else:
            df = pd.DataFrame([normalized])
        df = df.reindex(columns=TRIAL_COLUMNS, fill_value=None)
        df.to_csv(self.csv_path, index=False)

        json_rows = []
        if self.json_path.exists():
            try:
                json_rows = json.loads(self.json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                json_rows = []
        json_rows.append(normalized)
        self.json_path.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
        return normalized


def load_trial_results(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(csv_path) if csv_path is not None else DEFAULT_RESULT_DIR / "trial_results.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def aggregate_trials(rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> dict[str, Any]:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        frame = pd.DataFrame(list(rows))

    if frame.empty:
        return {
            "total_trials": 0,
            "successful_trials": 0,
            "failed_trials": 0,
            "success_rate": 0.0,
            "placement_success": 0,
            "mean_latency": 0.0,
            "median_latency": 0.0,
            "retry_count": 0,
            "failure_category_counts": {},
        }

    frame = frame.copy()
    for column in ("grasp_success", "placement_success", "retry_count"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        else:
            frame[column] = 0
    for column in ("total_execution_time_s", "perception_latency_ms"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        else:
            frame[column] = 0.0
    if "objects_detected" not in frame.columns:
        if "num_objects" in frame.columns:
            frame["objects_detected"] = pd.to_numeric(frame["num_objects"], errors="coerce").fillna(0).astype(float)
        else:
            frame["objects_detected"] = 0.0
    if "valid_detections" not in frame.columns:
        frame["valid_detections"] = pd.to_numeric(frame["objects_detected"], errors="coerce").fillna(0).astype(float)
    if "grasp_attempted" not in frame.columns:
        frame["grasp_attempted"] = frame.get("grasp_success", 0).fillna(0).astype(float) > 0

    success_count = int(frame["grasp_success"].astype(float).sum())
    placement_count = int(frame["placement_success"].astype(float).sum())
    total_trials = int(len(frame))
    total_time = float(frame["total_execution_time_s"].fillna(0.0).sum())
    latency_source = frame["perception_latency_ms"].fillna(0.0)
    if latency_source.sum() == 0 and total_trials:
        latency_source = frame["total_execution_time_s"].fillna(0.0)
    retry_total = int(frame["retry_count"].fillna(0).sum())

    from experiments.failure_analysis import group_failure_reasons

    return {
        "total_trials": total_trials,
        "successful_trials": success_count,
        "failed_trials": max(total_trials - success_count, 0),
        "success_rate": (success_count / total_trials) if total_trials else 0.0,
        "placement_success": placement_count,
        "mean_latency": float(latency_source.mean()) if total_trials else 0.0,
        "median_latency": float(latency_source.median()) if total_trials else 0.0,
        "mean_total_execution_time_s": float(total_time / total_trials) if total_trials else 0.0,
        "retry_count": retry_total,
        "failure_category_counts": group_failure_reasons(frame),
    }
