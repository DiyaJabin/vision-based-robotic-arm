from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_method_results(frame: pd.DataFrame) -> dict[str, object]:
    if frame is None or frame.empty:
        return {
            "total_trials": 0,
            "methods": [],
            "summary": {},
        }

    frame = frame.copy()
    for column in ("method", "objects_detected", "valid_detections", "grasp_attempted", "grasp_success", "placement_success", "perception_latency_ms", "total_execution_time_s", "retry_count", "failure_reason"):
        if column not in frame.columns:
            if column == "method":
                frame[column] = "unknown"
            elif column in {"objects_detected", "valid_detections"}:
                frame[column] = 0
            elif column == "grasp_attempted":
                frame[column] = False
            elif column in {"grasp_success", "placement_success"}:
                frame[column] = False
            elif column in {"perception_latency_ms", "total_execution_time_s"}:
                frame[column] = 0.0
            elif column == "retry_count":
                frame[column] = 0
            else:
                frame[column] = "unknown"

    summary: dict[str, dict[str, float | int | str]] = {}
    for method in sorted(frame["method"].dropna().unique().tolist()):
        subset = frame[frame["method"] == method].copy()
        total = len(subset)
        attempts = subset["grasp_attempted"].fillna(False).astype(bool).sum()
        successful = subset["grasp_success"].fillna(False).astype(bool).sum()
        placement_success = subset["placement_success"].fillna(False).astype(bool).sum()
        valid = subset["valid_detections"].fillna(0).astype(float)
        detections = subset["objects_detected"].fillna(0).astype(float)
        detection_count = float(detections.sum()) if not detections.empty else 0.0
        valid_rate = float(valid.sum() / max(total, 1)) if total else 0.0
        attempt_rate = float(attempts / max(total, 1)) if total else 0.0
        success_rate = float(successful / max(total, 1)) if total else 0.0
        placement_rate = float(placement_success / max(total, 1)) if total else 0.0
        avg_latency = float(subset["perception_latency_ms"].fillna(0.0).mean()) if total else 0.0
        if avg_latency == 0.0 and total:
            avg_latency = float(subset["total_execution_time_s"].fillna(0.0).mean()) if total else 0.0
        avg_total_time = float(subset["total_execution_time_s"].fillna(0.0).mean()) if total else 0.0
        retries = subset["retry_count"].fillna(0).astype(float).sum()
        retry_rate = float(retries / max(total, 1)) if total else 0.0
        failure_counts = int((subset["failure_reason"].notna() & subset["failure_reason"].astype(str).ne("unknown")).sum()) if total else 0
        summary[method] = {
            "total_trials": total,
            "detection_count": detection_count,
            "valid_detection_rate": valid_rate,
            "grasp_attempt_rate": attempt_rate,
            "grasp_success_rate": success_rate,
            "placement_success_rate": placement_rate,
            "average_perception_latency_ms": avg_latency,
            "average_total_execution_time_s": avg_total_time,
            "retry_rate": retry_rate,
            "failure_counts": failure_counts,
        }
    return {"total_trials": int(len(frame)), "methods": sorted(summary), "summary": summary}


def compare_methods_from_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[
            "total_trials",
            "detection_count",
            "valid_detection_rate",
            "grasp_attempt_rate",
            "grasp_success_rate",
            "placement_success_rate",
            "average_perception_latency_ms",
            "average_total_execution_time_s",
            "retry_rate",
            "failure_counts",
        ])

    frame = frame.copy()
    for column in ("method", "objects_detected", "valid_detections", "grasp_attempted", "grasp_success", "placement_success", "perception_latency_ms", "total_execution_time_s", "retry_count", "failure_reason"):
        if column not in frame.columns:
            if column == "method":
                frame[column] = "unknown"
            elif column in {"objects_detected", "valid_detections"}:
                frame[column] = 0
            elif column == "grasp_attempted":
                frame[column] = False
            elif column in {"grasp_success", "placement_success"}:
                frame[column] = False
            elif column in {"perception_latency_ms", "total_execution_time_s"}:
                frame[column] = 0.0
            elif column == "retry_count":
                frame[column] = 0
            else:
                frame[column] = "unknown"

    data = []
    for method in sorted(frame["method"].dropna().unique().tolist()):
        subset = frame[frame["method"] == method].copy()
        total = len(subset)
        detection_count = int(subset["objects_detected"].fillna(0).astype(float).sum())
        valid_rate = float(subset["valid_detections"].fillna(0).astype(float).sum() / max(total, 1))
        attempt_rate = float(subset["grasp_attempted"].fillna(False).astype(bool).sum() / max(total, 1))
        success_rate = float(subset["grasp_success"].fillna(False).astype(bool).sum() / max(total, 1))
        placement_rate = float(subset["placement_success"].fillna(False).astype(bool).sum() / max(total, 1))
        avg_latency = float(subset["perception_latency_ms"].fillna(0.0).mean()) if total else 0.0
        if avg_latency == 0.0 and total:
            avg_latency = float(subset["total_execution_time_s"].fillna(0.0).mean()) if total else 0.0
        avg_total_time = float(subset["total_execution_time_s"].fillna(0.0).mean()) if total else 0.0
        retry_rate = float(subset["retry_count"].fillna(0).astype(float).sum() / max(total, 1))
        failure_counts = int(subset["failure_reason"].fillna("unknown").astype(str).ne("unknown").sum())
        data.append({
            "method": method,
            "total_trials": total,
            "detection_count": detection_count,
            "valid_detection_rate": valid_rate,
            "grasp_attempt_rate": attempt_rate,
            "grasp_success_rate": success_rate,
            "success_rate": success_rate,
            "placement_success_rate": placement_rate,
            "average_perception_latency_ms": avg_latency,
            "average_total_execution_time_s": avg_total_time,
            "retry_rate": retry_rate,
            "failure_counts": failure_counts,
        })
    return pd.DataFrame(data).set_index("method")


def compare_methods(csv_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(csv_path) if csv_path is not None else Path(__file__).resolve().parent / "results" / "trial_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return compare_methods_from_dataframe(pd.read_csv(path))
