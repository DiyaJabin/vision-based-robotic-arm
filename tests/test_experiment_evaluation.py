import os
import tempfile
import unittest

import pandas as pd

from experiments.compare_methods import compare_methods_from_dataframe, summarize_method_results
from experiments.failure_analysis import classify_failure, group_failure_reasons
from experiments.logger import TrialLogger, aggregate_trials, load_trial_results
from experiments.run_trials import TrialConfig, describe_runtime_status
from experiments.plot_results import plot_results


class ExperimentLoggerTests(unittest.TestCase):
    def test_logger_writes_expected_schema_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrialLogger(output_dir=tmpdir)
            logger.log_trial({
                "trial_id": 1,
                "method": "opencv",
                "num_objects": 3,
                "objects_detected": 2,
                "valid_detections": 2,
                "selected_object": 7,
                "selected_class": "cube",
                "detection_confidence": 0.91,
                "validation_score": 0.88,
                "graspability_score": 0.73,
                "perception_latency_ms": 120.5,
                "ik_feasible": True,
                "grasp_attempted": True,
                "grasp_success": True,
                "placement_success": True,
                "retry_count": 1,
                "failure_reason": "destination_contact_verified",
                "total_execution_time_s": 5.4,
            })
            csv_path = os.path.join(tmpdir, "trial_results.csv")
            json_path = os.path.join(tmpdir, "trial_results.json")
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(json_path))
            frame = pd.read_csv(csv_path)
            self.assertIn("trial_id", frame.columns)
            self.assertIn("method", frame.columns)
            self.assertIn("failure_reason", frame.columns)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["method"], "opencv")

    def test_aggregation_and_failure_grouping(self):
        rows = [
            {"trial_id": 1, "method": "opencv", "grasp_success": 1, "placement_success": 1, "failure_reason": "destination_contact_verified", "retry_count": 0, "total_execution_time_s": 2.0},
            {"trial_id": 2, "method": "opencv", "grasp_success": 0, "placement_success": 0, "failure_reason": "low_confidence", "retry_count": 2, "total_execution_time_s": 4.0},
            {"trial_id": 3, "method": "hybrid", "grasp_success": 1, "placement_success": 0, "failure_reason": "placement_failure", "retry_count": 1, "total_execution_time_s": 3.0},
            {"trial_id": 4, "method": "hybrid", "grasp_success": 0, "placement_success": 0, "failure_reason": "runtime_unavailable", "retry_count": 0, "total_execution_time_s": 0.0},
        ]
        summary = aggregate_trials(rows)
        self.assertEqual(summary["total_trials"], 4)
        self.assertAlmostEqual(summary["success_rate"], 0.25, places=3)
        self.assertAlmostEqual(summary["mean_latency_ms"], 0.0, places=3)
        self.assertEqual(summary["failure_category_counts"]["low confidence"], 1)
        self.assertEqual(summary["failure_category_counts"]["runtime unavailable"], 1)

        grouped = group_failure_reasons(pd.DataFrame(rows))
        self.assertIn("low confidence", grouped)
        self.assertIn("runtime unavailable", grouped)

    def test_method_comparison_and_empty_data_handling(self):
        rows = [
            {"method": "opencv", "grasp_success": 1, "placement_success": 1, "failure_reason": "destination_contact_verified", "retry_count": 0, "total_execution_time_s": 2.0},
            {"method": "opencv", "grasp_success": 0, "placement_success": 0, "failure_reason": "low_confidence", "retry_count": 1, "total_execution_time_s": 3.0},
            {"method": "yolo", "grasp_success": 1, "placement_success": 1, "failure_reason": "destination_contact_verified", "retry_count": 0, "total_execution_time_s": 2.5},
        ]
        comparison = compare_methods_from_dataframe(pd.DataFrame(rows))
        self.assertIn("opencv", comparison.index)
        self.assertIn("yolo", comparison.index)
        self.assertAlmostEqual(float(comparison.loc["opencv", "success_rate"]), 0.5, places=3)

        empty = summarize_method_results(pd.DataFrame([]))
        self.assertEqual(empty["total_trials"], 0)

    def test_trial_config_and_runtime_status_reporting(self):
        config = TrialConfig(trials=3, seed=7, method="hybrid")
        self.assertEqual(config.trials, 3)
        self.assertEqual(config.seed, 7)
        self.assertEqual(config.method, "hybrid")

        self.assertEqual(describe_runtime_status(missing_runtime=True), "SKIPPED / NOT AVAILABLE")
        self.assertEqual(describe_runtime_status(missing_runtime=False), "AVAILABLE")

    def test_load_trial_results_handles_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(load_trial_results(os.path.join(tmpdir, "missing.csv")), [])

    def test_failure_categories_preserve_placement_and_timeout_semantics(self):
        self.assertEqual(classify_failure("motion_timeout"), "timeout")
        grouped = group_failure_reasons([
            {"grasp_success": True, "placement_success": False, "failure_reason": "placement_failure"},
            {"grasp_success": False, "placement_success": False, "failure_reason": "motion_timeout"},
            {"grasp_success": True, "placement_success": True, "failure_reason": "destination_contact_verified"},
        ])
        self.assertEqual(grouped["placement failure"], 1)
        self.assertEqual(grouped["timeout"], 1)
        self.assertNotIn("success", grouped)

    def test_plots_use_grasp_success_and_execution_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = pd.DataFrame([
                {"method": "opencv", "grasp_success": "True", "total_execution_time_s": 3.24},
                {"method": "opencv", "grasp_success": "False", "total_execution_time_s": 3.24},
                {"method": "yolo", "grasp_success": "True", "total_execution_time_s": 5.61},
                {"method": "hybrid", "grasp_success": "False", "total_execution_time_s": 4.03},
            ])
            plots = plot_results(frame, tmpdir)
            self.assertEqual([path.name for path in plots], [
                "grasp_success_rate_by_method.png", "mean_execution_time_by_method.png"
            ])
            self.assertTrue(all(path.exists() for path in plots))


if __name__ == "__main__":
    unittest.main()
