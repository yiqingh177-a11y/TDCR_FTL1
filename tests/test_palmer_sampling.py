"""Study bookkeeping checks without executing any optimizer or benchmark rollout."""

import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from continuum_ftl.benchmarks.palmer_s3 import load_settings
from continuum_ftl.benchmarks.palmer_s3_sampling import (
    changed_settings,
    sequence_difference,
    summary_row,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmarks/palmer_s3.json"


class SamplingStudyTests(unittest.TestCase):
    def test_only_optimization_spacing_changes(self):
        original = load_settings(CONFIG)
        for spacing in (0.005, 0.002, 0.001):
            new = changed_settings(original, spacing)
            before, after = asdict(original), asdict(new)
            self.assertEqual(after.pop("optimization_spacing_m"), spacing)
            before.pop("optimization_spacing_m")
            self.assertEqual(before, after)

    def test_common_prefix_and_rejected_trial_are_not_mixed_with_full_rollout(self):
        steps, diagnostics = [], []
        for accepted, magnitude in ((True, 1.0), (True, 9.0), (False, 999.0)):
            steps.append(
                {
                    "accepted": accepted,
                    "max_body_to_reference_m": magnitude / 1000,
                    "mean_body_to_reference_m": magnitude / 2000,
                    "max_body_to_previous_m": magnitude / 3000,
                    "tip_error_m": magnitude / 10000,
                    "solver_iterations": 1,
                }
            )
            diagnostics.append(
                {
                    "accepted": accepted,
                    "optimization_body_error_m": magnitude / 4000,
                    "fine_minus_optimization_m": magnitude / 12000,
                    "fine_near_tip_5mm_m": magnitude / 3000,
                    "optimization_near_tip_5mm_m": magnitude / 4000,
                    "exact_2mm_behind_tip_error_m": magnitude / 5000,
                }
            )
        result = {
            "steps": steps,
            "completed": False,
            "accepted_steps": 2,
            "failure_reason": "test failure",
            "total_step_time_s": 3.0,
            "mean_step_time_s": 1.0,
            "total_rollout_time_s": 4.0,
        }
        full = summary_row("5", result, diagnostics)
        prefix = summary_row("5", result, diagnostics, 1)
        self.assertEqual(full["body_to_reference_max_mm"], 9.0)
        self.assertAlmostEqual(full["body_to_reference_mean_frame_max_mm"], 5.0)
        self.assertEqual(prefix["body_to_reference_max_mm"], 1.0)
        self.assertEqual(prefix["evaluated_accepted_steps"], 1)
        self.assertEqual(prefix["accepted_steps"], 2)

    def test_baseline_sequence_has_zero_difference(self):
        settings = load_settings(CONFIG)
        rows = [
            {
                "accepted": True,
                "reference_progress_m": d,
                "configuration_rad": [0.0] * 4,
                "feed_m": d,
                "actual_tip_yz_m": [0.0, d],
            }
            for d in (0.002, 0.004)
        ]
        result = {"steps": rows, "reference_arc_lengths_m": np.linspace(0, 0.4, 1601).tolist()}
        difference = sequence_difference(settings, result, result)
        self.assertEqual(difference["common_accepted_steps"], 2)
        self.assertEqual(difference["max_section_angle_difference_deg"], 0.0)
        self.assertEqual(difference["max_full_body_world_position_difference_mm"], 0.0)

    def test_different_progress_cannot_be_silently_compared(self):
        row = {"accepted": True, "reference_progress_m": 0.002}
        old = {"steps": [row], "reference_arc_lengths_m": [0.0, 0.4]}
        current = {"steps": [{**row, "reference_progress_m": 0.004}]}
        with self.assertRaisesRegex(ValueError, "does not align"):
            sequence_difference(load_settings(CONFIG), current, old)
