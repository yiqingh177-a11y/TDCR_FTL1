"""Regression for a real frozen status-9 step and effective numerical settings."""

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from continuum_ftl.config import load_config
from continuum_ftl.deployment import compute_deployment
from continuum_ftl.environment_exposure import extract_environment_exposure
from continuum_ftl.ftl_optimizer import solve_ftl_step
from continuum_ftl.ftl_rollout import ftl_report, run_single_path_ftl
from continuum_ftl.models import FTLConfig, FTLStep
from continuum_ftl.reference_path import extract_reference_path, interpolate_reference_path

FIXTURE = Path(__file__).parent / "fixtures/status9_staircase.json"


class SolverPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data["config"]), encoding="utf-8")
            config = load_config(path)
        cls.config = replace(
            config,
            ftl=replace(
                config.ftl,
                body_reference_mode="previous_body",
                finite_difference_step=FTLConfig().finite_difference_step,
            ),
        )
        cls.reference = extract_reference_path(
            cls.config, np.array(data["reference_configuration_rad"])
        )
        old = data["previous_state"]
        state = compute_deployment(
            config.robot,
            config.deployment,
            old["distance_m"],
            np.array(old["configuration_rad"]),
            config.ftl.validation_body_spacing_m,
        )
        exposure = extract_environment_exposure(
            state,
            config.deployment.environment_entry_z_m,
            max_arc_spacing_m=config.ftl.validation_body_spacing_m,
            tolerance_m=config.deployment.geometry_tolerance_m,
        )
        values = data["previous_step"]
        for key in (
            "configuration_increment_rad",
            "desired_tip_position_m",
            "worst_body_point_m",
            "historical_tip_path_m",
            "comparison_path_m",
        ):
            if values[key] is not None:
                values[key] = np.asarray(values[key])
        values["optimized_section_indices"] = tuple(values["optimized_section_indices"])
        cls.previous = FTLStep(state=state, exposure=exposure, **values)
        cls.comparison = np.array(data["comparison_path_m"])
        cls.progress = data["desired_progress_m"]
        count = int(np.ceil(cls.reference.arc_lengths_m[-1] / config.ftl.comparison_spacing_m))
        cls.planned = interpolate_reference_path(
            cls.reference, np.linspace(0, cls.reference.arc_lengths_m[-1], count + 1)
        )

    def test_default_step_matches_scipy_numerical_scale_and_legacy_is_selectable(self):
        self.assertEqual(FTLConfig().finite_difference_step, np.sqrt(np.finfo(float).eps))
        legacy = replace(self.config.ftl, finite_difference_step=1e-5)
        self.assertEqual(legacy.finite_difference_step, 1e-5)

    def test_frozen_failed_step_now_converges_without_changing_problem_or_acceptance(self):
        old_curve = self.previous.state.forward.centerline_m.copy()
        comparison = self.comparison.copy()
        result = solve_ftl_step(
            self.config,
            self.reference,
            self.previous,
            self.progress,
            self.comparison,
            self.planned,
        )
        self.assertTrue(result.solver_success, result.solver_message)
        self.assertTrue(result.accepted, result.failure_reason)
        self.assertLess(result.solver_iterations, self.config.ftl.max_iterations)
        self.assertLess(result.tip_error_m, 1e-8)
        self.assertGreater(result.body_following_error_max_m, 0)  # Real error is not erased.
        self.assertLessEqual(result.feed_increment_m, self.config.ftl.max_feed_increment_m)
        self.assertTrue(
            np.all(
                np.abs(result.configuration_increment_rad)
                <= self.config.ftl.max_bend_increment_rad + 1e-8
            )
        )
        self.assertEqual(result.state.consistency_warnings, ())
        np.testing.assert_array_equal(old_curve, self.previous.state.forward.centerline_m)
        np.testing.assert_array_equal(comparison, self.comparison)

    def test_solver_failure_still_stops_and_effective_settings_are_archived(self):
        def failed(fun, x0, **kwargs):
            self.assertEqual(kwargs["options"]["eps"], self.config.ftl.finite_difference_step)
            return SimpleNamespace(x=x0, success=False, status=9, message="iteration limit", nit=1)

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=failed):
            result = run_single_path_ftl(self.config, self.reference)
        self.assertFalse(result.steps[-1].accepted)
        self.assertIn("SOLVER_FAILED", result.failure_reason)
        report = ftl_report(result, self.reference)
        self.assertEqual(report["effective_ftl_settings"], asdict(self.config.ftl))


if __name__ == "__main__":
    unittest.main()
