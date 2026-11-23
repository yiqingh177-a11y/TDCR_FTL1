"""Independent S3 geometry checks; no original research deployment involved."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from continuum_ftl.benchmarks.palmer_s3 import (
    geometry,
    load_settings,
    reference_geometry,
    validate_reference,
)
from continuum_ftl.benchmarks.palmer_s3_solver import (
    S3State,
    exposed_samples,
    initial_state,
    run_s3,
    solve_step,
)
from continuum_ftl.planar_geometry import evaluate_curve, sample_interval

CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmarks/palmer_s3.json"


class S3GeometryTests(unittest.TestCase):
    def test_reference_junctions_match_independent_integrals_and_given_coordinates(self):
        settings = load_settings(CONFIG)
        report = validate_reference(settings)
        self.assertLess(report["independent_circle_integral_residual_m"], 1e-14)
        assert_allclose(
            report["section_endpoints_yz_m"],
            [
                [0, 0],
                [0, 0.1],
                [0.0255873, 0.1954930],
                [0.0511746, 0.2909860],
                [0.0511746, 0.3909860],
            ],
            atol=1e-7,
        )
        curve = reference_geometry(settings)
        arcs = sample_interval(curve, 0, 0.4, settings.validation_spacing_m)
        self.assertTrue(report["sampling_contains_all_section_boundaries"])
        self.assertLessEqual(np.max(np.diff(arcs)), settings.validation_spacing_m + 1e-15)
        assert_allclose(evaluate_curve(curve, np.arange(5) * 0.1), curve.junction_positions_m)

    def test_initial_full_robot_has_no_passive_section_and_tip_at_entrance(self):
        curve = geometry(load_settings(CONFIG), np.zeros(4), 0.0)
        assert_allclose(curve.junction_positions_m[:, 0], 0)
        assert_allclose(curve.junction_positions_m[:, 1], [-0.4, -0.3, -0.2, -0.1, 0], atol=1e-15)
        self.assertEqual(len(curve.section_names), 4)

    def test_invalid_reference_and_increment_bounds_are_rejected(self):
        for key, value in (
            ("reference_bends_rad", [0, 0, 0]),
            ("max_feed_increment_m", 0),
            ("min_feed_increment_m", -0.002),
            ("max_iterations", 3.5),
        ):
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            data[key] = value
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_settings(path)


class S3SolverTests(unittest.TestCase):
    def setUp(self):
        self.settings = load_settings(CONFIG)

    def prescribed(self, control):
        def solve(fun, x0, **kwargs):
            self.assertEqual(len(x0), len(control))
            return SimpleNamespace(
                x=np.array(control), success=True, status=0, message="test", nit=1, nfev=1
            )

        return solve

    def test_first_entry_latches_straight_then_allows_partial_section_bending(self):
        state = S3State(
            0.098,
            np.zeros(4),
            np.array([False, False, False, True]),
            geometry(self.settings, np.zeros(4), 0.098),
        )
        with patch(
            "continuum_ftl.benchmarks.palmer_s3_solver.minimize",
            side_effect=self.prescribed([1.0, 0.0]),
        ):
            introduced, row = solve_step(self.settings, state, np.array([0, 0.1]), np.zeros(5))
        self.assertTrue(row["accepted"])
        self.assertEqual(row["newly_introduced_sections"], [2])
        self.assertTrue(introduced.introduced[2])
        self.assertEqual(introduced.bends_rad[2], 0)
        with patch(
            "continuum_ftl.benchmarks.palmer_s3_solver.minimize",
            side_effect=self.prescribed([1.0, 0.01 / 0.064, 0.0]),
        ):
            bent, row = solve_step(self.settings, introduced, np.array([0, 0.102]), np.zeros(5))
        self.assertTrue(row["accepted"])
        self.assertAlmostEqual(bent.bends_rad[2], 0.01)
        # Much of section 3 is still below the interface; no full-release test.
        self.assertLess(bent.curve.junction_positions_m[2, 1], 0)

    def test_real_s3_initial_step_and_fine_body_samples(self):
        state, row = solve_step(
            self.settings, initial_state(self.settings), np.array([0, 0.002]), np.zeros(5)
        )
        self.assertTrue(row["accepted"], row["failure_reason"])
        self.assertLess(row["tip_error_m"], 1e-6)
        points, arcs = exposed_samples(self.settings, state.curve, 0.00025)
        self.assertLessEqual(np.max(np.diff(arcs)), 0.00025 + 1e-12)
        self.assertAlmostEqual(points[0, 1], 0, places=8)
        self.assertEqual(len(state.curve.section_lengths_m), 4)

    def test_failure_does_not_advance_state_or_acceptance_statistics(self):
        failed = SimpleNamespace(
            x=np.array([1.0, 0.0]),
            success=False,
            status=9,
            message="Iteration limit reached",
            nit=500,
            nfev=1000,
        )
        with patch("continuum_ftl.benchmarks.palmer_s3_solver.minimize", return_value=failed):
            result = run_s3(self.settings)
        self.assertFalse(result["completed"])
        self.assertEqual(result["accepted_steps"], 0)
        self.assertEqual(result["rejected_steps"], 1)
        self.assertEqual(result["accepted_reference_progress_m"], 0)
        self.assertEqual(result["final_configuration_rad"], [0.0] * 4)
        self.assertIn("status=9", result["failure_reason"])

    def test_invalid_trial_cannot_pass_feed_increment_check(self):
        with patch(
            "continuum_ftl.benchmarks.palmer_s3_solver.minimize",
            side_effect=self.prescribed([2.0, 0.0]),
        ):
            _, row = solve_step(
                self.settings, initial_state(self.settings), np.array([0, 0.002]), np.zeros(5)
            )
        self.assertFalse(row["accepted"])
        self.assertEqual(row["failure_reason"], "CONTROL_OR_INTRODUCTION_BOUND_VIOLATION")
