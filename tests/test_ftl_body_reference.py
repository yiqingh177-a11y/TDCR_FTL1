"""Body-reference isolation, entry/release boundaries and independent A/B metrics."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from continuum_ftl.config import load_config
from continuum_ftl.deployment import compute_deployment
from continuum_ftl.environment_exposure import extract_environment_exposure
from continuum_ftl.ftl_demo import load_ftl_case
from continuum_ftl.ftl_evaluation import (
    evaluate_ftl_rollout,
    point_to_polyline_distances,
    summarize_ftl_evaluations,
)
from continuum_ftl.ftl_matching import (
    comparison_path,
    match_ordered_polyline,
    previous_body_comparison_path,
)
from continuum_ftl.ftl_rollout import ftl_report, run_single_path_ftl
from continuum_ftl.models import ExposureStatus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robot_2d.json"
FIXTURES = ROOT / "configs/ftl_validation.json"


class BodyReferenceTests(unittest.TestCase):
    def setUp(self):
        loaded = load_config(CONFIG)
        self.config = replace(
            loaded, ftl=replace(loaded.ftl, body_reference_mode="tip_history", max_steps=4)
        )

    def state_and_exposure(self, distance, bends):
        state = compute_deployment(
            self.config.robot, self.config.deployment, distance, np.array(bends), 0.001
        )
        exposure = extract_environment_exposure(
            state, 0.0, max_arc_spacing_m=0.001, tolerance_m=1e-9
        )
        return state, exposure

    def test_first_entry_is_single_point_then_real_geometric_extension(self):
        state, exposure = self.state_and_exposure(0.08, [0, 0, 0])
        self.assertEqual(exposure.status, ExposureStatus.EXPOSED)
        self.assertAlmostEqual(exposure.exposed_length_m, 0)
        waypoint = np.array([0, 0.001])
        path = previous_body_comparison_path(state, exposure, waypoint, 0.00025)
        assert_allclose(path, comparison_path(np.zeros((1, 2)), waypoint, 0.00025), atol=1e-15)
        # Newly exposed material can have a negative tip-anchored expected arc.
        # Its actual lateral discrepancy must still be measured, not set to zero.
        match = match_ordered_polyline(np.array([[0.0005, 0]]), path, np.array([-0.001]), 0.02)
        assert_allclose(match.distances_m, [0.0005], atol=1e-15)
        assert_allclose(match.matched_points_m, [[0, 0]], atol=1e-15)

    def test_partial_exposure_and_release_transitions_preserve_full_pcc(self):
        for distance in (0.0999, 0.1, 0.1001, 0.1799, 0.18, 0.1801, 0.22):
            with self.subTest(distance=distance):
                state, exposure = self.state_and_exposure(distance, [0, 0.02, 0.2])
                original = state.forward.centerline_m.copy()
                waypoint = state.forward.tip_position_m + [0.0002, 0.0005]
                path = previous_body_comparison_path(state, exposure, waypoint, 0.00025)
                assert_allclose(path[0], exposure.entry_position_m, atol=1e-12)
                assert_array_equal(path[-1], waypoint)
                self.assertTrue(np.all(path[:-1, 1] >= -1e-9))
                self.assertLessEqual(
                    np.linalg.norm(np.diff(path, axis=0), axis=1).max(), 0.00025 + 1e-12
                )
                self.assertAlmostEqual(state.section_lengths_m.sum(), 0.38)
                self.assertAlmostEqual(state.section_lengths_m[0], 0.2)
                assert_array_equal(state.forward.centerline_m, original)
                assert_array_equal(state.applied_bends_rad[~state.released], 0)
                self.assertEqual(state.section_names, ("Passive", "C", "B", "A"))

    def test_previous_pcc_resampled_exactly_not_coarse_chord_or_fixed_reference(self):
        state, exposure = self.state_and_exposure(0.22, [0.4, -0.1, 0.3])
        path = previous_body_comparison_path(state, exposure, state.forward.tip_position_m, 0.00025)
        # Curved old body supplies more than a single endpoint or its tip history.
        self.assertGreater(len(path), len(exposure.centerline_m))
        assert_allclose(path[-1], state.forward.tip_position_m, atol=1e-15)
        self.assertGreater(np.ptp(path[:, 0]), 0.005)
        # Merely densifying the old 1 mm chords cannot reproduce the PCC bow.
        self.assertGreater(point_to_polyline_distances(path, exposure.centerline_m).max(), 1e-8)
        with self.assertRaises(ValueError):
            previous_body_comparison_path(
                state, replace(exposure, status=ExposureStatus.AMBIGUOUS), path[-1], 0.00025
            )

    def test_passive_is_excluded_even_if_its_centreline_enters_environment(self):
        state, exposure = self.state_and_exposure(0.30, [0.4, -0.1, 0.3])
        self.assertGreater(exposure.section_exposed_lengths_m[0], 0)
        path = previous_body_comparison_path(state, exposure, state.forward.tip_position_m, 0.00025)
        assert_array_equal(path[0], state.active_base_position_m)
        self.assertTrue(np.all(path[:, 1] >= state.active_base_position_m[1] - 1e-12))

    def test_default_and_explicit_tip_history_reproduce_identical_states(self):
        cfg, reference = load_ftl_case(self.config, "single-bend", FIXTURES, "unused")
        with patch(
            "continuum_ftl.ftl_rollout.previous_body_comparison_path",
            side_effect=AssertionError("legacy mode must not use previous body"),
        ):
            implicit = run_single_path_ftl(cfg, reference)
            explicit = run_single_path_ftl(
                replace(cfg, ftl=replace(cfg.ftl, body_reference_mode="tip_history")), reference
            )
        self.assertEqual(implicit.failure_reason, explicit.failure_reason)
        for old, new in zip(implicit.steps, explicit.steps, strict=True):
            assert_array_equal(old.state.forward.centerline_m, new.state.forward.centerline_m)
            assert_array_equal(old.comparison_path_m, new.comparison_path_m)
            self.assertEqual(old.body_following_error_max_m, new.body_following_error_max_m)
            self.assertEqual(old.solver_iterations, new.solver_iterations)

    def test_previous_reference_frozen_per_sqp_and_rejected_trial_never_becomes_source(self):
        cfg, reference = load_ftl_case(self.config, "straight", FIXTURES, "unused")
        cfg = replace(cfg, ftl=replace(cfg.ftl, body_reference_mode="previous_body"))
        calls = 0

        def trial(fun, x0, **kwargs):
            nonlocal calls
            calls += 1
            fun(x0)
            fun(x0.copy())
            return SimpleNamespace(
                x=x0, success=calls < 3, status=0 if calls < 3 else 9, message="test stop", nit=2
            )

        with (
            patch("continuum_ftl.ftl_optimizer.minimize", side_effect=trial),
            patch(
                "continuum_ftl.ftl_rollout.previous_body_comparison_path",
                wraps=previous_body_comparison_path,
            ) as build,
        ):
            result = run_single_path_ftl(cfg, reference)
        ftl = [step for step in result.steps if step.phase == "FTL"]
        self.assertEqual(build.call_count, 3)  # Not called inside objective evaluations.
        self.assertFalse(ftl[-1].accepted)
        assert_array_equal(ftl[-1].historical_tip_path_m, ftl[-2].historical_tip_path_m)
        for call, step in zip(build.call_args_list, ftl, strict=True):
            previous = result.steps[step.step_index - 1]
            self.assertIs(call.args[0], previous.state)
            self.assertTrue(previous.accepted)
            assert_array_equal(step.comparison_path_m, previous_body_comparison_path(*call.args))
        report = ftl_report(result, reference)
        self.assertEqual(report["body_reference_mode"], "previous_body")
        evaluations = evaluate_ftl_rollout(cfg, reference, result)
        self.assertEqual(len(evaluations), 2)  # Excludes rejected geometry.
        self.assertEqual(summarize_ftl_evaluations(evaluations)["accepted_evaluated_frames"], 2)

    def test_independent_distance_uses_segments_and_endpoints_with_no_window(self):
        path = np.array([[0, 0], [0, 0.01], [0, 0.01]])
        points = np.array([[0.003, 0.005], [0, -0.002], [0, 0.014]])
        assert_allclose(point_to_polyline_distances(points, path), [0.003, 0.002, 0.004])
        assert_allclose(
            point_to_polyline_distances(points, np.zeros((2, 2))), np.linalg.norm(points, axis=1)
        )
        self.assertEqual(len(point_to_polyline_distances(np.empty((0, 2)), path)), 0)
        with self.assertRaises(ValueError):
            point_to_polyline_distances(points, np.empty((0, 2)))

    def test_visualization_displays_selected_mode_and_exports_pair(self):
        import matplotlib

        matplotlib.use("Agg")
        from continuum_ftl.deployment_viewer import DeploymentViewer
        from continuum_ftl.ftl_visualization import save_body_reference_comparison

        cfg, reference = load_ftl_case(self.config, "straight", FIXTURES, "unused")
        runs, evaluations = {}, {}
        for mode in ("tip_history", "previous_body"):
            selected = replace(cfg, ftl=replace(cfg.ftl, body_reference_mode=mode))
            result = run_single_path_ftl(selected, reference)
            runs[mode] = result
            evaluations[mode] = evaluate_ftl_rollout(selected, reference, result)
        selected = replace(cfg, ftl=replace(cfg.ftl, body_reference_mode="previous_body"))
        viewer = DeploymentViewer(
            selected,
            tuple(s.state for s in runs["previous_body"].steps),
            reference_path=reference,
            ftl_steps=runs["previous_body"].steps,
            exposures=tuple(s.exposure for s in runs["previous_body"].steps),
        )
        try:
            viewer.draw_frame(len(viewer.states) - 1)
            self.assertIn("body reference=previous_body", viewer.status.get_text())
        finally:
            viewer.close()
        with tempfile.TemporaryDirectory() as directory:
            save_body_reference_comparison(cfg, reference, runs, evaluations, directory)
            for filename in ("final_shapes.png", "independent_errors.png"):
                self.assertGreater((Path(directory) / filename).stat().st_size, 1000)

    def test_mode_validation_and_legacy_missing_field_default(self):
        data = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for mode in ("tip_history", "previous_body", "invalid", None):
                changed = copy.deepcopy(data)
                changed["ftl"]["body_reference_mode"] = mode
                path.write_text(json.dumps(changed), encoding="utf-8")
                if mode in ("tip_history", "previous_body"):
                    self.assertEqual(load_config(path).ftl.body_reference_mode, mode)
                else:
                    with self.assertRaisesRegex(ValueError, "body_reference_mode"):
                        load_config(path)
            del data["ftl"]["body_reference_mode"]
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(load_config(path).ftl.body_reference_mode, "tip_history")


if __name__ == "__main__":
    unittest.main()
