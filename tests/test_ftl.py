"""Stage 2C mathematical matching, A/B/C/D rollouts and computed-frame replay."""

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

from continuum_ftl.config import geometry_context, load_config
from continuum_ftl.deployment import compute_deployment
from continuum_ftl.evaluation import evaluate_target, result_report
from continuum_ftl.ftl_demo import load_ftl_case
from continuum_ftl.ftl_matching import (
    comparison_path,
    match_ordered_polyline,
    palmer_objective,
    polyline_arcs,
)
from continuum_ftl.ftl_optimizer import evaluate_ftl_trial, solve_ftl_step
from continuum_ftl.ftl_rollout import ftl_report, ftl_summary, run_single_path_ftl
from continuum_ftl.ik_search import find_collision_free_ik, search_result_report
from continuum_ftl.models import RectangleObstacle
from continuum_ftl.reference_path import (
    extract_reference_path,
    interpolate_reference_path,
    load_reference_path,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robot_2d.json"
FIXTURES = ROOT / "configs/ftl_validation.json"


class FTLMatchingTests(unittest.TestCase):
    def test_objective_is_tip_norm_plus_max_not_sum_of_squared_distances(self):
        self.assertAlmostEqual(palmer_objective(0.002, 0.004, 10), 0.024)

    def test_segment_projection_works_with_different_sampling_resolutions(self):
        points = np.array([[0.001, 0.003], [-0.002, 0.017]])
        path = np.array([[0.0, 0.0], [0.0, 0.02]])
        match = match_ordered_polyline(points, path, np.array([0.003, 0.017]), 0.005)
        assert_allclose(match.distances_m, [0.001, 0.002], atol=1e-15)
        assert_allclose(match.matched_arcs_m, [0.003, 0.017], atol=1e-15)

    def test_new_tip_extension_is_not_penalized_as_missing_old_history(self):
        history = np.array([[0.0, 0.0], [0.0, 0.01]])
        path = comparison_path(history, np.array([0.0, 0.012]), 0.00025)
        point = np.array([[0.0, 0.0115]])
        extended = match_ordered_polyline(point, path, np.array([0.0115]), 0.01)
        old_only = match_ordered_polyline(point, history, np.array([0.0115]), 0.01)
        self.assertLess(extended.distances_m[0], 1e-15)
        self.assertAlmostEqual(old_only.distances_m[0], 0.0015)
        assert_array_equal(path[-1], [0, 0.012])
        self.assertLessEqual(np.diff(polyline_arcs(path)).max(), 0.00025 + 1e-15)

    def test_local_windows_do_not_match_to_unrelated_loop(self):
        path = np.array([[0, 0], [0, 0.01], [0.01, 0.01], [0.01, 0], [0, 0], [0, -0.01]])
        match = match_ordered_polyline(np.array([[0, 0.0001]]), path, np.array([0.04]), 0.003)
        self.assertGreater(match.matched_arcs_m[0], 0.035)
        self.assertAlmostEqual(match.distances_m[0], 0.0001)

    def test_repeated_history_points_and_invalid_inputs(self):
        history = np.zeros((2, 2))
        path = comparison_path(history, np.array([0, 0.001]), 0.00025)
        self.assertTrue(np.all(np.diff(polyline_arcs(path)) > 0))
        for spacing in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                comparison_path(history, np.array([0, 0.001]), spacing)
        with self.assertRaises(ValueError):
            match_ordered_polyline(np.zeros((1, 3)), path, np.zeros(1), 0.01)


class FTLRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loaded = load_config(CONFIG)
        # Keep original feed geometry and strict A/B/C/D baselines explicit.
        cls.config = replace(
            loaded,
            ftl=replace(
                loaded.ftl,
                enforce_error_limits=True,
                reference_progress_mode="actual_tip",
                body_reference_mode="tip_history",
                min_feed_increment_m=0.0,
                reference_step_m=0.002,
                max_feed_increment_m=0.002,
                max_bend_increment_rad=0.064,
                finite_difference_step=1e-5,  # Preserve the original strict-mode regression.
            ),
            deployment=replace(
                loaded.deployment,
                feed_exit_m=np.array([0.0, -0.05]),
                feed_block_center_m=np.array([0.0, -0.105]),
            ),
        )
        cls.cases = {}
        # Run A, B, C in order. Fixture targets are derived in memory without IK.
        for name in ("straight", "single-bend", "s-bend"):
            config, reference = load_ftl_case(cls.config, name, FIXTURES, "unused")
            with (
                patch("continuum_ftl.inverse.solve_position_ik", side_effect=AssertionError("IK")),
                patch(
                    "continuum_ftl.collision.check_static_collision",
                    side_effect=AssertionError("collision"),
                ),
            ):
                result = run_single_path_ftl(config, reference)
            cls.cases[name] = config, reference, result
        # Existing Stage 1 is used solely to prepare the one fixed input to D.
        stage1 = find_collision_free_ik(cls.config)
        if not stage1.success:
            raise AssertionError("Existing Stage 1 did not provide the staircase test input")
        reference = extract_reference_path(cls.config, stage1.configuration)
        with (
            patch("continuum_ftl.inverse.solve_position_ik", side_effect=AssertionError("IK")),
            patch(
                "continuum_ftl.ik_search.find_collision_free_ik",
                side_effect=AssertionError("search"),
            ),
            patch(
                "continuum_ftl.collision.check_static_collision",
                side_effect=AssertionError("collision"),
            ),
        ):
            result = run_single_path_ftl(cls.config, reference)
        cls.cases["staircase"] = cls.config, reference, result
        cls.stage1_report = search_result_report(stage1)
        cls.stage1_report["geometry_context"] = geometry_context(cls.config)

    def test_a_straight_completes_with_near_zero_body_error(self):
        config, reference, result = self.cases["straight"]
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(
            np.linalg.norm(
                result.steps[-1].state.forward.tip_position_m - config.target.position_m
            ),
            config.ftl.tip_tolerance_m,
        )
        ftl_steps = [step for step in result.steps if step.phase == "FTL"]
        self.assertGreater(len(ftl_steps), 80)
        self.assertLess(max(s.body_following_error_max_m for s in ftl_steps), 1e-4)
        self.assertGreater(result.steps[-1].reference_progress_m, 0.179)
        self.assertAlmostEqual(reference.arc_lengths_m[-1], 0.18)

    def test_a_release_transitions_and_all_increment_limits(self):
        config, _, result = self.cases["straight"]
        for step in result.steps:
            state = step.state
            self.assertGreaterEqual(step.feed_increment_m, -config.ftl.increment_tolerance)
            self.assertLessEqual(
                step.feed_increment_m,
                config.ftl.max_feed_increment_m + config.ftl.increment_tolerance,
            )
            self.assertTrue(
                np.all(
                    np.abs(step.configuration_increment_rad)
                    <= config.ftl.max_bend_increment_rad + config.ftl.increment_tolerance
                )
            )
            assert_array_equal(state.applied_bends_rad[~state.released], 0)
            for i, threshold in enumerate(state.release_thresholds_m):
                if threshold <= state.distance_m < threshold + config.ftl.max_feed_increment_m:
                    allowed = (
                        config.ftl.max_bend_increment_rad
                        * (state.distance_m - threshold)
                        / config.ftl.max_feed_increment_m
                    )
                    self.assertLessEqual(
                        abs(state.applied_bends_rad[i]), allowed + config.ftl.increment_tolerance
                    )
        self.assertTrue(result.steps[-1].state.released.all())

    def test_b_bending_is_numerically_changed_without_discontinuous_jumps(self):
        config, _, result = self.cases["single-bend"]
        accepted = [s for s in result.steps if s.accepted and s.phase == "FTL"]
        self.assertGreater(len(accepted), 10)
        self.assertGreater(max(np.max(np.abs(s.state.applied_bends_rad)) for s in accepted), 0.1)
        self.assertTrue(any(s.body_following_error_max_m > 1e-5 for s in accepted))
        self.assertTrue(
            any(abs(s.state.distance_m - 0.05 - s.reference_progress_m) > 1e-4 for s in accepted)
        )
        for previous, step in zip(result.steps[:-1], result.steps[1:], strict=True):
            assert_allclose(
                step.state.applied_bends_rad - previous.state.applied_bends_rad,
                step.configuration_increment_rad,
                atol=1e-15,
            )
            self.assertTrue(
                np.all(
                    np.abs(step.configuration_increment_rad)
                    <= config.ftl.max_bend_increment_rad + config.ftl.increment_tolerance
                )
            )

    def test_c_s_bend_enters_second_curvature_region_and_reports_failure_honestly(self):
        config, _, result = self.cases["s-bend"]
        self.assertGreater(result.steps[-1].reference_progress_m, config.robot.sections[0].length_m)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "BODY_TOLERANCE_EXCEEDED")
        self.assertFalse(result.steps[-1].accepted)
        self.assertGreater(result.steps[-1].body_following_error_max_m, config.ftl.body_tolerance_m)
        assert_array_equal(
            result.steps[-1].historical_tip_path_m, result.steps[-2].historical_tip_path_m
        )

    def test_d_staircase_retains_failed_trial_and_metrics_not_synthetic_geometry(self):
        config, reference, result = self.cases["staircase"]
        self.assertGreater(len(result.steps), 26)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.failure_reason)
        failed = result.steps[-1]
        self.assertFalse(failed.accepted)
        self.assertGreater(failed.tip_error_m, config.ftl.tip_tolerance_m)
        self.assertGreater(
            np.linalg.norm(failed.state.forward.tip_position_m - failed.desired_tip_position_m),
            config.ftl.tip_tolerance_m,
        )
        self.assertTrue(failed.solver_success)  # Convergence is insufficient for acceptance.
        summary = ftl_summary(result, reference)
        self.assertGreater(summary["computation_time_s"], 0)
        self.assertFalse(summary["collision_checked"])
        self.assertGreater(summary["last_accepted_target_error_m"], 0.1)
        json.dumps(ftl_report(result, reference), allow_nan=False)

    def test_actual_geometry_and_body_metrics_are_recomputed_at_every_ftl_step(self):
        for name in ("single-bend", "staircase"):
            config, _, result = self.cases[name]
            for step in result.steps:
                state = step.state
                passive = state.forward.section_curves[0]
                assert_allclose(passive.points_m[:, 0], 0)
                self.assertAlmostEqual(passive.points_m[-1, 1] - passive.points_m[0, 1], 0.2)
                self.assertAlmostEqual(state.section_lengths_m.sum(), 0.38)
                if step.phase != "FTL":
                    continue
                actual = compute_deployment(
                    config.robot,
                    config.deployment,
                    state.distance_m,
                    state.applied_bends_rad,
                    config.ftl.validation_body_spacing_m,
                )
                assert_array_equal(actual.forward.centerline_m, state.forward.centerline_m)
                trial = evaluate_ftl_trial(
                    config,
                    state.distance_m,
                    state.applied_bends_rad,
                    step.desired_tip_position_m,
                    step.comparison_path_m,
                    config.ftl.validation_body_spacing_m,
                )
                self.assertAlmostEqual(
                    step.body_following_error_max_m, np.max(trial.body_match.distances_m)
                )
                self.assertAlmostEqual(
                    step.body_following_error_mean_m, np.mean(trial.body_match.distances_m)
                )
                self.assertGreaterEqual(
                    step.worst_body_material_arc_m, config.deployment.passive_length_m
                )
                self.assertGreaterEqual(
                    trial.body_points_m[:, 1].min(), -config.deployment.geometry_tolerance_m
                )
                self.assertGreaterEqual(
                    step.body_following_error_max_m, step.body_following_error_mean_m
                )

    def test_numerically_successful_solver_can_still_be_rejected_and_history_is_unchanged(self):
        config, reference, _ = self.cases["straight"]

        def false_success(fun, x0, **kwargs):
            return SimpleNamespace(
                x=np.zeros_like(x0),
                success=True,
                status=0,
                message="test false geometric success",
                nit=1,
            )

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=false_success):
            result = run_single_path_ftl(config, reference)
        self.assertEqual(result.failure_reason, "TIP_TOLERANCE_EXCEEDED")
        self.assertFalse(result.steps[-1].accepted)
        self.assertEqual(len(result.steps[-1].historical_tip_path_m), 1)
        self.assertAlmostEqual(result.steps[-1].reference_progress_m, 0)

    def test_solver_failure_and_budget_exhaustion_are_explicit(self):
        config, reference, _ = self.cases["straight"]

        def exhausted(fun, x0, **kwargs):
            return SimpleNamespace(x=x0, success=False, status=9, message="iteration limit", nit=1)

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=exhausted):
            result = run_single_path_ftl(config, reference)
        self.assertIn("SOLVER_FAILED: status=9", result.failure_reason)
        self.assertFalse(result.steps[-1].accepted)
        budget = replace(config, ftl=replace(config.ftl, max_steps=1))
        result = run_single_path_ftl(budget, reference)
        self.assertEqual(result.failure_reason, "FTL_STEP_BUDGET_EXHAUSTED")
        self.assertTrue(result.steps[-1].accepted)

    def test_warm_starts_new_sections_begin_straight_and_body_term_reaches_optimizer(self):
        config, reference, _ = self.cases["straight"]
        initial_guesses = []

        def inspect_objective(fun, x0, **kwargs):
            initial_guesses.append(x0.copy())
            self.assertTrue(np.isfinite(fun(x0)))
            return SimpleNamespace(x=x0, success=True, status=0, message="inspection", nit=1)

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=inspect_objective):
            result = run_single_path_ftl(
                replace(config, ftl=replace(config.ftl, max_steps=1)), reference
            )
        self.assertEqual(len(initial_guesses[0]), 2)  # Only feed and A; no passive/B/C DOFs.
        self.assertEqual(initial_guesses[0][1], 0)
        self.assertTrue(result.steps[-1].accepted)

    def test_scene_and_reference_are_not_mutated_and_stage1_report_is_reusable(self):
        before = copy.deepcopy(geometry_context(self.config))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "stage1.json"
            file.write_text(json.dumps(self.stage1_report), encoding="utf-8")
            config, reference = load_ftl_case(self.config, "staircase", FIXTURES, file)
            self.assertIs(config, self.config)
            self.assertEqual(geometry_context(config), before)
            points = reference.points_m.copy()
            with patch(
                "continuum_ftl.ftl_optimizer.minimize",
                side_effect=RuntimeError("programming error"),
            ):
                with self.assertRaisesRegex(RuntimeError, "programming error"):
                    run_single_path_ftl(config, reference)
            assert_array_equal(reference.points_m, points)

    def test_replay_uses_computed_accepted_and_failed_states_with_single_step_controls(self):
        import matplotlib

        matplotlib.use("Agg")
        from PIL import Image

        from continuum_ftl.deployment_viewer import DeploymentViewer
        from continuum_ftl.ftl_visualization import save_ftl_summary

        config, reference, result = self.cases["staircase"]
        selected = (result.steps[0], result.steps[26], result.steps[-2], result.steps[-1])
        viewer = DeploymentViewer(
            config,
            tuple(s.state for s in selected),
            reference_path=reference,
            exposures=tuple(s.exposure for s in selected),
            ftl_steps=selected,
        )
        try:
            with patch(
                "continuum_ftl.ftl_optimizer.minimize", side_effect=AssertionError("recomputed")
            ):
                for i, step in enumerate(selected):
                    viewer.draw_frame(i)
                    assert_array_equal(
                        np.array(viewer.tip.get_data()).ravel(), step.state.forward.tip_position_m
                    )
                    assert_array_equal(
                        np.array(viewer.history_line.get_data()).T, step.historical_tip_path_m
                    )
                    for line, curve in zip(
                        viewer.lines, step.state.forward.section_curves, strict=True
                    ):
                        assert_array_equal(np.array(line.get_data()).T, curve.points_m)
                self.assertIn(result.failure_reason, viewer.warning.get_text())
                viewer.step_frame(-1)
                self.assertEqual(viewer.current_index, 2)
                self.assertTrue(viewer.paused)
                viewer.step_frame(1)
                self.assertEqual(viewer.current_index, 3)
                with tempfile.TemporaryDirectory() as directory:
                    gif = Path(directory) / "ftl.gif"
                    viewer.save_gif(gif)
                    with Image.open(gif) as image:
                        self.assertEqual(image.n_frames, 4)
                    summary = Path(directory) / "summary.png"
                    save_ftl_summary(config, reference, result, summary)
                    self.assertGreater(summary.stat().st_size, 1000)
                viewer.replay()
                viewer.animation.event_source.stop()
                viewer.animation.event_source = None  # A completed finite animation.
                viewer.replay()
                self.assertEqual(viewer.current_index, 0)
        finally:
            viewer.close()


class FTLErrorDisplayTests(unittest.TestCase):
    def setUp(self):
        config = load_config(CONFIG)
        self.config, self.reference = load_ftl_case(config, "straight", FIXTURES, "unused")

    @staticmethod
    def bent_trial(fun, x0, **kwargs):
        control = x0.copy()
        control[1] = 0.05 / 0.064
        assert fun(control) > 0  # The error objective is still active.
        return SimpleNamespace(x=control, success=True, status=0, message="finite trial", nit=1)

    def test_errors_above_both_limits_are_recorded_and_accepted_when_limits_are_off(self):
        config = replace(
            self.config,
            ftl=replace(
                self.config.ftl,
                enforce_error_limits=False,
                tip_tolerance_m=1e-9,
                body_tolerance_m=1e-9,
                max_steps=1,
            ),
        )
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=self.bent_trial):
            result = run_single_path_ftl(config, self.reference)
        step = result.steps[-1]
        self.assertTrue(step.accepted)
        self.assertGreater(step.tip_error_m, config.ftl.tip_tolerance_m)
        self.assertGreater(step.body_following_error_max_m, config.ftl.body_tolerance_m)
        self.assertEqual(len(step.historical_tip_path_m), 2)
        self.assertEqual(result.failure_reason, "FTL_STEP_BUDGET_EXHAUSTED")
        self.assertFalse(ftl_summary(result, self.reference)["error_limits_enforced"])
        for tip_limit, expected in (
            (1e-9, "TIP_TOLERANCE_EXCEEDED"),
            (0.1, "BODY_TOLERANCE_EXCEEDED"),
        ):
            strict = replace(
                config,
                ftl=replace(config.ftl, enforce_error_limits=True, tip_tolerance_m=tip_limit),
            )
            with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=self.bent_trial):
                checked = run_single_path_ftl(strict, self.reference)
            self.assertEqual(checked.failure_reason, expected)

    def test_solver_failure_is_still_rejected_in_display_only_mode(self):
        def failed(fun, x0, **kwargs):
            return SimpleNamespace(x=x0, success=False, status=9, message="iteration limit", nit=1)

        self.assertFalse(self.config.ftl.enforce_error_limits)
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=failed):
            result = run_single_path_ftl(self.config, self.reference)
        self.assertFalse(result.steps[-1].accepted)
        self.assertIn("SOLVER_FAILED", result.failure_reason)

    def test_completed_reference_with_residual_is_not_claimed_as_target_reached(self):
        # A short prefix and a deliberately offset target isolate terminal reporting.
        arcs = np.array([0.0, 0.002])
        reference = replace(
            self.reference,
            points_m=interpolate_reference_path(self.reference, arcs),
            arc_lengths_m=arcs,
            section_boundary_arcs_m=arcs,
            target_position_m=np.array([0.001, 0.002]),
        )
        config = replace(
            self.config,
            target=replace(
                self.config.target,
                position_m=reference.target_position_m,
                tolerance_m=1e-9,
            ),
        )

        def straight_trial(fun, x0, **kwargs):
            return SimpleNamespace(x=x0, success=True, status=0, message="straight trial", nit=1)

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=straight_trial):
            result = run_single_path_ftl(config, reference)
        self.assertTrue(result.reference_completed)
        self.assertFalse(result.success)
        self.assertIsNone(result.failure_reason)
        report = ftl_summary(result, reference)
        self.assertFalse(report["target_reached"])
        self.assertGreater(report["last_accepted_target_error_m"], config.target.tolerance_m)


class FTLCollisionIsolationTests(unittest.TestCase):
    def setUp(self):
        loaded = load_config(CONFIG)
        config, self.reference = load_ftl_case(loaded, "straight", FIXTURES, "unused")
        self.clear = replace(config, ftl=replace(config.ftl, max_steps=3))
        enclosing_box = RectangleObstacle(
            "covers_entire_robot", np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        )
        self.blocked = replace(
            self.clear, environment=replace(self.clear.environment, obstacles=(enclosing_box,))
        )
        # Establish a genuinely colliding saved shape using unchanged Stage 1.
        result = evaluate_target(self.blocked)
        self.assertTrue(result.ik.target_reached)
        self.assertFalse(result.collision.collision_free)
        self.saved = {**result_report(result), "geometry_context": geometry_context(self.blocked)}

    def test_enclosing_obstacle_is_never_passed_to_numeric_ftl_or_used_to_stop_it(self):
        with (
            patch(
                "continuum_ftl.collision.check_static_collision",
                side_effect=AssertionError("collision must remain OFF"),
            ),
            patch("continuum_ftl.ftl_rollout.solve_ftl_step", wraps=solve_ftl_step) as solver,
        ):
            blocked = run_single_path_ftl(self.blocked, self.reference)
            clear = run_single_path_ftl(self.clear, self.reference)
        self.assertEqual(len(solver.call_args_list), 6)
        for call in solver.call_args_list:
            self.assertEqual(call.args[0].environment.obstacles, ())
        self.assertEqual(blocked.failure_reason, "FTL_STEP_BUDGET_EXHAUSTED")
        self.assertTrue(all(step.accepted for step in blocked.steps))
        for a, b in zip(blocked.steps, clear.steps, strict=True):
            assert_array_equal(a.state.forward.centerline_m, b.state.forward.centerline_m)
            self.assertEqual(a.tip_error_m, b.tip_error_m)
            self.assertEqual(a.body_following_error_max_m, b.body_following_error_max_m)
        self.assertEqual(
            len(self.blocked.environment.obstacles), 1
        )  # Still available to the viewer.
        self.assertFalse(ftl_summary(blocked, self.reference)["obstacles_used_for_optimization"])

    def test_ftl_ignores_reference_collision_but_checks_robot_and_target(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "colliding_stage1.json"
            path.write_text(json.dumps(self.saved), encoding="utf-8")
            # Stage 2B stays strict, including both scene and collision acceptance.
            with self.assertRaises(ValueError):
                load_reference_path(self.blocked, path)
            with self.assertRaises(ValueError):
                load_reference_path(self.clear, path)
            with patch(
                "continuum_ftl.collision.check_static_collision",
                side_effect=AssertionError("collision"),
            ):
                config, reference = load_ftl_case(self.clear, "staircase", FIXTURES, path)
            self.assertIs(config, self.clear)
            assert_array_equal(reference.points_m, self.reference.points_m)
            changed_target = replace(
                self.clear, target=replace(self.clear.target, tolerance_m=0.002)
            )
            with self.assertRaisesRegex(ValueError, "robot/target"):
                load_reference_path(changed_target, path, require_collision_free=False)
            changed_robot = replace(
                self.clear,
                robot=replace(
                    self.clear.robot,
                    sections=(
                        replace(self.clear.robot.sections[0], radius_m=0.003),
                        *self.clear.robot.sections[1:],
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "robot/target"):
                load_reference_path(changed_robot, path, require_collision_free=False)

    def test_one_colliding_rejected_diagnostic_can_supply_the_ftl_reference(self):
        data = {
            "success": False,
            "geometry_context": geometry_context(self.blocked),
            "accepted_final_configuration": None,
            "diagnostic_rejected_configuration": self.saved,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected_shape.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_reference_path(self.blocked, path)
            with patch(
                "continuum_ftl.ik_search.find_collision_free_ik",
                side_effect=AssertionError("new IK search"),
            ):
                _, reference = load_ftl_case(self.blocked, "staircase", FIXTURES, path)
            assert_array_equal(reference.points_m, self.reference.points_m)
            data["diagnostic_rejected_configuration"]["target_reached"] = False
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "target-reaching"):
                load_reference_path(self.blocked, path, require_collision_free=False)


class FTLReferenceProgressTests(unittest.TestCase):
    def test_scheduled_waypoints_continue_when_actual_tip_stalls_without_faking_motion(self):
        config, reference = load_ftl_case(load_config(CONFIG), "straight", FIXTURES, "unused")
        config = replace(
            config,
            ftl=replace(
                config.ftl,
                enforce_error_limits=False,
                reference_progress_mode="scheduled",
                max_steps=3,
            ),
        )

        def stationary(fun, x0, **kwargs):
            return SimpleNamespace(
                x=np.zeros_like(x0), success=True, status=0, message="stationary trial", nit=1
            )

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=stationary):
            result = run_single_path_ftl(config, reference)
        steps = [step for step in result.steps if step.phase == "FTL"]
        self.assertEqual(len(steps), 3)
        self.assertTrue(all(step.accepted and step.reference_progress_stalled for step in steps))
        assert_allclose(
            [s.desired_reference_progress_m for s in steps],
            config.ftl.reference_step_m * np.arange(1, 4),
        )
        for step in steps:
            assert_allclose(step.state.forward.tip_position_m, [0, 0], atol=1e-15)
            self.assertAlmostEqual(step.reference_progress_m, 0)
            assert_array_equal(step.state.forward.centerline_m, steps[0].state.forward.centerline_m)
        self.assertEqual(result.failure_reason, "FTL_STEP_BUDGET_EXHAUSTED")
        self.assertEqual(ftl_summary(result, reference)["stalled_tip_steps"], 3)
        self.assertFalse(result.success)

        actual_mode = replace(config, ftl=replace(config.ftl, reference_progress_mode="actual_tip"))
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=stationary):
            stopped = run_single_path_ftl(actual_mode, reference)
        self.assertEqual(stopped.failure_reason, "NO_FORWARD_REFERENCE_PROGRESS")
        self.assertFalse(stopped.steps[-1].accepted)

    def test_scheduled_staircase_passes_old_stop_and_reaches_actual_target(self):
        config = load_config(CONFIG)
        # Historical baseline: independent of the user's current body mode and
        # the corrected finite-difference default. New settings have own tests.
        config = replace(
            config,
            ftl=replace(
                config.ftl,
                body_reference_mode="tip_history",
                min_feed_increment_m=0.0,
                reference_step_m=0.002,
                max_feed_increment_m=0.002,
                max_bend_increment_rad=0.064,
                finite_difference_step=1e-5,
            ),
        )
        stage1 = find_collision_free_ik(config)
        self.assertTrue(stage1.success)
        reference = extract_reference_path(config, stage1.configuration)
        with (
            patch(
                "continuum_ftl.collision.check_static_collision",
                side_effect=AssertionError("collision"),
            ),
            patch("continuum_ftl.inverse.solve_position_ik", side_effect=AssertionError("IK")),
        ):
            result = run_single_path_ftl(config, reference)
        self.assertEqual(result.reference_progress_mode, "scheduled")
        self.assertTrue(result.reference_completed)
        self.assertTrue(result.success, result.failure_reason)
        self.assertIsNone(result.failure_reason)
        steps = [step for step in result.steps if step.phase == "FTL"]
        self.assertEqual(len(steps), 90)
        self.assertTrue(all(step.accepted for step in steps))
        self.assertGreater(result.steps[-1].state.distance_m, 0.18)
        self.assertLess(
            np.linalg.norm(
                result.steps[-1].state.forward.tip_position_m - config.target.position_m
            ),
            config.target.tolerance_m,
        )
        # Completion is accompanied by the measured body deviation, not hidden.
        self.assertGreater(
            max(s.body_following_error_max_m for s in steps), config.ftl.body_tolerance_m
        )
        assert_allclose(
            np.diff([0.0, *(s.desired_reference_progress_m for s in steps)]), 0.002, atol=1e-15
        )


class FTLConfigTests(unittest.TestCase):
    def test_invalid_settings_and_3d_ftl_are_rejected(self):
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        invalid = {
            "reference_progress_mode": "unknown",
            "enforce_error_limits": 1,
            "max_feed_increment_m": 0,
            "max_bend_increment_rad": -1,
            "max_iterations": True,
            "max_steps": 0,
            "tip_tolerance_m": -1,
            "comparison_spacing_m": 0.01,
            "validation_body_spacing_m": 0.02,
            "history_color": "invalid",
        }
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "config.json"
            for key, value in invalid.items():
                bad = copy.deepcopy(data)
                bad["ftl"][key] = value
                file.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_config(file)
            bad = json.loads((ROOT / "configs/example.json").read_text(encoding="utf-8"))
            bad["ftl"] = data["ftl"]
            file.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2D only"):
                load_config(file)


if __name__ == "__main__":
    unittest.main()
