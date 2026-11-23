"""Spatial PCC, rigid reference rotation, 3-zone deployment and genuine 3D SQP."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from continuum_ftl.config import load_config
from continuum_ftl.ftl_demo_3d import spatial_report
from continuum_ftl.ftl_matching import comparison_path
from continuum_ftl.inverse import bends_to_configuration
from continuum_ftl.kinematics import forward_kinematics, local_pose
from continuum_ftl.models import ExposureStatus, FTLStep, ReferencePath, TargetConfig
from continuum_ftl.planar_kinematics import forward_kinematics_2d
from continuum_ftl.reference_path import interpolate_reference_path
from continuum_ftl.spatial_deployment import (
    build_spatial_geometry,
    compute_spatial_deployment,
    sample_spatial_forward,
)
from continuum_ftl.spatial_ftl_optimizer import solve_spatial_ftl_step
from continuum_ftl.spatial_ftl_rollout import run_spatial_ftl
from continuum_ftl.spatial_geometry import (
    evaluate_spatial_curve,
    extract_spatial_exposure,
    local_positions,
    local_transform,
    spatial_entry_intersections,
)
from continuum_ftl.spatial_reference import load_spatial_case, reference_planarity

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robot_3d.json"


class SpatialGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_cartesian_pcc_matches_independent_existing_theta_phi_transforms(self):
        for bend in (np.zeros(2), np.array([1e-12, -2e-12]), np.array([0.6, -0.4])):
            theta, phi = bends_to_configuration(bend)[0]
            for arc in (0, 0.03, 0.08):
                expected = local_pose(0.08, theta, phi, arc)
                assert_allclose(local_transform(0.08, bend, arc), expected, atol=1e-14)
                assert_allclose(
                    local_positions(0.08, bend, np.array([arc]))[0], expected[:3, 3], atol=1e-14
                )

    def test_spatial_chain_can_be_nonplanar_and_matches_existing_3d_fk(self):
        bends = np.array([[0.5, 0], [0, -0.6], [0.2, 0.3]])
        lengths = np.array([s.length_m for s in self.config.robot.sections])
        geometry = build_spatial_geometry(("C", "B", "A"), lengths, bends, np.eye(4))
        forward = sample_spatial_forward(geometry, 0.001)
        expected = forward_kinematics(
            self.config.robot, bends_to_configuration(bends.ravel()), 0.001
        )
        assert_allclose(forward.section_endpoints_m, expected.section_endpoints_m, atol=1e-14)
        assert_allclose(forward.tip_orientation, expected.tip_orientation, atol=1e-14)
        self.assertGreater(np.linalg.svd(forward.centerline_m, compute_uv=False)[-1], 1e-4)
        assert_allclose(forward.tip_orientation.T @ forward.tip_orientation, np.eye(3), atol=1e-14)

    def test_each_original_reference_is_rigidly_rotated_without_length_or_shape_changes(self):
        planar = load_config(ROOT / "configs/robot_2d.json")
        angle = np.deg2rad(self.config.spatial_reference.plane_rotation_deg)
        for case, signed in self.config.spatial_reference.case_bends_rad.items():
            cfg, path = load_spatial_case(self.config, case)
            original = forward_kinematics_2d(planar.robot, signed, 0.001)
            expected = np.column_stack(
                (
                    -np.sin(angle) * original.centerline_m[:, 0],
                    np.cos(angle) * original.centerline_m[:, 0],
                    original.centerline_m[:, 1],
                )
            )
            actual = interpolate_reference_path(path, original.arc_lengths_m)
            assert_allclose(actual, expected, atol=2e-14)
            self.assertAlmostEqual(path.arc_lengths_m[-1], 0.18)
            self.assertLess(
                np.linalg.norm(actual[-1] - cfg.target.position_m), cfg.target.tolerance_m
            )
            assert_allclose(path.geometry.section_lengths_m, [0.08, 0.06, 0.04])

    def test_full_passive_lengths_release_rules_partial_exposure_and_negative_feed(self):
        config = self.config
        requested = np.array([[0.3, 0.1], [-0.2, 0.3], [0.2, 0.1]])
        for d in (0.039, 0.04, 0.08, 0.1, 0.18, 0.24, 0.179):
            state = compute_spatial_deployment(config.robot, config.deployment, d, requested, 0.001)
            assert_allclose(state.section_lengths_m, [0.2, 0.08, 0.06, 0.04])
            assert_allclose(state.release_thresholds_m, [0.18, 0.10, 0.04])
            assert_array_equal(state.applied_bends_rad[~state.released], 0)
            assert_allclose(state.forward.section_curves[0].points_m[:, :2], 0, atol=1e-15)
            self.assertAlmostEqual(state.forward.section_curves[0].arc_lengths_m[-1], 0.2)
        state = compute_spatial_deployment(
            config.robot, config.deployment, 0.1, np.array([[0, 0], [0, 0], [0.2, 0.1]]), 0.005
        )
        exposure = extract_spatial_exposure(state.geometry, 0, 0.001, 1e-9)
        self.assertEqual(exposure.status, ExposureStatus.EXPOSED)
        self.assertTrue(0 < exposure.section_exposed_lengths_m[-1] < 0.04)
        assert_allclose(exposure.centerline_m[0, 2], 0, atol=1e-9)
        self.assertEqual(exposure.centerline_m.shape[1], 3)

    def test_plane_intersections_find_reentry_and_coincident_curves_without_sampling(self):
        pose = np.eye(4)
        pose[2, 3] = -0.01
        curve = build_spatial_geometry(
            ("arc",), np.array([0.2]), np.array([[0.0, 2 * np.pi]]), pose
        )
        roots, coincident = spatial_entry_intersections(curve, 0, 1e-9)
        self.assertEqual(len(roots), 2)
        self.assertFalse(coincident)
        self.assertEqual(
            extract_spatial_exposure(curve, 0, 0.1, 1e-9).status, ExposureStatus.AMBIGUOUS
        )
        pose = np.eye(4)
        pose[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        curve = build_spatial_geometry(("horizontal",), np.array([0.2]), np.array([[0.5, 0]]), pose)
        self.assertTrue(spatial_entry_intersections(curve, 0, 1e-9)[1])

    def test_invalid_geometry_and_config_fail_explicitly(self):
        for bend, arc in ((np.zeros(3), 0.01), (np.zeros(2), -0.01)):
            with self.assertRaises(ValueError):
                local_positions(0.08, bend, np.array([arc]))
        with self.assertRaises(ValueError):
            compute_spatial_deployment(
                self.config.robot, self.config.deployment, 0.2, np.ones((3, 2)) * 2, 0.001
            )
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            data["ftl"]["max_reshape_attempts"] = 1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "2D only"):
                load_config(path)


class NonplanarReferenceTests(unittest.TestCase):
    def test_spatial_s_curve_is_nonplanar_with_original_lengths_and_bend_magnitudes(self):
        config, reference = load_spatial_case(load_config(CONFIG), "s-bend-3d")
        planar_config = load_config(ROOT / "configs/robot_2d.json")
        # Independent theta/phi FK checks chained local frames and global yaw.
        theta_phi = np.column_stack(
            (np.array([0.9, 1.3, 0.8]), np.deg2rad([90, -45, 110]) + np.pi / 4)
        )
        expected = forward_kinematics(config.robot, theta_phi, 0.001)
        points = interpolate_reference_path(reference, expected.arc_lengths_m)
        assert_allclose(points, expected.centerline_m, atol=1e-14)
        assert_allclose(config.target.position_m, expected.tip_position_m, atol=1e-14)
        assert_allclose(reference.geometry.section_lengths_m, [0.08, 0.06, 0.04])
        assert_allclose(
            reference.geometry.section_lengths_m,
            [section.length_m for section in planar_config.robot.sections],
        )
        assert_allclose(np.linalg.norm(reference.geometry.bends_rad, axis=1), [0.9, 1.3, 0.8])
        self.assertAlmostEqual(reference.arc_lengths_m[-1], 0.18)
        diagnostics = reference_planarity(reference.points_m, 1e-9)
        self.assertFalse(diagnostics["is_planar"])
        self.assertGreater(diagnostics["maximum_best_fit_plane_departure_m"], 0.005)
        _, old_reference = load_spatial_case(config, "s-bend")
        self.assertTrue(reference_planarity(old_reference.points_m, 1e-9)["is_planar"])

    def test_spatial_reference_config_rejects_invalid_shapes_limits_and_duplicate_names(self):
        for name, bends in (
            ("s-bend-3d", [0.1, 0.2, 0.3]),
            ("s-bend-3d", [[1.5, 1.5], [0, 0], [0, 0]]),
            ("s-bend-3d", [[float("nan"), 0], [0, 0], [0, 0]]),
            ("s-bend", [[0, 0], [0, 0], [0, 0]]),
        ):
            data = json.loads(CONFIG.read_text(encoding="utf-8"))
            data["spatial_reference"]["case_bends_xy_rad"] = {name: bends}
            with self.subTest(name=name, bends=bends), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_planarity_handles_short_paths_and_invalid_inputs(self):
        for points in (np.zeros((1, 3)), np.array([[0, 0, 0], [1, 1, 1]])):
            self.assertTrue(reference_planarity(points, 1e-9)["is_planar"])
        for points, tolerance in (
            (np.zeros((3, 2)), 1e-9),
            (np.empty((0, 3)), 1e-9),
            (np.eye(3), 0),
        ):
            with self.assertRaises(ValueError):
                reference_planarity(points, tolerance)


class SpatialFTLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config, reference = load_spatial_case(load_config(CONFIG), "straight")
        # An exact short prefix isolates completion without a full rollout per unit test.
        arcs = np.array([0.0, 0.005])
        points = interpolate_reference_path(reference, arcs)
        cls.reference = replace(
            reference, points_m=points, arc_lengths_m=arcs, target_position_m=points[-1]
        )
        cls.config = replace(config, target=TargetConfig(points[-1], 0.001))
        with (
            patch(
                "continuum_ftl.ftl_optimizer.solve_ftl_step",
                side_effect=AssertionError("planar SQP"),
            ),
            patch(
                "continuum_ftl.ftl_rollout.run_single_path_ftl",
                side_effect=AssertionError("planar rollout"),
            ),
        ):
            cls.result = run_spatial_ftl(cls.config, cls.reference)

    def previous_at(self, d, bends=None):
        bends = np.zeros((3, 2)) if bends is None else bends
        state = compute_spatial_deployment(
            self.config.robot, self.config.deployment, d, bends, 0.001
        )
        exposure = extract_spatial_exposure(state.geometry, 0, 0.001, 1e-9)
        return replace(
            self.result.steps[-1],
            state=state,
            exposure=exposure,
            configuration_increment_rad=np.zeros((3, 2)),
            historical_tip_path_m=np.array([[0, 0, 0], state.forward.tip_position_m]),
        )

    def solve(self, previous, config=None):
        config = config or self.config
        planned = interpolate_reference_path(self.reference, np.linspace(0, 0.005, 21))
        comparison = comparison_path(
            previous.historical_tip_path_m, self.reference.points_m[-1], 0.00025
        )
        return solve_spatial_ftl_step(config, self.reference, previous, 0.005, comparison, planned)

    def test_spatial_rollout_completes_without_calling_planar_solver_or_projecting_states(self):
        self.assertTrue(self.result.success, self.result.failure_reason)
        steps = [s for s in self.result.steps if s.phase == "FTL"]
        self.assertEqual(len(steps), 5)
        for step in steps:
            self.assertEqual(step.state.applied_bends_rad.shape, (3, 2))
            self.assertEqual(step.state.forward.centerline_m.shape[1], 3)
            self.assertLess(step.tip_error_m, 1e-6)
            direct = compute_spatial_deployment(
                self.config.robot,
                self.config.deployment,
                step.state.distance_m,
                step.state.applied_bends_rad,
                0.001,
            )
            assert_allclose(
                direct.forward.tip_position_m, step.state.forward.tip_position_m, atol=1e-15
            )

    def test_real_spatial_sqp_can_follow_waypoints_outside_configured_reference_plane(self):
        geometry = build_spatial_geometry(
            ("C", "B", "A"),
            np.array([0.08, 0.06, 0.04]),
            np.array([[0.4, 0.3], [-0.2, 0.4], [0.1, -0.2]]),
            np.eye(4),
        )
        arcs = np.array([0.0, 0.005])
        points = evaluate_spatial_curve(geometry, arcs)
        reference = ReferencePath(
            geometry, 0.0, points, arcs, np.array([0, 0]), arcs, points[-1], 0.0
        )
        config = replace(self.config, target=TargetConfig(points[-1], 0.001))
        result = run_spatial_ftl(config, reference)
        self.assertTrue(result.success, result.failure_reason)
        angle = np.deg2rad(config.spatial_reference.plane_rotation_deg)
        normal = np.array([np.cos(angle), np.sin(angle), 0.0])
        self.assertGreater(abs(result.steps[-1].state.forward.tip_position_m @ normal), 1e-5)
        self.assertLess(result.steps[-1].tip_error_m, 1e-6)

    def test_controls_have_seven_dofs_and_allow_nonplanar_shapes(self):
        previous = self.previous_at(0.22)

        def prescribed(fun, x0, **kwargs):
            self.assertEqual(len(x0), 7)
            x = np.zeros(7)
            x[1:3] = [0.6, 0.8]
            self.assertGreaterEqual(np.min(kwargs["constraints"][0]["fun"](x)), -1e-12)
            return SimpleNamespace(x=x, success=True, status=0, message="test", nit=1)

        with patch("continuum_ftl.spatial_ftl_optimizer.minimize", side_effect=prescribed):
            step = self.solve(previous)
        self.assertTrue(step.accepted, step.failure_reason)
        assert_allclose(step.state.applied_bends_rad[0], [0.0768, 0.1024])
        self.assertGreater(np.linalg.norm(step.state.forward.centerline_m[:, :2]), 0)

    def test_spatial_previous_body_reference_also_runs_without_planar_geometry(self):
        config = replace(
            self.config, ftl=replace(self.config.ftl, body_reference_mode="previous_body")
        )
        result = run_spatial_ftl(config, self.reference)
        self.assertTrue(result.success, result.failure_reason)
        self.assertEqual(result.body_reference_mode, "previous_body")
        self.assertTrue(
            all(s.comparison_path_m.shape[1] == 3 for s in result.steps if s.phase == "FTL")
        )

    def test_total_and_incremental_bend_limits_are_disks_not_component_boxes(self):
        previous = self.previous_at(0.22)
        for components, limit, reason in (
            ([0.9, 0.9], np.pi / 2, "INCREMENT_LIMIT_VIOLATION"),
            ([0.625, 0.625], 0.1, "BEND_LIMIT_VIOLATION"),
        ):
            config = replace(
                self.config,
                robot=replace(
                    self.config.robot,
                    sections=tuple(
                        replace(s, max_bend_rad=limit) for s in self.config.robot.sections
                    ),
                ),
            )

            def prescribed(fun, x0, components=components, **kwargs):
                x = np.zeros_like(x0)
                x[1:3] = components
                return SimpleNamespace(x=x, success=True, status=0, message="test", nit=1)

            with (
                self.subTest(reason=reason),
                patch("continuum_ftl.spatial_ftl_optimizer.minimize", side_effect=prescribed),
            ):
                step = self.solve(previous, config)
            self.assertFalse(step.accepted)
            self.assertEqual(step.failure_reason, reason)
            assert_array_equal(step.historical_tip_path_m, previous.historical_tip_path_m)

    def test_negative_feed_can_relock_only_after_both_bending_components_straighten(self):
        previous = self.previous_at(0.1805, np.array([[0.01, 0.01], [0, 0], [0, 0]]))

        def prescribed(fun, x0, **kwargs):
            x = np.zeros_like(x0)
            x[0] = -0.5
            x[1:3] = -0.01 / 0.128
            return SimpleNamespace(x=x, success=True, status=0, message="test", nit=1)

        with patch("continuum_ftl.spatial_ftl_optimizer.minimize", side_effect=prescribed):
            step = self.solve(previous)
        self.assertTrue(step.accepted, step.failure_reason)
        self.assertLess(step.feed_increment_m, 0)
        self.assertFalse(step.state.released[0])
        assert_allclose(step.state.applied_bends_rad[0], 0)

    def test_solver_failure_is_not_bypassed_and_failed_shape_stays_out_of_history(self):
        previous = self.previous_at(0.22)

        def failed(fun, x0, **kwargs):
            return SimpleNamespace(
                x=x0, success=False, status=9, message="iteration limit", nit=500
            )

        with patch("continuum_ftl.spatial_ftl_optimizer.minimize", side_effect=failed):
            step = self.solve(previous)
        self.assertFalse(step.accepted)
        self.assertIn("SOLVER_FAILED", step.failure_reason)
        assert_array_equal(step.historical_tip_path_m, previous.historical_tip_path_m)

    def test_frozen_spatial_angular_boundary_steps_pass_physical_not_mixed_unit_checks(self):
        for case in ("s-bend", "staircase", "straight"):
            data = json.loads(
                (ROOT / f"tests/fixtures/spatial_{case}_boundary.json").read_text(encoding="utf-8")
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(data["config"]), encoding="utf-8")
                config, reference = load_spatial_case(load_config(path), case)
            old = data["previous"]
            state = compute_spatial_deployment(
                config.robot,
                config.deployment,
                old["deployment_distance_m"],
                np.array(old["configuration_rad"]),
                0.001,
            )
            exposure = extract_spatial_exposure(state.geometry, 0, 0.001, 1e-9)
            kwargs = {
                f.name: old[f.name] for f in fields(FTLStep) if f.name not in ("state", "exposure")
            }
            for key in (
                "configuration_increment_rad",
                "desired_tip_position_m",
                "worst_body_point_m",
                "historical_tip_path_m",
                "comparison_path_m",
            ):
                if kwargs[key] is not None:
                    kwargs[key] = np.array(kwargs[key])
            previous = FTLStep(state=state, exposure=exposure, **kwargs)
            arcs = np.linspace(0, 0.18, 721)
            with self.subTest(case=case):
                step = solve_spatial_ftl_step(
                    config,
                    reference,
                    previous,
                    data["desired_progress_m"],
                    np.array(data["comparison_path_m"]),
                    interpolate_reference_path(reference, arcs),
                )
                self.assertTrue(step.accepted, step.failure_reason)
                self.assertTrue(step.solver_success)
                self.assertTrue(
                    np.all(np.linalg.norm(step.state.applied_bends_rad, axis=1) <= np.pi / 2 + 1e-8)
                )

    def test_report_and_spatial_replay_export_use_actual_sqp_states(self):
        import matplotlib

        matplotlib.use("Agg")
        from continuum_ftl.spatial_viewer import SpatialViewer

        report = spatial_report(self.config, self.reference, self.result)
        self.assertFalse(report["reference_plane_constraint"])
        self.assertTrue(report["reference_planarity"]["is_planar"])
        json.dumps(report, allow_nan=False)
        selected = (self.result.steps[0], self.result.steps[-1])
        viewer = SpatialViewer(
            self.config,
            tuple(s.state for s in selected),
            self.reference,
            tuple(s.exposure for s in selected),
            selected,
        )
        try:
            viewer.draw_frame(1)
            actual = np.array(viewer.tip.get_data_3d()).ravel()
            assert_allclose(actual, selected[-1].state.forward.tip_position_m, atol=0)
            viewer.axes.view_init(20, 12)
            viewer.replay()
            self.assertEqual(viewer.axes.azim, 12)
            with tempfile.TemporaryDirectory() as directory:
                viewer.save_gif(Path(directory) / "motion.gif")
                self.assertGreater((Path(directory) / "motion.gif").stat().st_size, 1000)
        finally:
            viewer.close()

    def test_nonplanar_reference_is_not_misreported_as_planar_tracking_error(self):
        from continuum_ftl.spatial_viewer import SpatialViewer

        config, reference = load_spatial_case(load_config(CONFIG), "s-bend-3d")
        report = spatial_report(config, reference, self.result)
        self.assertFalse(report["reference_planarity"]["is_planar"])
        self.assertIsNone(report["maximum_distance_from_reference_plane_m"])
        self.assertTrue(
            all(r["maximum_distance_from_reference_plane_m"] is None for r in report["steps"])
        )
        json.dumps(report, allow_nan=False)
        selected = (self.result.steps[0], self.result.steps[-1])
        viewer = SpatialViewer(
            config,
            tuple(s.state for s in selected),
            reference,
            tuple(s.exposure for s in selected),
            selected,
        )
        try:
            viewer.draw_frame(1)
            self.assertIn("Reference: nonplanar", viewer.status.get_text())
            self.assertNotIn("distance from reference plane", viewer.status.get_text())
            viewer.fig.canvas.draw()
        finally:
            viewer.close()

    def test_spatial_cli_produces_separate_files_without_touching_planar_configuration(self):
        original = (ROOT / "configs/robot_2d.json").read_bytes()
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        data["ftl"]["max_steps"] = 2
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps(data), encoding="utf-8")
            output = Path(directory) / "run"
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "continuum_ftl.ftl_demo_3d",
                    "--config",
                    str(config),
                    "--case",
                    "straight",
                    "--no-show",
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["dimension"], 3)
            self.assertEqual(result["accepted_ftl_steps"], 2)
            self.assertEqual(result["failure_reason"], "FTL_STEP_BUDGET_EXHAUSTED")
            self.assertTrue((output / "summary.png").is_file())
            self.assertTrue((output / "errors.png").is_file())
        self.assertEqual((ROOT / "configs/robot_2d.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
