"""Stage 2A fixed-feed geometry, release boundaries and exact viewer state reuse."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from math import fsum
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from continuum_ftl.config import load_config
from continuum_ftl.deployment import (
    compute_deployment,
    deployment_report,
    generate_deployment_states,
    manual_bending_schedule,
    release_thresholds,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/robot_2d.json"


class DeploymentGeometryTests(unittest.TestCase):
    def setUp(self):
        loaded = load_config(EXAMPLE)
        # Retain the original numerical fixture while the demo feed is movable.
        self.config = replace(
            loaded,
            deployment=replace(
                loaded.deployment,
                feed_exit_m=np.array([0.0, -0.05]),
                feed_block_center_m=np.array([0.0, -0.105]),
            ),
        )
        self.robot = self.config.robot
        self.settings = self.config.deployment

    def state(self, distance, angles=None, spacing=0.002):
        angles = self.settings.bend_targets_rad if angles is None else np.asarray(angles)
        return compute_deployment(self.robot, self.settings, distance, angles, spacing)

    def test_physical_order_lengths_and_tip_at_fixed_exit(self):
        state = self.state(0)
        self.assertEqual(state.section_names, ("Passive", "C", "B", "A"))
        assert_allclose(state.section_lengths_m, [0.2, 0.08, 0.06, 0.04])
        self.assertEqual(fsum(state.section_lengths_m), 0.38)
        assert_allclose(state.proximal_position_m, [0, -0.43], atol=1e-14)
        assert_allclose(state.forward.tip_position_m, [0, -0.05], atol=1e-14)
        assert_allclose(state.applied_bends_rad, 0)
        self.assertEqual(state.applied_bends_rad.shape, (3,))
        self.assertEqual(len(state.forward.section_curves), 4)
        self.assertEqual(state.forward.section_endpoints_m.shape, (5, 2))

    def test_full_release_thresholds_and_immediately_before(self):
        assert_allclose(release_thresholds(self.robot), [0.18, 0.10, 0.04], atol=0, rtol=0)
        for index, threshold in enumerate([0.18, 0.10, 0.04]):
            before = self.state(np.nextafter(threshold, -np.inf), [0.2, 0.2, 0.2])
            at = self.state(threshold, [0.2, 0.2, 0.2])
            self.assertFalse(before.released[index])
            self.assertEqual(before.applied_bends_rad[index], 0)
            self.assertTrue(at.released[index])
            self.assertEqual(at.applied_bends_rad[index], 0.2)

    def test_unreleased_sections_are_forced_straight(self):
        for distance, expected in (
            (0.02, [False, False, False]),
            (0.06, [False, False, True]),
            (0.12, [False, True, True]),
            (0.20, [True, True, True]),
        ):
            state = self.state(distance)
            self.assertEqual(state.released.tolist(), expected)
            assert_allclose(state.applied_bends_rad[~state.released], 0)

    def test_release_uses_feed_exit_not_environment_entry_and_never_splits_sections(self):
        state = self.state(0.065, [0, 0, 0.3])
        a_curve = state.forward.section_curves[-1]
        self.assertTrue(state.released[-1])
        self.assertAlmostEqual(state.applied_bends_rad[-1], 0.3)
        self.assertLess(a_curve.points_m[0, 1], 0)
        self.assertGreater(a_curve.points_m[-1, 1], 0)
        self.assertAlmostEqual(a_curve.arc_lengths_m[-1], 0.04)
        self.assertEqual(state.section_names.count("A"), 1)
        self.assertEqual(len(state.forward.section_curves), 4)

    def test_straight_passive_base_formula_and_complete_length_at_all_distances(self):
        for distance in (0, 0.06, 0.12, 0.2, 0.26, 0.38):
            state = self.state(distance)
            passive = state.forward.section_curves[0]
            assert_allclose(passive.points_m[:, 0], 0)
            self.assertTrue(np.all(np.diff(passive.points_m[:, 1]) > 0))
            self.assertAlmostEqual(passive.points_m[-1, 1] - passive.points_m[0, 1], 0.2)
            self.assertEqual(state.boundary_tangent_angles_rad[0], 0)
            self.assertEqual(state.boundary_tangent_angles_rad[1], 0)
            self.assertAlmostEqual(state.proximal_position_m[1], -0.05 - 0.38 + distance)
            self.assertAlmostEqual(state.forward.arc_lengths_m[-1], 0.38)
            self.assertTrue(np.all(np.diff(state.forward.arc_lengths_m) > 0))
            self.assertEqual(np.count_nonzero(state.forward.arc_lengths_m == 0.2), 1)
            assert_allclose(
                [curve.arc_lengths_m[-1] for curve in state.forward.section_curves],
                state.section_lengths_m,
            )

    def test_position_and_tangent_continuity_at_each_physical_connection(self):
        state = self.state(0.24, spacing=1e-5)
        assert_allclose(
            state.boundary_tangent_angles_rad,
            [0, 0, np.pi / 3, np.pi / 6, 5 * np.pi / 12],
            atol=1e-14,
        )
        curves = state.forward.section_curves
        for index, (first, second) in enumerate(zip(curves[:-1], curves[1:], strict=True)):
            assert_allclose(first.points_m[-1], second.points_m[0], atol=1e-14)
            left = first.points_m[-1] - first.points_m[-2]
            right = second.points_m[1] - second.points_m[0]
            left /= np.linalg.norm(left)
            right /= np.linalg.norm(right)
            angle = state.boundary_tangent_angles_rad[index + 1]
            expected = [np.sin(angle), np.cos(angle)]
            # Chord-vs-tangent error <= curvature * sample_spacing/2.
            assert_allclose(left, expected, atol=1e-4)
            assert_allclose(right, expected, atol=1e-4)

    def test_manual_schedule_is_continuous_bounded_and_zero_on_release(self):
        thresholds = release_thresholds(self.robot)
        for index, threshold in enumerate(thresholds):
            before = manual_bending_schedule(self.robot, self.settings, threshold - 1e-7)
            at = manual_bending_schedule(self.robot, self.settings, threshold)
            after = manual_bending_schedule(self.robot, self.settings, threshold + 1e-7)
            self.assertEqual(before[index], 0)
            self.assertEqual(at[index], 0)
            self.assertLess(abs(after[index]), 1e-9)
        states = generate_deployment_states(self.config)
        for a, b in zip(states[:-1], states[1:], strict=True):
            bound = (
                1.5
                * np.abs(self.settings.bend_targets_rad)
                / self.settings.bend_ramp_distances_m
                * (b.distance_m - a.distance_m)
            )
            self.assertTrue(
                np.all(np.abs(b.applied_bends_rad - a.applied_bends_rad) <= bound + 1e-12)
            )
        assert_allclose(states[-1].applied_bends_rad, self.settings.bend_targets_rad)

    def test_inconsistent_fold_back_is_reported_without_silent_shape_changes(self):
        q = np.full(3, np.pi / 2)
        state = self.state(0.18, q)
        self.assertTrue(state.consistency_warnings)
        self.assertIn("folds behind feed exit", state.consistency_warnings[-1])
        assert_allclose(state.applied_bends_rad, q)

    def test_state_generation_never_calls_ik_optimizer_or_collision_checker(self):
        with (
            patch("continuum_ftl.inverse.solve_position_ik", side_effect=AssertionError("IK")),
            patch("scipy.optimize.minimize", side_effect=AssertionError("optimization")),
            patch(
                "continuum_ftl.collision.check_static_collision",
                side_effect=AssertionError("collision"),
            ),
        ):
            states = generate_deployment_states(self.config)
        report = deployment_report(states)
        self.assertFalse(report["collision_checked"])
        self.assertEqual(report["state_count"], len(states))
        self.assertEqual(report["total_physical_length_m"], 0.38)
        self.assertIn("NOT FTL", report["stage"])
        self.assertEqual(states[0].distance_m, 0)
        self.assertEqual(states[-1].distance_m, self.settings.end_distance_m)
        self.assertFalse(any(s.consistency_warnings for s in states))
        for threshold in (0.04, 0.1, 0.18):
            self.assertTrue(any(s.distance_m == threshold for s in states))

    def test_invalid_inputs_and_3d_deployment_are_rejected(self):
        for distance in (-0.001, 0.381, float("nan")):
            with self.assertRaises(ValueError):
                self.state(distance)
        with self.assertRaises(ValueError):
            self.state(0.1, [0, 0])
        with self.assertRaises(ValueError):
            self.state(0.1, [0, 0, 2])
        with self.assertRaises(ValueError):
            self.state(0.1, spacing=0)
        with self.assertRaises(ValueError):
            release_thresholds(load_config(ROOT / "configs/example.json").robot)

    def test_target_obstacles_and_stage1_base_are_not_mutated(self):
        target = self.config.target.position_m.copy()
        base = self.robot.base_position_m.copy()
        objects = copy.deepcopy(self.config.environment.obstacles)
        generate_deployment_states(self.config)
        assert_allclose(self.config.target.position_m, target)
        assert_allclose(self.robot.base_position_m, base)
        for before, after in zip(objects, self.config.environment.obstacles, strict=True):
            self.assertEqual(before.name, after.name)
            assert_allclose(before.min_m, after.min_m)
            assert_allclose(before.max_m, after.max_m)


class DeploymentConfigTests(unittest.TestCase):
    def test_bad_deployment_settings_raise_explicit_errors(self):
        original = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        changes = {
            "passive_length_m": -1,
            "passive_radius_m": 0,
            "active_section_names": ["A", "A", "A"],
            "feed_exit_m": [0, 0, 0],
            "feed_block_center_m": [0, -0.2],
            "feed_block_size_m": [0.001, 0.11],
            "environment_entry_z_m": -0.1,
            "end_distance_m": 0.4,
            "distance_step_m": 0,
            "bend_targets_rad": [2, 0, 0],
            "bend_ramp_distances_m": [0, 0.04, 0.04],
            "frame_interval_ms": True,
            "geometry_tolerance_m": -1,
            "section_colors": ["invalid"],
            "feed_opacity": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for name, value in changes.items():
                data = copy.deepcopy(original)
                data["deployment"][name] = value
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    load_config(path)
            spatial = json.loads((ROOT / "configs/example.json").read_text(encoding="utf-8"))
            spatial["deployment"] = original["deployment"]
            path.write_text(json.dumps(spatial), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"shape \(3,\)"):
                load_config(path)


class DeploymentVisualizationTests(unittest.TestCase):
    def setUp(self):
        import matplotlib

        matplotlib.use("Agg")
        self.config = load_config(EXAMPLE)
        self.states = generate_deployment_states(self.config)

    def test_animation_uses_exact_same_stored_geometry_and_can_export_motion(self):
        from PIL import Image, ImageChops

        from continuum_ftl.deployment_viewer import DeploymentViewer

        selected = (self.states[0], self.states[len(self.states) // 2], self.states[-1])
        viewer = DeploymentViewer(self.config, selected)
        try:
            with patch(
                "continuum_ftl.planar_kinematics.forward_kinematics_2d",
                side_effect=AssertionError("recomputed"),
            ):
                for index, state in enumerate(selected):
                    viewer.draw_frame(index)
                    for line, curve in zip(viewer.lines, state.forward.section_curves, strict=True):
                        assert_allclose(np.array(line.get_data()).T, curve.points_m)
                    assert_allclose(
                        np.array(viewer.tip.get_data()).ravel(), state.forward.tip_position_m
                    )
                    self.assertIn(f"{state.distance_m * 1000:6.1f} mm", viewer.status.get_text())
                with tempfile.TemporaryDirectory() as directory:
                    gif = Path(directory) / "deployment.gif"
                    viewer.save_gif(gif)
                    with Image.open(gif) as image:
                        self.assertEqual(image.n_frames, 3)
                        first = image.convert("RGB")
                        image.seek(2)
                        self.assertIsNotNone(
                            ImageChops.difference(first, image.convert("RGB")).getbbox()
                        )
            viewer.replay()
            self.assertEqual(viewer.current_index, 0)
            viewer.toggle_pause()
            self.assertTrue(viewer.paused)
            viewer.toggle_pause()
            self.assertFalse(viewer.paused)
            self.assertEqual(viewer.axes.get_aspect(), 1)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
