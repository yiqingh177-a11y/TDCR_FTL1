"""Stage 2B exact PCC reference/exposure geometry and stored-state replay."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from continuum_ftl.config import geometry_context, load_config
from continuum_ftl.deployment import compute_deployment, generate_deployment_states
from continuum_ftl.environment_exposure import (
    exposure_report,
    extract_environment_exposure,
)
from continuum_ftl.evaluation import evaluate_target, result_report
from continuum_ftl.models import ExposureStatus, TargetConfig
from continuum_ftl.reference_path import (
    extract_reference_path,
    interpolate_reference_path,
    load_reference_path,
    reference_path_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/robot_2d.json"


class ReferencePathTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        # Independent geometric fixture. This is test input, not a demo solution.
        self.q = np.radians([90.0, -45.0, 45.0])
        first = 0.08 / (np.pi / 2) * np.array([1.0, 1.0])
        middle = 0.06 / (np.pi / 4) * np.array([np.sqrt(0.5), 1 - np.sqrt(0.5)])
        last = 0.04 / (np.pi / 4) * np.array([np.sqrt(0.5), 1 - np.sqrt(0.5)])
        target = first + middle + last
        self.config = replace(self.config, target=TargetConfig(target, 1e-9))

    def test_reference_exact_endpoints_boundaries_and_uniform_arc_sampling(self):
        path = extract_reference_path(self.config, self.q)
        assert_array_equal(path.points_m[0], [0, 0])
        assert_allclose(path.points_m[-1], self.config.target.position_m, atol=1e-15, rtol=0)
        assert_allclose(path.section_boundary_arcs_m, [0, 0.08, 0.14, 0.18], atol=1e-16)
        self.assertEqual(path.arc_lengths_m[0], 0)
        self.assertAlmostEqual(path.arc_lengths_m[-1], 0.18)
        for boundary in path.section_boundary_arcs_m:
            self.assertIn(boundary, path.arc_lengths_m)
        self.assertTrue(np.all(np.diff(path.arc_lengths_m) > 0))
        self.assertLessEqual(np.diff(path.arc_lengths_m).max(), 0.002 + 1e-15)
        assert_array_equal(
            interpolate_reference_path(path, np.array([0, path.arc_lengths_m[-1]])),
            path.points_m[[0, -1]],
        )
        self.assertGreaterEqual(path.points_m[:, 1].min(), 0)

    def test_arc_interpolation_uses_exact_circle_not_sampled_chords(self):
        path = extract_reference_path(self.config, self.q)
        arcs = np.array([0.0037, 0.031234, 0.0741])
        radius = 0.08 / (np.pi / 2)
        expected = radius * np.column_stack((1 - np.cos(arcs / radius), np.sin(arcs / radius)))
        assert_allclose(interpolate_reference_path(path, arcs), expected, atol=1e-15)
        for values in ([-1e-6], [0.181], [np.nan], [[0.01]]):
            with self.assertRaises(ValueError):
                interpolate_reference_path(path, np.array(values))

    def test_reference_clips_at_entry_without_extending_into_transition(self):
        robot = replace(self.config.robot, base_position_m=np.array([0.0, -0.01]))
        config = replace(self.config, robot=robot, target=TargetConfig(np.array([0, 0.17]), 1e-9))
        path = extract_reference_path(config, np.zeros(3))
        assert_allclose(path.points_m[0], [0, 0], atol=1e-14)
        assert_allclose(path.points_m[-1], [0, 0.17], atol=1e-14)
        self.assertAlmostEqual(path.entry_material_arc_m, 0.01)
        self.assertAlmostEqual(path.arc_lengths_m[-1], 0.17)
        assert_allclose(path.section_boundary_arcs_m, [0, 0.07, 0.13, 0.17])

    def test_near_zero_curvature_and_invalid_configuration(self):
        config = replace(self.config, target=TargetConfig(np.array([0, 0.18]), 1e-9))
        path = extract_reference_path(config, np.array([1e-12, -1e-12, 1e-12]))
        self.assertTrue(
            np.isfinite(interpolate_reference_path(path, np.array([0.001, 0.13]))).all()
        )
        with self.assertRaisesRegex(ValueError, "current target"):
            extract_reference_path(self.config, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "2D"):
            extract_reference_path(load_config(ROOT / "configs/example.json"), np.zeros((3, 2)))

    def test_load_one_accepted_stage1_result_checks_provenance_and_no_solver_calls(self):
        # Produce a genuine single-run Stage 1 report before entering Stage 2B.
        config = replace(
            self.config, robot=replace(self.config.robot, initial_configuration_rad=self.q)
        )
        accepted = evaluate_target(config)
        self.assertTrue(accepted.static_target_feasible)
        data = result_report(accepted)
        data["geometry_context"] = geometry_context(config)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "stage1.json"
            file.write_text(json.dumps(data), encoding="utf-8")
            with (
                patch("scipy.optimize.minimize", side_effect=AssertionError("optimizer")),
                patch(
                    "continuum_ftl.ik_search.find_collision_free_ik",
                    side_effect=AssertionError("search"),
                ),
                patch(
                    "continuum_ftl.collision.check_static_collision",
                    side_effect=AssertionError("collision"),
                ),
            ):
                path = load_reference_path(config, file)
                self.assertLessEqual(path.tip_error_m, config.target.tolerance_m)
                states = generate_deployment_states(config)
                for state in states:
                    extract_environment_exposure(
                        state, 0, max_arc_spacing_m=0.002, tolerance_m=1e-9
                    )
            # Stale target/environment reports must never be accepted silently.
            for key, value in (
                ("geometry_context", None),
                ("collision_status", "collision"),
                ("collision_free_with_margin", False),
                ("static_target_feasible", False),
                ("success", False),
            ):
                bad = {**data, key: value}
                file.write_text(json.dumps(bad), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    load_reference_path(config, file)
            file.write_text(json.dumps(data), encoding="utf-8")
            changed = replace(
                config, environment=replace(config.environment, safety_margin_m=0.002)
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_reference_path(changed, file)
            changed = replace(
                config, target=replace(config.target, position_m=np.array([0.1, 0.1]))
            )
            with self.assertRaisesRegex(ValueError, "stale"):
                load_reference_path(changed, file)
        json.dumps(reference_path_report(path), allow_nan=False)


class EnvironmentExposureTests(unittest.TestCase):
    def setUp(self):
        loaded = load_config(CONFIG)
        # Exact intersection expectations below use the original 5 cm feed gap.
        self.config = replace(
            loaded,
            deployment=replace(
                loaded.deployment,
                feed_exit_m=np.array([0.0, -0.05]),
                feed_block_center_m=np.array([0.0, -0.105]),
            ),
        )

    def state(self, distance, q=(0, 0, 0), spacing=0.002, robot=None):
        return compute_deployment(
            self.config.robot if robot is None else robot,
            self.config.deployment,
            distance,
            np.asarray(q),
            spacing,
        )

    def expose(self, state, entry=0.0, spacing=0.002):
        return extract_environment_exposure(
            state,
            entry,
            max_arc_spacing_m=spacing,
            tolerance_m=self.config.deployment.geometry_tolerance_m,
        )

    def test_empty_before_tip_enters_environment(self):
        for distance in (0, 0.02, 0.049):
            exposed = self.expose(self.state(distance))
            self.assertEqual(exposed.status, ExposureStatus.NOT_ENTERED)
            self.assertFalse(exposed.tip_entered)
            self.assertEqual(exposed.centerline_m.shape, (0, 2))
            self.assertEqual(exposed.exposed_length_m, 0)
            self.assertIsNone(exposed.entry_material_arc_m)
            assert_array_equal(exposed.section_exposed_lengths_m, np.zeros(4))

    def test_first_forward_contact_at_tip_and_straight_length(self):
        contact = self.expose(self.state(0.05))
        self.assertEqual(contact.status, ExposureStatus.EXPOSED)
        self.assertTrue(contact.tip_entered)
        self.assertAlmostEqual(contact.exposed_length_m, 0)
        assert_allclose(contact.entry_position_m, [0, 0], atol=1e-14)
        for distance in (0.06, 0.12, 0.26, 0.38):
            exposed = self.expose(self.state(distance))
            self.assertEqual(exposed.status, ExposureStatus.EXPOSED)
            self.assertAlmostEqual(exposed.exposed_length_m, distance - 0.05)
            self.assertAlmostEqual(exposed.entry_material_arc_m, 0.43 - distance)
            self.assertAlmostEqual(exposed.arc_lengths_m[0], 0)
            assert_allclose(exposed.centerline_m[-1], exposed.tip_position_m)

    def test_bent_length_is_exact_arc_not_deployment_minus_gap(self):
        distance, theta, length = 0.065, np.pi / 3, 0.04
        state = self.state(distance, [0, 0, theta])
        exposed = self.expose(state)
        curvature = theta / length
        start_z = distance - 0.09
        expected_local_entry = np.arcsin(-start_z * curvature) / curvature
        self.assertAlmostEqual(exposed.entry_material_arc_m, 0.34 + expected_local_entry, places=11)
        self.assertAlmostEqual(exposed.exposed_length_m, length - expected_local_entry, places=11)
        self.assertGreater(abs(exposed.exposed_length_m - (distance - 0.05)), 0.001)
        expected_y = (1 - np.cos(curvature * expected_local_entry)) / curvature
        assert_allclose(exposed.entry_position_m, [expected_y, 0], atol=1e-11)
        assert_allclose(exposed.section_exposed_lengths_m[:3], 0)
        self.assertEqual(exposed.section_exposed_lengths_m[-1], exposed.exposed_length_m)
        self.assertTrue(state.released[-1])

    def test_partial_section_b_participates_without_waiting_for_whole_section(self):
        # B is .06 m with k=10 /m; arrange exactly .04 m of B below entry.
        distance = 0.15 - np.sin(0.4) / 10
        state = self.state(distance, [0, 0.6, 0])
        before = state.forward.centerline_m.copy()
        exposed = self.expose(state)
        self.assertTrue(state.released[1])
        assert_allclose(exposed.section_exposed_lengths_m, [0, 0, 0.02, 0.04], atol=1e-11)
        self.assertAlmostEqual(exposed.entry_material_arc_m, 0.32, places=11)
        assert_array_equal(state.forward.centerline_m, before)
        self.assertEqual(state.applied_bends_rad[1], 0.6)
        self.assertEqual(len(state.forward.section_curves), 4)
        # The same analytic B arc spans transition and environment; no added joint.
        b_start = state.forward.section_endpoints_m[2]
        expected = b_start + np.array([(1 - np.cos(0.4)) / 10, np.sin(0.4) / 10])
        assert_allclose(exposed.entry_position_m, expected, atol=1e-11)
        local = exposed.material_arc_lengths_m[exposed.section_indices == 2] - 0.28
        expected_curve = b_start + np.column_stack(
            ((1 - np.cos(10 * local)) / 10, np.sin(10 * local) / 10)
        )
        assert_allclose(
            exposed.centerline_m[exposed.section_indices == 2], expected_curve, atol=1e-14
        )

    def test_section_accounting_identity_and_crossing_independent_of_original_sampling(self):
        fine_state = self.state(0.14, [0, 0.6, -0.2], spacing=1e-4)
        coarse_state = self.state(0.14, [0, 0.6, -0.2], spacing=0.5)
        fine, coarse = self.expose(fine_state), self.expose(coarse_state)
        assert_allclose(coarse.centerline_m, fine.centerline_m, atol=1e-14)
        self.assertAlmostEqual(fine.section_exposed_lengths_m.sum(), fine.exposed_length_m)
        self.assertAlmostEqual(fine.arc_lengths_m[-1], fine.exposed_length_m)
        assert_allclose(
            fine.material_arc_lengths_m,
            fine.arc_lengths_m + fine.entry_material_arc_m,
        )
        self.assertTrue(np.all(np.diff(fine.material_arc_lengths_m) > 0))
        for boundary in (0.28, 0.34):
            if boundary >= fine.entry_material_arc_m:
                self.assertIn(boundary, fine.material_arc_lengths_m)
        report = exposure_report(fine)
        self.assertEqual(report["sample_section_names"][0], "B")
        self.assertEqual(report["sample_section_names"][-1], "A")
        json.dumps(report, allow_nan=False)

    def test_multiple_crossings_return_ambiguity_not_unrelated_inside_points(self):
        state = self.state(0.18, np.full(3, np.pi / 2))
        exposed = self.expose(state)
        self.assertEqual(exposed.status, ExposureStatus.AMBIGUOUS)
        self.assertGreaterEqual(len(exposed.intersection_material_arcs_m), 2)
        self.assertEqual(exposed.centerline_m.shape, (0, 2))
        self.assertIsNone(exposed.exposed_length_m)
        self.assertIsNone(exposed.section_exposed_lengths_m)
        self.assertIn("Single forward entry", exposed.diagnostic)

    def test_multiple_crossings_within_one_coarsely_sampled_arc_are_detected(self):
        robot = replace(
            self.config.robot,
            sections=tuple(
                replace(section, max_bend_rad=2 * np.pi) for section in self.config.robot.sections
            ),
        )
        state = self.state(0.23, [2 * np.pi, 0, 0], spacing=0.5, robot=robot)
        exposed = self.expose(state)
        self.assertEqual(exposed.status, ExposureStatus.AMBIGUOUS)
        assert_allclose(exposed.intersection_material_arcs_m, [0.2, 0.24, 0.28], atol=1e-12)

    def test_tangent_and_plane_coincident_geometry_are_diagnosed(self):
        robot = replace(
            self.config.robot,
            sections=tuple(
                replace(section, max_bend_rad=np.pi) for section in self.config.robot.sections
            ),
        )
        radius = 0.08 / np.pi
        exposed = self.expose(self.state(0.23 - radius, [np.pi, 0, 0], robot=robot))
        self.assertEqual(exposed.status, ExposureStatus.AMBIGUOUS)
        self.assertIn("tangential", exposed.diagnostic)
        state = self.state(0.18, [np.pi / 2, 0, 0])
        exposed = self.expose(state, entry=state.forward.section_endpoints_m[2, 1])
        self.assertEqual(exposed.status, ExposureStatus.AMBIGUOUS)
        self.assertIn("plane-coincident section=True", exposed.diagnostic)

    def test_default_states_preserved_and_all_exposures_accounted_for(self):
        config_before = copy.deepcopy(geometry_context(self.config))
        states = generate_deployment_states(self.config)
        exposures = [self.expose(state) for state in states]
        self.assertEqual(len(states), 131)
        self.assertFalse(any(e.status == ExposureStatus.AMBIGUOUS for e in exposures))
        self.assertAlmostEqual(
            next(s.distance_m for s, e in zip(states, exposures, strict=True) if e.tip_entered),
            0.052,
        )
        for state, exposed in zip(states, exposures, strict=True):
            self.assertAlmostEqual(state.section_lengths_m.sum(), 0.38)
            self.assertAlmostEqual(
                exposed.section_exposed_lengths_m.sum(), exposed.exposed_length_m
            )
        self.assertEqual(geometry_context(self.config), config_before)
        for spacing in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.expose(states[0], spacing=spacing)


class ExposureVisualizationTests(unittest.TestCase):
    def test_visible_pipeline_preserves_actual_shapes_and_reference_is_fixed(self):
        import matplotlib

        matplotlib.use("Agg")
        from PIL import Image

        from continuum_ftl.deployment_viewer import DeploymentViewer

        config = load_config(CONFIG)
        # Generate one Stage 1 solution using its existing single-run API, not
        # the Stage 2B demo. The optimizer/checker are forbidden during replay.
        reference_config = replace(
            config, robot=replace(config.robot, initial_configuration_rad=np.radians([90, -45, 45]))
        )
        result = evaluate_target(reference_config)
        self.assertTrue(result.static_target_feasible)
        path = extract_reference_path(config, result.ik.configuration_rad)
        all_states = generate_deployment_states(config)
        states = tuple(all_states[i] for i in (0, 33, 70, 130))
        exposures = tuple(
            extract_environment_exposure(state, 0, max_arc_spacing_m=0.002, tolerance_m=1e-9)
            for state in states
        )
        viewer = DeploymentViewer(config, states, reference_path=path, exposures=exposures)
        try:
            with (
                patch(
                    "continuum_ftl.planar_kinematics.forward_kinematics_2d",
                    side_effect=AssertionError("FK"),
                ),
                patch("scipy.optimize.minimize", side_effect=AssertionError("optimizer")),
                patch(
                    "continuum_ftl.collision.check_static_collision",
                    side_effect=AssertionError("collision"),
                ),
            ):
                for index, state in enumerate(states):
                    viewer.draw_frame(index)
                    for line, curve in zip(viewer.lines, state.forward.section_curves, strict=True):
                        assert_array_equal(np.array(line.get_data()).T, curve.points_m)
                    assert_array_equal(np.array(viewer.reference_line.get_data()).T, path.points_m)
                    self.assertIn("Exposed length", viewer.status.get_text())
                    self.assertIn("NOT FTL", viewer.fig._suptitle.get_text())
                    for line in viewer.exposed_lines:
                        points = np.array(line.get_data()).T
                        if len(points):
                            self.assertGreaterEqual(points[:, 1].min(), -1e-9)
                with tempfile.TemporaryDirectory() as directory:
                    gif = Path(directory) / "exposure.gif"
                    viewer.save_gif(gif)
                    with Image.open(gif) as image:
                        self.assertEqual(image.n_frames, len(states))
                viewer.replay()
                self.assertEqual(viewer.current_index, 0)
                viewer.toggle_pause()
                self.assertTrue(viewer.paused)
                self.assertEqual(viewer.axes.get_aspect(), 1)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
