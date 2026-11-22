"""Stage 3A geometry, complete-zone coverage, provenance and visual replay."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from continuum_ftl.collision import (
    check_static_collision,
    segment_box_distance,
    segment_box_signed_distance,
)
from continuum_ftl.collision_postprocess import analyze_saved_ftl, collision_timeline_report
from continuum_ftl.config import load_config
from continuum_ftl.deployment import compute_deployment
from continuum_ftl.models import CircleObstacle, CollisionStatus, RectangleObstacle, SectionConfig

ROOT = Path(__file__).resolve().parents[1]


class SignedDistanceTests(unittest.TestCase):
    def test_inside_depth_is_not_saturated_at_zero_and_outside_is_unchanged(self):
        center, size = np.zeros(2), np.array([0.02, 0.02])
        start, end = np.array([0, -0.03]), np.array([0, 0.03])
        self.assertAlmostEqual(segment_box_distance(start, end, center, size), 0)
        self.assertAlmostEqual(segment_box_signed_distance(start, end, center, size), -0.01)
        outside = np.array([0.02, 0.0])
        self.assertAlmostEqual(segment_box_signed_distance(outside, outside, center, size), 0.01)
        boundary = np.array([0.01, 0])
        self.assertAlmostEqual(segment_box_signed_distance(boundary, boundary, center, size), 0)

    def test_signed_minimum_matches_independent_dense_point_sdf_in_2d_and_3d(self):
        random = np.random.default_rng(7)
        for dimension in (2, 3):
            center = random.uniform(-0.1, 0.1, dimension)
            size = random.uniform(0.01, 0.05, dimension)
            for _ in range(8):
                start = center + random.uniform(-0.05, 0.05, dimension)
                end = center + random.uniform(-0.05, 0.05, dimension)
                samples = np.linspace(start, end, 20001)
                q = np.abs(samples - center) - size / 2
                sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
                exact = segment_box_signed_distance(start, end, center, size)
                self.assertLessEqual(exact, sdf.min() + 1e-12)
                self.assertLessEqual(sdf.min() - exact, np.linalg.norm(end - start) / 40000 + 1e-12)


class CollisionPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        config = load_config(ROOT / "configs/robot_2d.json")
        self.config = replace(config, environment=replace(config.environment, obstacles=()))

    def saved_run(self, distances=(0.0, 0.08), bends=None):
        if bends is None:
            bends = [np.zeros(3) for _ in distances]
        steps = []
        for index, (distance, q) in enumerate(zip(distances, bends, strict=True)):
            state = compute_deployment(
                self.config.robot, self.config.deployment, distance, q, 0.001
            )
            steps.append(
                {
                    "step_index": index,
                    "accepted": True,
                    "phase": "approach" if index == 0 else "FTL",
                    "deployment_distance_m": distance,
                    "configuration_rad": state.applied_bends_rad.tolist(),
                    "actual_tip_position_m": state.forward.tip_position_m.tolist(),
                    "section_endpoints_m": state.forward.section_endpoints_m.tolist(),
                    "desired_reference_progress_m": index * 0.001,
                }
            )
        return {
            "steps": steps,
            "reference_path": {"points_m": [[0, 0], [0, 0.18]], "target_position_m": [0, 0.18]},
        }

    def analyze(self, data, config=None):
        path = self.folder / "input.json"
        raw = json.dumps(data, allow_nan=False).encode()
        path.write_bytes(raw)
        result = analyze_saved_ftl(config or self.config, path)
        self.assertEqual(path.read_bytes(), raw)
        return result

    def obstacles(self, *obstacles, margin=0.001):
        return replace(
            self.config,
            environment=replace(
                self.config.environment, obstacles=obstacles, safety_margin_m=margin
            ),
        )

    def test_full_passive_and_transition_bodies_below_entry_are_checked_without_optimization(self):
        config = self.obstacles(
            CircleObstacle("passive", np.array([0, -0.25]), 0.001),
            CircleObstacle("transition", np.array([0, -0.02]), 0.001),
        )
        data = self.saved_run((0.08,))
        original = copy.deepcopy(data)
        with (
            patch("scipy.optimize.minimize", side_effect=AssertionError("SQP")),
            patch(
                "continuum_ftl.ftl_rollout.run_single_path_ftl", side_effect=AssertionError("FTL")
            ),
        ):
            timeline = self.analyze(data, config)
        self.assertEqual(data, original)
        contact_ids = {c.section_index for c in timeline.frames[0].collision.contacts}
        self.assertIn(0, contact_ids)  # Passive entirely below environment entry.
        self.assertIn(3, contact_ids)  # A in the transition zone, tip just at entry.
        self.assertAlmostEqual(timeline.frames[0].state.forward.tip_position_m[1], 0)
        self.assertEqual(timeline.frames[0].state.section_names, ("Passive", "C", "B", "A"))

    def test_signed_penetration_uses_radius_and_existing_collision_decisions(self):
        config = self.obstacles(
            RectangleObstacle("deep", np.array([-0.01, -0.03]), np.array([0.01, -0.01]))
        )
        timeline = self.analyze(self.saved_run((0.08,)), config)
        report = collision_timeline_report(timeline, config)
        self.assertEqual(report["collision_frame_count"], 1)
        self.assertAlmostEqual(report["maximum_penetration_depth_estimate_m"], 0.014)
        frame = timeline.frames[0]
        robot = replace(
            config.robot, sections=(SectionConfig(0.2, 0.004, 0), *config.robot.sections)
        )
        legacy = check_static_collision(robot, frame.state.forward, config.environment)
        self.assertEqual(legacy.status, frame.collision.status)
        self.assertAlmostEqual(legacy.minimum_clearance_m, -0.004)

    def test_tube_grazing_and_safety_margin_are_separate_from_centerline_collision(self):
        data = self.saved_run((0.08,))
        for low, expected in (
            (0.004, CollisionStatus.COLLISION),
            (0.0045, CollisionStatus.SAFETY_MARGIN_VIOLATION),
            (0.006, CollisionStatus.CLEAR),
        ):
            config = self.obstacles(
                RectangleObstacle("wall", np.array([low, -0.03]), np.array([low + 0.01, -0.01]))
            )
            with self.subTest(low=low):
                timeline = self.analyze(data, config)
                self.assertEqual(timeline.frames[0].collision.status, expected)
                self.assertAlmostEqual(
                    timeline.frames[0].collision.minimum_clearance_m, low - 0.004
                )

    def test_empty_environment_clearance_is_null_and_rejected_trials_are_excluded(self):
        data = self.saved_run()
        data["steps"].append({"accepted": False})
        timeline = self.analyze(data)
        report = collision_timeline_report(timeline, self.config)
        self.assertEqual(report["checked_frame_count"], 2)
        self.assertEqual(report["excluded_rejected_frames"], 1)
        self.assertIsNone(report["minimum_clearance_lower_bound_m"])
        self.assertIsNone(report["first_collision"])
        self.assertEqual(report["maximum_penetration_depth_estimate_m"], 0)
        json.dumps(report, allow_nan=False)

    def test_collision_frames_episodes_and_replay_time_are_unambiguous(self):
        config = self.obstacles(CircleObstacle("hit", np.array([0, -0.02]), 0.001))
        timeline = self.analyze(self.saved_run((0, 0.08, 0.08, 0, 0.08)), config)
        report = collision_timeline_report(timeline, config)
        self.assertEqual(report["collision_frame_count"], 3)
        self.assertEqual(report["collision_episode_count"], 2)
        self.assertEqual(report["first_collision"]["source_frame_number"], 2)
        self.assertEqual(report["first_collision"]["source_step_index"], 1)
        self.assertAlmostEqual(report["first_collision"]["replay_time_s"], 0.06)
        self.assertIsNone(report["first_collision"]["physical_time_s"])
        self.assertFalse(report["physical_time_available"])

    def test_mismatched_pose_configuration_or_corrupt_saved_pose_is_rejected(self):
        data = self.saved_run()
        config = replace(
            self.config,
            deployment=replace(self.config.deployment, feed_exit_m=np.array([0, -0.09])),
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.analyze(data, config)
        data["steps"][0]["section_endpoints_m"][1][0] = 0.01
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.analyze(data)

    def test_replay_highlights_collision_restores_colors_and_exports_two_frame_gif(self):
        import matplotlib

        matplotlib.use("Agg")
        from continuum_ftl.collision_replay import CollisionReplay, save_clearance_plot

        config = self.obstacles(CircleObstacle("hit", np.array([0, -0.02]), 0.001))
        timeline = self.analyze(self.saved_run(), config)
        viewer = CollisionReplay(config, timeline)
        try:
            viewer.jump_to_first_collision()
            self.assertEqual(viewer.current_index, 1)
            self.assertEqual(viewer.lines[3].get_color(), "#d62728")
            self.assertIn("FIRST COLLISION HERE", viewer.warning.get_text())
            viewer.draw_frame(0)
            self.assertEqual(viewer.lines[3].get_color(), config.deployment.section_colors[3])
            viewer.save_gif(self.folder / "motion.gif")
            self.assertGreater((self.folder / "motion.gif").stat().st_size, 1000)
            save_clearance_plot(timeline, config, self.folder / "clearance.png")
            self.assertGreater((self.folder / "clearance.png").stat().st_size, 1000)
        finally:
            viewer.close()

    def test_cli_exports_diagnostics_without_changing_input_or_config(self):
        data = self.saved_run()
        source = self.folder / "ftl.json"
        source.write_text(json.dumps(data), encoding="utf-8")
        raw = json.loads((ROOT / "configs/robot_2d.json").read_text(encoding="utf-8"))
        raw["environment"]["obstacles"] = [
            {"kind": "circle", "name": "hit", "center_m": [0, -0.02], "radius_m": 0.001}
        ]
        config = self.folder / "config.json"
        config.write_text(json.dumps(raw), encoding="utf-8")
        original_source, original_config = source.read_bytes(), config.read_bytes()
        output = self.folder / "stage3a"
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "continuum_ftl.collision_demo",
                "--input",
                str(source),
                "--config",
                str(config),
                "--output-dir",
                str(output),
                "--no-show",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads((output / "collision_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["collision_frame_count"], 1)
        for name in ("frames.csv", "config_snapshot.json", "clearance.png", "first_collision.png"):
            self.assertTrue((output / name).is_file())
        self.assertEqual(source.read_bytes(), original_source)
        self.assertEqual(config.read_bytes(), original_config)


if __name__ == "__main__":
    unittest.main()
