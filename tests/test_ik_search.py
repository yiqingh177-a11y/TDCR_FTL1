"""Multi-start success, honest exhaustion, duplicates, references and tip-only FK."""

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
from numpy.testing import assert_allclose

from continuum_ftl.config import load_config
from continuum_ftl.evaluation import evaluate_configuration
from continuum_ftl.ik_search import find_collision_free_ik, search_result_report
from continuum_ftl.inverse import solve_position_ik
from continuum_ftl.kinematics import forward_kinematics, tip_pose
from continuum_ftl.models import CircleObstacle, CollisionStatus, IKSearchConfig, SphereObstacle
from continuum_ftl.planar_kinematics import forward_kinematics_2d, tip_pose_2d

ROOT = Path(__file__).resolve().parents[1]


class IKSearchTests(unittest.TestCase):
    def setUp(self):
        self.planar = load_config(ROOT / "configs/robot_2d.json")
        self.spatial = load_config(ROOT / "configs/example.json")
        self.reference = np.deg2rad([90, -45, 45])

    def test_obstacle_free_2d_and_3d_stop_on_first_valid_solution(self):
        for config in (self.planar, self.spatial):
            config = replace(config, environment=replace(config.environment, obstacles=()))
            with patch(
                "continuum_ftl.ik_search.solve_position_ik", wraps=solve_position_ik
            ) as solver:
                result = find_collision_free_ik(config)
            self.assertTrue(result.success)
            self.assertEqual(result.attempts, 1)
            self.assertEqual(solver.call_count, 1)
            self.assertLessEqual(result.tip_error, config.target.tolerance_m)
            self.assertIsNone(result.minimum_clearance)
            self.assertIsNone(result.failure_reason)
            self.assertGreaterEqual(result.computation_time, 0)

    def test_staircase_without_reference_rejects_collision_then_finds_clear_shape(self):
        result = find_collision_free_ik(self.planar)
        self.assertTrue(result.success)
        self.assertGreater(result.attempts, 1)
        self.assertEqual(result.history[0].collision_status, CollisionStatus.COLLISION)
        self.assertEqual(result.history[-1].collision_status, CollisionStatus.CLEAR)
        self.assertTrue(result.diagnostic.static_target_feasible)
        self.assertGreater(result.minimum_clearance, self.planar.environment.safety_margin_m)
        # The test reference never enters the unseeded search.
        self.assertTrue(all(a.seed_source != "reference" for a in result.history))
        self.assertFalse(search_result_report(result)["motion_checked"])

    def test_reference_is_first_and_accurately_checked_not_hard_coded(self):
        with patch(
            "continuum_ftl.ik_search.evaluate_configuration",
            wraps=evaluate_configuration,
        ) as checker:
            result = find_collision_free_ik(self.planar, self.reference)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.history[0].seed_source, "reference")
        assert_allclose(result.history[0].seed_configuration_rad, self.reference)
        self.assertEqual(checker.call_count, 1)
        self.assertGreater(result.minimum_clearance, 0.005)
        # A different user reference must be used literally, including colliding ones.
        result = find_collision_free_ik(
            self.planar, np.zeros(3), settings=IKSearchConfig(max_attempts=1)
        )
        self.assertFalse(result.success)
        self.assertEqual(result.history[0].collision_status, CollisionStatus.COLLISION)

    def test_actual_budget_exhaustion_returns_failure_code_in_both_dimensions(self):
        for config in (self.planar, self.spatial):
            obstacle_type = CircleObstacle if config.dimension == 2 else SphereObstacle
            obstacle = obstacle_type("covers_target", config.target.position_m, 0.02)
            blocked = replace(
                config, environment=replace(config.environment, obstacles=(obstacle,))
            )
            result = find_collision_free_ik(blocked, settings=IKSearchConfig(max_attempts=4))
            self.assertFalse(result.success)
            self.assertEqual(result.attempts, 4)
            self.assertEqual(result.failure_reason, "NO_COLLISION_FREE_IK_FOUND")
            self.assertEqual(result.termination, "attempt_budget_exhausted")
            self.assertIsNone(result.configuration)
            self.assertIsNone(result.minimum_clearance)
            self.assertIsNotNone(result.diagnostic)
            self.assertIsNone(search_result_report(result)["accepted_final_configuration"])

    def test_random_seeds_and_result_are_reproducible_and_within_limits(self):
        for config in (self.planar, self.spatial):
            obstacle_type = CircleObstacle if config.dimension == 2 else SphereObstacle
            blocked = replace(
                config,
                environment=replace(
                    config.environment,
                    obstacles=(obstacle_type("covers_target", config.target.position_m, 0.02),),
                ),
            )
            settings = IKSearchConfig(max_attempts=4, random_seed=32)
            first = find_collision_free_ik(blocked, settings=settings)
            second = find_collision_free_ik(blocked, settings=settings)
            self.assertEqual(first.attempts, second.attempts)
            limits = np.array([s.max_bend_rad for s in config.robot.sections])
            for a, b in zip(first.history, second.history, strict=True):
                assert_allclose(a.seed_configuration_rad, b.seed_configuration_rad)
                assert_allclose(a.configuration_rad, b.configuration_rad)
                theta = (
                    np.abs(a.seed_configuration_rad)
                    if config.dimension == 2
                    else a.seed_configuration_rad[:, 0]
                )
                self.assertTrue(np.all(theta <= limits))

    def test_duplicate_seed_and_repeated_candidate_are_not_rechecked(self):
        initial = self.planar.robot.initial_configuration_rad
        solved = solve_position_ik(
            self.planar.robot, self.planar.target, initial, self.planar.solver
        )
        # Both distinct seeds return the same rejected endpoint: only one body check.
        with (
            patch("continuum_ftl.ik_search.solve_position_ik", return_value=solved) as solver,
            patch(
                "continuum_ftl.ik_search.evaluate_configuration", wraps=evaluate_configuration
            ) as checker,
        ):
            result = find_collision_free_ik(
                self.planar, initial.copy(), settings=IKSearchConfig(max_attempts=2)
            )
        self.assertEqual(solver.call_count, 2)
        self.assertEqual(checker.call_count, 1)
        self.assertEqual(result.skipped_duplicate_seeds, 1)
        self.assertEqual(result.skipped_duplicate_candidates, 1)
        self.assertTrue(result.history[-1].duplicate_candidate)

    def test_3d_phi_wraps_and_straight_directions_do_not_create_new_seeds(self):
        original = self.spatial.robot.initial_configuration_rad
        reference = original.copy()
        reference[:, 1] += 2 * np.pi
        blocked = replace(
            self.spatial,
            environment=replace(
                self.spatial.environment,
                obstacles=(SphereObstacle("covers_target", self.spatial.target.position_m, 0.02),),
            ),
        )
        result = find_collision_free_ik(blocked, reference, settings=IKSearchConfig(max_attempts=2))
        self.assertEqual(result.skipped_duplicate_seeds, 1)
        self.assertEqual(result.history[0].seed_source, "reference")
        self.assertEqual(result.history[1].seed_source, "random")

    def test_seed_draw_limit_is_bounded_when_all_proposals_are_duplicates(self):
        settings = IKSearchConfig(max_attempts=5, max_seed_draws=3, seed_distance_tolerance_rad=100)
        result = find_collision_free_ik(self.planar, settings=settings)
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.skipped_duplicate_seeds, 3)
        self.assertEqual(result.termination, "seed_draw_budget_exhausted")

    def test_tip_misses_skip_full_sampling_and_collision(self):
        far = replace(self.planar, target=replace(self.planar.target, position_m=np.array([0, 2])))
        with patch("continuum_ftl.ik_search.evaluate_configuration") as checker:
            result = find_collision_free_ik(far)
        checker.assert_not_called()
        self.assertFalse(result.success)
        self.assertIsNone(result.diagnostic)
        self.assertEqual(result.attempts, 1)

    def test_programming_errors_are_not_hidden_and_invalid_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            find_collision_free_ik(self.planar, np.ones(3) * 2)
        with self.assertRaises(ValueError):
            find_collision_free_ik(self.planar, np.zeros((3, 2)))
        with patch("continuum_ftl.ik_search.solve_position_ik", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                find_collision_free_ik(self.planar)

    def test_tip_only_fk_matches_complete_fk_in_2d_and_3d(self):
        q2 = np.array([0.8, -0.6, 0.4])
        q3 = np.array([[0.8, 0.2], [0.6, -1.1], [0.4, 2.0]])
        for robot, q, full_fk, tip_fk in (
            (self.planar.robot, q2, forward_kinematics_2d, tip_pose_2d),
            (self.spatial.robot, q3, forward_kinematics, tip_pose),
        ):
            full = full_fk(robot, q, 0.002)
            position, orientation = tip_fk(robot, q)
            assert_allclose(position, full.tip_position_m, atol=1e-14)
            assert_allclose(orientation, full.tip_orientation, atol=1e-14)
        # Ordinary solver must never invoke full sampled FK, even at trial states.
        with (
            patch(
                "continuum_ftl.kinematics.forward_kinematics",
                side_effect=AssertionError("sampling"),
            ),
            patch(
                "continuum_ftl.planar_kinematics.forward_kinematics_2d",
                side_effect=AssertionError("sampling"),
            ),
        ):
            for config in (self.planar, self.spatial):
                solved = solve_position_ik(
                    config.robot,
                    config.target,
                    config.robot.initial_configuration_rad,
                    config.solver,
                )
                self.assertTrue(solved.target_reached)

    def test_search_configuration_validation_and_reference_loading(self):
        raw = json.loads((ROOT / "configs/robot_2d.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            raw["ik_search"]["reference_configuration_rad"] = self.reference.tolist()
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(path)
            self.assertTrue(find_collision_free_ik(config).success)
            for name, value in (
                ("max_attempts", 0),
                ("max_seed_draws", -1),
                ("random_seed", True),
                ("seed_distance_tolerance_rad", -1),
                ("candidate_distance_tolerance_rad", float("nan")),
                ("reference_configuration_rad", [0, 0]),
                ("reference_configuration_rad", [-2, 0, 0]),
                ("unexpected", 1),
            ):
                data = copy.deepcopy(raw)
                data["ik_search"][name] = value
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(field=name), self.assertRaises(ValueError):
                    load_config(path)

    def test_headless_search_cli_and_original_single_run_are_separate(self):
        for mode in ("--search", "--single-run"):
            script = (
                "import sys; from continuum_ftl.__main__ import main; "
                "sys.argv=['continuum_ftl','--config','configs/robot_2d.json','--headless',"
                f"'{mode}']; main(); assert 'matplotlib' not in sys.modules"
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            if mode == "--search":
                self.assertTrue(report["success"])
                self.assertGreater(report["attempts"], 1)
            else:
                self.assertTrue(report["target_reached"])
                self.assertFalse(report["static_target_feasible"])


if __name__ == "__main__":
    unittest.main()
