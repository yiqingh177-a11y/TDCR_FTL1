"""Planar config, single-start IK, and shared static body collision checks."""

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
from scipy.optimize import OptimizeResult

from continuum_ftl.collision import check_static_collision, segment_box_distance
from continuum_ftl.config import load_config
from continuum_ftl.evaluation import evaluate_target, result_report
from continuum_ftl.inverse import solve_position_ik
from continuum_ftl.models import (
    CircleObstacle,
    CollisionStatus,
    IKStatus,
    RectangleObstacle,
    SphereObstacle,
)
from continuum_ftl.planar_kinematics import forward_kinematics_2d

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs/robot_2d.json"


class PlanarConfigTests(unittest.TestCase):
    def test_example_and_cross_dimension_validation(self):
        config = load_config(EXAMPLE)
        self.assertEqual(config.dimension, 2)
        self.assertEqual(config.robot.initial_configuration_rad.shape, (3,))
        self.assertFalse(config.target.position_m.flags.writeable)
        original = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        changes = [
            ("robot", "base_position_m", [0, 0, 0]),
            ("robot", "base_angle_rad", float("nan")),
            ("robot", "initial_configuration_rad", [[0, 0], [0, 0], [0, 0]]),
            ("robot", "initial_configuration_rad", [-2, 0, 0]),
            ("target", "position_m", [0, 0, 0]),
            ("environment", "workspace_bounds_m", [[0, 1]] * 3),
            (
                "environment",
                "obstacles",
                [{"kind": "sphere", "name": "bad", "center_m": [0, 0], "radius_m": 0.01}],
            ),
            (
                "environment",
                "obstacles",
                [{"kind": "circle", "name": "bad", "center_m": [0, 0, 0], "radius_m": 0.01}],
            ),
            (
                "environment",
                "obstacles",
                [{"kind": "circle", "name": "bad", "center_m": [0, 0], "radius_m": -1}],
            ),
            (
                "environment",
                "obstacles",
                [{"kind": "rectangle", "name": "bad", "min_m": [0, 1], "max_m": [1, 0]}],
            ),
            (
                "environment",
                "obstacles",
                [{"kind": "rectangle", "name": "bad", "min_m": [0, 0, 0], "max_m": [1, 1, 1]}],
            ),
            ("solver", "method", "not-a-solver"),
            ("visualization", "body_alpha", 2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for group, name, value in changes:
                data = copy.deepcopy(original)
                data[group][name] = value
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(field=f"{group}.{name}"), self.assertRaises(ValueError):
                    load_config(path)
            for dimension in (True, 1, "2", 4):
                data = {**original, "dimension": dimension}
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_legacy_3d_default_and_solver_remain_supported(self):
        config = load_config(ROOT / "configs/example.json")
        self.assertEqual(config.dimension, 3)
        self.assertEqual(config.solver.method, "SLSQP")


class PlanarIKTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(EXAMPLE)

    def solve(self, robot=None, target=None, settings=None):
        robot = self.config.robot if robot is None else robot
        return solve_position_ik(
            robot,
            self.config.target if target is None else target,
            np.zeros(robot.section_count),
            self.config.solver if settings is None else settings,
        )

    def test_positive_and_negative_single_section_targets(self):
        robot = replace(self.config.robot, sections=self.config.robot.sections[:1])
        theta = 0.7
        radius = robot.sections[0].length_m / theta
        for sign in (-1, 1):
            target = replace(
                self.config.target,
                position_m=np.array([sign * radius * (1 - np.cos(theta)), radius * np.sin(theta)]),
                tolerance_m=1e-6,
            )
            solved = self.solve(robot=robot, target=target)
            self.assertTrue(solved.target_reached)
            assert_allclose(solved.configuration_rad, [sign * theta], atol=1e-4)

    def test_multisection_example_both_methods_and_determinism(self):
        first, second = self.solve(), self.solve()
        self.assertTrue(first.target_reached)
        assert_allclose(first.configuration_rad, second.configuration_rad)
        alternative = self.solve(settings=replace(self.config.solver, method="L-BFGS-B"))
        self.assertTrue(alternative.target_reached)
        for count in (2, 4):
            robot = replace(self.config.robot, sections=(self.config.robot.sections[0],) * count)
            q = np.linspace(0.2, 0.5, count)
            target = replace(
                self.config.target, position_m=forward_kinematics_2d(robot, q, 0.01).tip_position_m
            )
            self.assertTrue(self.solve(robot=robot, target=target).target_reached)

    def test_length_exclusion_and_bending_limits(self):
        far = self.solve(target=replace(self.config.target, position_m=np.array([0, 2])))
        self.assertEqual(far.status, IKStatus.OUTSIDE_LENGTH_ENVELOPE)
        section = replace(self.config.robot.sections[0], max_bend_rad=0.1)
        robot = replace(self.config.robot, sections=(section,))
        target = replace(self.config.target, position_m=np.array([0.04, 0.06]))
        solved = self.solve(robot=robot, target=target)
        self.assertFalse(solved.target_reached)
        self.assertLessEqual(abs(solved.configuration_rad[0]), 0.1)
        self.assertNotEqual(solved.status, IKStatus.OUTSIDE_LENGTH_ENVELOPE)

    def test_convergence_is_not_geometric_success_and_failure_is_structured(self):
        for success, values, expected in (
            (True, np.zeros(3), IKStatus.TARGET_NOT_REACHED),
            (False, np.zeros(3), IKStatus.OPTIMIZATION_FAILED),
            (False, np.full(3, np.nan), IKStatus.OPTIMIZATION_FAILED),
            (True, np.full(3, -3.0), IKStatus.OPTIMIZATION_FAILED),
        ):
            fake = OptimizeResult(x=values, success=success, message="diagnostic", nit=1, nfev=1)
            with patch("continuum_ftl.inverse.minimize", return_value=fake):
                solved = self.solve()
            self.assertEqual(solved.status, expected)
            self.assertTrue(np.isfinite(solved.configuration_rad).all())

    def test_target_reached_but_colliding_and_report_orientation(self):
        obstacle = CircleObstacle("at_target", self.config.target.position_m, 0.01)
        config = replace(
            self.config, environment=replace(self.config.environment, obstacles=(obstacle,))
        )
        result = evaluate_target(config)
        self.assertTrue(result.ik.target_reached)
        self.assertFalse(result.static_target_feasible)
        report = result_report(result)
        self.assertEqual(report["position_coordinates"], "yz")
        self.assertIn("tip_orientation_angle_rad", report)
        self.assertNotIn("tip_orientation_matrix", report)
        self.assertFalse(report["motion_checked"])

    def test_dimension_mismatch_is_not_silently_converted(self):
        with self.assertRaises(ValueError):
            self.solve(target=replace(self.config.target, position_m=np.zeros(3)))

    def test_planar_headless_cli_does_not_import_matplotlib(self):
        script = (
            "import sys; from continuum_ftl.__main__ import main; "
            "sys.argv=['continuum_ftl','--config','configs/robot_2d.json',"
            "'--headless','--single-run']; "
            "main(); assert 'matplotlib' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        # The editable scene may deliberately obstruct the single IK solution.
        # This test verifies headless computation, not obstacle-aware IK.
        self.assertTrue(json.loads(result.stdout)["target_reached"])


class PlanarCollisionTests(unittest.TestCase):
    def setUp(self):
        config = load_config(EXAMPLE)
        section = replace(config.robot.sections[0], length_m=0.2, radius_m=0.01)
        self.robot = replace(config.robot, sections=(section,))
        self.forward = forward_kinematics_2d(self.robot, np.zeros(1), 1)
        self.environment = replace(config.environment, obstacles=(), safety_margin_m=0)

    def check(self, obstacle, margin=0):
        return check_static_collision(
            self.robot,
            self.forward,
            replace(self.environment, obstacles=(obstacle,), safety_margin_m=margin),
        )

    def test_circle_crossing_between_endpoints_and_radius(self):
        # Robot sample points are only (0,0) and (0,.2); obstacle is in between.
        obstacle = CircleObstacle("middle", np.array([0, 0.1]), 0.005)
        self.assertEqual(self.check(obstacle).status, CollisionStatus.COLLISION)
        self.assertAlmostEqual(self.check(obstacle).minimum_clearance_m, -0.015)
        body_only = replace(obstacle, center_m=np.array([0.012, 0.1]))
        self.assertEqual(self.check(body_only).status, CollisionStatus.COLLISION)
        clear = replace(obstacle, center_m=np.array([0.025, 0.1]))
        self.assertTrue(self.check(clear).collision_free)
        self.assertAlmostEqual(self.check(clear).minimum_clearance_m, 0.01)

    def test_rectangle_crossing_tangent_and_safety_margin(self):
        rectangle = RectangleObstacle("middle", np.array([-0.005, 0.09]), np.array([0.005, 0.11]))
        self.assertEqual(self.check(rectangle).status, CollisionStatus.COLLISION)
        outside = replace(rectangle, min_m=np.array([0.015, 0.09]), max_m=np.array([0.025, 0.11]))
        self.assertAlmostEqual(self.check(outside).minimum_clearance_m, 0.005)
        self.assertEqual(self.check(outside, 0.006).status, CollisionStatus.SAFETY_MARGIN_VIOLATION)
        tangent = CircleObstacle("tangent", np.array([0.02, 0.1]), 0.01)
        self.assertEqual(self.check(tangent).status, CollisionStatus.COLLISION)
        tangent_rect = replace(
            rectangle, min_m=np.array([0.01, 0.09]), max_m=np.array([0.02, 0.11])
        )
        self.assertFalse(self.check(tangent_rect).collision_free)

    def test_exact_rectangle_distance_and_dimension_rejection(self):
        distance = segment_box_distance(
            np.array([-2.0, 2.0]), np.array([2.0, 2.0]), np.zeros(2), np.ones(2) * 2
        )
        self.assertAlmostEqual(distance, 1)
        with self.assertRaises(ValueError):
            segment_box_distance(np.zeros(2), np.ones(2), np.zeros(3), np.ones(3))
        with self.assertRaises(ValueError):
            self.check(SphereObstacle("wrong_dimension", np.zeros(3), 0.01))

    def test_negative_arc_sagitta_covers_contact(self):
        robot = replace(self.robot, sections=(replace(self.robot.sections[0], radius_m=0.001),))
        radius = robot.sections[0].length_m / (np.pi / 2)
        midpoint = np.array([-radius * (1 - np.cos(np.pi / 4)), radius * np.sin(np.pi / 4)])
        curve = forward_kinematics_2d(robot, np.array([-np.pi / 2]), 1)
        obstacle = CircleObstacle("on_arc", midpoint, 0.0001)
        result = check_static_collision(
            robot, curve, replace(self.environment, obstacles=(obstacle,))
        )
        self.assertFalse(result.collision_free)


if __name__ == "__main__":
    unittest.main()
