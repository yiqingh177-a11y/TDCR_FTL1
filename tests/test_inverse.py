"""Single-start IK acceptance, failure semantics, and no hidden GUI imports."""

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult

from continuum_ftl.config import load_config
from continuum_ftl.evaluation import evaluate_target
from continuum_ftl.inverse import bends_to_configuration, configuration_to_bends, solve_position_ik
from continuum_ftl.kinematics import forward_kinematics
from continuum_ftl.models import IKStatus, SphereObstacle

ROOT = Path(__file__).resolve().parents[1]


class InverseTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/example.json")

    def solve(self, target):
        return solve_position_ik(
            self.config.robot,
            target,
            self.config.robot.initial_configuration_rad,
            self.config.solver,
        )

    def test_reachable_example_and_repeatability(self):
        first, second = self.solve(self.config.target), self.solve(self.config.target)
        self.assertTrue(first.target_reached)
        self.assertLessEqual(first.position_error_m, self.config.target.tolerance_m)
        assert_allclose(first.configuration_rad, second.configuration_rad)
        self.assertTrue(
            np.all(
                first.configuration_rad[:, 0]
                <= [s.max_bend_rad for s in self.config.robot.sections]
            )
        )
        tip = forward_kinematics(self.config.robot, first.configuration_rad, 0.01).tip_position_m
        assert_allclose(tip, first.tip_position_m)

    def test_straight_target(self):
        point = self.config.robot.base_position_m + np.array(
            [0, 0, sum(section.length_m for section in self.config.robot.sections)]
        )
        target = replace(self.config.target, position_m=point)
        self.assertTrue(self.solve(target).target_reached)

    def test_single_section_ik_and_bending_bound(self):
        section = self.config.robot.sections[0]
        robot = replace(self.config.robot, sections=(section,))
        theta, phi = 0.8, 0.6
        radius = section.length_m / theta
        point = robot.base_position_m + radius * np.array(
            [(1 - np.cos(theta)) * np.cos(phi), (1 - np.cos(theta)) * np.sin(phi), np.sin(theta)]
        )
        target = replace(self.config.target, position_m=point, tolerance_m=1e-6)
        result = solve_position_ik(robot, target, np.zeros((1, 2)), self.config.solver)
        self.assertTrue(result.target_reached)
        assert_allclose(result.configuration_rad, [[theta, phi]], atol=1e-4)
        # Same target remains within the total-length sphere but violates a
        # tightened section bend limit: this is different from length exclusion.
        robot = replace(robot, sections=(replace(section, max_bend_rad=0.2),))
        result = solve_position_ik(robot, target, np.zeros((1, 2)), self.config.solver)
        self.assertFalse(result.target_reached)
        self.assertNotEqual(result.status, IKStatus.OUTSIDE_LENGTH_ENVELOPE)
        self.assertLessEqual(result.configuration_rad[0, 0], 0.2)

    def test_outside_total_length_is_explicit(self):
        result = self.solve(replace(self.config.target, position_m=np.array([0, 0, 2.0])))
        self.assertEqual(result.status, IKStatus.OUTSIDE_LENGTH_ENVELOPE)
        self.assertFalse(result.target_reached)
        self.assertEqual(result.function_evaluations, 0)

    def test_convergence_alone_does_not_mean_target_reached(self):
        fake = OptimizeResult(x=np.zeros(6), success=True, message="Converged", nit=1, nfev=1)
        with patch("continuum_ftl.inverse.minimize", return_value=fake):
            result = self.solve(self.config.target)
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.status, IKStatus.TARGET_NOT_REACHED)

    def test_optimizer_failure_and_invalid_iterate_are_structured(self):
        for values in (np.zeros(6), np.full(6, np.nan), np.ones(6) * 3):
            fake = OptimizeResult(
                x=values, success=False, message="Budget exhausted", nit=1, nfev=1
            )
            with patch("continuum_ftl.inverse.minimize", return_value=fake):
                result = self.solve(self.config.target)
            self.assertEqual(result.status, IKStatus.OPTIMIZATION_FAILED)
            self.assertTrue(np.isfinite(result.configuration_rad).all())

    def test_target_reached_but_colliding_is_not_static_success(self):
        obstacle = SphereObstacle("blocks_tip", self.config.target.position_m, 0.02)
        config = replace(
            self.config, environment=replace(self.config.environment, obstacles=(obstacle,))
        )
        result = evaluate_target(config)
        self.assertTrue(result.ik.target_reached)
        self.assertFalse(result.collision.collision_free)
        self.assertFalse(result.static_target_feasible)

    def test_configuration_coordinate_roundtrip(self):
        configuration = np.array([[0, 2], [0.4, -1.3], [0.6, 1.2]])
        bends = configuration_to_bends(configuration)
        assert_allclose(configuration_to_bends(bends_to_configuration(bends)), bends)
        with self.assertRaises(ValueError):
            bends_to_configuration(np.zeros(3))

    def test_headless_cli_never_imports_matplotlib(self):
        script = (
            "import sys; from continuum_ftl.__main__ import main; "
            "sys.argv=['continuum_ftl','--headless']; main(); "
            "assert 'matplotlib' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertTrue(report["static_target_feasible"])
        self.assertFalse(report["motion_checked"])


if __name__ == "__main__":
    unittest.main()
