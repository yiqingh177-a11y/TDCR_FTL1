"""Signed planar geometry and fixed-coordinate comparison with original 3D PCC."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_ftl.config import load_config
from continuum_ftl.kinematics import forward_kinematics
from continuum_ftl.models import RobotConfig
from continuum_ftl.planar_kinematics import forward_kinematics_2d

ROOT = Path(__file__).resolve().parents[1]


class PlanarKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_config(ROOT / "configs/robot_2d.json").robot

    def test_straight_uniform_arc_length_and_boundaries(self):
        result = forward_kinematics_2d(self.robot, np.zeros(3), 0.003)
        assert_allclose(result.centerline_m[:, 0], 0)
        assert_allclose(result.centerline_m[:, 1], result.arc_lengths_m)
        assert_allclose(np.diff(result.arc_lengths_m), np.diff(result.arc_lengths_m)[0])
        self.assertTrue(np.all(np.diff(result.arc_lengths_m) > 0))
        self.assertAlmostEqual(result.arc_lengths_m[-1], 0.18)
        assert_allclose(result.section_endpoints_m[:, 1], [0, 0.08, 0.14, 0.18])
        self.assertEqual(result.tip_orientation, 0)

    def test_positive_negative_quarter_circles_and_tangents(self):
        robot = replace(self.robot, sections=(self.robot.sections[0],))
        radius = robot.sections[0].length_m / (np.pi / 2)
        for sign in (-1, 1):
            result = forward_kinematics_2d(robot, np.array([sign * np.pi / 2]), 0.002)
            assert_allclose(result.tip_position_m, [sign * radius, radius], atol=1e-14)
            self.assertAlmostEqual(result.tip_orientation, sign * np.pi / 2)
            assert_allclose(
                np.linalg.norm(result.centerline_m - [sign * radius, 0], axis=1), radius
            )

    def test_chaining_uses_tangent_and_configurable_base(self):
        robot = replace(self.robot, sections=self.robot.sections[:2])
        result = forward_kinematics_2d(robot, np.array([np.pi / 2, 0]), 0.002)
        radius = robot.sections[0].length_m / (np.pi / 2)
        assert_allclose(result.tip_position_m, [radius + robot.sections[1].length_m, radius])
        assert_allclose(result.section_curves[0].points_m[-1], result.section_curves[1].points_m[0])
        moved = replace(robot, base_position_m=np.array([0.12, -0.05]), base_angle_rad=np.pi / 2)
        straight = forward_kinematics_2d(moved, np.zeros(2), 0.002)
        assert_allclose(straight.tip_position_m, moved.base_position_m + [0.14, 0], atol=1e-14)

    def test_signed_near_zero_and_length_convergence(self):
        for sign in (-1, 1):
            result = forward_kinematics_2d(self.robot, np.full(3, sign * 1e-12), 0.002)
            assert_allclose(result.tip_position_m, [0, 0.18], atol=1e-12)
            self.assertTrue(np.isfinite(result.centerline_m).all())
        q = np.array([0.7, -1.2, 0.9])
        coarse = forward_kinematics_2d(self.robot, q, 0.02)
        fine = forward_kinematics_2d(self.robot, q, 0.001)
        for section, a, b in zip(
            self.robot.sections, coarse.section_curves, fine.section_curves, strict=True
        ):
            first = np.linalg.norm(np.diff(a.points_m, axis=0), axis=1).sum()
            second = np.linalg.norm(np.diff(b.points_m, axis=0), axis=1).sum()
            self.assertLess(first, second)
            self.assertLess(second, section.length_m)
            self.assertAlmostEqual(b.arc_lengths_m[-1], section.length_m)

    def test_arbitrary_counts_and_invalid_inputs(self):
        for count in (1, 2, 4, 5):
            robot = replace(self.robot, sections=(self.robot.sections[0],) * count)
            result = forward_kinematics_2d(robot, np.zeros(count), 0.01)
            self.assertEqual(result.section_endpoints_m.shape, (count + 1, 2))
        for q in (np.zeros((3, 2)), np.zeros(2), np.full(3, np.nan), np.array([-2.0, 0, 0])):
            with self.assertRaises(ValueError):
                forward_kinematics_2d(self.robot, q, 0.002)
        with self.assertRaises(ValueError):
            forward_kinematics_2d(self.robot, np.zeros(3), 0)


class PCCConsistencyTests(unittest.TestCase):
    def compare(self, q, base_angle=0.0, base_yz=None):
        """1e-12 m allows accumulated double rounding, well below a micron."""
        robot = load_config(ROOT / "configs/robot_2d.json").robot
        base = np.zeros(2) if base_yz is None else np.asarray(base_yz)
        robot = replace(
            robot,
            sections=robot.sections[: len(q)],
            base_position_m=base,
            base_angle_rad=base_angle,
        )
        q = np.asarray(q, dtype=float)
        # Verified from existing 3D local_pose: phi=+pi/2 bends to local +Y.
        spatial_q = np.column_stack((np.abs(q), np.where(q >= 0, np.pi / 2, -np.pi / 2)))
        spatial_robot = RobotConfig(
            robot.sections, np.r_[0.0, base], np.array([-base_angle, 0, 0]), spatial_q
        )
        planar = forward_kinematics_2d(robot, q, 0.003)
        spatial = forward_kinematics(spatial_robot, spatial_q, 0.003)
        assert_allclose(planar.arc_lengths_m, spatial.arc_lengths_m, atol=0, rtol=0)
        error = np.linalg.norm(planar.centerline_m - spatial.centerline_m[:, 1:], axis=1).max()
        x_error = np.abs(spatial.centerline_m[:, 0]).max()
        diagnostic = f"q={q}; max_yz_error_m={error:.6e}; max_abs_x_m={x_error:.6e}"
        self.assertLessEqual(error, 1e-12, diagnostic)
        self.assertLessEqual(x_error, 1e-12, diagnostic)
        assert_allclose(planar.section_endpoints_m, spatial.section_endpoints_m[:, 1:], atol=1e-12)
        assert_allclose(planar.tip_position_m, spatial.tip_position_m[1:], atol=1e-12)
        assert_allclose(
            [0, np.sin(planar.tip_orientation), np.cos(planar.tip_orientation)],
            spatial.tip_orientation[:, 2],
            atol=1e-12,
        )
        for planar_curve, spatial_curve in zip(
            planar.section_curves, spatial.section_curves, strict=True
        ):
            assert_allclose(planar_curve.points_m, spatial_curve.points_m[:, 1:], atol=1e-12)

    def test_required_seven_planar_spatial_cases(self):
        cases = {
            "straight": [0, 0, 0],
            "single_positive": [0.8],
            "single_negative": [-0.8],
            "same_direction": [0.3, 0.7, 0.9],
            "s_bend": [0.8, -1.1, 0.6],
            "near_zero": [1e-12, -1e-12, 0],
            "maximum": [np.pi / 2, -np.pi / 2, np.pi / 2],
        }
        for name, q in cases.items():
            with self.subTest(case=name):
                self.compare(q)

    def test_nonzero_matched_base_pose(self):
        self.compare([0.6, -0.9, 0.4], base_angle=0.35, base_yz=[0.1, -0.02])


if __name__ == "__main__":
    unittest.main()
