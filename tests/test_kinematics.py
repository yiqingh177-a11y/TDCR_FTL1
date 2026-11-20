"""Independent geometric checks, not comparisons against old project code."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from continuum_ftl.config import load_config
from continuum_ftl.kinematics import forward_kinematics, local_pose

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "example.json"


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_config(EXAMPLE).robot

    def test_straight_robot_and_uniform_global_arc_sampling(self):
        result = forward_kinematics(self.robot, np.zeros((3, 2)), 0.007)
        length = sum(s.length_m for s in self.robot.sections)
        assert_allclose(result.tip_position_m, self.robot.base_position_m + [0, 0, length])
        assert_allclose(result.tip_orientation, np.eye(3))
        assert_allclose(result.centerline_m[:, :2], 0)
        assert_allclose(
            result.centerline_m[:, 2], self.robot.base_position_m[2] + result.arc_lengths_m
        )
        assert_allclose(np.diff(result.arc_lengths_m), np.diff(result.arc_lengths_m)[0])
        self.assertLessEqual(np.diff(result.arc_lengths_m).max(), 0.007)

    def test_single_quarter_circle_radius_and_tangent(self):
        section = replace(self.robot.sections[0], length_m=0.2)
        robot = replace(self.robot, sections=(section,), base_position_m=np.zeros(3))
        result = forward_kinematics(robot, np.array([[np.pi / 2, 0]]), 0.005)
        radius = section.length_m / (np.pi / 2)
        assert_allclose(result.tip_position_m, [radius, 0, radius], atol=1e-14)
        assert_allclose(result.tip_orientation[:, 2], [1, 0, 0], atol=1e-14)
        # Circle center is (R,0,0): every sample must lie on that circle.
        assert_allclose(np.linalg.norm(result.centerline_m - [radius, 0, 0], axis=1), radius)

    def test_zero_and_near_zero_curvature(self):
        for theta in (0.0, 1e-14, 1e-9):
            pose = local_pose(0.2, theta, 1.3, 0.2)
            assert_allclose(pose[:3, 3], [0, 0, 0.2], atol=2e-10)
            assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0)

    def test_serial_connection_with_independent_expected_pose(self):
        # First quarter circle bends to +X, rotating local +Z into world +X.
        # The second straight section must extend along +X, not world +Z.
        first, second = self.robot.sections[:2]
        robot = replace(self.robot, sections=(first, second), base_position_m=np.zeros(3))
        result = forward_kinematics(robot, np.array([[np.pi / 2, 0], [0, 0]]), 0.01)
        radius = first.length_m / (np.pi / 2)
        assert_allclose(result.section_endpoints_m[1], [radius, 0, radius], atol=1e-14)
        assert_allclose(result.tip_position_m, [radius + second.length_m, 0, radius], atol=1e-14)
        assert_allclose(result.section_curves[0].points_m[-1], result.section_curves[1].points_m[0])

    def test_base_pose_and_mixed_bending_with_independent_rotations(self):
        robot = replace(
            self.robot,
            base_position_m=np.array([0.1, -0.2, 0.3]),
            base_rpy_rad=np.array([0.2, -0.3, 0.4]),
        )
        configuration = np.array([[0.3, 0.2], [0.7, -1.1], [0.2, 2.0]])
        rotation = Rotation.from_euler("xyz", robot.base_rpy_rad).as_matrix()
        position = robot.base_position_m.copy()
        for section, (theta, phi) in zip(robot.sections, configuration, strict=True):
            radius = section.length_m / theta
            local_position = radius * np.array(
                [
                    (1 - np.cos(theta)) * np.cos(phi),
                    (1 - np.cos(theta)) * np.sin(phi),
                    np.sin(theta),
                ]
            )
            position += rotation @ local_position
            rotation = (
                rotation
                @ Rotation.from_rotvec(theta * np.array([-np.sin(phi), np.cos(phi), 0])).as_matrix()
            )
        result = forward_kinematics(robot, configuration, 0.004)
        assert_allclose(result.tip_position_m, position, atol=1e-14)
        assert_allclose(result.tip_orientation, rotation, atol=1e-14)

    def test_arc_lengths_are_physical_lengths_and_chords_converge(self):
        configuration = np.array([[1.1, 0.2], [0.7, -1.0], [1.3, 2.0]])
        coarse = forward_kinematics(self.robot, configuration, 0.02)
        fine = forward_kinematics(self.robot, configuration, 0.002)
        for section, (theta, _), a, b in zip(
            self.robot.sections,
            configuration,
            coarse.section_curves,
            fine.section_curves,
            strict=True,
        ):
            self.assertAlmostEqual(b.arc_lengths_m[-1], section.length_m)
            chord_length = np.linalg.norm(np.diff(b.points_m, axis=0), axis=1).sum()
            coarse_length = np.linalg.norm(np.diff(a.points_m, axis=0), axis=1).sum()
            self.assertGreater(chord_length, coarse_length)
            self.assertLess(chord_length, section.length_m)
            self.assertLess(section.length_m - chord_length, 1e-5)
            count = len(b.points_m) - 1
            expected = section.length_m / theta * (1 - np.cos(theta / (2 * count)))
            self.assertAlmostEqual(b.chord_deviation_bound_m, expected, places=14)

    def test_arbitrary_section_count(self):
        for count in (1, 2, 4, 5):
            robot = replace(self.robot, sections=(self.robot.sections[0],) * count)
            result = forward_kinematics(robot, np.zeros((count, 2)), 0.01)
            self.assertEqual(result.section_endpoints_m.shape, (count + 1, 3))
            self.assertAlmostEqual(
                result.tip_position_m[2],
                self.robot.base_position_m[2] + count * self.robot.sections[0].length_m,
            )

    def test_invalid_inputs(self):
        for configuration in (
            np.zeros((2, 2)),
            np.full((3, 2), np.nan),
            np.array([[-0.1, 0], [0, 0], [0, 0]]),
            np.array([[2, 0], [0, 0], [0, 0]]),
        ):
            with self.assertRaises(ValueError):
                forward_kinematics(self.robot, configuration, 0.01)
        with self.assertRaises(ValueError):
            forward_kinematics(self.robot, np.zeros((3, 2)), 0)


if __name__ == "__main__":
    unittest.main()
