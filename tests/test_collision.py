"""Whole-body radius, conservative curved geometry, and exact segment checks."""

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from continuum_ftl.collision import (
    check_static_collision,
    segment_box_distance,
    segment_point_distance,
)
from continuum_ftl.config import load_config
from continuum_ftl.kinematics import forward_kinematics
from continuum_ftl.models import BoxObstacle, CollisionStatus, SphereObstacle


class CollisionTests(unittest.TestCase):
    def setUp(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/example.json")
        # Fix geometry for these independent distance examples, not the editable demo.
        self.robot = replace(
            config.robot,
            sections=tuple(
                replace(section, length_m=0.12, radius_m=0.008) for section in config.robot.sections
            ),
        )
        self.environment = replace(config.environment, obstacles=(), safety_margin_m=0)
        self.forward = forward_kinematics(self.robot, np.zeros((3, 2)), 0.01)

    def check(self, obstacle, margin=0):
        environment = replace(self.environment, obstacles=(obstacle,), safety_margin_m=margin)
        return check_static_collision(self.robot, self.forward, environment)

    def test_point_segment_distance_and_zero_length(self):
        self.assertAlmostEqual(
            segment_point_distance(np.zeros(3), np.ones(3), np.ones(3) * 2), np.sqrt(3)
        )
        self.assertAlmostEqual(
            segment_point_distance(np.zeros(3), np.zeros(3), np.ones(3)), np.sqrt(3)
        )

    def test_box_distance_crossing_parallel_endpoint_and_corner(self):
        center, size = np.zeros(3), np.ones(3) * 2
        examples = [
            ([-2, 0, 0], [2, 0, 0], 0),
            ([-2, 2, 0], [2, 2, 0], 1),
            ([2, 0, 0], [3, 0, 0], 1),
            ([2, 2, 2], [3, 3, 3], np.sqrt(3)),
            ([2, 2, 2], [2, 2, 2], np.sqrt(3)),
            ([-2, 2, 0], [2, -2, 0], 0),
        ]
        for start, end, expected in examples:
            self.assertAlmostEqual(
                segment_box_distance(
                    np.array(start, dtype=float), np.array(end, dtype=float), center, size
                ),
                expected,
            )

    def test_sphere_collision_along_body_not_tip(self):
        obstacle = SphereObstacle("middle", np.array([0.01, 0, 0.13]), 0.004)
        result = self.check(obstacle)
        self.assertEqual(result.status, CollisionStatus.COLLISION)
        self.assertAlmostEqual(result.minimum_clearance_m, -0.002)
        self.assertGreater(np.linalg.norm(self.forward.tip_position_m - obstacle.center_m), 0.2)

    def test_box_collision_and_clearance(self):
        obstacle = BoxObstacle("wall", np.array([0.02, 0, 0.20]), np.array([0.02, 0.03, 0.02]))
        clear = self.check(obstacle)
        self.assertEqual(clear.status, CollisionStatus.CLEAR)
        self.assertAlmostEqual(clear.minimum_clearance_m, 0.002)
        self.assertEqual(
            self.check(obstacle, 0.003).status, CollisionStatus.SAFETY_MARGIN_VIOLATION
        )
        hit = replace(obstacle, center_m=np.array([0.015, 0, 0.20]))
        self.assertEqual(self.check(hit).status, CollisionStatus.COLLISION)

    def test_tangent_contact_and_no_obstacles(self):
        touching = SphereObstacle("tangent", np.array([0.012, 0, 0.13]), 0.004)
        self.assertFalse(self.check(touching).collision_free)
        result = check_static_collision(self.robot, self.forward, self.environment)
        self.assertTrue(result.collision_free)
        self.assertIsNone(result.minimum_clearance_m)

    def test_sagitta_inflation_catches_contact_between_coarse_samples(self):
        section = replace(self.robot.sections[0], length_m=0.2, radius_m=0.001)
        robot = replace(self.robot, sections=(section,), base_position_m=np.zeros(3))
        forward = forward_kinematics(robot, np.array([[np.pi / 2, 0.0]]), 1.0)
        radius = section.length_m / (np.pi / 2)
        arc_midpoint = np.array([radius * (1 - np.cos(np.pi / 4)), 0, radius * np.sin(np.pi / 4)])
        obstacle = SphereObstacle("on_arc", arc_midpoint, 0.0001)
        # An uninflated chord would miss the obstacle.
        chord = forward.section_curves[0].points_m
        self.assertGreater(segment_point_distance(chord[0], chord[1], arc_midpoint), 0.01)
        result = check_static_collision(
            robot, forward, replace(self.environment, obstacles=(obstacle,))
        )
        self.assertFalse(result.collision_free)

    def test_box_distance_against_dense_independent_sampling(self):
        rng = np.random.default_rng(23)
        for _ in range(20):
            start, end = rng.uniform(-2, 2, (2, 3))
            center, size = rng.uniform(-0.5, 0.5, 3), rng.uniform(0.1, 1, 3)
            points = start + np.linspace(0, 1, 10001)[:, None] * (end - start)
            low, high = center - size / 2, center + size / 2
            sampled = np.linalg.norm(points - np.clip(points, low, high), axis=1).min()
            exact = segment_box_distance(start, end, center, size)
            self.assertLessEqual(exact, sampled + 1e-12)
            assert_allclose(exact, sampled, atol=np.linalg.norm(end - start) / 10000)


if __name__ == "__main__":
    unittest.main()
