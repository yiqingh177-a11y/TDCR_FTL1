"""Signed feed increments, physical end stops and continuous section re-entry."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from continuum_ftl.config import load_config
from continuum_ftl.deployment import compute_deployment
from continuum_ftl.environment_exposure import extract_environment_exposure
from continuum_ftl.ftl_demo import load_ftl_case
from continuum_ftl.ftl_matching import comparison_path
from continuum_ftl.ftl_optimizer import solve_ftl_step
from continuum_ftl.ftl_rollout import ftl_report, run_single_path_ftl
from continuum_ftl.reference_path import interpolate_reference_path

ROOT = Path(__file__).resolve().parents[1]


class FeedBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(ROOT / "configs/robot_2d.json")
        cls.config, cls.reference = load_ftl_case(
            config, "straight", ROOT / "configs/ftl_validation.json", "unused"
        )
        cls.config = replace(
            cls.config,
            ftl=replace(
                cls.config.ftl,
                min_feed_increment_m=-0.002,
                max_feed_increment_m=0.002,
                max_bend_increment_rad=0.128,
                max_steps=1,
                enforce_error_limits=False,
                reference_progress_mode="scheduled",
            ),
        )
        cls.planned = interpolate_reference_path(cls.reference, np.linspace(0, 0.18, 721))
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=cls.stationary):
            cls.template = run_single_path_ftl(cls.config, cls.reference).steps[-1]

    @staticmethod
    def stationary(fun, x0, **kwargs):
        return SimpleNamespace(
            x=np.zeros_like(x0), success=True, status=0, message="test stationary", nit=1
        )

    def previous(self, distance, bends=(0.0, 0.0, 0.0)):
        state = compute_deployment(
            self.config.robot, self.config.deployment, distance, np.asarray(bends), 0.001
        )
        exposure = extract_environment_exposure(
            state, 0.0, max_arc_spacing_m=0.001, tolerance_m=1e-9
        )
        return replace(
            self.template,
            state=state,
            exposure=exposure,
            feed_increment_m=-0.001,
            configuration_increment_rad=np.zeros(3),
            reference_progress_m=0.0,
            historical_tip_path_m=np.array([[0.0, 0.0], state.forward.tip_position_m]),
        )

    def solve(self, previous, *, config=None, progress=0.1):
        config = config or self.config
        target = interpolate_reference_path(self.reference, np.array([progress]))[0]
        comparison = comparison_path(np.zeros((1, 2)), target, 0.00025)
        return solve_ftl_step(config, self.reference, previous, progress, comparison, self.planned)

    def prescribed(self, feed, increments=(0.0, 0.0, 0.0), *, check_constraints=True):
        """Inject a physical trial to isolate validation from optimizer behaviour."""

        def solver(fun, x0, **kwargs):
            # Test configuration uses symmetric 2 mm bounds and all sections eligible.
            control = np.r_[feed / 0.002, np.asarray(increments) / 0.128]
            self.assertEqual(len(control), len(x0))
            if check_constraints:
                for constraint in kwargs["constraints"]:
                    self.assertGreaterEqual(np.min(constraint["fun"](control)), -1e-12)
            return SimpleNamespace(
                x=control, success=True, status=0, message="test prescribed trial", nit=1
            )

        return solver

    def test_real_sqp_can_select_negative_feed_without_synthetic_geometry(self):
        previous = self.previous(0.22)
        original = previous.state.forward.centerline_m.copy()
        result = self.solve(previous, progress=0.139)
        self.assertTrue(result.accepted, result.failure_reason)
        self.assertTrue(result.solver_success)
        self.assertLess(result.feed_increment_m, -0.0009)
        self.assertGreaterEqual(result.feed_increment_m, -0.002)
        self.assertLess(result.tip_error_m, 1e-7)
        assert_allclose(
            result.state.proximal_position_m - previous.state.proximal_position_m,
            [0, result.feed_increment_m],
            atol=1e-15,
        )
        assert_array_equal(previous.state.forward.centerline_m, original)
        self.assertEqual(len(result.historical_tip_path_m), len(previous.historical_tip_path_m) + 1)

    def test_asymmetric_bounds_are_normalized_and_zero_restores_forward_only(self):
        for lower, expected in ((-0.005, (-1.0, 0.4)), (0.0, (0.0, 1.0))):
            config = replace(self.config, ftl=replace(self.config.ftl, min_feed_increment_m=lower))

            def inspect(fun, x0, expected=expected, **kwargs):
                assert_allclose(kwargs["bounds"][0], expected)
                self.assertGreaterEqual(x0[0], expected[0])
                self.assertLessEqual(x0[0], expected[1])
                return self.stationary(fun, x0, **kwargs)

            with (
                self.subTest(lower=lower),
                patch("continuum_ftl.ftl_optimizer.minimize", side_effect=inspect),
            ):
                self.solve(self.previous(0.22), config=config)

    def test_symmetric_lower_endpoint_is_accepted_and_excess_is_rejected(self):
        previous = self.previous(0.22)
        for feed, accepted in ((-0.002, True), (-0.0021, False), (0.0021, False)):
            with (
                self.subTest(feed=feed),
                patch("continuum_ftl.ftl_optimizer.minimize", side_effect=self.prescribed(feed)),
            ):
                result = self.solve(previous)
            self.assertEqual(result.accepted, accepted)
            if not accepted:
                self.assertEqual(result.failure_reason, "INCREMENT_LIMIT_VIOLATION")
                assert_array_equal(result.historical_tip_path_m, previous.historical_tip_path_m)

    def test_physical_zero_and_total_length_clip_feed_bounds_before_sqp(self):
        for distance, expected in ((0.0005, (-0.25, 1.0)), (0.38, (-1.0, 0.0))):

            def inspect(fun, x0, expected=expected, **kwargs):
                assert_allclose(kwargs["bounds"][0], expected, atol=1e-14)
                return self.stationary(fun, x0, **kwargs)

            with (
                self.subTest(distance=distance),
                patch("continuum_ftl.ftl_optimizer.minimize", side_effect=inspect),
            ):
                self.solve(self.previous(distance))

    def test_crossing_back_below_release_requires_straight_requested_section(self):
        previous = self.previous(0.1805, (0.01, 0, 0))
        with patch(
            "continuum_ftl.ftl_optimizer.minimize",
            side_effect=self.prescribed(-0.001, (-0.01, 0, 0)),
        ):
            result = self.solve(previous)
        self.assertTrue(result.accepted, result.failure_reason)
        self.assertFalse(result.state.released[0])
        self.assertEqual(result.state.applied_bends_rad[0], 0)
        self.assertAlmostEqual(result.configuration_increment_rad[0], -0.01)
        with patch(
            "continuum_ftl.ftl_optimizer.minimize",
            side_effect=self.prescribed(-0.001, check_constraints=False),
        ):
            invalid = self.solve(previous)
        self.assertFalse(invalid.accepted)
        self.assertEqual(invalid.failure_reason, "UNRELEASED_BENDING_REQUEST")

    def test_mature_bent_section_must_straighten_gradually_when_reentering(self):
        previous = self.previous(0.1825, (0.4, 0, 0))
        # At d=.181, reverse ramp allows (.4+.128)*.5=.264 rad, not .4.
        with patch(
            "continuum_ftl.ftl_optimizer.minimize",
            side_effect=self.prescribed(-0.0015, check_constraints=False),
        ):
            invalid = self.solve(previous)
        self.assertEqual(invalid.failure_reason, "RELEASE_CONTINUITY_VIOLATION")
        with patch(
            "continuum_ftl.ftl_optimizer.minimize",
            side_effect=self.prescribed(-0.001, (-0.1, 0, 0)),
        ):
            valid = self.solve(previous)
        self.assertTrue(valid.accepted, valid.failure_reason)
        self.assertAlmostEqual(valid.state.applied_bends_rad[0], 0.3)

    def test_negative_warm_start_and_settings_survive_report(self):
        def inspect(fun, x0, **kwargs):
            self.assertAlmostEqual(x0[0], -0.5)
            return self.stationary(fun, x0, **kwargs)

        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=inspect):
            self.solve(self.previous(0.22))
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=self.stationary):
            rollout = run_single_path_ftl(self.config, self.reference)
        report = ftl_report(rollout, self.reference)
        self.assertEqual(report["effective_ftl_settings"]["min_feed_increment_m"], -0.002)
        self.assertTrue(
            all(s.feed_increment_m >= 0 for s in rollout.steps if s.phase == "approach")
        )

    def test_multistep_reentry_keeps_reachable_bends_then_relocks_without_jump(self):
        previous = self.previous(0.1825, (0.4, 0, 0))
        for feed, bend_increment in (
            (-0.001, -0.1),
            (-0.0005, -0.1),
            (-0.0005, -0.1),
            (-0.0005, -0.1),
            (-0.0005, 0),
        ):
            with patch(
                "continuum_ftl.ftl_optimizer.minimize",
                side_effect=self.prescribed(feed, (bend_increment, 0, 0)),
            ):
                step = self.solve(previous)
            self.assertTrue(step.accepted, step.failure_reason)
            assert_allclose(step.configuration_increment_rad, [bend_increment, 0, 0], atol=1e-15)
            previous = step
        self.assertFalse(previous.state.released[0])
        self.assertEqual(previous.state.applied_bends_rad[0], 0)

    def test_config_accepts_signed_lower_bound_and_legacy_omission(self):
        data = json.loads((ROOT / "configs/robot_2d.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for lower in (-0.005, 0.0, None):
                if lower is None:
                    data["ftl"].pop("min_feed_increment_m")
                else:
                    data["ftl"]["min_feed_increment_m"] = lower
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(load_config(path).ftl.min_feed_increment_m, lower or 0.0)
            for lower in (0.001, float("nan"), float("inf"), True):
                data["ftl"]["min_feed_increment_m"] = lower
                path.write_text(json.dumps(data), encoding="utf-8")
                with (
                    self.subTest(lower=lower),
                    self.assertRaisesRegex(ValueError, "min_feed_increment_m"),
                ):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
