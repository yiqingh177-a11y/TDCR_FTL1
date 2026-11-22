"""Fixed-waypoint attempts, quality gates, history ownership and replay states."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from continuum_ftl.config import load_config
from continuum_ftl.ftl_demo import load_ftl_case
from continuum_ftl.ftl_optimizer import solve_ftl_step
from continuum_ftl.ftl_reshape import reshape_at_waypoint
from continuum_ftl.ftl_rollout import ftl_report, run_single_path_ftl
from continuum_ftl.models import FTLConfig
from continuum_ftl.reference_path import interpolate_reference_path

ROOT = Path(__file__).resolve().parents[1]


class ReshapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg, cls.reference = load_ftl_case(
            load_config(ROOT / "configs/robot_2d.json"),
            "straight",
            ROOT / "configs/ftl_validation.json",
            "unused",
        )
        cls.config = replace(
            cfg,
            ftl=replace(cfg.ftl, max_steps=1, max_reshape_attempts=0, enforce_error_limits=False),
        )
        cls.original = run_single_path_ftl(cls.config, cls.reference)
        cls.previous = replace(cls.original.steps[-1], body_following_error_max_m=0.006)
        cls.config = replace(cls.config, ftl=replace(cls.config.ftl, max_reshape_attempts=3))
        cls.planned = interpolate_reference_path(cls.reference, np.linspace(0, 0.18, 721))

    def reshape(self, previous=None, config=None):
        return reshape_at_waypoint(
            config or self.config, self.reference, previous or self.previous, self.planned
        )

    def trial(self, seed, body, **changes):
        return replace(
            seed,
            step_index=seed.step_index + 1,
            phase="reshape",
            body_following_error_max_m=body,
            accepted=True,
            historical_tip_path_m=np.vstack(
                (seed.historical_tip_path_m, seed.state.forward.tip_position_m)
            ),
            **changes,
        )

    def test_disabled_and_below_threshold_do_not_call_solver(self):
        with patch("continuum_ftl.ftl_reshape.solve_ftl_step", side_effect=AssertionError("SQP")):
            self.assertEqual(
                self.reshape(
                    config=replace(
                        self.config, ftl=replace(self.config.ftl, max_reshape_attempts=0)
                    )
                ),
                ((), ()),
            )
            self.assertEqual(
                self.reshape(previous=replace(self.previous, body_following_error_max_m=0.005)),
                ((), ()),
            )
        self.assertEqual(FTLConfig().max_reshape_attempts, 0)

    def test_fixed_reference_zero_warm_start_and_finite_budget(self):
        def improve(config, reference, seed, progress, comparison, planned, **kwargs):
            self.assertEqual(progress, self.previous.desired_reference_progress_m)
            assert_array_equal(comparison, self.previous.comparison_path_m)
            self.assertTrue(kwargs["hold_waypoint"])
            self.assertEqual(seed.feed_increment_m, 0)
            assert_array_equal(seed.configuration_increment_rad, np.zeros(3))
            return self.trial(seed, seed.body_following_error_max_m - 0.00002)

        with patch("continuum_ftl.ftl_reshape.solve_ftl_step", side_effect=improve):
            accepted, attempts = self.reshape()
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual([a.attempt_index for a in attempts], [1, 2, 3])
        self.assertTrue(all(a.outcome == "RESHAPE_ACCEPTED" for a in attempts))
        self.assertEqual(
            len(accepted[-1].historical_tip_path_m), len(self.previous.historical_tip_path_m) + 3
        )

    def test_reaching_body_tolerance_ends_episode_early(self):
        def improve(config, reference, seed, *args, **kwargs):
            return self.trial(seed, 0.0049)

        with patch("continuum_ftl.ftl_reshape.solve_ftl_step", side_effect=improve):
            accepted, attempts = self.reshape()
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(attempts), 1)

    def test_no_improvement_tip_drift_and_solver_failure_are_not_executed(self):
        cases = (
            ({"body": 0.005999}, "RESHAPE_NO_BODY_IMPROVEMENT"),
            ({"body": 0.005, "tip_error_m": 0.001}, "RESHAPE_TIP_TOLERANCE_EXCEEDED"),
            (
                {
                    "body": 0.005,
                    "accepted": False,
                    "solver_success": False,
                    "failure_reason": "SOLVER_FAILED: status=9",
                },
                "SOLVER_FAILED: status=9",
            ),
        )
        for values, reason in cases:

            def rejected(config, reference, seed, *args, values=values, **kwargs):
                values = dict(values)
                body = values.pop("body")
                return replace(self.trial(seed, body), **values)

            with (
                self.subTest(reason=reason),
                patch("continuum_ftl.ftl_reshape.solve_ftl_step", side_effect=rejected),
            ):
                accepted, attempts = self.reshape()
            self.assertEqual(accepted, ())
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0].outcome, reason)
            self.assertFalse(attempts[0].trial.accepted)
            assert_array_equal(
                attempts[0].trial.historical_tip_path_m, self.previous.historical_tip_path_m
            )

    def test_only_hold_calls_can_stall_in_actual_tip_mode(self):
        config = replace(
            self.config, ftl=replace(self.config.ftl, reference_progress_mode="actual_tip")
        )
        previous = self.original.steps[-1]

        def stationary(fun, x0, **kwargs):
            return SimpleNamespace(
                x=np.zeros_like(x0), success=True, status=0, message="test", nit=1
            )

        args = (
            config,
            self.reference,
            previous,
            previous.desired_reference_progress_m,
            previous.comparison_path_m,
            self.planned,
        )
        with patch("continuum_ftl.ftl_optimizer.minimize", side_effect=stationary):
            normal = solve_ftl_step(*args)
            hold = solve_ftl_step(*args, hold_waypoint=True)
        self.assertEqual(normal.failure_reason, "NO_FORWARD_REFERENCE_PROGRESS")
        self.assertTrue(hold.accepted, hold.failure_reason)
        self.assertEqual(hold.phase, "reshape")
        assert_array_equal(hold.state.forward.centerline_m, previous.state.forward.centerline_m)

    def test_rollout_inserts_only_accepted_reshape_then_advances_from_it(self):
        config = replace(self.config, ftl=replace(self.config.ftl, max_steps=2))
        seen_progress = []

        def forward(config, reference, previous, progress, *args):
            seen_progress.append(progress)
            return replace(
                self.previous,
                step_index=previous.step_index + 1,
                desired_reference_progress_m=progress,
            )

        def improve(config, reference, seed, *args, **kwargs):
            return self.trial(seed, 0.0049)

        with (
            patch("continuum_ftl.ftl_rollout.solve_ftl_step", side_effect=forward),
            patch("continuum_ftl.ftl_reshape.solve_ftl_step", side_effect=improve),
        ):
            result = run_single_path_ftl(config, self.reference)
        motion = [s for s in result.steps if s.phase != "approach"]
        self.assertEqual([s.phase for s in motion], ["FTL", "reshape", "FTL", "reshape"])
        np.testing.assert_allclose(seen_progress, [0.001, 0.002])
        self.assertEqual(
            motion[0].desired_reference_progress_m, motion[1].desired_reference_progress_m
        )
        report = ftl_report(result, self.reference)
        self.assertEqual(report["accepted_reshape_steps"], 2)
        self.assertEqual(report["attempted_ftl_steps"], 2)
        self.assertEqual(len(report["reshape_attempts"]), 2)
        json.dumps(report, allow_nan=False)

        import matplotlib

        matplotlib.use("Agg")
        from continuum_ftl.deployment_viewer import DeploymentViewer

        viewer = DeploymentViewer(
            config,
            tuple(s.state for s in result.steps),
            ftl_steps=result.steps,
            reference_path=self.reference,
            exposures=tuple(s.exposure for s in result.steps),
        )
        try:
            viewer.draw_frame(len(result.steps) - 1)
            self.assertIn("Phase=reshape", viewer.status.get_text())
            assert_array_equal(
                np.array(viewer.tip.get_data()).ravel(),
                result.steps[-1].state.forward.tip_position_m,
            )
        finally:
            viewer.close()

    def test_invalid_reshape_settings_are_rejected(self):
        data = json.loads((ROOT / "configs/robot_2d.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for name, value in (
                ("max_reshape_attempts", -1),
                ("max_reshape_attempts", True),
                ("max_reshape_attempts", 1.5),
                ("reshape_min_improvement_m", 0),
            ):
                candidate = json.loads(json.dumps(data))
                candidate["ftl"][name] = value
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
