"""Bounded optional reshaping at one fixed waypoint, reusing the same SQP."""

from dataclasses import replace

import numpy as np

from .ftl_optimizer import solve_ftl_step
from .models import FloatArray, FTLConfig, FTLReshapeAttempt, FTLStep, ProjectConfig, ReferencePath


def reshape_at_waypoint(
    config: ProjectConfig,
    reference: ReferencePath,
    previous: FTLStep,
    planned_polyline_m: FloatArray,
) -> tuple[tuple[FTLStep, ...], tuple[FTLReshapeAttempt, ...]]:
    """Try up to max_reshape_attempts at previous desired s, without advancing it.

    Trigger on native fine-grid body error > body_tolerance_m. Freeze the body's
    comparison curve for this episode so a new reference cannot fake improvement.
    Keep the original weighted objective, geometry, release and increment limits.
    Feed can be zero/negative/positive; it is not fixed. Accept only solver-valid
    trials with tip error <= tip_tolerance_m and fine body improvement >= the
    configured minimum. Stop on first rejected trial or when body tolerance is met.
    Rejected trials never enter accepted history or physical replay. Their failure
    is recorded; normal waypoint advancement then resumes from the retained state.
    Returned accepted steps can be inserted directly into the actual-state replay.
    """
    settings = config.ftl or FTLConfig()
    if (
        not previous.accepted
        or settings.max_reshape_attempts == 0
        or previous.body_following_error_max_m is None
        or previous.body_following_error_max_m <= settings.body_tolerance_m
    ):
        return (), ()
    accepted, attempts = [], []
    comparison = previous.comparison_path_m.copy()
    current = previous
    for index in range(settings.max_reshape_attempts):
        # Already at this waypoint: start with no physical motion, not the last
        # forward increment. This changes the warm start only for reshape trials.
        seed = replace(
            current,
            feed_increment_m=0.0,
            configuration_increment_rad=np.zeros_like(current.configuration_increment_rad),
        )
        trial = solve_ftl_step(
            config,
            reference,
            seed,
            previous.desired_reference_progress_m,
            comparison,
            planned_polyline_m,
            hold_waypoint=True,
        )
        before = current.body_following_error_max_m
        outcome = trial.failure_reason
        if trial.accepted:
            if trial.tip_error_m > settings.tip_tolerance_m:
                outcome = "RESHAPE_TIP_TOLERANCE_EXCEEDED"
            elif before - trial.body_following_error_max_m < settings.reshape_min_improvement_m:
                outcome = "RESHAPE_NO_BODY_IMPROVEMENT"
            else:
                outcome = "RESHAPE_ACCEPTED"
        if outcome != "RESHAPE_ACCEPTED":
            trial = replace(
                trial,
                accepted=False,
                failure_reason=outcome,
                historical_tip_path_m=current.historical_tip_path_m.copy(),
            )
        attempts.append(FTLReshapeAttempt(index + 1, before, trial, outcome))
        if not trial.accepted:
            break
        accepted.append(trial)
        current = trial
        if current.body_following_error_max_m <= settings.body_tolerance_m:
            break
    return tuple(accepted), tuple(attempts)
