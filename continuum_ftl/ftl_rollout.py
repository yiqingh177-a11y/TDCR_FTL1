"""Single fixed-path FTL sequencing, history ownership and serializable diagnostics."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .deployment import compute_deployment, release_thresholds
from .environment_exposure import exposure_report, extract_environment_exposure
from .ftl_matching import comparison_path, previous_body_comparison_path
from .ftl_optimizer import solve_ftl_step
from .ftl_reshape import reshape_at_waypoint
from .models import FTLConfig, FTLResult, FTLStep, PlanarRobotConfig, ProjectConfig, ReferencePath
from .reference_path import interpolate_reference_path, reference_path_report


def run_single_path_ftl(config: ProjectConfig, reference: ReferencePath) -> FTLResult:
    """Approach straight to entry, then optimize increments until target/failure.

    Does not invoke Stage 1 IK/search or any collision checker. No waypoint is
    skipped, no failed trial enters history, and no reference-aligned substitute
    geometry is generated. Maximum feed is physical total length, not the old
    manual-demo end_distance_m. Physical q vectors use C/B/A order.
    """
    if not isinstance(config.robot, PlanarRobotConfig) or config.deployment is None:
        raise ValueError("Stage 2C supports 2D with deployment settings only")
    # Keep visible scene data with the caller; numerical FTL receives no obstacles.
    config = replace(config, environment=replace(config.environment, obstacles=()))
    started = perf_counter()
    settings, ftl = config.deployment, config.ftl or FTLConfig()
    approach_distance = settings.environment_entry_z_m - settings.feed_exit_m[1]
    entry = np.array([settings.feed_exit_m[0], settings.environment_entry_z_m])
    if np.linalg.norm(entry - reference.points_m[0]) > settings.geometry_tolerance_m:
        raise ValueError("Reference entry does not coincide with straight +Z feed approach")
    total = settings.passive_length_m + sum(s.length_m for s in config.robot.sections)
    if approach_distance > total:
        raise ValueError("Straight robot cannot reach environment entry within its physical length")
    distances = np.linspace(
        0, approach_distance, max(1, int(np.ceil(approach_distance / ftl.max_feed_increment_m))) + 1
    )
    thresholds = release_thresholds(config.robot)
    distances = np.unique(
        np.r_[distances, thresholds[(thresholds > 0) & (thresholds < approach_distance)]]
    )
    steps = []
    for index, distance in enumerate(distances):
        frame_started = perf_counter()
        state = compute_deployment(
            config.robot,
            settings,
            float(distance),
            np.zeros(config.robot.section_count),
            config.sampling.max_arc_spacing_m,
        )
        exposure = extract_environment_exposure(
            state,
            settings.environment_entry_z_m,
            max_arc_spacing_m=ftl.validation_body_spacing_m,
            tolerance_m=settings.geometry_tolerance_m,
        )
        history = (
            state.forward.tip_position_m[None, :].copy()
            if index == len(distances) - 1
            else np.empty((0, 2))
        )
        steps.append(
            FTLStep(
                index,
                "approach",
                state,
                exposure,
                float(distance - distances[index - 1]) if index else 0.0,
                np.zeros(config.robot.section_count),
                (),
                0.0,
                0.0,
                state.forward.tip_position_m.copy(),
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                None,
                None,
                None,
                "Straight feed approach; no optimizer",
                0,
                perf_counter() - frame_started,
                True,
                None,
                history,
                np.empty((0, 2)),
            )
        )
    reference_arcs = np.linspace(
        0,
        reference.arc_lengths_m[-1],
        max(1, int(np.ceil(reference.arc_lengths_m[-1] / ftl.comparison_spacing_m))) + 1,
    )
    planned = interpolate_reference_path(reference, reference_arcs)
    reshape_attempts = []
    for _ in range(ftl.max_steps):
        previous = steps[-1]
        # Observation mode advances the desired sequence even if actual tip stalls.
        # Actual projected progress remains measured and is reported separately.
        previous_progress = (
            previous.reference_progress_m
            if ftl.reference_progress_mode == "actual_tip"
            else previous.desired_reference_progress_m
        )
        progress = min(reference.arc_lengths_m[-1], previous_progress + ftl.reference_step_m)
        target = interpolate_reference_path(reference, np.array([progress]))[0]
        if ftl.body_reference_mode == "tip_history":
            comparison = comparison_path(
                previous.historical_tip_path_m, target, ftl.comparison_spacing_m
            )
        else:
            comparison = previous_body_comparison_path(
                previous.state, previous.exposure, target, ftl.comparison_spacing_m
            )
        step = solve_ftl_step(config, reference, previous, float(progress), comparison, planned)
        steps.append(step)
        if not step.accepted:
            return FTLResult(
                False,
                tuple(steps),
                step.failure_reason,
                perf_counter() - started,
                error_limits_enforced=ftl.enforce_error_limits,
                reference_progress_mode=ftl.reference_progress_mode,
                body_reference_mode=ftl.body_reference_mode,
                effective_ftl_settings=ftl,
                reshape_attempts=tuple(reshape_attempts),
            )
        if ftl.max_reshape_attempts:
            reshaped, attempts = reshape_at_waypoint(config, reference, step, planned)
            steps.extend(reshaped)
            reshape_attempts.extend(attempts)
            step = steps[-1]
        if progress >= reference.arc_lengths_m[-1] - settings.geometry_tolerance_m:
            error = float(
                np.linalg.norm(step.state.forward.tip_position_m - config.target.position_m)
            )
            reached = error <= config.target.tolerance_m
            if reached or not ftl.enforce_error_limits:
                return FTLResult(
                    reached,
                    tuple(steps),
                    None,
                    perf_counter() - started,
                    reference_completed=True,
                    error_limits_enforced=ftl.enforce_error_limits,
                    reference_progress_mode=ftl.reference_progress_mode,
                    body_reference_mode=ftl.body_reference_mode,
                    effective_ftl_settings=ftl,
                    reshape_attempts=tuple(reshape_attempts),
                )
            return FTLResult(
                False,
                tuple(steps),
                "FINAL_TARGET_TOLERANCE_EXCEEDED",
                perf_counter() - started,
                reference_completed=True,
                error_limits_enforced=ftl.enforce_error_limits,
                reference_progress_mode=ftl.reference_progress_mode,
                body_reference_mode=ftl.body_reference_mode,
                effective_ftl_settings=ftl,
                reshape_attempts=tuple(reshape_attempts),
            )
    return FTLResult(
        False,
        tuple(steps),
        "FTL_STEP_BUDGET_EXHAUSTED",
        perf_counter() - started,
        error_limits_enforced=ftl.enforce_error_limits,
        reference_progress_mode=ftl.reference_progress_mode,
        body_reference_mode=ftl.body_reference_mode,
        effective_ftl_settings=ftl,
        reshape_attempts=tuple(reshape_attempts),
    )


def ftl_summary(result: FTLResult, reference: ReferencePath) -> dict:
    """Quantitative outcome; failed trial and last accepted state are distinguished."""
    accepted = [step for step in result.steps if step.accepted]
    ftl_steps = [step for step in result.steps if step.phase == "FTL"]
    motion_steps = [step for step in result.steps if step.phase in ("FTL", "reshape")]
    last = accepted[-1]
    errors = [
        step.body_following_error_max_m
        for step in motion_steps
        if step.body_following_error_max_m is not None
    ]
    return {
        "stage": "2C: Palmer-inspired single-path incremental SQP FTL",
        "success": result.success,
        "target_reached": result.success,
        "reference_completed": result.reference_completed,
        "error_limits_enforced": result.error_limits_enforced,
        "reference_progress_mode": result.reference_progress_mode,
        "body_reference_mode": result.body_reference_mode,
        "native_body_error_reference": result.body_reference_mode + " plus intended tip extension",
        "effective_ftl_settings": asdict(result.effective_ftl_settings)
        if result.effective_ftl_settings is not None
        else None,
        "stalled_tip_steps": sum(step.reference_progress_stalled for step in ftl_steps),
        "failure_reason": result.failure_reason,
        "frame_count": len(result.steps),
        "attempted_ftl_steps": len(ftl_steps),
        "accepted_ftl_steps": sum(step.accepted for step in ftl_steps),
        "attempted_reshape_steps": len(result.reshape_attempts),
        "accepted_reshape_steps": sum(a.trial.accepted for a in result.reshape_attempts),
        "computation_time_s": result.computation_time_s,
        "last_accepted_target_error_m": float(
            np.linalg.norm(last.state.forward.tip_position_m - reference.target_position_m)
        ),
        "last_accepted_waypoint_error_m": last.tip_error_m,
        "last_trial_tip_error_m": result.steps[-1].tip_error_m,
        "maximum_body_error_m": max(errors, default=0.0),
        "maximum_accepted_body_error_m": max(
            (step.body_following_error_max_m for step in motion_steps if step.accepted), default=0.0
        ),
        "final_reference_progress_m": last.reference_progress_m,
        "reference_length_m": float(reference.arc_lengths_m[-1]),
        "final_deployment_distance_m": last.state.distance_m,
        "collision_checked": False,
        "obstacles_used_for_optimization": False,
        "continuous_motion_checked": False,
    }


def ftl_report(result: FTLResult, reference: ReferencePath) -> dict:
    """Full replay data, per-step metrics, history and rejected trial, SI units."""
    return {
        **ftl_summary(result, reference),
        "reference_path": reference_path_report(reference),
        "configuration_order": [*result.steps[0].state.section_names[1:]],
        "steps": [_step_report(step) for step in result.steps],
        "reshape_attempts": [
            {
                "attempt_index": attempt.attempt_index,
                "body_error_before_m": attempt.body_error_before_m,
                "body_improvement_m": (
                    attempt.body_error_before_m - attempt.trial.body_following_error_max_m
                    if attempt.trial.body_following_error_max_m is not None
                    else None
                ),
                "outcome": attempt.outcome,
                "trial": _step_report(attempt.trial),
            }
            for attempt in result.reshape_attempts
        ],
    }


def _step_report(step: FTLStep) -> dict:
    """Serialize an executed frame or a separately retained reshape trial."""
    return {
        "step_index": step.step_index,
        "phase": step.phase,
        "deployment_distance_m": step.state.distance_m,
        "feed_increment_m": step.feed_increment_m,
        "configuration_rad": step.state.applied_bends_rad.tolist(),
        "configuration_increment_rad": step.configuration_increment_rad.tolist(),
        "optimized_section_indices": list(step.optimized_section_indices),
        "section_release_status": step.state.released.tolist(),
        "environment_exposed_length_m": step.exposure.exposed_length_m,
        "reference_progress_m": step.reference_progress_m,
        "desired_reference_progress_m": step.desired_reference_progress_m,
        "reference_progress_stalled": step.reference_progress_stalled,
        "desired_tip_position_m": step.desired_tip_position_m.tolist(),
        "actual_tip_position_m": step.state.forward.tip_position_m.tolist(),
        "tip_error_m": step.tip_error_m,
        "body_following_error_max_m": step.body_following_error_max_m,
        "body_following_error_mean_m": step.body_following_error_mean_m,
        "planned_path_deviation_max_m": step.planned_path_deviation_max_m,
        "worst_body_point_m": step.worst_body_point_m.tolist()
        if step.worst_body_point_m is not None
        else None,
        "worst_body_material_arc_m": step.worst_body_material_arc_m,
        "solver_success": step.solver_success,
        "solver_status": step.solver_status,
        "solver_message": step.solver_message,
        "solver_iterations": step.solver_iterations,
        "computation_time_s": step.computation_time_s,
        "accepted": step.accepted,
        "failure_reason": step.failure_reason,
        "actual_centerline_m": step.state.forward.centerline_m.tolist(),
        "section_endpoints_m": step.state.forward.section_endpoints_m.tolist(),
        "historical_tip_path_m": step.historical_tip_path_m.tolist(),
        "comparison_path_m": step.comparison_path_m.tolist(),
        "exposure": exposure_report(step.exposure),
    }
