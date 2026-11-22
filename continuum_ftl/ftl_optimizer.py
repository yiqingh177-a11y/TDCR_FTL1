"""Incremental feed/bending SQP on actual Stage 2A/2B geometry; no IK or collisions."""

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .deployment import compute_deployment, release_thresholds
from .environment_exposure import extract_environment_exposure
from .ftl_matching import match_ordered_polyline, palmer_objective, polyline_arcs
from .models import (
    ExposureStatus,
    FloatArray,
    FTLConfig,
    FTLStep,
    FTLTrial,
    PlanarRobotConfig,
    ProjectConfig,
    ReferencePath,
)
from .reference_path import interpolate_reference_path


def evaluate_ftl_trial(
    config: ProjectConfig,
    distance_m: float,
    bends_rad: FloatArray,
    desired_tip_m: FloatArray,
    comparison_m: FloatArray,
    body_spacing_m: float,
) -> FTLTrial:
    """Evaluate real full physical PCC shape and environment-only active-body error.

    The passive section stays in the deployment state/exposure display but is not
    a body-following sample. Distances use local ordered polyline projections.
    Geometry failures remain explicit; no synthetic path-aligned shape is made.
    """
    if not isinstance(config.robot, PlanarRobotConfig) or config.deployment is None:
        raise ValueError("FTL trial requires 2D robot and deployment settings")
    settings, ftl = config.deployment, config.ftl or FTLConfig()
    state = compute_deployment(config.robot, settings, distance_m, bends_rad, body_spacing_m)
    exposure = extract_environment_exposure(
        state,
        settings.environment_entry_z_m,
        max_arc_spacing_m=body_spacing_m,
        tolerance_m=settings.geometry_tolerance_m,
    )
    tip_error = float(np.linalg.norm(state.forward.tip_position_m - desired_tip_m))
    failure = None
    if state.consistency_warnings:
        failure = "INCONSISTENT_DEPLOYMENT: " + "; ".join(state.consistency_warnings)
    elif exposure.status != ExposureStatus.EXPOSED:
        failure = f"INVALID_EXPOSURE: {exposure.status.value}: {exposure.diagnostic}"
    active = exposure.section_indices > 0
    body_points = exposure.centerline_m[active]
    material_arcs = exposure.material_arc_lengths_m[active]
    match = None
    if failure is None:
        expected = polyline_arcs(comparison_m)[-1] - (state.section_lengths_m.sum() - material_arcs)
        match = match_ordered_polyline(body_points, comparison_m, expected, ftl.matching_window_m)
    return FTLTrial(state, exposure, tip_error, body_points, material_arcs, match, failure)


def _body_max(trial: FTLTrial) -> float:
    return float(np.max(trial.body_match.distances_m, initial=0.0))


def solve_ftl_step(
    config: ProjectConfig,
    reference: ReferencePath,
    previous: FTLStep,
    desired_progress_m: float,
    comparison_m: FloatArray,
    planned_polyline_m: FloatArray,
    *,
    hold_waypoint: bool = False,
) -> FTLStep:
    """One warm-started SQP step over bounded dimensionless incremental variables.

    Physical increments are [Delta d, Delta q_C, Delta q_B, Delta q_A]. Only
    released or potentially released sections are variables. Feed can retract,
    bounded by min_feed_increment_m and the physical d >= 0 limit. Release and
    re-entry ramps require zero bend at/below a section's full-release threshold.
    Independent fine-grid acceptance checks are rerun after SLSQP terminates.
    hold_waypoint labels a reshape trial and permits zero reference progress;
    it changes neither the objective nor physical motion/geometry constraints.
    """
    if not isinstance(config.robot, PlanarRobotConfig) or config.deployment is None:
        raise ValueError("Stage 2C requires 2D robot and Stage 2A deployment settings")
    started = perf_counter()
    settings = config.ftl or FTLConfig()
    old = previous.state
    target = interpolate_reference_path(reference, np.array([desired_progress_m]))[0]
    total = float(old.section_lengths_m.sum())
    min_feed = max(settings.min_feed_increment_m, -old.distance_m)
    max_feed = min(settings.max_feed_increment_m, max(0.0, total - old.distance_m))
    thresholds = release_thresholds(config.robot)
    eligible = np.flatnonzero(old.released | (old.distance_m + max_feed > thresholds))
    scales = np.r_[
        max(-settings.min_feed_increment_m, settings.max_feed_increment_m),
        np.full(len(eligible), settings.max_bend_increment_rad),
    ]
    limits = np.array([section.max_bend_rad for section in config.robot.sections])
    lower_q = np.maximum(
        -settings.max_bend_increment_rad, -limits[eligible] - old.applied_bends_rad[eligible]
    )
    upper_q = np.minimum(
        settings.max_bend_increment_rad, limits[eligible] - old.applied_bends_rad[eligible]
    )
    bounds = [
        (min_feed / scales[0], max_feed / scales[0]),
        *zip(lower_q / scales[1:], upper_q / scales[1:], strict=True),
    ]

    def unpack(control: FloatArray) -> tuple[float, FloatArray]:
        increments = control * scales
        distance = float(np.clip(old.distance_m + increments[0], 0, total))
        bends = old.applied_bends_rad.copy()
        bends[eligible] += increments[1:]
        return distance, np.clip(bends, -limits, limits)

    last_control, last_trial = None, None

    def trial(control: FloatArray) -> FTLTrial:
        nonlocal last_control, last_trial
        if last_control is None or not np.array_equal(control, last_control):
            distance, bends = unpack(control)
            last_trial = evaluate_ftl_trial(
                config, distance, bends, target, comparison_m, settings.optimization_body_spacing_m
            )
            last_control = control.copy()
        return last_trial

    def objective(control: FloatArray) -> float:
        evaluated = trial(control)
        body_error = (
            _body_max(evaluated)
            if evaluated.geometry_failure is None
            else settings.invalid_geometry_penalty_m
        )
        return (
            palmer_objective(evaluated.tip_error_m, body_error, settings.tip_weight)
            / settings.objective_scale_m
        )

    ramp_ends = thresholds[eligible] + settings.max_feed_increment_m
    old_ramp_fractions = np.clip(
        (old.distance_m - thresholds[eligible]) / settings.max_feed_increment_m, 0, 1
    )
    # A retracted mature bend can exceed the small introduction envelope. Keep
    # treating it as re-entry on later steps, without adding hidden mode state.
    entering = (old.distance_m < ramp_ends) & (
        np.abs(old.applied_bends_rad[eligible])
        <= settings.max_bend_increment_rad * old_ramp_fractions + settings.increment_tolerance
    )
    reentering = ~entering & (old.distance_m + min_feed < ramp_ends)
    ramp_mask = entering | reentering
    ramp_indices = eligible[ramp_mask]
    ramp_amplitudes = np.where(
        entering[ramp_mask],
        settings.max_bend_increment_rad,
        np.abs(old.applied_bends_rad[ramp_indices]) + settings.max_bend_increment_rad,
    )
    ramp_widths = np.where(
        entering[ramp_mask],
        settings.max_feed_increment_m,
        np.minimum(settings.max_feed_increment_m, old.distance_m - thresholds[ramp_indices]),
    )

    def release_constraints(control: FloatArray) -> FloatArray:
        distance, bends = unpack(control)
        # Preserve the original introduction ramp. For a mature section returning
        # to the release band, taper from its reachable angular bound to zero;
        # do not impose the small introduction cap on its existing large bend.
        allowed = ramp_amplitudes * np.clip(
            (distance - thresholds[ramp_indices]) / ramp_widths, 0, 1
        )
        # Dimensionless, with one inequality for each sign; straight at release.
        return (
            np.r_[allowed - bends[ramp_indices], allowed + bends[ramp_indices]]
            / settings.max_bend_increment_rad
        )

    feed_guess = (
        previous.feed_increment_m
        if previous.feed_increment_m < 0
        else min(
            max_feed,
            max(previous.feed_increment_m, np.linalg.norm(target - old.forward.tip_position_m)),
        )
    )
    guess = np.r_[
        feed_guess / scales[0],
        previous.configuration_increment_rad[eligible] / scales[1:],
    ]
    guess = np.clip(guess, [b[0] for b in bounds], [b[1] for b in bounds])
    result = minimize(
        objective,
        guess,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": release_constraints}] if len(ramp_indices) else (),
        options={
            "maxiter": settings.max_iterations,
            "ftol": settings.solver_tolerance,
            "eps": settings.finite_difference_step,
        },
    )
    finite_control = bool(np.isfinite(result.x).all())
    distance, bends = unpack(result.x if finite_control else np.zeros_like(guess))
    # Recompute using a finer independent body grid; never trust cached objective/flags.
    checked = evaluate_ftl_trial(
        config, distance, bends, target, comparison_m, settings.validation_body_spacing_m
    )
    state = checked.state
    delta_feed = distance - old.distance_m
    delta_q = state.applied_bends_rad - old.applied_bends_rad
    failure = checked.geometry_failure
    tolerance = settings.increment_tolerance
    if not finite_control:
        failure = "NONFINITE_SOLVER_RESULT: no finite failed shape was available"
    elif np.any(np.abs(bends - state.applied_bends_rad) > tolerance):
        failure = "UNRELEASED_BENDING_REQUEST"
    elif (
        not (min_feed - tolerance <= result.x[0] * scales[0] <= max_feed + tolerance)
        or not (min_feed - tolerance <= delta_feed <= max_feed + tolerance)
    ) or np.any(np.abs(delta_q) > settings.max_bend_increment_rad + tolerance):
        failure = "INCREMENT_LIMIT_VIOLATION"
    elif len(ramp_indices) and np.min(release_constraints(result.x)) < -tolerance:
        failure = "RELEASE_CONTINUITY_VIOLATION"
    elif failure is None and not result.success:
        failure = f"SOLVER_FAILED: status={result.status}: {result.message}"
    elif (
        failure is None
        and settings.enforce_error_limits
        and checked.tip_error_m > settings.tip_tolerance_m
    ):
        failure = "TIP_TOLERANCE_EXCEEDED"
    elif (
        failure is None
        and settings.enforce_error_limits
        and _body_max(checked) > settings.body_tolerance_m
    ):
        failure = "BODY_TOLERANCE_EXCEEDED"

    # Project actual tip only onto a local forward neighbourhood of planned path.
    planned_arcs = polyline_arcs(planned_polyline_m)
    # Convert between dense polyline chord coordinates and exact reference s.
    reference_arcs = np.linspace(0, reference.arc_lengths_m[-1], len(planned_polyline_m))
    projected = match_ordered_polyline(
        state.forward.tip_position_m[None, :],
        planned_polyline_m,
        np.array([np.interp(desired_progress_m, reference_arcs, planned_arcs)]),
        settings.matching_window_m,
    ).matched_arcs_m[0]
    progress = float(
        np.clip(
            np.interp(projected, planned_arcs, reference_arcs),
            previous.reference_progress_m,
            desired_progress_m,
        )
    )
    terminal = (
        desired_progress_m >= reference.arc_lengths_m[-1] - config.deployment.geometry_tolerance_m
    )
    stalled = (
        not terminal and progress - previous.reference_progress_m < settings.minimum_progress_m
    )
    if (
        failure is None
        and stalled
        and settings.reference_progress_mode == "actual_tip"
        and not hold_waypoint
    ):
        failure = "NO_FORWARD_REFERENCE_PROGRESS"
    planned_deviation = None
    maximum, mean, worst_point, worst_arc = None, None, None, None
    if checked.body_match is not None:
        maximum = _body_max(checked)
        mean = float(np.mean(checked.body_match.distances_m)) if len(checked.body_points_m) else 0.0
        if len(checked.body_points_m):
            worst = int(np.argmax(checked.body_match.distances_m))
            worst_point, worst_arc = (
                checked.body_points_m[worst].copy(),
                float(checked.body_material_arcs_m[worst]),
            )
        expected = np.interp(
            progress - (total - checked.body_material_arcs_m), reference_arcs, planned_arcs
        )
        planned_deviation = float(
            np.max(
                match_ordered_polyline(
                    checked.body_points_m, planned_polyline_m, expected, settings.matching_window_m
                ).distances_m,
                initial=0.0,
            )
        )
    accepted = failure is None
    history = previous.historical_tip_path_m.copy()
    if accepted:
        history = np.vstack((history, state.forward.tip_position_m))
    return FTLStep(
        previous.step_index + 1,
        "reshape" if hold_waypoint else "FTL",
        state,
        checked.exposure,
        delta_feed,
        delta_q,
        tuple(int(i) for i in eligible),
        progress,
        desired_progress_m,
        target,
        checked.tip_error_m,
        maximum,
        mean,
        planned_deviation,
        worst_point,
        worst_arc,
        bool(result.success),
        int(result.status),
        str(result.message),
        int(result.nit),
        perf_counter() - started,
        accepted,
        failure,
        history,
        comparison_m.copy(),
        stalled,
    )
