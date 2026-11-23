"""Genuine spatial incremental SQP: feed plus two local bends per released section."""

from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .ftl_matching import match_ordered_polyline, palmer_objective, polyline_arcs
from .models import (
    ExposureStatus,
    FloatArray,
    FTLConfig,
    FTLStep,
    FTLTrial,
    ProjectConfig,
    ReferencePath,
    RobotConfig,
)
from .reference_path import interpolate_reference_path
from .spatial_deployment import compute_spatial_deployment
from .spatial_geometry import extract_spatial_exposure


def evaluate_spatial_trial(
    config: ProjectConfig,
    distance_m: float,
    bends_rad: FloatArray,
    target_m: FloatArray,
    comparison_m: FloatArray,
    spacing_m: float,
) -> FTLTrial:
    """Full spatial FK/exposure and Euclidean XYZ errors; no plane projection.

    SQP may evaluate infeasible angular trial states. Temporary FK limits are
    relaxed for those evaluations only; final acceptance checks original bounds.
    Passive geometry stays in FK but not in the environment-only active body metric.
    """
    settings = config.ftl or FTLConfig()
    robot = replace(
        config.robot,
        sections=tuple(
            replace(s, max_bend_rad=max(s.max_bend_rad, float(np.linalg.norm(b))))
            for s, b in zip(config.robot.sections, bends_rad, strict=True)
        ),
    )
    state = compute_spatial_deployment(robot, config.deployment, distance_m, bends_rad, spacing_m)
    exposure = extract_spatial_exposure(
        state.geometry,
        config.deployment.environment_entry_z_m,
        spacing_m,
        config.deployment.geometry_tolerance_m,
    )
    error = float(np.linalg.norm(state.forward.tip_position_m - target_m))
    failure = None
    if state.consistency_warnings:
        failure = "INCONSISTENT_DEPLOYMENT: " + "; ".join(state.consistency_warnings)
    elif exposure.status != ExposureStatus.EXPOSED:
        failure = "INVALID_EXPOSURE: " + exposure.diagnostic
    active = exposure.section_indices > 0
    points, arcs = exposure.centerline_m[active], exposure.material_arc_lengths_m[active]
    match = None
    if failure is None:
        expected = polyline_arcs(comparison_m)[-1] - (state.section_lengths_m.sum() - arcs)
        match = match_ordered_polyline(points, comparison_m, expected, settings.matching_window_m)
    return FTLTrial(state, exposure, error, points, arcs, match, failure)


def _body_error(trial: FTLTrial) -> float:
    return float(np.max(trial.body_match.distances_m, initial=0.0))


def solve_spatial_ftl_step(
    config: ProjectConfig,
    reference: ReferencePath,
    previous: FTLStep,
    desired_progress_m: float,
    comparison_m: FloatArray,
    planned_m: FloatArray,
) -> FTLStep:
    """Optimize delta_d and (delta_bend_x,delta_bend_y) for each eligible section.

    Absolute and incremental bend bounds are Euclidean disks, not independent
    rigid-joint boxes. Release/re-entry ramps taper the total bend magnitude to
    zero. The fixed reference may be planar; there is no constraint on leaving it.
    Same unsmoothed weighted tip/max-body objective as planar Stage 2C.
    """
    if not isinstance(config.robot, RobotConfig) or config.deployment is None:
        raise ValueError("Spatial FTL requires 3D robot and deployment")
    started = perf_counter()
    ftl = config.ftl or FTLConfig()
    old = previous.state
    target = interpolate_reference_path(reference, np.array([desired_progress_m]))[0]
    total = float(old.section_lengths_m.sum())
    lower = max(ftl.min_feed_increment_m, -old.distance_m)
    upper = min(ftl.max_feed_increment_m, max(0.0, total - old.distance_m))
    thresholds = old.release_thresholds_m
    eligible = np.flatnonzero(old.released | (old.distance_m + upper > thresholds))
    feed_scale = max(-ftl.min_feed_increment_m, ftl.max_feed_increment_m)
    limits = np.array([s.max_bend_rad for s in config.robot.sections])
    count = len(eligible)
    bounds = [(lower / feed_scale, upper / feed_scale), *[(-1.0, 1.0)] * (2 * count)]
    old_norms = np.linalg.norm(old.applied_bends_rad, axis=1)
    fraction = np.clip((old.distance_m - thresholds[eligible]) / ftl.max_feed_increment_m, 0, 1)
    entering = (old.distance_m < thresholds[eligible] + ftl.max_feed_increment_m) & (
        old_norms[eligible] <= ftl.max_bend_increment_rad * fraction + ftl.increment_tolerance
    )
    returning = ~entering & (
        old.distance_m + lower < thresholds[eligible] + ftl.max_feed_increment_m
    )
    ramp_mask = entering | returning
    ramp_ids = eligible[ramp_mask]
    ramp_caps = np.where(
        entering[ramp_mask],
        ftl.max_bend_increment_rad,
        old_norms[ramp_ids] + ftl.max_bend_increment_rad,
    )
    ramp_widths = np.where(
        entering[ramp_mask],
        ftl.max_feed_increment_m,
        np.minimum(ftl.max_feed_increment_m, old.distance_m - thresholds[ramp_ids]),
    )

    def unpack(control):
        distance = float(np.clip(old.distance_m + control[0] * feed_scale, 0, total))
        bends = old.applied_bends_rad.copy()
        bends[eligible] += control[1:].reshape(count, 2) * ftl.max_bend_increment_rad
        return distance, bends

    def constraints(control):
        distance, bends = unpack(control)
        # Disk constraints in stable Cartesian bend coordinates. Squared norms
        # are differentiable at straight sections (phi is not a solver variable).
        # Same angular disks, scaled by one-step angles so SLSQP's dimensionless
        # feasibility tolerance cannot hide a larger physical angle violation.
        absolute = (
            limits[eligible] ** 2 - np.sum(bends[eligible] ** 2, axis=1)
        ) / ftl.max_bend_increment_rad**2
        incremental = 1 - np.sum(control[1:].reshape(count, 2) ** 2, axis=1)
        allowed = ramp_caps * np.clip((distance - thresholds[ramp_ids]) / ramp_widths, 0, 1)
        # A squared zero-radius release disk has zero gradient and can hide small
        # nonzero bends. Use the actual radial slack and redundant linear facets.
        # Both describe the same isotropic disk, including its exact straight tip.
        angular_scale = min(
            ftl.max_bend_increment_rad, 0.5 * ftl.increment_tolerance / ftl.solver_tolerance
        )
        release = (allowed - np.linalg.norm(bends[ramp_ids], axis=1)) / angular_scale
        facets = np.r_[
            allowed[:, None] - bends[ramp_ids], allowed[:, None] + bends[ramp_ids]
        ].ravel()
        return np.r_[absolute, incremental, release, facets / angular_scale]

    last_control, last_trial = None, None

    def objective(control):
        nonlocal last_control, last_trial
        if last_control is None or not np.array_equal(control, last_control):
            distance, bends = unpack(control)
            last_trial = evaluate_spatial_trial(
                config, distance, bends, target, comparison_m, ftl.optimization_body_spacing_m
            )
            last_control = control.copy()
        body = (
            _body_error(last_trial)
            if last_trial.geometry_failure is None
            else ftl.invalid_geometry_penalty_m
        )
        return (
            palmer_objective(last_trial.tip_error_m, body, ftl.tip_weight) / ftl.objective_scale_m
        )

    feed_guess = (
        previous.feed_increment_m
        if previous.feed_increment_m < 0
        else min(
            upper,
            max(previous.feed_increment_m, np.linalg.norm(target - old.forward.tip_position_m)),
        )
    )
    guess = np.r_[
        feed_guess / feed_scale,
        previous.configuration_increment_rad[eligible].ravel() / ftl.max_bend_increment_rad,
    ]
    guess = np.clip(guess, [b[0] for b in bounds], [b[1] for b in bounds])
    result = minimize(
        objective,
        guess,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraints}] if count else (),
        options={
            "maxiter": ftl.max_iterations,
            "ftol": ftl.solver_tolerance,
            "eps": ftl.finite_difference_step,
        },
    )
    finite = bool(np.isfinite(result.x).all())
    control = result.x if finite else np.zeros_like(guess)
    distance, bends = unpack(control)
    checked = evaluate_spatial_trial(
        config, distance, bends, target, comparison_m, ftl.validation_body_spacing_m
    )
    state = checked.state
    delta = state.applied_bends_rad - old.applied_bends_rad
    delta_feed = distance - old.distance_m
    tol = ftl.increment_tolerance
    failure = checked.geometry_failure
    if not finite:
        failure = "NONFINITE_SOLVER_RESULT"
    elif np.any(np.abs(bends - state.applied_bends_rad) > tol):
        failure = "UNRELEASED_BENDING_REQUEST"
    elif not lower - tol <= control[0] * feed_scale <= upper + tol or np.any(
        np.linalg.norm(delta, axis=1) > ftl.max_bend_increment_rad + tol
    ):
        failure = "INCREMENT_LIMIT_VIOLATION"
    elif np.any(np.linalg.norm(bends, axis=1) > limits + tol):
        failure = "BEND_LIMIT_VIOLATION"
    elif len(ramp_ids) and np.any(
        np.linalg.norm(bends[ramp_ids], axis=1)
        > ramp_caps * np.clip((distance - thresholds[ramp_ids]) / ramp_widths, 0, 1) + tol
    ):
        failure = "RELEASE_CONTINUITY_VIOLATION"
    elif failure is None and not result.success:
        failure = f"SOLVER_FAILED: status={result.status}: {result.message}"
    elif failure is None and ftl.enforce_error_limits and checked.tip_error_m > ftl.tip_tolerance_m:
        failure = "TIP_TOLERANCE_EXCEEDED"
    elif (
        failure is None and ftl.enforce_error_limits and _body_error(checked) > ftl.body_tolerance_m
    ):
        failure = "BODY_TOLERANCE_EXCEEDED"
    chords = polyline_arcs(planned_m)
    exact_arcs = np.linspace(0, reference.arc_lengths_m[-1], len(planned_m))
    projected = match_ordered_polyline(
        state.forward.tip_position_m[None, :],
        planned_m,
        np.array([np.interp(desired_progress_m, exact_arcs, chords)]),
        ftl.matching_window_m,
    ).matched_arcs_m[0]
    progress = float(
        np.clip(
            np.interp(projected, chords, exact_arcs),
            previous.reference_progress_m,
            desired_progress_m,
        )
    )
    terminal = (
        desired_progress_m >= reference.arc_lengths_m[-1] - config.deployment.geometry_tolerance_m
    )
    stalled = bool(
        not terminal and progress - previous.reference_progress_m < ftl.minimum_progress_m
    )
    if failure is None and stalled and ftl.reference_progress_mode == "actual_tip":
        failure = "NO_FORWARD_REFERENCE_PROGRESS"
    maximum, mean, worst_point, worst_arc, deviation = None, None, None, None, None
    if checked.body_match is not None:
        maximum = _body_error(checked)
        mean = float(np.mean(checked.body_match.distances_m)) if len(checked.body_points_m) else 0.0
        if len(checked.body_points_m):
            worst = int(np.argmax(checked.body_match.distances_m))
            worst_point, worst_arc = (
                checked.body_points_m[worst].copy(),
                float(checked.body_material_arcs_m[worst]),
            )
        expected = np.interp(progress - (total - checked.body_material_arcs_m), exact_arcs, chords)
        deviation = float(
            np.max(
                match_ordered_polyline(
                    checked.body_points_m, planned_m, expected, ftl.matching_window_m
                ).distances_m,
                initial=0.0,
            )
        )
    accepted = failure is None
    history = (
        np.vstack((previous.historical_tip_path_m, state.forward.tip_position_m))
        if accepted
        else previous.historical_tip_path_m.copy()
    )
    return FTLStep(
        previous.step_index + 1,
        "FTL",
        state,
        checked.exposure,
        delta_feed,
        delta,
        tuple(int(i) for i in eligible),
        progress,
        desired_progress_m,
        target,
        checked.tip_error_m,
        maximum,
        mean,
        deviation,
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
