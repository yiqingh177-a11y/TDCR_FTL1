"""S3-only incremental SQP with first-entry section introduction, no original deployment."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from ..ftl_evaluation import point_to_polyline_distances
from ..ftl_matching import (
    comparison_path,
    match_ordered_polyline,
    palmer_objective,
    polyline_arcs,
)
from ..models import PlanarCurveGeometry
from ..planar_geometry import entry_intersections, evaluate_curve, sample_interval
from .palmer_s3 import S3Settings, geometry, reference_geometry, validate_reference


@dataclass(frozen=True)
class S3State:
    """Full four-section state: feed m, signed bends rad, latched first-entry flags."""

    feed_m: float
    bends_rad: np.ndarray
    introduced: np.ndarray
    curve: PlanarCurveGeometry


def exposed_samples(
    settings: S3Settings, curve: PlanarCurveGeometry, spacing_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the connected centreline at/above ideal Z=0, preserving exact entry.

    This benchmark has one ideal interface, not our research robot's three-zone
    exposure or release rules. A crossing away from Y=0 is allowed (no guide).
    Multiple crossings are rejected rather than silently selecting body points.
    """
    tolerance = settings.geometry_tolerance_m
    if curve.junction_positions_m[-1, 1] < -tolerance:
        raise ValueError("TIP_BEHIND_IDEAL_ENTRANCE")
    roots, coincident = entry_intersections(curve, 0.0, tolerance)
    if coincident or len(roots) > 1:
        raise ValueError("AMBIGUOUS_IDEAL_ENTRANCE")
    if len(roots):
        start = roots[0]
    elif curve.junction_positions_m[0, 1] >= -tolerance:
        start = 0.0
    else:
        raise ValueError("NO_IDEAL_ENTRANCE_CROSSING")
    arcs = sample_interval(curve, float(start), 4 * settings.section_length_m, spacing_m)
    return evaluate_curve(curve, arcs), arcs


def initial_state(settings: S3Settings) -> S3State:
    """Base (0,-0.4), straight tip at entrance; only distal section introduced."""
    return S3State(
        0.0,
        np.zeros(4),
        np.array([False, False, False, True]),
        geometry(settings, np.zeros(4), 0.0),
    )


def solve_step(
    settings: S3Settings,
    old: S3State,
    target_m: np.ndarray,
    previous_increment: np.ndarray,
) -> tuple[S3State, dict]:
    """Warm-start bounded SLSQP on feed and introduced bends; frozen old-body path.

    All q remain fixed-length full-section PCC angles. A newly entering section
    stays exactly straight in this solve; its first bend variable appears only
    in the following solve. No original full-section release/ramp is used.
    """
    started = perf_counter()
    previous_points, _ = exposed_samples(settings, old.curve, settings.comparison_spacing_m)
    comparison = comparison_path(previous_points, target_m, settings.comparison_spacing_m)
    ids = np.flatnonzero(old.introduced)
    scales = np.r_[
        settings.max_feed_increment_m, np.full(len(ids), settings.max_bend_increment_rad)
    ]
    lower = np.maximum(
        -settings.max_bend_increment_rad, -settings.max_bend_rad - old.bends_rad[ids]
    )
    upper = np.minimum(settings.max_bend_increment_rad, settings.max_bend_rad - old.bends_rad[ids])
    bounds = [
        (settings.min_feed_increment_m / scales[0], 1.0),
        *zip(lower / scales[1:], upper / scales[1:], strict=True),
    ]

    def unpack(control):
        feed = old.feed_m + float(control[0] * scales[0])
        bends = old.bends_rad.copy()
        bends[ids] += control[1:] * scales[1:]
        # Clip arithmetic roundoff at a bound; optimizer variables themselves
        # stay box-constrained and are independently checked below.
        return feed, np.clip(bends, -settings.max_bend_rad, settings.max_bend_rad)

    def evaluate(control, spacing):
        feed, bends = unpack(control)
        curve = geometry(settings, bends, feed)
        points, arcs = exposed_samples(settings, curve, spacing)
        expected = polyline_arcs(comparison)[-1] - (4 * settings.section_length_m - arcs)
        match = match_ordered_polyline(points, comparison, expected, settings.matching_window_m)
        tip_error = float(np.linalg.norm(curve.junction_positions_m[-1] - target_m))
        return curve, points, arcs, match, tip_error

    def objective(control):
        try:
            _, _, _, match, error = evaluate(control, settings.optimization_spacing_m)
        except ValueError as exc:
            if str(exc) not in (
                "TIP_BEHIND_IDEAL_ENTRANCE",
                "AMBIGUOUS_IDEAL_ENTRANCE",
                "NO_IDEAL_ENTRANCE_CROSSING",
            ):
                raise
            # Invalid geometry only; not an alternative FTL objective.
            return 1e6
        return (
            palmer_objective(error, float(np.max(match.distances_m)), settings.tip_weight)
            / settings.objective_scale_m
        )

    guess = (
        np.r_[
            min(
                settings.max_feed_increment_m,
                max(
                    previous_increment[0],
                    np.linalg.norm(target_m - old.curve.junction_positions_m[-1]),
                ),
            ),
            previous_increment[1:][ids],
        ]
        / scales
    )
    guess = np.clip(guess, [x[0] for x in bounds], [x[1] for x in bounds])
    result = minimize(
        objective,
        guess,
        method="SLSQP",
        bounds=bounds,
        options={
            "maxiter": settings.max_iterations,
            "ftol": settings.solver_tolerance,
            "eps": settings.finite_difference_step,
        },
    )
    failure = None
    finite = np.isfinite(result.x).all()
    feed, bends = unpack(result.x if finite else np.zeros_like(guess))
    if not result.success or not finite:
        failure = f"SOLVER_FAILED: status={result.status}: {result.message}"
    delta = np.r_[feed - old.feed_m, bends - old.bends_rad]
    tolerance = settings.acceptance_tolerance
    if failure is None and (
        not settings.min_feed_increment_m - tolerance
        <= delta[0]
        <= settings.max_feed_increment_m + tolerance
        or np.any(np.abs(delta[1:]) > settings.max_bend_increment_rad + tolerance)
        or np.any(np.abs(bends) > settings.max_bend_rad + tolerance)
        or np.any(bends[~old.introduced] != 0)
    ):
        failure = "CONTROL_OR_INTRODUCTION_BOUND_VIOLATION"
    curve = geometry(settings, bends, feed)
    # First crossing is latched only after acceptance. At that moment the new
    # bend is still zero because it was absent from this solve's variable list.
    introduced = old.introduced | (
        curve.junction_positions_m[1:, 1] >= -settings.geometry_tolerance_m
    )
    state = S3State(feed, bends, introduced, curve)
    row = {
        "feed_m": feed,
        "base_yz_m": curve.junction_positions_m[0].tolist(),
        "configuration_rad": bends.tolist(),
        "introduced": introduced.tolist(),
        "newly_introduced_sections": np.flatnonzero(introduced & ~old.introduced).tolist(),
        "feed_increment_m": float(delta[0]),
        "bend_increment_rad": delta[1:].tolist(),
        "desired_tip_yz_m": target_m.tolist(),
        "actual_tip_yz_m": curve.junction_positions_m[-1].tolist(),
        "actual_tip_displacement_m": float(
            np.linalg.norm(curve.junction_positions_m[-1] - old.curve.junction_positions_m[-1])
        ),
        "tip_error_m": float(np.linalg.norm(curve.junction_positions_m[-1] - target_m)),
        "comparison_path_yz_m": comparison.tolist(),
        "section_endpoints_yz_m": curve.junction_positions_m.tolist(),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "objective_function_evaluations": int(result.nfev),
    }
    try:
        _, points, arcs, match, _ = evaluate(
            result.x if finite else np.zeros_like(guess), settings.validation_spacing_m
        )
        row.update(
            body_yz_m=points.tolist(),
            material_arcs_m=arcs.tolist(),
            max_body_to_previous_m=float(np.max(match.distances_m)),
            mean_body_to_previous_m=float(np.mean(match.distances_m)),
        )
    except ValueError as exc:
        if str(exc) not in (
            "TIP_BEHIND_IDEAL_ENTRANCE",
            "AMBIGUOUS_IDEAL_ENTRANCE",
            "NO_IDEAL_ENTRANCE_CROSSING",
        ):
            raise
        failure = failure or str(exc)
        row.update(
            body_yz_m=[],
            material_arcs_m=[],
            max_body_to_previous_m=None,
            mean_body_to_previous_m=None,
        )
    row.update(
        accepted=failure is None,
        failure_reason=failure,
        computation_time_s=perf_counter() - started,
    )
    return state, row


def run_s3(
    settings: S3Settings, progress_callback: Callable[[dict, int], None] | None = None
) -> dict:
    """Run only S3; stop on first rejected candidate, never tune/retry failures."""
    validation = validate_reference(settings)
    reference = reference_geometry(settings)
    total = 4 * settings.section_length_m
    reference_arcs = sample_interval(reference, 0, total, settings.validation_spacing_m)
    fixed = evaluate_curve(reference, reference_arcs)
    state = initial_state(settings)
    initial = {
        "feed_m": 0.0,
        "configuration_rad": [0.0] * 4,
        "introduced": state.introduced.tolist(),
        "actual_tip_yz_m": [0.0, 0.0],
        "section_endpoints_yz_m": state.curve.junction_positions_m.tolist(),
        "body_yz_m": [[0.0, 0.0]],
        "accepted": True,
        "reference_progress_m": 0.0,
    }
    rows = []
    previous_increment = np.zeros(5)
    started = perf_counter()
    count = int(np.ceil(total / settings.desired_path_step_m))
    for index in range(1, count + 1):
        progress = min(index * settings.desired_path_step_m, total)
        target = evaluate_curve(reference, np.array([progress]))[0]
        candidate, row = solve_step(settings, state, target, previous_increment)
        row.update(
            step_index=index,
            reference_progress_m=progress,
            desired_path_increment_m=progress - (rows[-1]["reference_progress_m"] if rows else 0.0),
        )
        if row["body_yz_m"]:
            distances = point_to_polyline_distances(np.array(row["body_yz_m"]), fixed)
            row.update(
                max_body_to_reference_m=float(np.max(distances)),
                mean_body_to_reference_m=float(np.mean(distances)),
            )
        else:
            row.update(max_body_to_reference_m=None, mean_body_to_reference_m=None)
        rows.append(row)
        if progress_callback is not None:
            progress_callback(row, count)
        if not row["accepted"]:
            break
        state = candidate
        previous_increment = np.r_[row["feed_increment_m"], row["bend_increment_rad"]]
    accepted = [row for row in rows if row["accepted"]]
    return {
        "benchmark": "Palmer 2014 S3 only",
        "completed": len(accepted) == count,
        "failure_reason": rows[-1]["failure_reason"],
        "geometry_validation": validation,
        "initial_state": initial,
        "reference_yz_m": fixed.tolist(),
        "reference_arc_lengths_m": reference_arcs.tolist(),
        "accepted_steps": len(accepted),
        "rejected_steps": len(rows) - len(accepted),
        "attempted_steps": len(rows),
        "accepted_reference_progress_m": accepted[-1]["reference_progress_m"] if accepted else 0.0,
        "final_tip_yz_m": state.curve.junction_positions_m[-1].tolist(),
        "final_configuration_rad": state.bends_rad.tolist(),
        "final_feed_m": state.feed_m,
        "total_rollout_time_s": perf_counter() - started,
        "total_step_time_s": sum(row["computation_time_s"] for row in rows),
        "mean_step_time_s": float(np.mean([row["computation_time_s"] for row in rows])),
        "max_body_to_reference_m": max(
            (row["max_body_to_reference_m"] for row in accepted), default=None
        ),
        "mean_of_frame_max_body_to_reference_m": float(
            np.mean([row["max_body_to_reference_m"] for row in accepted])
        )
        if accepted
        else None,
        "mean_of_frame_mean_body_to_reference_m": float(
            np.mean([row["mean_body_to_reference_m"] for row in accepted])
        )
        if accepted
        else None,
        "max_body_to_previous_m": max(
            (row["max_body_to_previous_m"] for row in accepted), default=None
        ),
        "max_tip_error_m": max((row["tip_error_m"] for row in accepted), default=None),
        "mean_tip_error_m": float(np.mean([row["tip_error_m"] for row in accepted]))
        if accepted
        else None,
        "steps": rows,
    }
