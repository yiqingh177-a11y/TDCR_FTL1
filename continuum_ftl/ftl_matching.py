"""Frozen history/previous-body references and deterministic local curve matching."""

import numpy as np

from .models import (
    DeploymentState,
    EnvironmentExposure,
    ExposureStatus,
    FloatArray,
    PathMatch,
    PlanarCurveGeometry,
)
from .planar_geometry import evaluate_curve, sample_interval


def polyline_arcs(points_m: FloatArray) -> FloatArray:
    """Cumulative chord length (M,) m of a finite world YZ/XYZ polyline."""
    points = np.asarray(points_m, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] not in (2, 3)
        or len(points) == 0
        or not np.isfinite(points).all()
    ):
        raise ValueError("A finite nonempty (M,2) or (M,3) polyline is required")
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def comparison_path(
    history_m: FloatArray, next_waypoint_m: FloatArray, max_spacing_m: float
) -> FloatArray:
    """Densify old accepted history plus intended straight incremental extension.

    Extension is only a comparison target, never an actual robot/tip position.
    Historical geometry is preserved; no spline smoothing or reference copying.
    """
    polyline_arcs(history_m)
    waypoint = np.asarray(next_waypoint_m, dtype=float)
    if waypoint.shape != (history_m.shape[1],) or not np.isfinite(waypoint).all():
        raise ValueError("Next waypoint must be finite and match path dimension")
    if not np.isfinite(max_spacing_m) or max_spacing_m <= 0:
        raise ValueError("Positive finite comparison spacing required")
    points = np.vstack((history_m, waypoint))
    pieces = [points[:1]]
    for start, end in zip(points[:-1], points[1:], strict=True):
        length = np.linalg.norm(end - start)
        if length > 0:
            pieces.append(
                np.linspace(start, end, max(1, int(np.ceil(length / max_spacing_m))) + 1)[1:]
            )
    return np.vstack(pieces)


def previous_body_comparison_path(
    state: DeploymentState,
    exposure: EnvironmentExposure,
    next_waypoint_m: FloatArray,
    max_spacing_m: float,
) -> FloatArray:
    """Freeze previous accepted exposed active PCC body + new tip extension.

    Returns world YZ points (M,2), metres, proximal to distal. The Stage 2B
    entry material coordinate clips the actual continuous full-length PCC chain;
    the passive section is excluded even if it enters the environment. Sample
    the old PCC at comparison resolution, preserving junctions and exact ends.

    At first entry the old exposed body is one point. Newly exposed candidate
    points are still matched geometrically by the existing matcher, including
    its entry endpoint; negative expected arcs only clip the search window.
    They are never matched by material ID or given an automatic zero error.
    No candidate trial geometry is used or retained here.
    """
    if exposure.status != ExposureStatus.EXPOSED or exposure.entry_material_arc_m is None:
        raise ValueError("Previous body requires an accepted, unambiguous exposed interval")
    geometry = PlanarCurveGeometry(
        state.section_names,
        state.section_lengths_m,
        state.forward.section_endpoints_m,
        state.boundary_tangent_angles_rad,
    )
    start = max(exposure.entry_material_arc_m, state.section_lengths_m[0])
    end = float(np.sum(state.section_lengths_m))
    arcs = sample_interval(geometry, start, end, max_spacing_m)
    points = evaluate_curve(geometry, arcs)
    return comparison_path(points, next_waypoint_m, max_spacing_m)


def match_ordered_polyline(
    points_m: FloatArray, path_m: FloatArray, expected_arcs_m: FloatArray, window_m: float
) -> PathMatch:
    """Exact point-to-segment distance, constrained to local ordered path windows.

    expected_arcs (M,) selects search neighbourhoods only, not same-index pairs.
    Body points arrive proximal-to-distal. Correspondences may stay at one path
    location but cannot go backwards onto an unrelated loop. All lengths m.
    """
    points = np.asarray(points_m, dtype=float)
    expected = np.asarray(expected_arcs_m, dtype=float)
    arcs = polyline_arcs(path_m)
    if (
        points.shape != (len(expected), path_m.shape[1])
        or expected.ndim != 1
        or not np.isfinite(points).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Finite body points (M,d) and arcs (M,) must match path dimension d")
    if not np.isfinite(window_m) or window_m <= 0:
        raise ValueError("Positive finite local matching window required")
    distances, matched, progress = (
        np.empty(len(points)),
        np.empty_like(points),
        np.empty(len(points)),
    )
    if len(path_m) == 1:
        matched[:] = path_m[0]
        return PathMatch(np.linalg.norm(points - matched, axis=1), matched, np.zeros(len(points)))
    vectors = np.diff(path_m, axis=0)
    lengths = np.diff(arcs)
    previous_arc = 0.0
    for index, point in enumerate(points):
        lower = max(0.0, previous_arc, min(arcs[-1], expected[index]) - window_m)
        upper = min(arcs[-1], max(0.0, expected[index]) + window_m)
        if upper < lower:
            raise ValueError("Ordered matching windows are inconsistent")
        eligible = (arcs[:-1] <= upper) & (arcs[1:] >= lower) & (lengths > 0)
        ids = np.flatnonzero(eligible)
        starts, segments, sizes = path_m[ids], vectors[ids], lengths[ids]
        fraction = np.sum((point - starts) * segments, axis=1) / sizes**2
        fraction = np.clip(
            fraction,
            np.maximum(0, (lower - arcs[ids]) / sizes),
            np.minimum(1, (upper - arcs[ids]) / sizes),
        )
        projections = starts + fraction[:, None] * segments
        errors = np.linalg.norm(projections - point, axis=1)
        best = int(np.argmin(errors))
        distances[index], matched[index] = errors[best], projections[best]
        previous_arc = float(arcs[ids[best]] + fraction[best] * sizes[best])
        progress[index] = previous_arc
    return PathMatch(distances, matched, progress)


def palmer_objective(tip_error_m: float, maximum_body_error_m: float, tip_weight: float) -> float:
    """Unsmoothed weighted Euclidean tip error plus maximum body deviation (m)."""
    return float(tip_weight * tip_error_m + maximum_body_error_m)
