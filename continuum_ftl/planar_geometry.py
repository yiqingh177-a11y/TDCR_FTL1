"""Continuous PCC evaluation and entry-plane intersections, independent of sampling."""

import numpy as np
from scipy.optimize import brentq

from .models import FloatArray, PlanarCurveGeometry
from .planar_kinematics import planar_rotation, section_points_2d


def material_boundaries(geometry: PlanarCurveGeometry) -> FloatArray:
    """Material junction coordinates (n+1,), m from the physical proximal end."""
    return np.r_[0.0, np.cumsum(geometry.section_lengths_m)]


def evaluate_curve(geometry: PlanarCurveGeometry, material_arcs_m: FloatArray) -> FloatArray:
    """Exact world YZ points (M,2) at material arc coordinates (M,), metres.

    Internal junctions belong to the distal section. Queries outside the physical
    curve are errors; no extrapolation or projection onto a desired path occurs.
    """
    arcs = np.asarray(material_arcs_m, dtype=float)
    boundaries = material_boundaries(geometry)
    if (
        arcs.ndim != 1
        or not np.isfinite(arcs).all()
        or np.any((arcs < 0) | (arcs > boundaries[-1]))
    ):
        raise ValueError("Material arc coordinates must be a finite vector inside the curve")
    indices = section_indices(geometry, arcs)
    points = np.empty((len(arcs), 2))
    for index in np.unique(indices):
        mask = indices == index
        length = geometry.section_lengths_m[index]
        local = np.clip(arcs[mask] - boundaries[index], 0.0, length)
        bend = geometry.junction_angles_rad[index + 1] - geometry.junction_angles_rad[index]
        points[mask] = (
            section_points_2d(length, bend, local)
            @ planar_rotation(geometry.junction_angles_rad[index]).T
            + geometry.junction_positions_m[index]
        )
    # Preserve stored exact junctions, particularly start/tip, without rounding.
    for index, boundary in enumerate(boundaries):
        points[arcs == boundary] = geometry.junction_positions_m[index]
    return points


def section_indices(geometry: PlanarCurveGeometry, arcs_m: FloatArray) -> np.ndarray:
    """Zero-based physical section IDs (M,); junction ownership is distal."""
    return np.clip(
        np.searchsorted(material_boundaries(geometry), arcs_m, side="right") - 1,
        0,
        len(geometry.section_names) - 1,
    ).astype(np.int64)


def sample_interval(
    geometry: PlanarCurveGeometry, start_m: float, end_m: float, max_spacing_m: float
) -> FloatArray:
    """Material coordinates in [start,end], m; include exact endpoints and junctions.

    Uniform physical arc sampling is supplemented by section junctions, so the
    spacing never exceeds max_spacing_m. Does not measure sampled chord length.
    """
    total = material_boundaries(geometry)[-1]
    if not np.isfinite([start_m, end_m, max_spacing_m]).all() or not (
        0 <= start_m <= end_m <= total and max_spacing_m > 0
    ):
        raise ValueError("Valid physical interval and positive finite sampling spacing required")
    boundaries = material_boundaries(geometry)
    internal = boundaries[(boundaries > start_m) & (boundaries < end_m)]
    anchors = np.r_[start_m, internal, end_m]
    if start_m == end_m:
        return np.array([start_m])
    return np.concatenate(
        [
            np.linspace(a, b, max(1, int(np.ceil((b - a) / max_spacing_m))) + 1)[:-1]
            for a, b in zip(anchors[:-1], anchors[1:], strict=True)
        ]
        + [np.array([end_m])]
    )


def entry_intersections(
    geometry: PlanarCurveGeometry, entry_z_m: float, tolerance_m: float
) -> tuple[FloatArray, bool]:
    """All plane intersections in material m, plus a plane-coincident-arc flag.

    Split each exact PCC arc at dz/ds=cos(tangent)=0, then bracket scalar roots
    on monotone intervals. This catches multiple crossings within ONE coarse
    sampled chord and tangential contacts; it does not filter sampled z values.
    brentq is only a scalar geometric root finder, not an IK/FTL optimizer.
    """
    if not np.isfinite([entry_z_m, tolerance_m]).all() or tolerance_m <= 0:
        raise ValueError("Finite entry height and positive finite geometry tolerance required")
    boundaries = material_boundaries(geometry)
    roots = []
    coincident = False
    for index, length in enumerate(geometry.section_lengths_m):
        angle = geometry.junction_angles_rad[index]
        bend = geometry.junction_angles_rad[index + 1] - angle
        start = geometry.junction_positions_m[index]
        rotation = planar_rotation(angle)

        def height(
            local_arc: float,
            length: float = length,
            bend: float = bend,
            rotation: FloatArray = rotation,
            start: FloatArray = start,
        ) -> float:
            point = section_points_2d(length, bend, np.array([local_arc]))[0] @ rotation.T + start
            return float(point[1] - entry_z_m)

        if bend == 0 and abs(np.cos(angle)) <= 32 * np.finfo(float).eps:
            coincident |= abs(height(0.0)) <= tolerance_m
            continue
        knots = [0.0, float(length)]
        if bend != 0:
            lower, upper = sorted((angle, angle + bend))
            first = int(np.ceil((lower - np.pi / 2) / np.pi))
            last = int(np.floor((upper - np.pi / 2) / np.pi))
            for turn in range(first, last + 1):
                local_arc = (np.pi / 2 + turn * np.pi - angle) * length / bend
                if 0 < local_arc < length:
                    knots.append(float(local_arc))
        knots = sorted(knots)
        heights = [height(a) for a in knots]
        for arc, value in zip(knots, heights, strict=True):
            if abs(value) <= tolerance_m:
                roots.append(boundaries[index] + arc)
        for a, b, za, zb in zip(knots[:-1], knots[1:], heights[:-1], heights[1:], strict=True):
            if abs(za) > tolerance_m and abs(zb) > tolerance_m and np.signbit(za) != np.signbit(zb):
                root = brentq(height, a, b, xtol=tolerance_m * 0.01)
                roots.append(boundaries[index] + root)
    ordered = []
    for root in sorted(roots):
        if not ordered or root - ordered[-1] > tolerance_m:
            ordered.append(float(root))
    return np.array(ordered), bool(coincident)
