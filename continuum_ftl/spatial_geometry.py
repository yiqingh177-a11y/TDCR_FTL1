"""Vectorized spatial PCC geometry and exact horizontal-plane exposure."""

import numpy as np
from scipy.optimize import brentq

from .models import EnvironmentExposure, ExposureStatus, FloatArray, SpatialCurveGeometry
from .planar_geometry import material_boundaries, sample_interval, section_indices


def local_positions(length_m: float, bend_xy_rad: FloatArray, arcs_m: FloatArray) -> FloatArray:
    """Exact local XYZ points (M,3), integrated bends (2,), arcs (M,), SI units.

    bend_x/bend_y point toward local +X/+Y. The analytic Cartesian form has
    finite derivatives at zero bend and avoids the undefined straight phi.
    """
    bend = np.asarray(bend_xy_rad, dtype=float)
    arcs = np.asarray(arcs_m, dtype=float)
    if (
        bend.shape != (2,)
        or arcs.ndim != 1
        or not np.isfinite(bend).all()
        or not np.isfinite(arcs).all()
        or not np.isfinite(length_m)
        or length_m <= 0
        or np.any((arcs < 0) | (arcs > length_m))
    ):
        raise ValueError("Require finite bends (2,), arcs (M,) in [0,length], and positive length")
    angle = np.linalg.norm(bend) * arcs / length_m
    radial = arcs**2 / (2 * length_m) * np.sinc(angle / (2 * np.pi)) ** 2
    return np.column_stack((radial[:, None] * bend, arcs * np.sinc(angle / np.pi)))


def local_transform(length_m: float, bend_xy_rad: FloatArray, arc_m: float) -> FloatArray:
    """Local cross-section pose (4,4), no independent torsion; lengths m, angles rad."""
    position = local_positions(length_m, bend_xy_rad, np.array([arc_m]))[0]
    x, y = np.asarray(bend_xy_rad) * (arc_m / length_m)
    skew = np.array([[0.0, 0.0, x], [0.0, 0.0, y], [-x, -y, 0.0]])
    angle = np.hypot(x, y)
    transform = np.eye(4)
    transform[:3, :3] += np.sinc(angle / np.pi) * skew + 0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (
        skew @ skew
    )
    transform[:3, 3] = position
    return transform


def evaluate_spatial_curve(geometry: SpatialCurveGeometry, arcs_m: FloatArray) -> FloatArray:
    """Evaluate exact world XYZ at physical material arcs (M,), not reference indices."""
    arcs = np.asarray(arcs_m, dtype=float)
    boundaries = material_boundaries(geometry)
    if (
        arcs.ndim != 1
        or not np.isfinite(arcs).all()
        or np.any((arcs < 0) | (arcs > boundaries[-1]))
    ):
        raise ValueError("Material arc coordinates must lie within the complete physical chain")
    indices = section_indices(geometry, arcs)
    points = np.empty((len(arcs), 3))
    for i in np.unique(indices):
        mask = indices == i
        length = geometry.section_lengths_m[i]
        local = np.clip(arcs[mask] - boundaries[i], 0, length)
        pose = geometry.junction_transforms[i]
        points[mask] = (
            local_positions(length, geometry.bends_rad[i], local) @ pose[:3, :3].T + pose[:3, 3]
        )
    for i, boundary in enumerate(boundaries):
        points[arcs == boundary] = geometry.junction_transforms[i, :3, 3]
    return points


def spatial_entry_intersections(
    geometry: SpatialCurveGeometry, entry_z_m: float, tolerance_m: float
) -> tuple[FloatArray, bool]:
    """All exact PCC crossings of world Z=entry, including tangencies/re-entry.

    For each spatial circle, dz/ds=a*sin(theta*s/L)+b*cos(theta*s/L). Split at
    its analytic stationary points, then bracket scalar roots. Sampling does
    not decide which material is exposed; root-finding is geometric, not IK.
    """
    if not np.isfinite([entry_z_m, tolerance_m]).all() or tolerance_m <= 0:
        raise ValueError("Finite entry height and positive geometry tolerance required")
    boundaries = material_boundaries(geometry)
    roots = []
    coincident = False
    for i, length in enumerate(geometry.section_lengths_m):
        pose, bend = geometry.junction_transforms[i], geometry.bends_rad[i]
        theta = float(np.linalg.norm(bend))
        b = pose[2, 2]
        a = float(pose[2, :2] @ (bend / theta)) if theta > 0 else 0.0

        def height(s, length=length, bend=bend, pose=pose):
            return float(
                (local_positions(length, bend, np.array([s])) @ pose[:3, :3].T)[0, 2]
                + pose[2, 3]
                - entry_z_m
            )

        if (theta == 0 and abs(b) <= 32 * np.finfo(float).eps) or (
            theta > 0 and np.hypot(a, b) <= 32 * np.finfo(float).eps
        ):
            coincident |= abs(height(0.0)) <= tolerance_m
            continue
        knots = [0.0, float(length)]
        if theta > 0:
            first = np.arctan2(-b, a)
            for k in range(
                int(np.ceil(-first / np.pi)), int(np.floor((theta - first) / np.pi)) + 1
            ):
                angle = first + k * np.pi
                if 0 < angle < theta:
                    knots.append(float(length * angle / theta))
        knots = sorted(knots)
        heights = [height(s) for s in knots]
        for s, value in zip(knots, heights, strict=True):
            if abs(value) <= tolerance_m:
                roots.append(boundaries[i] + s)
        for left, right, za, zb in zip(
            knots[:-1], knots[1:], heights[:-1], heights[1:], strict=True
        ):
            if abs(za) > tolerance_m and abs(zb) > tolerance_m and np.signbit(za) != np.signbit(zb):
                roots.append(boundaries[i] + brentq(height, left, right, xtol=tolerance_m * 0.01))
    unique = []
    for value in sorted(roots):
        if not unique or value - unique[-1] > tolerance_m:
            unique.append(float(value))
    return np.array(unique), bool(coincident)


def extract_spatial_exposure(
    geometry: SpatialCurveGeometry, entry_z_m: float, spacing_m: float, tolerance_m: float
) -> EnvironmentExposure:
    """Continuous single forward entry-to-tip domain; full physical sections stay intact."""
    if not np.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("Positive finite exposure spacing required")
    roots, coincident = spatial_entry_intersections(geometry, entry_z_m, tolerance_m)
    tip = geometry.junction_transforms[-1, :3, 3].copy()
    entered = bool(tip[2] >= entry_z_m - tolerance_m)
    count = len(geometry.section_names)

    def no_interval(status, diagnostic):
        empty = status == ExposureStatus.NOT_ENTERED
        return EnvironmentExposure(
            status,
            diagnostic,
            entered,
            None,
            None,
            np.empty((0, 3)),
            np.empty(0),
            np.empty(0),
            0.0 if empty else None,
            geometry.section_names,
            np.zeros(count) if empty else None,
            np.empty(0, dtype=np.int64),
            tip,
            roots,
        )

    if coincident or len(roots) > 1:
        return no_interval(ExposureStatus.AMBIGUOUS, "Multiple/plane-coincident entry crossings")
    if not len(roots):
        return no_interval(
            ExposureStatus.NOT_ENTERED if not entered else ExposureStatus.AMBIGUOUS,
            "No unique entry crossing",
        )
    entry = float(roots[0])
    boundaries = material_boundaries(geometry)
    i = int(section_indices(geometry, np.array([entry]))[0])
    local = float(np.clip(entry - boundaries[i], 0, geometry.section_lengths_m[i]))
    tangent = (
        geometry.junction_transforms[i]
        @ local_transform(geometry.section_lengths_m[i], geometry.bends_rad[i], local)
    )[:3, 2]
    if tangent[2] <= np.sqrt(np.finfo(float).eps) or not entered:
        return no_interval(ExposureStatus.AMBIGUOUS, "Entry is tangential or backward")
    arcs = sample_interval(geometry, entry, boundaries[-1], spacing_m)
    points = evaluate_spatial_curve(geometry, arcs)
    exposed = np.maximum(0.0, boundaries[1:] - np.maximum(boundaries[:-1], entry))
    return EnvironmentExposure(
        ExposureStatus.EXPOSED,
        "One forward spatial entry interval",
        entered,
        points[0].copy(),
        entry,
        points,
        arcs - entry,
        arcs,
        float(boundaries[-1] - entry),
        geometry.section_names,
        exposed,
        section_indices(geometry, arcs),
        tip,
        roots,
    )
