"""Full spatial PCC feed-in/release geometry, independent of FTL and rendering."""

from math import fsum

import numpy as np

from .kinematics import base_transform
from .models import (
    DeploymentConfig,
    FloatArray,
    ForwardResult,
    RobotConfig,
    SectionCurve,
    SpatialCurveGeometry,
    SpatialDeploymentState,
)
from .planar_geometry import material_boundaries, sample_interval
from .spatial_geometry import evaluate_spatial_curve, local_positions, local_transform


def build_spatial_geometry(
    names: tuple[str, ...], lengths_m: FloatArray, bends_rad: FloatArray, base_pose: FloatArray
) -> SpatialCurveGeometry:
    """Chain local PCC poses from the preceding tip, world XYZ; bends shape (n,2)."""
    lengths, bends, pose = (
        np.asarray(lengths_m, float),
        np.asarray(bends_rad, float),
        np.asarray(base_pose, float),
    )
    if (
        lengths.shape != (len(names),)
        or bends.shape != (len(names), 2)
        or pose.shape != (4, 4)
        or not np.isfinite(lengths).all()
        or not np.isfinite(bends).all()
        or not np.isfinite(pose).all()
        or np.any(lengths <= 0)
    ):
        raise ValueError("Finite physical lengths, local bends (n,2), base pose (4,4) required")
    poses = [pose.copy()]
    for length, bend in zip(lengths, bends, strict=True):
        poses.append(poses[-1] @ local_transform(length, bend, length))
    return SpatialCurveGeometry(names, lengths.copy(), np.array(poses), bends.copy())


def sample_spatial_forward(geometry: SpatialCurveGeometry, spacing_m: float) -> ForwardResult:
    """Exact spatial PCC samples with per-subarc conservative chord bounds, SI units."""
    boundaries = material_boundaries(geometry)
    arcs = sample_interval(geometry, 0.0, boundaries[-1], spacing_m)
    centerline = evaluate_spatial_curve(geometry, arcs)
    curves = []
    for i, length in enumerate(geometry.section_lengths_m):
        theta = np.linalg.norm(geometry.bends_rad[i])
        count = max(1, int(np.ceil(length / spacing_m)), int(np.ceil(theta / np.pi)))
        local = np.linspace(0, length, count + 1)
        pose = geometry.junction_transforms[i]
        points = (
            local_positions(length, geometry.bends_rad[i], local) @ pose[:3, :3].T + pose[:3, 3]
        )
        sagitta = length * theta / (8 * count**2) * np.sinc(theta / (4 * count * np.pi)) ** 2
        curves.append(SectionCurve(points, local, float(sagitta)))
    return ForwardResult(
        geometry.junction_transforms[-1, :3, 3].copy(),
        geometry.junction_transforms[-1, :3, :3].copy(),
        geometry.junction_transforms[:, :3, 3].copy(),
        centerline,
        arcs,
        tuple(curves),
    )


def compute_spatial_deployment(
    robot: RobotConfig,
    settings: DeploymentConfig,
    distance_m: float,
    requested_bends_rad: FloatArray,
    spacing_m: float,
) -> SpatialDeploymentState:
    """Feed in world +Z; unreleased complete sections straight; no passive bend.

    Active variables are integrated local bends (n,2), radians. Their norms
    obey section total-angle bounds. All physical lengths remain unchanged.
    """
    if not isinstance(robot, RobotConfig):
        raise ValueError("Spatial deployment requires RobotConfig")
    bends = np.asarray(requested_bends_rad, float)
    limits = np.array([s.max_bend_rad for s in robot.sections])
    if (
        bends.shape != (robot.section_count, 2)
        or not np.isfinite(bends).all()
        or np.any(np.linalg.norm(bends, axis=1) > limits + 1e-12)
    ):
        raise ValueError("Spatial bends must be finite (n,2) within total-angle limits")
    lengths = np.r_[settings.passive_length_m, [s.length_m for s in robot.sections]]
    total = float(np.sum(lengths))
    if not np.isfinite(distance_m) or not 0 <= distance_m <= total:
        raise ValueError("Feed distance must be in [0,total physical length]")
    pose = base_transform(robot)
    if not np.allclose(pose[:3, 2], [0, 0, 1], atol=settings.geometry_tolerance_m, rtol=0):
        raise ValueError("This spatial deployment uses fixed world +Z feed (no base tilt)")
    thresholds = np.array(
        [fsum(s.length_m for s in robot.sections[i:]) for i in range(robot.section_count)]
    )
    released = distance_m >= thresholds
    applied = np.where(released[:, None], bends, 0.0)
    proximal = settings.feed_exit_m + [0.0, 0.0, distance_m - total]
    pose[:3, 3] = proximal
    geometry = build_spatial_geometry(
        ("Passive", *settings.active_section_names),
        lengths,
        np.vstack((np.zeros(2), applied)),
        pose,
    )
    forward = sample_spatial_forward(geometry, spacing_m)
    warnings = tuple(
        f"Released section {settings.active_section_names[i]} folds behind feed exit"
        for i, curve in enumerate(forward.section_curves[1:])
        if released[i]
        and curve.points_m[:, 2].min() < settings.feed_exit_m[2] - settings.geometry_tolerance_m
    )
    return SpatialDeploymentState(
        float(distance_m),
        proximal,
        geometry.junction_transforms[1, :3, 3].copy(),
        bends.copy(),
        applied,
        released,
        thresholds,
        geometry.section_names,
        lengths,
        geometry,
        forward,
        warnings,
    )
