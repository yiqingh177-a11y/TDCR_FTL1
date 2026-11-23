"""Planar and nonplanar spatial PCC references; no trajectory optimization."""

from dataclasses import replace

import numpy as np

from .kinematics import base_transform
from .models import ExposureStatus, ProjectConfig, ReferencePath, RobotConfig, TargetConfig
from .planar_geometry import material_boundaries
from .spatial_deployment import build_spatial_geometry
from .spatial_geometry import extract_spatial_exposure


def load_spatial_case(config: ProjectConfig, case: str) -> tuple[ProjectConfig, ReferencePath]:
    """Create a configured PCC reference, globally rotated about world +Z.

    Legacy signed bends define a YZ curve. XY case arrays instead specify
    local integrated bending vectors (n,2), allowing a nonplanar chain. The
    global yaw preserves both shapes and lengths; FTL stays unconstrained in 3D.
    """
    if (
        not isinstance(config.robot, RobotConfig)
        or config.spatial_reference is None
        or config.deployment is None
    ):
        raise ValueError("Spatial case requires 3D robot, deployment and spatial_reference")
    settings = config.spatial_reference
    if case not in settings.case_bends_rad and case not in settings.case_bends_xy_rad:
        raise ValueError(f"Unknown spatial reference case: {case}")
    angle = np.deg2rad(settings.plane_rotation_deg)
    world_axis = np.array([-np.sin(angle), np.cos(angle), 0.0])
    pose = base_transform(config.robot)
    if not np.allclose(pose[:3, 2], [0, 0, 1], atol=config.deployment.geometry_tolerance_m, rtol=0):
        raise ValueError("Reference/feed axis must remain world +Z")
    if case in settings.case_bends_rad:
        axis_local = pose[:3, :3].T @ world_axis
        bends = settings.case_bends_rad[case][:, None] * axis_local[:2]
    else:
        # Conjugating every local PCC bend by the same yaw rigidly rotates the
        # full chain. Each next section still starts in its predecessor's pose.
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        local_rotation = pose[:3, :3].T @ rotation
        bends = settings.case_bends_xy_rad[case] @ local_rotation[:2, :2].T
    geometry = build_spatial_geometry(
        config.deployment.active_section_names,
        np.array([s.length_m for s in config.robot.sections]),
        bends,
        pose,
    )
    tip = geometry.junction_transforms[-1, :3, 3]
    target = (
        config.target
        if case == "staircase"
        else TargetConfig(tip.copy(), config.target.tolerance_m)
    )
    error = float(np.linalg.norm(tip - target.position_m))
    if error > target.tolerance_m:
        raise ValueError(
            "Spatial reference shape does not reach configured target; check lengths/target"
        )
    exposure = extract_spatial_exposure(
        geometry,
        config.deployment.environment_entry_z_m,
        config.sampling.max_arc_spacing_m,
        config.deployment.geometry_tolerance_m,
    )
    if exposure.status != ExposureStatus.EXPOSED or not exposure.exposed_length_m:
        raise ValueError(f"Spatial reference has no valid exposed path: {exposure.diagnostic}")
    boundaries = material_boundaries(geometry)
    retained = (
        boundaries[boundaries >= exposure.entry_material_arc_m] - exposure.entry_material_arc_m
    )
    path = ReferencePath(
        geometry,
        exposure.entry_material_arc_m,
        exposure.centerline_m,
        exposure.arc_lengths_m,
        exposure.section_indices,
        np.unique(np.r_[0.0, retained]),
        target.position_m.copy(),
        error,
    )
    return replace(config, target=target), path


def reference_planarity(points_m: np.ndarray, tolerance_m: float) -> dict:
    """Fit a plane to world XYZ reference samples (k,3), with SI diagnostics.

    The smallest singular value measures the third spatial dimension. A line
    also counts as planar, but has no unique plane normal. Best-fit departure
    describes reference geometry, not FTL body-following error.
    """
    points = np.asarray(points_m, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.all(np.isfinite(points))
        or not np.isfinite(tolerance_m)
        or tolerance_m <= 0
    ):
        raise ValueError("Finite nonempty XYZ samples and positive planarity tolerance required")
    origin = points.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(points - origin, full_matrices=False)
    # With fewer than three samples a plane always exists; obtain its null axis.
    if len(points) < 3:
        _, _, axes = np.linalg.svd(points - origin, full_matrices=True)
    normal = axes[-1]
    departure = np.abs((points - origin) @ normal)
    return {
        "is_planar": bool(np.max(departure) <= tolerance_m),
        "tolerance_m": tolerance_m,
        "maximum_best_fit_plane_departure_m": float(np.max(departure)),
        "rms_best_fit_plane_departure_m": float(np.sqrt(np.mean(departure**2))),
        "singular_values_m": singular_values.tolist(),
        "best_fit_plane_origin_m": origin.tolist(),
        "best_fit_plane_normal": normal.tolist(),
    }
