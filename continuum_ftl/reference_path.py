"""Stage 2B fixed reference curve from ONE already accepted Stage 1 IK result."""

import json
from pathlib import Path

import numpy as np

from .config import geometry_context
from .environment_exposure import extract_curve_exposure
from .models import (
    ExposureStatus,
    FloatArray,
    PlanarCurveGeometry,
    PlanarRobotConfig,
    ProjectConfig,
    ReferencePath,
)
from .planar_geometry import evaluate_curve, material_boundaries
from .planar_kinematics import forward_kinematics_2d


def extract_reference_path(config: ProjectConfig, configuration_rad: FloatArray) -> ReferencePath:
    """Fixed active PCC path in original environment YZ, s in metres from entry.

    Caller supplies a previously collision-validated final IK configuration.
    This function validates FK tip tolerance and entry geometry only; it does
    NOT optimize or check collisions. It preserves the exact FK tip, including
    any reported IK residual, rather than inserting an artificial target segment.
    """
    if not isinstance(config.robot, PlanarRobotConfig) or config.deployment is None:
        raise ValueError("Reference extraction requires 2D robot and deployment settings")
    forward = forward_kinematics_2d(
        config.robot, configuration_rad, config.sampling.max_arc_spacing_m
    )
    tip_error = float(np.linalg.norm(forward.tip_position_m - config.target.position_m))
    if tip_error > config.target.tolerance_m:
        raise ValueError(
            "Supplied final IK configuration does not satisfy current target tolerance"
        )
    geometry = PlanarCurveGeometry(
        config.deployment.active_section_names,
        np.array([s.length_m for s in config.robot.sections]),
        forward.section_endpoints_m,
        np.r_[
            config.robot.base_angle_rad, config.robot.base_angle_rad + np.cumsum(configuration_rad)
        ],
    )
    exposed = extract_curve_exposure(
        geometry,
        config.deployment.environment_entry_z_m,
        max_arc_spacing_m=config.sampling.max_arc_spacing_m,
        tolerance_m=config.deployment.geometry_tolerance_m,
    )
    if exposed.status != ExposureStatus.EXPOSED or exposed.exposed_length_m == 0:
        raise ValueError(f"Final IK curve does not define a reference path: {exposed.diagnostic}")
    boundaries = material_boundaries(geometry)
    retained = boundaries[boundaries >= exposed.entry_material_arc_m] - exposed.entry_material_arc_m
    return ReferencePath(
        geometry,
        exposed.entry_material_arc_m,
        exposed.centerline_m,
        exposed.arc_lengths_m,
        exposed.section_indices,
        np.unique(np.r_[0.0, retained]),
        config.target.position_m.copy(),
        tip_error,
    )


def interpolate_reference_path(path: ReferencePath, arc_lengths_m: FloatArray) -> FloatArray:
    """Exact PCC interpolation (M,2) at reference s (M,) from entry to IK tip, m.

    No extrapolation, deployment-distance substitution or nearest-point matching.
    The stored sample spacing does not affect interpolated geometry.
    """
    arcs = np.asarray(arc_lengths_m, dtype=float)
    if (
        arcs.ndim != 1
        or not np.isfinite(arcs).all()
        or np.any((arcs < 0) | (arcs > path.arc_lengths_m[-1]))
    ):
        raise ValueError("Reference s must be a finite vector in [0,reference_length]")
    material = np.clip(
        path.entry_material_arc_m + arcs,
        path.entry_material_arc_m,
        material_boundaries(path.geometry)[-1],
    )
    points = evaluate_curve(path.geometry, material)
    points[arcs == 0] = path.points_m[0]
    points[arcs == path.arc_lengths_m[-1]] = path.points_m[-1]
    return points


def load_reference_path(config: ProjectConfig, result_path: str | Path) -> ReferencePath:
    """Read one Stage 1 report accepted as collision-free in the same scene.

    Reject stale/missing provenance and unsuccessful/colliding reports. Static
    acceptance is inherited from Stage 1; Stage 2B never reruns its checker/solver.
    Re-export old reports with the Stage 1 CLI to attach geometry_context.
    """
    data = json.loads(Path(result_path).read_text(encoding="utf-8-sig"))
    if data.get("geometry_context") != geometry_context(config):
        raise ValueError(
            "Stage 1 report is stale or lacks geometry_context; re-export for this scene"
        )
    if "success" in data:
        if data["success"] is not True:
            raise ValueError("Stage 1 search report did not succeed")
        accepted = data.get("accepted_final_configuration")
    else:
        accepted = data
    if not isinstance(accepted, dict) or not (
        accepted.get("static_target_feasible") is True
        and accepted.get("collision_free_with_margin") is True
        and accepted.get("target_reached") is True
        and accepted.get("collision_status") == "clear"
    ):
        raise ValueError("A collision-free, target-reaching Stage 1 final result is required")
    return extract_reference_path(config, np.asarray(accepted["configuration_rad"], dtype=float))


def reference_path_report(path: ReferencePath) -> dict:
    """Serializable fixed reference including continuous PCC interpolation data."""
    geometry = path.geometry
    return {
        "coordinate_frame": "Original Stage 1 world YZ, metres",
        "arc_origin": "Environment entry; increases along final active centreline to IK tip",
        "points_m": path.points_m.tolist(),
        "arc_lengths_m": path.arc_lengths_m.tolist(),
        "section_boundary_arcs_m": path.section_boundary_arcs_m.tolist(),
        "sample_section_indices": path.section_indices.tolist(),
        "target_position_m": path.target_position_m.tolist(),
        "tip_error_m": path.tip_error_m,
        "entry_material_arc_m": path.entry_material_arc_m,
        "section_names": list(geometry.section_names),
        "section_lengths_m": geometry.section_lengths_m.tolist(),
        "junction_positions_m": geometry.junction_positions_m.tolist(),
        "junction_angles_rad": geometry.junction_angles_rad.tolist(),
        "interpolation": "Exact piecewise PCC by physical arc length; no transition-zone extension",
    }
