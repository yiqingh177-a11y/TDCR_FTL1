"""Stage 2A fixed-exit geometry and a manual, NON-FTL bending schedule.

No optimization, path following, collisions, forces or passive bending.
Full physical PCC sections remain intact at both feed and environment planes.
"""

from dataclasses import replace
from math import fsum

import numpy as np

from .models import (
    DeploymentConfig,
    DeploymentState,
    FloatArray,
    ForwardResult,
    PlanarRobotConfig,
    ProjectConfig,
    SectionCurve,
)
from .planar_kinematics import forward_kinematics_2d, validate_planar_configuration


def release_thresholds(robot: PlanarRobotConfig) -> FloatArray:
    """Full-release distances (n,) m, in proximal-to-distal active order.

    Sum each section and all sections distal to it: C/B/A .08/.06/.04 gives
    [.18,.10,.04]. These are measured from feed exit, NOT environment entry.
    """
    if not isinstance(robot, PlanarRobotConfig):
        raise ValueError("Deployment requires the existing 2D planar robot")
    return np.array(
        [fsum(s.length_m for s in robot.sections[i:]) for i in range(robot.section_count)]
    )


def compute_deployment(
    robot: PlanarRobotConfig,
    settings: DeploymentConfig,
    distance_m: float,
    requested_bends_rad: FloatArray,
    max_arc_spacing_m: float,
) -> DeploymentState:
    """Entire chain at d (m) beyond fixed +Z feed exit; n requested bends in rad.

    Unreleased sections are forced straight. Passive has NO bending parameter.
    Proximal base is feed_exit + [0,d-total_length]. Active PCC starts after the
    straight passive line, using original FK. Stage 1 base pose is not the feed
    boundary. Full physical lengths are retained regardless of viewer clipping.
    Geometrically released arcs folding behind feed exit are explicitly warned.
    """
    if not isinstance(robot, PlanarRobotConfig):
        raise ValueError("Stage 2A is 2D only")
    requested = validate_planar_configuration(robot, requested_bends_rad)
    if not np.isfinite(max_arc_spacing_m) or max_arc_spacing_m <= 0:
        raise ValueError("Positive finite centreline spacing required")
    total_length = fsum([settings.passive_length_m, *(s.length_m for s in robot.sections)])
    if not np.isfinite(distance_m) or not 0 <= distance_m <= total_length:
        raise ValueError("d must lie in [0,total physical length]")
    thresholds = release_thresholds(robot)
    released = distance_m >= thresholds
    applied = np.where(released, requested, 0.0)
    proximal = settings.feed_exit_m + [0.0, distance_m - total_length]
    active_base = proximal + [0.0, settings.passive_length_m]
    active_robot = replace(robot, base_position_m=active_base, base_angle_rad=0.0)
    active = forward_kinematics_2d(active_robot, applied, max_arc_spacing_m)
    passive_arcs = np.linspace(
        0.0,
        settings.passive_length_m,
        max(1, int(np.ceil(settings.passive_length_m / max_arc_spacing_m))) + 1,
    )
    passive_points = proximal + np.column_stack((np.zeros_like(passive_arcs), passive_arcs))
    forward = ForwardResult(
        active.tip_position_m.copy(),
        active.tip_orientation,
        np.vstack((proximal, active.section_endpoints_m)),
        np.vstack((passive_points[:-1], active.centerline_m)),
        np.r_[passive_arcs[:-1], settings.passive_length_m + active.arc_lengths_m],
        (SectionCurve(passive_points, passive_arcs, 0.0), *active.section_curves),
    )
    warnings = []
    for index, curve in enumerate(active.section_curves):
        if (
            released[index]
            and curve.points_m[:, 1].min() < settings.feed_exit_m[1] - settings.geometry_tolerance_m
        ):
            warnings.append(
                f"Released section {settings.active_section_names[index]} folds behind feed exit; "
                "ideal straight-feed geometry is inconsistent here"
            )
    return DeploymentState(
        float(distance_m),
        proximal,
        active_base,
        requested.copy(),
        applied,
        released,
        thresholds,
        ("Passive", *settings.active_section_names),
        np.array([settings.passive_length_m, *(s.length_m for s in robot.sections)]),
        np.r_[0.0, 0.0, np.cumsum(applied)],
        forward,
        tuple(warnings),
    )


def manual_bending_schedule(
    robot: PlanarRobotConfig, settings: DeploymentConfig, distance_m: float
) -> FloatArray:
    """C1 smooth bounded ramps after release; signed q (n,), rad. NOT FTL or IK."""
    thresholds = release_thresholds(robot)
    if not np.isfinite(distance_m):
        raise ValueError("Deployment distance must be finite")
    phase = np.clip((distance_m - thresholds) / settings.bend_ramp_distances_m, 0.0, 1.0)
    return settings.bend_targets_rad * phase**2 * (3 - 2 * phase)


def generate_deployment_states(config: ProjectConfig) -> tuple[DeploymentState, ...]:
    """Precompute replay states, including exact release/ramp-end distances.

    Same physical lengths and geometry function at every d. No IK or collision
    checking is invoked. Exceptions propagate for inconsistent inputs.
    """
    if config.deployment is None:
        raise ValueError("Add a deployment block to the 2D JSON configuration")
    settings = config.deployment
    uniform = np.linspace(
        0.0,
        settings.end_distance_m,
        max(1, int(np.ceil(settings.end_distance_m / settings.distance_step_m))) + 1,
    )
    thresholds = release_thresholds(config.robot)
    events = np.r_[thresholds, thresholds + settings.bend_ramp_distances_m]
    distances = np.unique(
        np.r_[uniform, events[(events >= 0) & (events <= settings.end_distance_m)]]
    )
    return tuple(
        compute_deployment(
            config.robot,
            settings,
            float(d),
            manual_bending_schedule(config.robot, settings, float(d)),
            config.sampling.max_arc_spacing_m,
        )
        for d in distances
    )


def deployment_report(states: tuple[DeploymentState, ...]) -> dict:
    """Serializable physical geometry and releases; lengths m, angles rad."""
    if not states:
        raise ValueError("Nonempty deployment state sequence required")
    return {
        "stage": "2A: manual axial deployment, NOT FTL",
        "section_order": list(states[0].section_names),
        "section_lengths_m": states[0].section_lengths_m.tolist(),
        "total_physical_length_m": float(fsum(states[0].section_lengths_m)),
        "release_thresholds_m": states[0].release_thresholds_m.tolist(),
        "collision_checked": False,
        "state_count": len(states),
        "states": [
            {
                "d_m": s.distance_m,
                "proximal_position_m": s.proximal_position_m.tolist(),
                "active_base_position_m": s.active_base_position_m.tolist(),
                "applied_bends_rad": s.applied_bends_rad.tolist(),
                "released": s.released.tolist(),
                "tip_position_m": s.forward.tip_position_m.tolist(),
                "centreline_m": s.forward.centerline_m.tolist(),
                "arc_lengths_m": s.forward.arc_lengths_m.tolist(),
                "section_endpoints_m": s.forward.section_endpoints_m.tolist(),
                "boundary_tangent_angles_rad": s.boundary_tangent_angles_rad.tolist(),
                "consistency_warnings": list(s.consistency_warnings),
            }
            for s in states
        ],
    }
