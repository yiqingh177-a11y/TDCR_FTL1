"""Signed-angle PCC in world YZ: local +Z straight, positive theta towards +Y.

Planar orientation beta is measured from +Z towards +Y. Thus the tip tangent
is [sin(beta), cos(beta)], and the rotation of YZ column vectors is
[[cos(beta), sin(beta)], [-sin(beta), cos(beta)]]. No phi parameter is used.
"""

import numpy as np

from .models import FloatArray, ForwardResult, PlanarRobotConfig, SectionCurve


def validate_planar_configuration(
    robot: PlanarRobotConfig, configuration: FloatArray
) -> FloatArray:
    """Validate finite signed q (n,) in rad, with abs(q_i) <= max_bend_i."""
    configuration = np.asarray(configuration, dtype=float)
    if configuration.shape != (robot.section_count,) or not np.isfinite(configuration).all():
        raise ValueError("2D configuration must be finite with shape (section_count,)")
    if np.any(np.abs(configuration) > [s.max_bend_rad for s in robot.sections]):
        raise ValueError("Signed section bend lies outside [-max_bend_rad, max_bend_rad]")
    return configuration


def planar_rotation(angle_rad: float) -> FloatArray:
    """Return (2,2) local-to-parent rotation for angle from +Z towards +Y."""
    if not np.isfinite(angle_rad):
        raise ValueError("Planar orientation must be finite")
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cosine, sine], [-sine, cosine]])


def _local_points(length_m: float, theta: float, arcs_m: FloatArray) -> FloatArray:
    angle = theta * arcs_m / length_m
    # Signed sinc form of (1-cos(k*s))/k and sin(k*s)/k, stable at k=0.
    return np.column_stack(
        (
            arcs_m * angle / 2 * np.sinc(angle / (2 * np.pi)) ** 2,
            arcs_m * np.sinc(angle / np.pi),
        )
    )


def section_points_2d(length_m: float, bend_rad: float, arc_lengths_m: FloatArray) -> FloatArray:
    """Exact local PCC points (M,2) at physical arc coordinates (M,), in metres.

    Origin is section start, neutral tangent +Z; positive bend is toward +Y.
    Reuses FK's stable signed formula, including straight/near-straight cases.
    """
    arcs = np.asarray(arc_lengths_m, dtype=float)
    if not np.isfinite(length_m) or length_m <= 0 or not np.isfinite(bend_rad):
        raise ValueError("Positive finite section length and finite bend required")
    if arcs.ndim != 1 or not np.isfinite(arcs).all() or np.any((arcs < 0) | (arcs > length_m)):
        raise ValueError("Section arc coordinates must be a finite vector in [0,length]")
    return _local_points(length_m, bend_rad, arcs)


def forward_kinematics_2d(
    robot: PlanarRobotConfig, configuration: FloatArray, max_arc_spacing_m: float
) -> ForwardResult:
    """Return exact planar tip, signed tangent angle, endpoints and YZ centreline.

    Input q is (n,) in rad. Global points (M,2) are uniformly sampled in physical
    arc length, with no duplicate boundaries. Per-section curves (K,2) additionally
    include exact boundaries for collision. Tip angle = base_angle + sum(q),
    unwrapped in rad. Distances are m. No IK or obstacle calculation occurs here.
    """
    configuration = validate_planar_configuration(robot, configuration)
    if not np.isfinite(max_arc_spacing_m) or max_arc_spacing_m <= 0:
        raise ValueError("max_arc_spacing_m must be positive and finite")
    lengths = np.array([s.length_m for s in robot.sections])
    boundaries = np.r_[0.0, np.cumsum(lengths)]
    total = boundaries[-1]
    arcs = np.linspace(0.0, total, max(1, int(np.ceil(total / max_arc_spacing_m))) + 1)
    centerline = np.empty((len(arcs), 2))
    position = np.asarray(robot.base_position_m, dtype=float).copy()
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("Planar base position must be finite world (y,z)")
    angle = robot.base_angle_rad
    endpoints, curves = [position.copy()], []
    for index, (section, theta) in enumerate(zip(robot.sections, configuration, strict=True)):
        rotation = planar_rotation(angle)
        start, end = boundaries[index : index + 2]
        mask = (arcs >= start) & (arcs <= end if index == robot.section_count - 1 else arcs < end)
        local_arcs = np.clip(arcs[mask] - start, 0.0, section.length_m)
        centerline[mask] = (
            _local_points(section.length_m, theta, local_arcs) @ rotation.T + position
        )
        count = max(
            1, int(np.ceil(section.length_m / max_arc_spacing_m)), int(np.ceil(abs(theta) / np.pi))
        )
        section_arcs = np.linspace(0.0, section.length_m, count + 1)
        points = _local_points(section.length_m, theta, section_arcs) @ rotation.T + position
        deviation = (
            section.length_m
            * abs(theta)
            / (8 * count**2)
            * np.sinc(theta / (4 * count * np.pi)) ** 2
        )
        curves.append(SectionCurve(points, section_arcs, float(deviation)))
        position = points[-1].copy()
        angle += float(theta)
        endpoints.append(position.copy())
    return ForwardResult(
        position, float(angle), np.array(endpoints), centerline, arcs, tuple(curves)
    )


def tip_pose_2d(robot: PlanarRobotConfig, configuration: FloatArray) -> tuple[FloatArray, float]:
    """Exact world YZ tip (2,) and tangent angle (rad), without curve sampling.

    Reuses full FK's signed local PCC formula, section ordering and base pose.
    Input configuration is (n,), rad; positions and lengths are m.
    """
    configuration = validate_planar_configuration(robot, configuration)
    position = np.asarray(robot.base_position_m, dtype=float).copy()
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("Planar base position must be finite world (y,z)")
    angle = robot.base_angle_rad
    for section, theta in zip(robot.sections, configuration, strict=True):
        local_tip = _local_points(section.length_m, theta, np.array([section.length_m]))[0]
        position += planar_rotation(angle) @ local_tip
        angle += float(theta)
    return position, float(angle)
