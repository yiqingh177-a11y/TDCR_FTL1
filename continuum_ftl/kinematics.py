"""Analytical PCC transforms and arc-length sampling, independent of rendering."""

import numpy as np

from .models import FloatArray, ForwardResult, RobotConfig, SectionCurve


def validate_configuration(robot: RobotConfig, configuration: FloatArray) -> FloatArray:
    """Check finite (n,2) theta/phi in rad and nonnegative bounded theta."""
    configuration = np.asarray(configuration, dtype=float)
    if configuration.shape != (robot.section_count, 2) or not np.isfinite(configuration).all():
        raise ValueError("Configuration must be finite with shape (section_count, 2)")
    theta = configuration[:, 0]
    if np.any(theta < 0) or np.any(theta > [s.max_bend_rad for s in robot.sections]):
        raise ValueError("Section theta lies outside [0, max_bend_rad]")
    return configuration


def base_transform(robot: RobotConfig) -> FloatArray:
    """Return world-from-base homogeneous transform (4,4), R=Rz @ Ry @ Rx."""
    roll, pitch, yaw = robot.base_rpy_rad
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    transform = np.eye(4)
    transform[:3, :3] = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    transform[:3, 3] = robot.base_position_m
    return transform


def local_pose(length_m: float, theta: float, phi: float, arc_m: float) -> FloatArray:
    """Base-from-cross-section transform (4,4) at arc_m in one PCC section.

    All lengths in m, angles in rad. phi=0 bends toward local +X. The section
    has no independent twist. sinc identities avoid zero-curvature cancellation.
    """
    if not np.isfinite([length_m, theta, phi, arc_m]).all() or length_m <= 0:
        raise ValueError("Finite pose inputs and positive section length required")
    if theta < 0 or not 0 <= arc_m <= length_m:
        raise ValueError("Require theta >= 0 and 0 <= arc_m <= length_m")
    angle = theta * arc_m / length_m
    axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    transform = np.eye(4)
    # 1-cos(angle) is evaluated as 2*sin(angle/2)^2 near zero.
    transform[:3, :3] += np.sin(angle) * skew + 2 * np.sin(angle / 2) ** 2 * (skew @ skew)
    radial = arc_m * angle / 2 * np.sinc(angle / (2 * np.pi)) ** 2
    transform[:3, 3] = [radial * np.cos(phi), radial * np.sin(phi), arc_m * np.sinc(angle / np.pi)]
    return transform


def forward_kinematics(
    robot: RobotConfig, configuration: FloatArray, max_arc_spacing_m: float
) -> ForwardResult:
    """Compute multi-section PCC FK and world centreline for q (n,2), radians.

    Uses T_world_tip = T_world_base @ T1 @ ... @ Tn. Global centreline spacing is
    uniform and <= max_arc_spacing_m. Collision curves additionally include every
    section boundary. Physical lengths are arc lengths, not chord-polyline lengths.
    """
    configuration = validate_configuration(robot, configuration)
    if not np.isfinite(max_arc_spacing_m) or max_arc_spacing_m <= 0:
        raise ValueError("max_arc_spacing_m must be positive and finite")
    lengths = np.array([s.length_m for s in robot.sections])
    boundaries = np.r_[0.0, np.cumsum(lengths)]
    total = boundaries[-1]
    arc_lengths = np.linspace(0.0, total, max(1, int(np.ceil(total / max_arc_spacing_m))) + 1)
    centerline = np.empty((len(arc_lengths), 3))
    transform = base_transform(robot)
    endpoints = [transform[:3, 3].copy()]
    curves = []
    for index, (section, (theta, phi)) in enumerate(
        zip(robot.sections, configuration, strict=True)
    ):
        start, end = boundaries[index : index + 2]
        mask = (arc_lengths >= start) & (
            arc_lengths <= end if index == robot.section_count - 1 else arc_lengths < end
        )
        for row in np.flatnonzero(mask):
            local_arc = float(np.clip(arc_lengths[row] - start, 0.0, section.length_m))
            centerline[row] = (transform @ local_pose(section.length_m, theta, phi, local_arc))[
                :3, 3
            ]
        count = max(
            1, int(np.ceil(section.length_m / max_arc_spacing_m)), int(np.ceil(theta / np.pi))
        )
        local_arcs = np.linspace(0.0, section.length_m, count + 1)
        points = np.array(
            [
                (transform @ local_pose(section.length_m, theta, phi, float(s)))[:3, 3]
                for s in local_arcs
            ]
        )
        # A circular subarc deviates from its chord by R*(1-cos(delta_angle/2)).
        # Inflate capsules by this bound so curved centreline geometry is covered.
        deviation = (
            section.length_m * theta / (8 * count**2) * np.sinc(theta / (4 * count * np.pi)) ** 2
        )
        curves.append(SectionCurve(points, local_arcs, float(deviation)))
        transform = transform @ local_pose(section.length_m, theta, phi, section.length_m)
        endpoints.append(transform[:3, 3].copy())
    return ForwardResult(
        transform[:3, 3].copy(),
        transform[:3, :3].copy(),
        np.array(endpoints),
        centerline,
        arc_lengths,
        tuple(curves),
    )


def tip_pose(robot: RobotConfig, configuration: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Exact world tip (3,) and rotation (3,3) for q (n,2), without sampling curves.

    Uses the same local_pose and base transform as full PCC FK. Lengths m,
    configuration angles rad. Intended for repeated position IK evaluations.
    """
    configuration = validate_configuration(robot, configuration)
    transform = base_transform(robot)
    for section, (theta, phi) in zip(robot.sections, configuration, strict=True):
        transform = transform @ local_pose(section.length_m, theta, phi, section.length_m)
    return transform[:3, 3].copy(), transform[:3, :3].copy()
