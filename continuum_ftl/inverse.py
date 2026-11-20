"""Single-start position IK using SciPy SLSQP; no collision-aware optimization."""

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from .kinematics import tip_pose, validate_configuration
from .models import (
    FloatArray,
    IKResult,
    IKStatus,
    PlanarRobotConfig,
    RobotConfig,
    SolverConfig,
    TargetConfig,
)
from .planar_kinematics import tip_pose_2d, validate_planar_configuration


def configuration_to_bends(configuration: FloatArray) -> FloatArray:
    """Convert theta/phi (n,2) to flat integrated bends (2n,), in radians.

    Cartesian bend coordinates remove the undefined direction of a straight
    section during optimization. They are not independent rigid-joint rotations.
    """
    configuration = np.asarray(configuration, dtype=float)
    if (
        configuration.ndim != 2
        or configuration.shape[1] != 2
        or configuration.shape[0] == 0
        or not np.isfinite(configuration).all()
        or np.any(configuration[:, 0] < 0)
    ):
        raise ValueError("Configuration must be finite (n,2) with n > 0 and theta >= 0")
    theta, phi = configuration.T
    return np.column_stack((theta * np.cos(phi), theta * np.sin(phi))).ravel()


def bends_to_configuration(bends: FloatArray) -> FloatArray:
    """Convert finite flat integrated bends (2n,) to theta/phi (n,2), in radians."""
    bends = np.asarray(bends, dtype=float)
    if bends.ndim != 1 or bends.size == 0 or bends.size % 2 or not np.isfinite(bends).all():
        raise ValueError("Bends must be a finite nonempty vector with even length")
    pairs = bends.reshape(-1, 2)
    theta = np.linalg.norm(pairs, axis=1)
    phi = np.arctan2(pairs[:, 1], pairs[:, 0])
    return np.column_stack((theta, np.where(theta > 0, phi, 0.0)))


def position_objective(bends: FloatArray, robot: RobotConfig, target_m: FloatArray) -> float:
    """Half squared world tip error (m^2) for flat integrated bends (2n,).

    Trial states can exceed bending limits while SLSQP evaluates constraints.
    The objective uses the same exact FK, with relaxed temporary limits only
    for those trials. Final acceptance always uses the actual robot limits.
    """
    configuration = bends_to_configuration(bends)
    error = _trial_tip_position(robot, configuration) - target_m
    return float(0.5 * (error @ error))


def position_objective_2d(
    configuration: FloatArray, robot: PlanarRobotConfig, target_m: FloatArray
) -> float:
    """Half squared YZ tip error (m^2) for signed section bends (n,), in rad."""
    error = _trial_tip_position(robot, configuration) - target_m
    return float(0.5 * (error @ error))


def _trial_tip_position(
    robot: RobotConfig | PlanarRobotConfig, configuration: FloatArray
) -> FloatArray:
    # The exact same local PCC transforms are used, without building any curves.
    planar = isinstance(robot, PlanarRobotConfig)
    angles = np.abs(configuration) if planar else configuration[:, 0]
    sections = tuple(
        replace(section, max_bend_rad=max(section.max_bend_rad, float(theta)))
        for section, theta in zip(robot.sections, angles, strict=True)
    )
    forward_tip = tip_pose_2d if planar else tip_pose
    return forward_tip(replace(robot, sections=sections), configuration)[0]


def solve_position_ik(
    robot: RobotConfig | PlanarRobotConfig,
    target: TargetConfig,
    initial_configuration: FloatArray,
    settings: SolverConfig,
) -> IKResult:
    """Find one bounded PCC configuration for a world target (m), single start.

    Planar robots take signed q (n,) and YZ target (2,); spatial robots take
    theta/phi (n,2) and XYZ target (3,). Planar bounds support SLSQP or L-BFGS-B;
    spatial disk constraints use SLSQP. No obstacle penalties are included.
    Optimization is deterministic and uses no random sampling. A solver failure
    is not a global unreachability proof. Collision checking is a separate stage.
    """
    planar = isinstance(robot, PlanarRobotConfig)
    validate = validate_planar_configuration if planar else validate_configuration
    initial = validate(robot, initial_configuration)
    methods = ("SLSQP", "L-BFGS-B") if planar else ("SLSQP",)
    if settings.method not in methods:
        raise ValueError(f"Supported methods for this robot: {methods}")
    target_m = np.asarray(target.position_m, dtype=float)
    if (
        target_m.shape != ((2,) if planar else (3,))
        or not np.isfinite(target_m).all()
        or not np.isfinite(target.tolerance_m)
        or target.tolerance_m <= 0
    ):
        raise ValueError("Target must match robot dimension and have finite positive tolerance")
    limits = np.array([s.max_bend_rad for s in robot.sections])
    maximum_reach = sum(s.length_m for s in robot.sections)
    if np.linalg.norm(target_m - robot.base_position_m) > maximum_reach + target.tolerance_m:
        tip = _trial_tip_position(robot, initial)
        return IKResult(
            initial.copy(),
            tip,
            float(np.linalg.norm(tip - target_m)),
            IKStatus.OUTSIDE_LENGTH_ENVELOPE,
            False,
            "Target exceeds total backbone length plus target tolerance from fixed base.",
            0,
            0,
        )

    def bend_constraints(bends: FloatArray) -> FloatArray:
        # Normalized disk inequalities, one per section: theta_i <= theta_max_i.
        return 1.0 - np.sum((bends.reshape(-1, 2) / limits[:, None]) ** 2, axis=1)

    result = minimize(
        position_objective_2d if planar else position_objective,
        initial.copy() if planar else configuration_to_bends(initial),
        args=(robot, target_m),
        method=settings.method,
        jac="3-point",
        bounds=[(-limit, limit) for limit in (limits if planar else np.repeat(limits, 2))],
        constraints=() if planar else {"type": "ineq", "fun": bend_constraints},
        options={"maxiter": settings.max_iterations, "ftol": settings.objective_tolerance},
    )
    if not np.isfinite(result.x).all():
        tip = _trial_tip_position(robot, initial)
        return IKResult(
            initial.copy(),
            tip,
            float(np.linalg.norm(tip - target_m)),
            IKStatus.OPTIMIZATION_FAILED,
            False,
            f"Nonfinite optimizer output; returned initial configuration. {result.message}",
            int(result.nit),
            int(result.nfev),
        )
    configuration = np.array(result.x, copy=True) if planar else bends_to_configuration(result.x)
    angles = np.abs(configuration) if planar else configuration[:, 0]
    within_limits = bool(np.all(angles <= limits + settings.bend_limit_tolerance_rad))
    # Remove only roundoff overshoot before invoking strict public FK. Substantial
    # violations remain a failure even if the displayed diagnostic is projected.
    if planar:
        configuration = np.clip(configuration, -limits, limits)
    else:
        configuration[:, 0] = np.minimum(configuration[:, 0], limits)
    tip = _trial_tip_position(robot, configuration)
    error = float(np.linalg.norm(tip - target_m))
    if within_limits and error <= target.tolerance_m:
        status = IKStatus.TARGET_REACHED
    elif not result.success or not within_limits:
        status = IKStatus.OPTIMIZATION_FAILED
    else:
        status = IKStatus.TARGET_NOT_REACHED
    message = str(result.message)
    if not within_limits:
        message += " Bend constraints violated; diagnostic configuration projected to bounds."
    return IKResult(
        configuration,
        tip,
        error,
        status,
        bool(result.success),
        message,
        int(result.nit),
        int(result.nfev),
    )
