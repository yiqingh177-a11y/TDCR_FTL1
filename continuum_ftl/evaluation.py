"""Stage 1 orchestration: position IK, exact FK, then final-shape collision checks."""

from .collision import check_static_collision
from .inverse import solve_position_ik
from .kinematics import forward_kinematics
from .models import EvaluationResult, IKResult, PlanarRobotConfig, ProjectConfig
from .planar_kinematics import forward_kinematics_2d


def evaluate_target(config: ProjectConfig) -> EvaluationResult:
    """Evaluate one configured world target; no path or motion claim is made.

    A colliding target-reaching IK result is retained and explicitly reported.
    This single-start solver does not search for an alternative collision-free IK.
    """
    ik = solve_position_ik(
        config.robot, config.target, config.robot.initial_configuration_rad, config.solver
    )
    return evaluate_configuration(config, ik)


def evaluate_configuration(config: ProjectConfig, ik: IKResult) -> EvaluationResult:
    """Run full sampled FK and accurate static body collision for one IK result.

    No IK is rerun. The caller decides whether a tip-reaching check is required
    first. Configuration angles rad; world positions and clearances m.
    """
    forward_function = (
        forward_kinematics_2d if isinstance(config.robot, PlanarRobotConfig) else forward_kinematics
    )
    forward = forward_function(
        config.robot, ik.configuration_rad, config.sampling.max_arc_spacing_m
    )
    collision = check_static_collision(config.robot, forward, config.environment)
    return EvaluationResult(ik, forward, collision)


def result_report(result: EvaluationResult) -> dict:
    """Return a JSON-compatible diagnostic report; lengths m, angles rad, world poses."""
    orientation = result.forward.tip_orientation
    orientation_report = (
        {"tip_orientation_angle_rad": float(orientation)}
        if isinstance(orientation, float)
        else {"tip_orientation_matrix": orientation.tolist()}
    )
    return {
        "dimension": len(result.forward.tip_position_m),
        "position_coordinates": "yz" if len(result.forward.tip_position_m) == 2 else "xyz",
        "ik_status": result.ik.status.value,
        "target_reached": result.ik.target_reached,
        "configuration_rad": result.ik.configuration_rad.tolist(),
        "tip_position_m": result.forward.tip_position_m.tolist(),
        **orientation_report,
        "position_error_m": result.ik.position_error_m,
        "optimizer_success": result.ik.optimizer_success,
        "solver_message": result.ik.message,
        "solver_iterations": result.ik.iterations,
        "solver_function_evaluations": result.ik.function_evaluations,
        "collision_status": result.collision.status.value,
        "collision_free_with_margin": result.collision.collision_free,
        "minimum_clearance_lower_bound_m": result.collision.minimum_clearance_m,
        "contacts": [
            {
                "obstacle": c.obstacle_name,
                "section_index": c.section_index,
                "clearance_lower_bound_m": c.clearance_m,
            }
            for c in result.collision.contacts
        ],
        "static_target_feasible": result.static_target_feasible,
        "motion_checked": False,
        "self_collision_checked": False,
        "scope": "One final PCC shape only. Failure is not a global unreachability proof, "
        "except the explicit total-length exclusion. No collision-free navigation claim.",
    }
