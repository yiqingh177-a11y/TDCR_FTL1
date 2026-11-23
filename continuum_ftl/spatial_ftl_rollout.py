"""Spatial Stage 2C sequencing and state ownership; planar solver is never called."""

from math import fsum
from time import perf_counter

import numpy as np

from .ftl_matching import comparison_path
from .models import (
    FTLConfig,
    FTLResult,
    FTLStep,
    ProjectConfig,
    ReferencePath,
    RobotConfig,
    SpatialDeploymentState,
)
from .planar_geometry import material_boundaries, sample_interval
from .reference_path import interpolate_reference_path
from .spatial_deployment import compute_spatial_deployment
from .spatial_ftl_optimizer import solve_spatial_ftl_step
from .spatial_geometry import evaluate_spatial_curve, extract_spatial_exposure


def generate_spatial_deployment_states(config: ProjectConfig) -> tuple[SpatialDeploymentState, ...]:
    """Stage 2A manual feed/release ramps in XYZ, not FTL or path tracking."""
    settings = config.deployment
    thresholds = np.array(
        [
            fsum(s.length_m for s in config.robot.sections[i:])
            for i in range(config.robot.section_count)
        ]
    )
    uniform = np.linspace(
        0,
        settings.end_distance_m,
        max(1, int(np.ceil(settings.end_distance_m / settings.distance_step_m))) + 1,
    )
    events = np.r_[thresholds, thresholds + settings.bend_ramp_distances_m]
    distances = np.unique(
        np.r_[uniform, events[(events >= 0) & (events <= settings.end_distance_m)]]
    )
    states = []
    for distance in distances:
        phase = np.clip((distance - thresholds) / settings.bend_ramp_distances_m, 0, 1)
        bends = settings.bend_targets_rad * (phase**2 * (3 - 2 * phase))[:, None]
        states.append(
            compute_spatial_deployment(
                config.robot, settings, float(distance), bends, config.sampling.max_arc_spacing_m
            )
        )
    return tuple(states)


def run_spatial_ftl(config: ProjectConfig, reference: ReferencePath) -> FTLResult:
    """Straight approach, then true spatial SQP with two bends per active section.

    Maintains original desired path arc scheduling and actual tip history. No
    projection of robot states onto the path plane, no collisions, no IK search.
    A failed trial is stored and stops the run; it never enters accepted history.
    """
    if not isinstance(config.robot, RobotConfig) or config.deployment is None:
        raise ValueError("Spatial FTL requires 3D robot and deployment settings")
    ftl = config.ftl or FTLConfig()
    if ftl.max_reshape_attempts:
        raise ValueError("The optional reshaping experiment is currently 2D only")
    if config.environment.obstacles:
        raise ValueError("This initial spatial baseline requires environment.obstacles=[]")
    started = perf_counter()
    settings = config.deployment
    gap = settings.environment_entry_z_m - settings.feed_exit_m[2]
    entry = settings.feed_exit_m.copy()
    entry[2] = settings.environment_entry_z_m
    if np.linalg.norm(entry - reference.points_m[0]) > settings.geometry_tolerance_m:
        raise ValueError("Spatial reference start does not match the straight +Z entry")
    total = settings.passive_length_m + sum(s.length_m for s in config.robot.sections)
    if not 0 <= gap <= total:
        raise ValueError("Environment entry must be reachable by straight feed")
    distances = np.linspace(0, gap, max(1, int(np.ceil(gap / ftl.max_feed_increment_m))) + 1)
    steps = []
    for index, distance in enumerate(distances):
        frame_started = perf_counter()
        state = compute_spatial_deployment(
            config.robot,
            settings,
            float(distance),
            np.zeros((config.robot.section_count, 2)),
            config.sampling.max_arc_spacing_m,
        )
        exposure = extract_spatial_exposure(
            state.geometry,
            settings.environment_entry_z_m,
            ftl.validation_body_spacing_m,
            settings.geometry_tolerance_m,
        )
        history = (
            state.forward.tip_position_m[None, :].copy()
            if index == len(distances) - 1
            else np.empty((0, 3))
        )
        steps.append(
            FTLStep(
                index,
                "approach",
                state,
                exposure,
                float(distance - distances[index - 1]) if index else 0.0,
                np.zeros_like(state.applied_bends_rad),
                (),
                0.0,
                0.0,
                state.forward.tip_position_m.copy(),
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                None,
                None,
                None,
                "Straight +Z feed approach; no optimizer",
                0,
                perf_counter() - frame_started,
                True,
                None,
                history,
                np.empty((0, 3)),
            )
        )
    arcs = np.linspace(
        0,
        reference.arc_lengths_m[-1],
        max(1, int(np.ceil(reference.arc_lengths_m[-1] / ftl.comparison_spacing_m))) + 1,
    )
    planned = interpolate_reference_path(reference, arcs)

    def finish(success, reason, completed=False):
        return FTLResult(
            success,
            tuple(steps),
            reason,
            perf_counter() - started,
            reference_completed=completed,
            error_limits_enforced=ftl.enforce_error_limits,
            reference_progress_mode=ftl.reference_progress_mode,
            body_reference_mode=ftl.body_reference_mode,
            effective_ftl_settings=ftl,
        )

    for _ in range(ftl.max_steps):
        previous = steps[-1]
        old_progress = (
            previous.reference_progress_m
            if ftl.reference_progress_mode == "actual_tip"
            else previous.desired_reference_progress_m
        )
        progress = min(reference.arc_lengths_m[-1], old_progress + ftl.reference_step_m)
        target = interpolate_reference_path(reference, np.array([progress]))[0]
        if ftl.body_reference_mode == "tip_history":
            comparison = comparison_path(
                previous.historical_tip_path_m, target, ftl.comparison_spacing_m
            )
        else:
            geometry = previous.state.geometry
            start = max(previous.exposure.entry_material_arc_m, geometry.section_lengths_m[0])
            material = sample_interval(
                geometry, start, material_boundaries(geometry)[-1], ftl.comparison_spacing_m
            )
            comparison = comparison_path(
                evaluate_spatial_curve(geometry, material), target, ftl.comparison_spacing_m
            )
        step = solve_spatial_ftl_step(
            config, reference, previous, float(progress), comparison, planned
        )
        steps.append(step)
        if not step.accepted:
            return finish(False, step.failure_reason)
        if progress >= reference.arc_lengths_m[-1] - settings.geometry_tolerance_m:
            reached = (
                np.linalg.norm(step.state.forward.tip_position_m - config.target.position_m)
                <= config.target.tolerance_m
            )
            return finish(
                bool(reached),
                None
                if reached or not ftl.enforce_error_limits
                else "FINAL_TARGET_TOLERANCE_EXCEEDED",
                True,
            )
    return finish(False, "FTL_STEP_BUDGET_EXHAUSTED")
