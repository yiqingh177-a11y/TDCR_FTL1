"""Bounded multi-start orchestration for collision-free FINAL PCC shapes only."""

from collections.abc import Iterator
from dataclasses import replace
from time import perf_counter

import numpy as np

from .evaluation import evaluate_configuration, result_report
from .inverse import configuration_to_bends, solve_position_ik
from .kinematics import validate_configuration
from .models import (
    FloatArray,
    IKSearchAttempt,
    IKSearchConfig,
    IKSearchResult,
    IKStatus,
    PlanarRobotConfig,
    ProjectConfig,
)
from .planar_kinematics import validate_planar_configuration


def _bend_vector(config: ProjectConfig, configuration: FloatArray) -> FloatArray:
    # In 3D, wrapping phi or changing phi at theta=0 must not create a new shape.
    if isinstance(config.robot, PlanarRobotConfig):
        return np.asarray(configuration, dtype=float).copy()
    return configuration_to_bends(configuration)


def _near_any(vector: FloatArray, previous: list[FloatArray], tolerance: float) -> bool:
    return any(np.linalg.norm(vector - other) <= tolerance for other in previous)


def _initial_guesses(
    config: ProjectConfig, settings: IKSearchConfig, reference: FloatArray | None
) -> Iterator[tuple[str, FloatArray]]:
    if reference is not None:
        yield "reference", reference.copy()
    yield "initial", config.robot.initial_configuration_rad.copy()
    rng = np.random.default_rng(settings.random_seed)
    limits = np.array([s.max_bend_rad for s in config.robot.sections])
    for _ in range(settings.max_seed_draws):
        if isinstance(config.robot, PlanarRobotConfig):
            guess = rng.uniform(-limits, limits)
        else:
            # Uniform area in each section's bend disk, not Cartesian workspace.
            theta = limits * np.sqrt(rng.random(config.robot.section_count))
            phi = rng.uniform(-np.pi, np.pi, config.robot.section_count)
            guess = np.column_stack((theta, phi))
        yield "random", guess


def find_collision_free_ik(
    config: ProjectConfig,
    reference_configuration: FloatArray | None = None,
    *,
    settings: IKSearchConfig | None = None,
) -> IKSearchResult:
    """Find one target-reaching, statically collision-free PCC shape in 2D or 3D.

    Reference is tried first, then configured initial q, then reproducible random
    seeds. Explicit settings/reference override config.ik_search values. Angles
    are rad: reference shape (n,) in 2D, (n,2) in 3D. No trajectories are checked.

    attempts counts actual calls to the existing single-run IK (including failed
    or duplicate solutions). Near-duplicate seeds skip IK; near-duplicate final
    solutions skip repeated body checks. Only target-reaching candidates receive
    accurate full-body collision checks. A successful result always includes one.
    Failure code is NO_COLLISION_FREE_IK_FOUND, never a global infeasibility claim.
    Invalid inputs raise ValueError; unexpected solver/programming errors propagate.
    """
    started = perf_counter()
    settings = settings if settings is not None else config.ik_search or IKSearchConfig()
    reference = (
        reference_configuration
        if reference_configuration is not None
        else settings.reference_configuration_rad
    )
    validate = (
        validate_planar_configuration
        if isinstance(config.robot, PlanarRobotConfig)
        else validate_configuration
    )
    validate(config.robot, config.robot.initial_configuration_rad)
    if reference is not None:
        reference = validate(config.robot, reference)

    tried_seeds: list[FloatArray] = []
    tried_candidates: list[FloatArray] = []
    history: list[IKSearchAttempt] = []
    diagnostic = None
    best_error = None
    duplicate_seeds = duplicate_candidates = 0

    def finish(success: bool, termination: str) -> IKSearchResult:
        return IKSearchResult(
            success=success,
            configuration=diagnostic.ik.configuration_rad.copy() if success else None,
            tip_error=diagnostic.ik.position_error_m if success else best_error,
            minimum_clearance=diagnostic.collision.minimum_clearance_m if success else None,
            attempts=len(history),
            computation_time=perf_counter() - started,
            failure_reason=None if success else "NO_COLLISION_FREE_IK_FOUND",
            termination=termination,
            history=tuple(history),
            diagnostic=diagnostic,
            skipped_duplicate_seeds=duplicate_seeds,
            skipped_duplicate_candidates=duplicate_candidates,
        )

    for source, seed in _initial_guesses(config, settings, reference):
        vector = _bend_vector(config, seed)
        if _near_any(vector, tried_seeds, settings.seed_distance_tolerance_rad):
            duplicate_seeds += 1
            continue
        tried_seeds.append(vector)
        ik = solve_position_ik(config.robot, config.target, seed, config.solver)
        best_error = (
            ik.position_error_m if best_error is None else min(best_error, ik.position_error_m)
        )
        collision_status = None
        duplicate = False
        evaluation = None
        if ik.target_reached and ik.position_error_m <= config.target.tolerance_m:
            candidate = _bend_vector(config, ik.configuration_rad)
            duplicate = _near_any(
                candidate, tried_candidates, settings.candidate_distance_tolerance_rad
            )
            if duplicate:
                duplicate_candidates += 1
            else:
                tried_candidates.append(candidate)
                evaluation = evaluate_configuration(config, ik)
                # Independently confirm the accepted body's FK tip, not just metadata.
                error = float(
                    np.linalg.norm(evaluation.forward.tip_position_m - config.target.position_m)
                )
                ik = replace(
                    ik,
                    tip_position_m=evaluation.forward.tip_position_m.copy(),
                    position_error_m=error,
                    status=IKStatus.TARGET_REACHED
                    if error <= config.target.tolerance_m
                    else IKStatus.TARGET_NOT_REACHED,
                )
                evaluation = replace(evaluation, ik=ik)
                collision_status = evaluation.collision.status
                if diagnostic is None or error < diagnostic.ik.position_error_m:
                    diagnostic = evaluation
        history.append(
            IKSearchAttempt(
                source,
                seed.copy(),
                ik.configuration_rad.copy(),
                ik.status,
                ik.position_error_m,
                collision_status,
                duplicate,
            )
        )
        if evaluation is not None and evaluation.static_target_feasible:
            diagnostic = evaluation
            return finish(True, "collision_free_solution_found")
        if len(history) >= settings.max_attempts:
            return finish(False, "attempt_budget_exhausted")
        if ik.status == IKStatus.OUTSIDE_LENGTH_ENVELOPE:
            return finish(False, "outside_length_envelope")
    return finish(False, "seed_draw_budget_exhausted")


def search_result_report(result: IKSearchResult) -> dict:
    """JSON-compatible search report; angles rad, lengths m, time s.

    A failed search's configuration/minimum_clearance stay null. Diagnostic
    failure shapes are separately labelled and never masquerade as accepted IK.
    """
    return {
        "success": result.success,
        "configuration": result.configuration.tolist()
        if result.configuration is not None
        else None,
        "tip_error": result.tip_error,
        "minimum_clearance": result.minimum_clearance,
        "attempts": result.attempts,
        "computation_time": result.computation_time,
        "failure_reason": result.failure_reason,
        "termination": result.termination,
        "units": {
            "configuration": "rad",
            "tip_error": "m",
            "minimum_clearance": "m",
            "computation_time": "s",
        },
        "skipped_duplicate_seeds": result.skipped_duplicate_seeds,
        "skipped_duplicate_candidates": result.skipped_duplicate_candidates,
        "attempt_history": [
            {
                "seed_source": a.seed_source,
                "seed_configuration_rad": a.seed_configuration_rad.tolist(),
                "configuration_rad": a.configuration_rad.tolist(),
                "ik_status": a.ik_status.value,
                "tip_error_m": a.tip_error_m,
                "collision_status": a.collision_status.value
                if a.collision_status is not None
                else None,
                "duplicate_candidate": a.duplicate_candidate,
            }
            for a in result.history
        ],
        "accepted_final_configuration": result_report(result.diagnostic)
        if result.success
        else None,
        "diagnostic_rejected_configuration": (
            result_report(result.diagnostic) if not result.success and result.diagnostic else None
        ),
        "motion_checked": False,
        "self_collision_checked": False,
        "scope": "Multi-start static final-shape search only. Search failure is not proof "
        "of mathematical unreachability. No collision-free navigation claim.",
    }
