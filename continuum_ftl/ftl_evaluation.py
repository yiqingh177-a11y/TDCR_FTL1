"""Common offline geometric metrics for FTL modes; no optimization or collisions."""

import numpy as np

from .environment_exposure import extract_environment_exposure
from .ftl_matching import comparison_path
from .models import (
    FloatArray,
    FTLFrameEvaluation,
    FTLResult,
    ProjectConfig,
    ReferencePath,
    SpatialDeploymentState,
)
from .reference_path import interpolate_reference_path
from .spatial_geometry import extract_spatial_exposure


def point_to_polyline_distances(points_m: FloatArray, path_m: FloatArray) -> FloatArray:
    """Unrestricted nearest-segment Euclidean distances (M,), YZ/XYZ metres.

    Both inputs have shape (M,d)/(K,d), d=2 or 3. A point or all-repeated path is
    valid. No material IDs, matching windows or monotonicity assumptions enter
    this independent metric. Chunks bound working memory, not search accuracy.
    """
    points, path = np.asarray(points_m, dtype=float), np.asarray(path_m, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] not in (2, 3)
        or path.ndim != 2
        or path.shape[1] != points.shape[1]
        or not len(path)
        or not np.isfinite(points).all()
        or not np.isfinite(path).all()
    ):
        raise ValueError("Finite world YZ points (M,2) and nonempty path (K,2) required")
    vectors = np.diff(path, axis=0)
    squared_lengths = np.sum(vectors**2, axis=1)
    valid = squared_lengths > 0
    if not np.any(valid):
        return np.linalg.norm(points - path[0], axis=1)
    starts, vectors, squared_lengths = path[:-1][valid], vectors[valid], squared_lengths[valid]
    distances = np.empty(len(points))
    for offset in range(0, len(points), 128):
        block = points[offset : offset + 128]
        fraction = np.clip(
            np.sum((block[:, None] - starts) * vectors, axis=2) / squared_lengths, 0, 1
        )
        projections = starts + fraction[:, :, None] * vectors
        distances[offset : offset + len(block)] = np.sqrt(
            np.min(np.sum((block[:, None] - projections) ** 2, axis=2), axis=1)
        )
    return distances


def evaluate_ftl_rollout(
    config: ProjectConfig, reference: ReferencePath, result: FTLResult
) -> tuple[FTLFrameEvaluation, ...]:
    """Evaluate accepted FTL frames identically for both body-reference modes.

    Body: exact PCC/exposure sampled at validation_body_spacing_m. Fixed path:
    exact PCC sampled at comparison_spacing_m. History: accepted FK tips joined
    by straight edges; history queries densify those edges at the same comparison
    spacing. Every metric is a directed point-to-polyline distance, in metres.
    Failed trials are excluded; their native diagnostics remain in ftl_report.
    These sampled metrics do not certify a continuous supremum or collisions.
    """
    if config.ftl is None or config.deployment is None:
        raise ValueError("FTL evaluation requires deployment and FTL settings")
    settings = config.ftl
    count = max(1, int(np.ceil(reference.arc_lengths_m[-1] / settings.comparison_spacing_m)))
    fixed = interpolate_reference_path(
        reference, np.linspace(0, reference.arc_lengths_m[-1], count + 1)
    )
    evaluations = []
    for step in result.steps:
        if step.phase not in ("FTL", "reshape") or not step.accepted:
            continue
        if isinstance(step.state, SpatialDeploymentState):
            exposure = extract_spatial_exposure(
                step.state.geometry,
                config.deployment.environment_entry_z_m,
                settings.validation_body_spacing_m,
                config.deployment.geometry_tolerance_m,
            )
        else:
            exposure = extract_environment_exposure(
                step.state,
                config.deployment.environment_entry_z_m,
                max_arc_spacing_m=settings.validation_body_spacing_m,
                tolerance_m=config.deployment.geometry_tolerance_m,
            )
        body = exposure.centerline_m[exposure.section_indices > 0]
        if not len(body):
            raise ValueError("Accepted FTL frame has no exposed active body samples")
        history = step.historical_tip_path_m
        dense_history = comparison_path(history[:-1], history[-1], settings.comparison_spacing_m)
        history_error = point_to_polyline_distances(dense_history, fixed)
        body_history = point_to_polyline_distances(body, history)
        body_fixed = point_to_polyline_distances(body, fixed)
        evaluations.append(
            FTLFrameEvaluation(
                step.step_index,
                step.reference_progress_m,
                step.desired_reference_progress_m,
                float(np.max(history_error)),
                float(np.mean(history_error)),
                float(np.max(body_history)),
                float(np.mean(body_history)),
                float(np.max(body_fixed)),
                float(np.mean(body_fixed)),
                step.body_following_error_max_m,
                step.tip_error_m,
            )
        )
    return tuple(evaluations)


def summarize_ftl_evaluations(evaluations: tuple[FTLFrameEvaluation, ...]) -> dict:
    """Separate final, worst-frame and unweighted mean-of-frame statistics, SI.

    Empty accepted sequences have no curve statistics, reported explicitly as
    null alongside the count; the rollout report retains the failure reason.
    """
    summary = {"accepted_evaluated_frames": len(evaluations)}
    for prefix in ("history_to_reference", "body_to_history", "body_to_reference"):
        maxima = [getattr(frame, prefix + "_max_m") for frame in evaluations]
        means = [getattr(frame, prefix + "_mean_m") for frame in evaluations]
        summary[prefix] = (
            {
                "maximum_over_accepted_frames_m": max(maxima),
                "mean_of_frame_maxima_m": float(np.mean(maxima)),
                "mean_of_frame_means_m": float(np.mean(means)),
                "final_maximum_m": maxima[-1],
                "final_mean_m": means[-1],
            }
            if maxima
            else None
        )
    return summary
