"""Palmer S3 geometry in SI units, signed bends, proximal-to-distal order."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import PlanarCurveGeometry, PlanarRobotConfig, SectionConfig
from ..planar_geometry import evaluate_curve, sample_interval
from ..planar_kinematics import forward_kinematics_2d


@dataclass(frozen=True)
class S3Settings:
    """Published geometry/control bounds plus explicit numerical reconstruction settings."""

    section_length_m: float
    reference_bends_rad: list[float]
    max_bend_rad: float
    desired_path_step_m: float
    min_feed_increment_m: float
    max_feed_increment_m: float
    max_bend_increment_rad: float
    tip_weight: float
    optimization_spacing_m: float
    comparison_spacing_m: float
    validation_spacing_m: float
    matching_window_m: float
    objective_scale_m: float
    max_iterations: int
    solver_tolerance: float
    finite_difference_step: float
    acceptance_tolerance: float
    geometry_tolerance_m: float
    frame_interval_ms: int


def load_settings(path: Path) -> S3Settings:
    """Load an S3-only config; reject nonfinite, invalid bounds and non-four-section data."""
    settings = S3Settings(**json.loads(path.read_text(encoding="utf-8")))
    for name, value in vars(settings).items():
        if name == "reference_bends_rad":
            if np.shape(value) != (4,) or not np.isfinite(value).all():
                raise ValueError("S3 needs four finite reference bends")
        elif isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Finite numerical setting required: {name}")
        elif name != "min_feed_increment_m" and value <= 0:
            raise ValueError(f"Positive setting required: {name}")
    if not 0 <= settings.min_feed_increment_m < settings.max_feed_increment_m:
        raise ValueError("This initial reconstruction uses forward variable feed")
    if np.any(np.abs(settings.reference_bends_rad) > settings.max_bend_rad):
        raise ValueError("Reference exceeds absolute bend limits")
    for name in ("max_iterations", "frame_interval_ms"):
        if type(getattr(settings, name)) is not int:
            raise ValueError(f"Integer setting required: {name}")
    return settings


def geometry(settings: S3Settings, bends: np.ndarray, feed_m: float) -> PlanarCurveGeometry:
    """Full fixed-length four-section PCC, base Y=0, Z=feed-4L; world +Z feed.

    No passive section, environment zones, collision radius or original release
    rules. Radius is an unused required FK data field, not a paper-derived value.
    """
    robot = PlanarRobotConfig(
        tuple(
            SectionConfig(settings.section_length_m, 0.0, settings.max_bend_rad) for _ in range(4)
        ),
        np.array([0.0, feed_m - 4 * settings.section_length_m]),
        0.0,
        np.zeros(4),
    )
    forward = forward_kinematics_2d(robot, bends, settings.optimization_spacing_m)
    return PlanarCurveGeometry(
        ("1", "2", "3", "4"),
        np.full(4, settings.section_length_m),
        forward.section_endpoints_m,
        np.r_[0.0, np.cumsum(bends)],
    )


def reference_geometry(settings: S3Settings) -> PlanarCurveGeometry:
    """Fixed S3 at base (0,0); construction uses existing stable PCC FK."""
    return geometry(settings, np.array(settings.reference_bends_rad), 4 * settings.section_length_m)


def validate_reference(settings: S3Settings) -> dict:
    """Independent circle-integral junction check, independent of shared sinc FK."""
    position, angle = np.zeros(2), 0.0
    expected = [position.copy()]
    length = settings.section_length_m
    for bend in settings.reference_bends_rad:
        if bend == 0:
            displacement = length * np.array([np.sin(angle), np.cos(angle)])
        else:
            curvature = bend / length
            displacement = (
                np.array(
                    [np.cos(angle) - np.cos(angle + bend), np.sin(angle + bend) - np.sin(angle)]
                )
                / curvature
            )
        position += displacement
        angle += bend
        expected.append(position.copy())
    curve = reference_geometry(settings)
    residual = float(np.max(np.abs(curve.junction_positions_m - expected)))
    if residual > 1e-12:
        raise ArithmeticError("Independent S3 circle-integral FK check failed")
    arcs = sample_interval(curve, 0.0, 4 * length, settings.validation_spacing_m)
    return {
        "section_endpoints_yz_m": curve.junction_positions_m.tolist(),
        "independent_circle_integral_residual_m": residual,
        "arc_length_m": 4 * length,
        "sample_count": len(evaluate_curve(curve, arcs)),
        "sampling_contains_all_section_boundaries": all(
            np.any(arcs == x) for x in np.arange(5) * length
        ),
    }
