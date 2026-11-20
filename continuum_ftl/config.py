"""Load and validate the one JSON interface without importing GUI dependencies."""

import json
import re
from pathlib import Path

import numpy as np

from .models import (
    BoxObstacle,
    CircleObstacle,
    EnvironmentConfig,
    FloatArray,
    IKSearchConfig,
    PlanarRobotConfig,
    PlanarVisualizationConfig,
    ProjectConfig,
    RectangleObstacle,
    RobotConfig,
    SamplingConfig,
    SectionConfig,
    SolverConfig,
    SphereObstacle,
    TargetConfig,
    VisualizationConfig,
)


def _array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    if not allow_zero and value == 0:
        raise ValueError(f"{name} must be positive")


def _integer(value: int, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def load_config(path: str | Path) -> ProjectConfig:
    """Read SI-unit JSON, validate dimensions/settings, return immutable configuration.

    Arrays are read-only. Raises ValueError for malformed settings, JSON, or
    unknown fields, and OSError for file access failures. Bounds only set the plot.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return _parse_config(data)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid configuration {path}: {exc}") from exc


def _parse_config(data: dict) -> ProjectConfig:
    data = dict(data)
    dimension = data.pop("dimension", 3)
    raw_search = data.pop("ik_search", None)
    if type(dimension) is not int or dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3; omitted means legacy 3D")
    if set(data) != {
        "description",
        "robot",
        "target",
        "environment",
        "solver",
        "sampling",
        "visualization",
    }:
        raise ValueError(
            "Expected description, robot, target, environment, solver, sampling, visualization"
        )
    robot = _parse_robot(data["robot"], dimension)
    raw_target = dict(data["target"])
    raw_target["position_m"] = _array(raw_target["position_m"], (dimension,), "target.position_m")
    target = TargetConfig(**raw_target)
    _positive(target.tolerance_m, "target.tolerance_m")
    environment = _parse_environment(data["environment"], dimension)
    solver = SolverConfig(**data["solver"])
    _integer(solver.max_iterations, "max_iterations")
    _positive(solver.objective_tolerance, "objective_tolerance")
    _positive(solver.bend_limit_tolerance_rad, "bend_limit_tolerance_rad", allow_zero=True)
    methods = ("SLSQP", "L-BFGS-B") if dimension == 2 else ("SLSQP",)
    if solver.method not in methods:
        raise ValueError(f"{dimension}D solver.method must be one of {methods}")
    sampling = SamplingConfig(**data["sampling"])
    _positive(sampling.max_arc_spacing_m, "max_arc_spacing_m")
    visualization = _parse_visualization(data["visualization"], dimension)
    search = _parse_search(raw_search, robot, dimension) if raw_search is not None else None
    if not isinstance(data["description"], str):
        raise ValueError("description must be a string")
    return ProjectConfig(
        data["description"],
        robot,
        target,
        environment,
        solver,
        sampling,
        visualization,
        search,
    )


def _parse_search(
    data: dict, robot: RobotConfig | PlanarRobotConfig, dimension: int
) -> IKSearchConfig:
    data = dict(data)
    reference = data.get("reference_configuration_rad")
    if reference is not None:
        shape = (robot.section_count,) if dimension == 2 else (robot.section_count, 2)
        reference = _array(reference, shape, "ik_search.reference_configuration_rad")
        limits = np.array([s.max_bend_rad for s in robot.sections])
        angles = np.abs(reference) if dimension == 2 else reference[:, 0]
        if np.any(angles < 0) or np.any(angles > limits):
            raise ValueError("Reference configuration exceeds section bend limits")
        data["reference_configuration_rad"] = reference
    return IKSearchConfig(**data)


def _parse_robot(data: dict, dimension: int) -> RobotConfig | PlanarRobotConfig:
    raw_robot = dict(data)
    sections = tuple(SectionConfig(**item) for item in raw_robot.pop("sections"))
    if not sections:
        raise ValueError("At least one section is required")
    for section in sections:
        for name in ("length_m", "radius_m", "max_bend_rad"):
            _positive(getattr(section, name), f"section.{name}")
    raw_robot["base_position_m"] = _array(
        raw_robot["base_position_m"], (dimension,), "base_position_m"
    )
    if dimension == 2:
        if not np.isfinite(raw_robot["base_angle_rad"]):
            raise ValueError("base_angle_rad must be finite")
        raw_robot["initial_configuration_rad"] = _array(
            raw_robot["initial_configuration_rad"], (len(sections),), "initial_configuration_rad"
        )
        robot = PlanarRobotConfig(sections=sections, **raw_robot)
        if np.any(np.abs(robot.initial_configuration_rad) > [s.max_bend_rad for s in sections]):
            raise ValueError("Initial signed bends must be within [-max_bend_rad, max_bend_rad]")
        return robot
    raw_robot["base_rpy_rad"] = _array(raw_robot["base_rpy_rad"], (3,), "base_rpy_rad")
    raw_robot["initial_configuration_rad"] = _array(
        raw_robot["initial_configuration_rad"], (len(sections), 2), "initial_configuration_rad"
    )
    robot = RobotConfig(sections=sections, **raw_robot)
    theta = robot.initial_configuration_rad[:, 0]
    if np.any(theta < 0) or np.any(theta > [s.max_bend_rad for s in sections]):
        raise ValueError("Initial theta must be within each section's [0, max_bend_rad]")
    return robot


def _parse_environment(data: dict, dimension: int) -> EnvironmentConfig:
    raw_environment = dict(data)
    obstacles = []
    for item in raw_environment.pop("obstacles"):
        item = dict(item)
        kind = item.pop("kind")
        if kind != "rectangle":
            item["center_m"] = _array(item["center_m"], (dimension,), "obstacle.center_m")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ValueError("Obstacle names must be nonempty strings")
        if (dimension, kind) in ((3, "sphere"), (2, "circle")):
            obstacle = SphereObstacle(**item) if dimension == 3 else CircleObstacle(**item)
            _positive(obstacle.radius_m, "sphere.radius_m")
        elif dimension == 3 and kind == "box":
            item["size_m"] = _array(item["size_m"], (3,), "box.size_m")
            if np.any(item["size_m"] <= 0):
                raise ValueError("Box full sizes must be positive")
            obstacle = BoxObstacle(**item)
        elif dimension == 2 and kind == "rectangle":
            item["min_m"] = _array(item["min_m"], (2,), "rectangle.min_m")
            item["max_m"] = _array(item["max_m"], (2,), "rectangle.max_m")
            if np.any(item["min_m"] >= item["max_m"]):
                raise ValueError("Rectangle min_m must be strictly below max_m")
            obstacle = RectangleObstacle(**item)
        else:
            raise ValueError(f"Unsupported obstacle kind: {kind}")
        obstacles.append(obstacle)
    if len({o.name for o in obstacles}) != len(obstacles):
        raise ValueError("Obstacle names must be unique")
    bounds = _array(raw_environment["workspace_bounds_m"], (dimension, 2), "workspace_bounds_m")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Workspace lower bounds must be below upper bounds")
    raw_environment["workspace_bounds_m"] = bounds
    environment = EnvironmentConfig(obstacles=tuple(obstacles), **raw_environment)
    _positive(environment.safety_margin_m, "safety_margin_m", allow_zero=True)
    return environment


def _parse_visualization(
    data: dict, dimension: int
) -> VisualizationConfig | PlanarVisualizationConfig:
    raw_visualization = dict(data)
    raw_visualization["figure_size_inches"] = _array(
        raw_visualization["figure_size_inches"], (2,), "figure_size_inches"
    )
    raw_visualization["section_colors"] = tuple(raw_visualization["section_colors"])
    visualization = (
        PlanarVisualizationConfig(**raw_visualization)
        if dimension == 2
        else VisualizationConfig(**raw_visualization)
    )
    if np.any(visualization.figure_size_inches <= 0):
        raise ValueError("Figure dimensions must be positive")
    colors = (
        *visualization.section_colors,
        visualization.target_color,
        visualization.obstacle_color,
    )
    if not visualization.section_colors or any(
        not isinstance(c, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", c) is None for c in colors
    ):
        raise ValueError("Colors must use #RRGGBB; a nonempty section palette is required")
    if dimension == 3:
        if not np.isfinite([visualization.elevation_deg, visualization.azimuth_deg]).all():
            raise ValueError("Finite view angles required")
        _integer(visualization.surface_samples, "surface_samples", 4)
    elif not np.isfinite(visualization.body_alpha) or not 0 <= visualization.body_alpha <= 1:
        raise ValueError("body_alpha must be in [0,1]")
    if not np.isfinite(visualization.obstacle_alpha) or not 0 <= visualization.obstacle_alpha <= 1:
        raise ValueError("obstacle_alpha must be in [0,1]")
    _positive(visualization.centerline_width, "centerline_width")
    _positive(visualization.base_axis_length_m, "base_axis_length_m")
    return visualization
