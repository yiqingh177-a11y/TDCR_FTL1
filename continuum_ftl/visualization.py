"""Optional Matplotlib adapter. Core algorithms never import this module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .kinematics import base_transform
from .models import (
    BoxObstacle,
    CircleObstacle,
    EvaluationResult,
    FloatArray,
    PlanarRobotConfig,
    ProjectConfig,
)
from .planar_kinematics import planar_rotation

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _sphere_surface(center: FloatArray, radius: float, count: int) -> tuple:
    azimuth = np.linspace(0, 2 * np.pi, count)
    polar = np.linspace(0, np.pi, count)
    return (
        center[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
        center[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
        center[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)),
    )


def create_figure(config: ProjectConfig, result: EvaluationResult) -> Figure:
    """Create a 2D YZ or 3D XYZ final-shape figure according to robot configuration.

    Target wire sphere shows position tolerance. Centerlines do not depict robot
    radius; the numerical collision checker includes radius and arc inflation.
    Nothing is animated or treated as a verified path.
    """
    if isinstance(config.robot, PlanarRobotConfig):
        return _create_planar_figure(config, result)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    settings = config.visualization
    figure = plt.figure(figsize=tuple(settings.figure_size_inches), layout="constrained")
    axes = figure.add_subplot(111, projection="3d")
    for index, curve in enumerate(result.forward.section_curves):
        axes.plot(
            *curve.points_m.T,
            color=settings.section_colors[index % len(settings.section_colors)],
            linewidth=settings.centerline_width,
            label=f"Section {index + 1}",
        )
    endpoints = result.forward.section_endpoints_m
    axes.scatter(*endpoints.T, color=settings.section_colors[0], label="Section boundaries")
    axes.scatter(
        *result.forward.tip_position_m, color=settings.target_color, marker="x", label="Tip"
    )
    axes.scatter(*config.target.position_m, color=settings.target_color, marker="*", label="Target")
    axes.plot_wireframe(
        *_sphere_surface(
            config.target.position_m, config.target.tolerance_m, settings.surface_samples
        ),
        color=settings.target_color,
        alpha=settings.obstacle_alpha,
    )
    for obstacle in config.environment.obstacles:
        if isinstance(obstacle, BoxObstacle):
            signs = np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
            )
            corners = obstacle.center_m + signs * obstacle.size_m / 2
            faces = [
                [corners[i] for i in ids]
                for ids in [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (0, 1, 5, 4),
                    (1, 2, 6, 5),
                    (2, 3, 7, 6),
                    (3, 0, 4, 7),
                ]
            ]
            axes.add_collection3d(
                Poly3DCollection(
                    faces, facecolor=settings.obstacle_color, alpha=settings.obstacle_alpha
                )
            )
        else:
            axes.plot_surface(
                *_sphere_surface(obstacle.center_m, obstacle.radius_m, settings.surface_samples),
                color=settings.obstacle_color,
                alpha=settings.obstacle_alpha,
            )
        axes.text(*obstacle.center_m, obstacle.name)
    transform = base_transform(config.robot)
    for index, (name, color) in enumerate(zip(("X", "Y", "Z"), ("r", "g", "b"), strict=True)):
        vector = transform[:3, index] * settings.base_axis_length_m
        axes.quiver(*config.robot.base_position_m, *vector, color=color)
        axes.text(*(config.robot.base_position_m + vector), name, color=color)
    bounds = config.environment.workspace_bounds_m
    axes.set(
        xlim=bounds[0],
        ylim=bounds[1],
        zlim=bounds[2],
        xlabel="World X (m)",
        ylabel="World Y (m)",
        zlabel="World Z (m)",
    )
    axes.set_box_aspect(bounds[:, 1] - bounds[:, 0])
    axes.view_init(elev=settings.elevation_deg, azim=settings.azimuth_deg)
    clearance = result.collision.minimum_clearance_m
    clearance_text = "no obstacles" if clearance is None else f">= {clearance * 1000:.2f} mm"
    axes.set_title(
        f"Stage 1: final PCC shape only\n"
        f"IK: {result.ik.status.value}; error: {result.ik.position_error_m * 1000:.2f} mm\n"
        f"Collision: {result.collision.status.value}; clearance {clearance_text}\n"
        f"Static target feasible: {result.static_target_feasible}; motion NOT checked"
    )
    axes.legend(loc="upper left")
    return figure


def _create_planar_figure(config: ProjectConfig, result: EvaluationResult) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection, PolyCollection
    from matplotlib.patches import Circle, Rectangle

    settings = config.visualization
    figure, axes = plt.subplots(figsize=tuple(settings.figure_size_inches), layout="constrained")
    for index, (section, curve) in enumerate(
        zip(config.robot.sections, result.forward.section_curves, strict=True)
    ):
        color = settings.section_colors[index % len(settings.section_colors)]
        points = curve.points_m
        directions = np.diff(points, axis=0)
        lengths = np.linalg.norm(directions, axis=1)
        # Offsets and round end caps are expressed in metres, not screen linewidth.
        normals = np.column_stack((-directions[:, 1], directions[:, 0]))
        normals = normals / lengths[:, None] * section.radius_m
        quads = np.stack(
            (
                points[:-1] + normals,
                points[1:] + normals,
                points[1:] - normals,
                points[:-1] - normals,
            ),
            axis=1,
        )
        axes.add_collection(PolyCollection(quads, facecolors=color, alpha=settings.body_alpha))
        caps = [Circle(point, section.radius_m) for point in points]
        axes.add_collection(PatchCollection(caps, facecolors=color, alpha=settings.body_alpha))
        axes.plot(
            *points.T,
            color=color,
            linewidth=settings.centerline_width,
            label=f"Section {index + 1}",
        )
    axes.scatter(
        *result.forward.section_endpoints_m.T,
        color=settings.section_colors[0],
        label="Section boundaries",
        zorder=3,
    )
    axes.scatter(
        *config.robot.base_position_m, marker="s", color=settings.section_colors[0], label="Base"
    )
    axes.scatter(
        *config.target.position_m, marker="*", color=settings.target_color, label="Target", zorder=4
    )
    axes.scatter(
        *result.forward.tip_position_m,
        marker="x",
        color=settings.target_color,
        label="Tip",
        zorder=5,
    )
    axes.add_patch(
        Circle(
            config.target.position_m,
            config.target.tolerance_m,
            fill=False,
            edgecolor=settings.target_color,
        )
    )
    for obstacle in config.environment.obstacles:
        if isinstance(obstacle, CircleObstacle):
            patch = Circle(obstacle.center_m, obstacle.radius_m)
            center = obstacle.center_m
        else:
            patch = Rectangle(obstacle.min_m, *(obstacle.max_m - obstacle.min_m))
            center = (obstacle.min_m + obstacle.max_m) / 2
        patch.set(facecolor=settings.obstacle_color, alpha=settings.obstacle_alpha)
        axes.add_patch(patch)
        axes.text(*center, obstacle.name)
    rotation = planar_rotation(config.robot.base_angle_rad)
    for index, (name, color) in enumerate(zip(("Y", "Z"), ("g", "b"), strict=True)):
        end = config.robot.base_position_m + rotation[:, index] * settings.base_axis_length_m
        axes.annotate(
            "",
            xy=end,
            xytext=config.robot.base_position_m,
            arrowprops={"arrowstyle": "->", "color": color},
            color=color,
        )
        axes.text(*end, name, color=color)
    bounds = config.environment.workspace_bounds_m
    axes.set(xlim=bounds[0], ylim=bounds[1], xlabel="World y (m)", ylabel="World z (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True)
    clearance = result.collision.minimum_clearance_m
    clearance_text = "no obstacles" if clearance is None else f">= {clearance * 1000:.2f} mm"
    axes.set_title(
        f"Stage 1: planar YZ final shape only\n"
        f"IK: {result.ik.status.value}; error: {result.ik.position_error_m * 1000:.2f} mm\n"
        f"Collision: {result.collision.status.value}; clearance {clearance_text}\n"
        f"Static target feasible: {result.static_target_feasible}; motion NOT checked"
    )
    axes.legend(loc="best")
    return figure
