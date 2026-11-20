"""Matplotlib replay of stored Stage 2A geometry. No FK, IK or collisions here."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import (
    CircleObstacle,
    DeploymentState,
    EnvironmentExposure,
    ExposureStatus,
    ProjectConfig,
    ReferencePath,
)

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation


class DeploymentViewer:
    """Visible manual deployment with Replay/Pause, exact precomputed state frames."""

    def __init__(
        self,
        config: ProjectConfig,
        states: tuple[DeploymentState, ...],
        *,
        reference_path: ReferencePath | None = None,
        exposures: tuple[EnvironmentExposure, ...] | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.widgets import Button

        if config.deployment is None or not states:
            raise ValueError("Deployment settings and nonempty state sequence required")
        if (reference_path is None) != (exposures is None):
            raise ValueError(
                "Provide both a reference path and actual exposure sequence, or neither"
            )
        if exposures is not None and len(exposures) != len(states):
            raise ValueError("Exactly one exposure result per actual deployment state required")
        self.config, self.states = config, states
        self.reference_path, self.exposures = reference_path, exposures
        self.settings = config.deployment
        self.animation = None
        self.paused = False
        self.current_index = 0
        self.fig = plt.figure(figsize=tuple(config.visualization.figure_size_inches))
        self.axes = self.fig.add_axes([0.10, 0.18, 0.83, 0.62])
        self.fig.suptitle(
            "Stage 2B: deployment / exposure — NOT FTL"
            if exposures is not None
            else "Stage 2A: manual feed-in / release demo — NOT FTL"
        )
        self.status = self.fig.text(
            0.10,
            0.94,
            "",
            va="top",
            family="monospace",
            fontsize=9 if exposures is not None else 10,
        )
        self.warning = self.fig.text(0.10, 0.13, "", color="#c62828", fontsize=8)
        self.fig.text(
            0.10,
            0.015,
            "Obstacles/target: context only. Collision NOT checked. Passive may be clipped.",
            fontsize=8,
        )
        for obstacle in config.environment.obstacles:
            if isinstance(obstacle, CircleObstacle):
                patch = Circle(obstacle.center_m, obstacle.radius_m)
                center = obstacle.center_m
            else:
                patch = Rectangle(obstacle.min_m, *(obstacle.max_m - obstacle.min_m))
                center = (obstacle.min_m + obstacle.max_m) / 2
            patch.set(
                facecolor=config.visualization.obstacle_color,
                alpha=config.visualization.obstacle_alpha,
            )
            self.axes.add_patch(patch)
            self.axes.text(*center, obstacle.name, fontsize=8)
        self.axes.scatter(
            *config.target.position_m,
            marker="*",
            color=config.visualization.target_color,
            label="Stage 1 target\n(not tracked)"
            if exposures is not None
            else "Stage 1 target (not tracked)",
        )
        settings = self.settings
        self.reference_line = None
        if reference_path is not None:
            (self.reference_line,) = self.axes.plot(
                *reference_path.points_m.T,
                "--",
                color=settings.reference_path_color,
                linewidth=config.visualization.centerline_width,
                label="Fixed reference tip path\n(not followed)",
                zorder=2,
            )
        self.feed_block = Rectangle(
            settings.feed_block_center_m - settings.feed_block_size_m / 2,
            *settings.feed_block_size_m,
            facecolor=settings.feed_color,
            alpha=settings.feed_opacity,
            label="Feed block (visual only)"
            if exposures is not None
            else "Ideal feed block (visual only)",
        )
        self.axes.add_patch(self.feed_block)
        self.axes.axhline(
            settings.environment_entry_z_m,
            color="#666666",
            linestyle="--",
            label=f"Environment entry z={settings.environment_entry_z_m:g} m",
        )
        self.axes.scatter(
            *settings.feed_exit_m,
            marker="s",
            color=settings.feed_color,
            zorder=5,
            label=f"Fixed feed exit z={settings.feed_exit_m[1]:g} m",
        )
        self.lines = [
            self.axes.plot(
                [], [], color=color, linewidth=config.visualization.centerline_width + 1, label=name
            )[0]
            for name, color in zip(states[0].section_names, settings.section_colors, strict=True)
        ]
        self.exposed_lines = []
        if exposures is not None:
            self.exposed_lines = [
                self.axes.plot(
                    [],
                    [],
                    color=color,
                    linewidth=(config.visualization.centerline_width + 1)
                    * settings.exposed_line_width_factor,
                    label="Exposed body\n(thick, section colors)" if i == 0 else None,
                    solid_capstyle="round",
                    zorder=4,
                )[0]
                for i, color in enumerate(settings.section_colors)
            ]
        (self.boundaries,) = self.axes.plot([], [], "o", color="#333333", markersize=4, zorder=5)
        (self.tip,) = self.axes.plot(
            [], [], "o", color="#c62828", markersize=6, label="Tip", zorder=6
        )
        active_points = np.vstack(
            [curve.points_m for state in states for curve in state.forward.section_curves[1:]]
        )
        if reference_path is not None:
            active_points = np.vstack((active_points, reference_path.points_m))
        bounds = config.environment.workspace_bounds_m
        lower_z = min(bounds[1, 0], active_points[:, 1].min() - settings.visible_passive_length_m)
        upper_z = max(bounds[1, 1], active_points[:, 1].max())
        padding = settings.plot_padding_m
        self.axes.set(
            xlim=(
                min(bounds[0, 0], active_points[:, 0].min()) - padding,
                max(bounds[0, 1], active_points[:, 0].max()) + padding,
            ),
            ylim=(lower_z - padding, upper_z + padding),
            xlabel="World Y (m)",
            ylabel="World Z (m)",
        )
        self.axes.set_aspect("equal", adjustable="box")
        self.axes.grid(True, alpha=0.3)
        if exposures is not None:
            self.fig.legend(
                *self.axes.get_legend_handles_labels(),
                loc="upper left",
                bbox_to_anchor=(0.005, 0.80),
                fontsize=7,
            )
        else:
            self.axes.legend(loc="upper left", fontsize=7)
        self.replay_button = Button(self.fig.add_axes([0.25, 0.055, 0.2, 0.05]), "Replay")
        self.pause_button = Button(self.fig.add_axes([0.55, 0.055, 0.2, 0.05]), "Pause")
        self.replay_button.on_clicked(self.replay)
        self.pause_button.on_clicked(self.toggle_pause)
        self.draw_frame(0)

    def draw_frame(self, index: int) -> tuple:
        """Render exactly one stored state; index order is the deployment order."""
        state = self.states[index]
        self.current_index = index
        for line, curve in zip(self.lines, state.forward.section_curves, strict=True):
            line.set_data(curve.points_m[:, 0], curve.points_m[:, 1])
        self.boundaries.set_data(*state.forward.section_endpoints_m.T)
        self.tip.set_data([state.forward.tip_position_m[0]], [state.forward.tip_position_m[1]])
        flags = " | ".join(
            f"{name}: {'RELEASED' if flag else 'locked'} ({np.degrees(angle):+.1f} deg)"
            for name, flag, angle in reversed(
                list(
                    zip(
                        state.section_names[1:],
                        state.released,
                        state.applied_bends_rad,
                        strict=True,
                    )
                )
            )
        )
        text = (
            f"d = {state.distance_m * 1000:6.1f} mm | frame {index + 1}/{len(self.states)}\n"
            f"{flags}\nTip YZ = ({state.forward.tip_position_m[0]:+.4f}, "
            f"{state.forward.tip_position_m[1]:+.4f}) m | physical length = "
            f"{state.section_lengths_m.sum():.2f} m"
        )
        warnings = list(state.consistency_warnings)
        if self.exposures is not None:
            exposed = self.exposures[index]
            boundaries = np.r_[0.0, np.cumsum(state.section_lengths_m)]
            for i, line in enumerate(self.exposed_lines):
                # Include both ends of each physical section for connected drawing,
                # even though internal sample IDs use distal ownership.
                mask = (exposed.material_arc_lengths_m >= boundaries[i]) & (
                    exposed.material_arc_lengths_m <= boundaries[i + 1]
                )
                points = exposed.centerline_m[mask]
                line.set_data(points[:, 0], points[:, 1])
            if exposed.exposed_length_m is None:
                text += "\nExposure: AMBIGUOUS (no interval inferred)"
                warnings.append(exposed.diagnostic)
            else:
                lengths = " | ".join(
                    f"{name}={length * 1000:.1f}"
                    for name, length in zip(
                        exposed.section_names, exposed.section_exposed_lengths_m, strict=True
                    )
                )
                text += (
                    f"\nExposed length = {exposed.exposed_length_m * 1000:.1f} mm"
                    f" | tip entered: {exposed.tip_entered}\nExposed mm: {lengths}"
                )
            self.status.set_color(
                "#c62828" if exposed.status == ExposureStatus.AMBIGUOUS else "#222222"
            )
        self.status.set_text(text)
        self.warning.set_text("\n".join(warnings))
        return (
            *self.lines,
            *self.exposed_lines,
            self.boundaries,
            self.tip,
            self.status,
            self.warning,
        )

    def animate(self) -> FuncAnimation:
        """Create/retain a finite visible replay, using stored states only."""
        from matplotlib.animation import FuncAnimation

        if self.animation is None:
            self.animation = FuncAnimation(
                self.fig,
                self.draw_frame,
                frames=len(self.states),
                init_func=lambda: self.draw_frame(0),
                interval=self.settings.frame_interval_ms,
                repeat=False,
                blit=False,
            )
        return self.animation

    def replay(self, event: object = None) -> None:
        """Restart all stored deployment frames without recomputing geometry."""
        animation = self.animate()
        animation.frame_seq = animation.new_frame_seq()
        self.paused = False
        self.pause_button.label.set_text("Pause")
        self.draw_frame(0)
        animation.event_source.start()
        self.fig.canvas.draw_idle()

    def toggle_pause(self, event: object = None) -> None:
        """Pause/resume animation without changing release or geometry states."""
        animation = self.animate()
        self.paused = not self.paused
        if self.paused:
            animation.event_source.stop()
        else:
            animation.event_source.start()
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def save_gif(self, path: str | Path) -> None:
        """Save the same replay frames to GIF using Matplotlib's Pillow writer."""
        from matplotlib.animation import PillowWriter

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.animate().save(
            str(path), writer=PillowWriter(fps=1000 / self.settings.frame_interval_ms)
        )

    def run(self) -> None:
        """Open the visible animation; Replay and Pause remain available afterwards."""
        import matplotlib.pyplot as plt

        self.animate()
        plt.show()

    def close(self) -> None:
        """Stop playback and close this figure only."""
        import matplotlib.pyplot as plt

        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        plt.close(self.fig)
