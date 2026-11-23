"""Matplotlib spatial replay of computed full-section PCC states, no optimization."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .models import (
    EnvironmentExposure,
    FTLStep,
    ProjectConfig,
    ReferencePath,
    SpatialDeploymentState,
)
from .spatial_reference import reference_planarity
from .sweep_visualization import CentrelineSweep

if TYPE_CHECKING:
    from matplotlib.animation import FuncAnimation


class SpatialViewer:
    """Orbitable world XYZ animation with replay/pause/previous/next controls."""

    def __init__(
        self,
        config: ProjectConfig,
        states: tuple[SpatialDeploymentState, ...],
        reference: ReferencePath | None = None,
        exposures: tuple[EnvironmentExposure, ...] | None = None,
        steps: tuple[FTLStep, ...] | None = None,
        failure_reason: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if (
            not states
            or (exposures is not None and len(exposures) != len(states))
            or (steps is not None and len(steps) != len(states))
        ):
            raise ValueError("Nonempty states and matching per-frame diagnostics required")
        self.config, self.states, self.reference = config, states, reference
        self.exposures, self.steps, self.failure_reason = exposures, steps, failure_reason
        self.planarity = (
            reference_planarity(reference.points_m, config.deployment.geometry_tolerance_m)
            if reference is not None
            else None
        )
        self.animation = None
        self.current_index, self.paused = 0, False
        self.fig = plt.figure(figsize=tuple(config.visualization.figure_size_inches))
        self.axes = self.fig.add_axes([0.22, 0.19, 0.76, 0.60], projection="3d")
        self.sweep = (
            CentrelineSweep(
                self.axes,
                tuple(e.centerline_m for e in exposures),
                tuple(s.accepted for s in steps),
                dimension=3,
            )
            if steps is not None and exposures is not None
            else None
        )
        self.fig.suptitle(
            "Spatial Stage 2C: incremental SQP FTL"
            if steps is not None
            else "Spatial Stage 2B: reference / actual exposure"
            if reference is not None
            else "Spatial Stage 2A: manual feed / full-section release"
        )
        self.status = self.fig.text(0.05, 0.94, "", va="top", family="monospace", fontsize=8)
        self.warning = self.fig.text(0.05, 0.135, "", color="#c62828", fontsize=8)
        self.fig.text(
            0.05,
            0.015,
            "World XYZ; no external obstacles or collisions. Mouse drag rotates view.",
            fontsize=8,
        )
        self.lines = [
            self.axes.plot([], [], [], color=color, linewidth=3, label=name)[0]
            for name, color in zip(
                states[0].section_names, config.deployment.section_colors, strict=True
            )
        ]
        self.exposed_lines = (
            [
                self.axes.plot([], [], [], color=color, linewidth=5)[0]
                for color in config.deployment.section_colors
            ]
            if exposures is not None
            else []
        )
        (self.boundaries,) = self.axes.plot([], [], [], "o", color="#444444", markersize=4)
        (self.tip,) = self.axes.plot(
            [], [], [], "o", color="#c62828", markersize=6, label="Actual tip"
        )
        (self.history,) = self.axes.plot(
            [],
            [],
            [],
            "--",
            color=config.ftl.history_color if config.ftl else "#158c91",
            linewidth=1.5,
            label="Actual tip history",
        )
        (self.waypoint,) = self.axes.plot(
            [], [], [], "x", color="#159947", markersize=8, label="Desired waypoint"
        )
        if reference is not None:
            self.axes.plot(
                *reference.points_m.T,
                "--",
                color=config.deployment.reference_path_color,
                linewidth=2,
                label="Fixed spatial reference",
            )
            self.axes.scatter(
                *reference.target_position_m,
                marker="*",
                color="#159947",
                s=70,
                label="Final target",
            )
        feed = config.deployment.feed_exit_m
        self.axes.scatter(
            *feed, marker="s", color=config.deployment.feed_color, s=30, label="Fixed feed exit"
        )
        center, size = config.deployment.feed_block_center_m, config.deployment.feed_block_size_m
        corners = (
            center
            + np.array([[x, y, z] for z in (-0.5, 0.5) for y in (-0.5, 0.5) for x in (-0.5, 0.5)])
            * size
        )
        faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
        self.axes.add_collection3d(
            Poly3DCollection(
                [corners[f] for f in faces],
                facecolor=config.deployment.feed_color,
                alpha=config.deployment.feed_opacity,
                edgecolor="none",
            )
        )
        # The entry plane is a visual coordinate aid, not an external obstacle.
        bounds = config.environment.workspace_bounds_m
        x, y = np.meshgrid(bounds[0], bounds[1])
        self.axes.plot_wireframe(
            x,
            y,
            np.full_like(x, config.deployment.environment_entry_z_m),
            color="#aaaaaa",
            alpha=0.5,
            linewidth=0.5,
        )
        points = np.vstack([s.forward.centerline_m for s in states])
        if reference is not None:
            points = np.vstack((points, reference.points_m))
        low = np.minimum(bounds[:, 0], points.min(axis=0)) - config.deployment.plot_padding_m
        high = np.maximum(bounds[:, 1], points.max(axis=0)) + config.deployment.plot_padding_m
        self.axes.set(
            xlim=tuple([low[0], high[0]]),
            ylim=tuple([low[1], high[1]]),
            zlim=tuple([low[2], high[2]]),
            xlabel="World X (m)",
            ylabel="World Y (m)",
            zlabel="World Z (m)",
        )
        self.axes.set_box_aspect(high - low)
        self.axes.view_init(config.visualization.elevation_deg, config.visualization.azimuth_deg)
        self.fig.legend(
            *self.axes.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.005, 0.77),
            fontsize=7,
        )
        self.replay_button = Button(self.fig.add_axes([0.10, 0.055, 0.17, 0.05]), "Replay")
        self.pause_button = Button(self.fig.add_axes([0.31, 0.055, 0.17, 0.05]), "Pause")
        self.previous_button = Button(self.fig.add_axes([0.52, 0.055, 0.17, 0.05]), "Previous")
        self.next_button = Button(self.fig.add_axes([0.73, 0.055, 0.17, 0.05]), "Next")
        self.replay_button.on_clicked(self.replay)
        self.pause_button.on_clicked(self.toggle_pause)
        self.previous_button.on_clicked(lambda event: self.step_frame(-1))
        self.next_button.on_clicked(lambda event: self.step_frame(1))
        self.draw_frame(0)

    def draw_frame(self, index: int) -> tuple:
        """Draw actual precomputed XYZ geometry; do not project it onto the reference plane."""
        state = self.states[index]
        self.current_index = index
        if self.sweep is not None:
            self.sweep.draw_frame(index)
        for line, curve in zip(self.lines, state.forward.section_curves, strict=True):
            line.set_data_3d(*curve.points_m.T)
        self.boundaries.set_data_3d(*state.forward.section_endpoints_m.T)
        self.tip.set_data_3d(*state.forward.tip_position_m[:, None])
        flags = " | ".join(
            f"{n}:{'free' if f else 'locked'} {np.degrees(np.linalg.norm(b)):.1f}deg"
            for n, f, b in zip(
                state.section_names[1:], state.released, state.applied_bends_rad, strict=True
            )
        )
        text = (
            f"Frame {index + 1}/{len(self.states)} | d={state.distance_m * 1000:.2f} mm\n{flags}\n"
            f"Tip XYZ (m): {np.array2string(state.forward.tip_position_m, precision=5)}\n"
            f"Local bends (rad): {np.array2string(state.applied_bends_rad.ravel(), precision=3)}\n"
        )
        if self.exposures is not None:
            exposure = self.exposures[index]
            bounds = np.r_[0.0, np.cumsum(state.section_lengths_m)]
            for i, line in enumerate(self.exposed_lines):
                mask = (exposure.material_arc_lengths_m >= bounds[i]) & (
                    exposure.material_arc_lengths_m <= bounds[i + 1]
                )
                line.set_data_3d(*exposure.centerline_m[mask].T)
            text += f"Exposure: {exposure.status.value}"
            if exposure.exposed_length_m is not None:
                text += f" | active+passive exposed={exposure.exposed_length_m * 1000:.2f} mm"
        failed = None
        if self.steps is not None:
            step = self.steps[index]
            self.history.set_data_3d(*step.historical_tip_path_m.T)
            self.waypoint.set_data_3d(*step.desired_tip_position_m[:, None])
            body = (
                "N/A"
                if step.body_following_error_max_m is None
                else f"{step.body_following_error_max_m * 1000:.4g}"
            )
            text += (
                f"\nPhase={step.phase} | solver={step.solver_status} | "
                f"{'ACCEPTED' if step.accepted else 'FAILED'} | "
                f"desired s={step.desired_reference_progress_m * 1000:.2f} mm\n"
                f"Tip error={step.tip_error_m * 1000:.4g} mm | max body error={body} mm"
            )
            if self.planarity is not None and self.planarity["is_planar"]:
                angle = np.deg2rad(self.config.spatial_reference.plane_rotation_deg)
                normal = np.array([np.cos(angle), np.sin(angle), 0.0])
                origin = self.reference.points_m[0]
                if (
                    np.max(np.abs((self.reference.points_m - origin) @ normal))
                    > self.planarity["tolerance_m"]
                ):
                    normal = np.array(self.planarity["best_fit_plane_normal"])
                    origin = np.array(self.planarity["best_fit_plane_origin_m"])
                off_plane = np.max(np.abs((state.forward.centerline_m - origin) @ normal))
                text += (
                    f"\nMax distance from reference plane={off_plane * 1000:.4g} mm (unconstrained)"
                )
            elif self.planarity is not None:
                departure_mm = self.planarity["maximum_best_fit_plane_departure_m"] * 1000
                text += f"\nReference: nonplanar (best-fit plane departure={departure_mm:.3f} mm)"
            failed = step.failure_reason if not step.accepted else None
        self.status.set_text(text)
        self.status.set_color("#c62828" if failed else "#222222")
        warning = failed or (self.failure_reason if index == len(self.states) - 1 else None)
        self.warning.set_text("\n".join(textwrap.wrap(warning or "", width=115)))
        return (
            *self.lines,
            *self.exposed_lines,
            self.boundaries,
            self.tip,
            self.history,
            self.waypoint,
            self.status,
            self.warning,
        )

    def animate(self) -> FuncAnimation:
        """Create a replay animation without resimulating any state."""
        from matplotlib.animation import FuncAnimation

        if self.animation is None or self.animation.event_source is None:
            self.animation = FuncAnimation(
                self.fig,
                self.draw_frame,
                frames=len(self.states),
                init_func=lambda: self.draw_frame(0),
                interval=self.config.deployment.frame_interval_ms,
                repeat=False,
                blit=False,
            )
        return self.animation

    def replay(self, event: object = None) -> None:
        """Restart saved frames, retaining the current mouse-controlled camera."""
        animation = self.animate()
        animation.frame_seq = animation.new_frame_seq()
        self.paused = False
        self.pause_button.label.set_text("Pause")
        self.draw_frame(0)
        animation.event_source.start()
        self.fig.canvas.draw_idle()

    def step_frame(self, offset: int) -> None:
        """Pause and inspect an adjacent computed state."""
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        self.paused = True
        self.pause_button.label.set_text("Resume")
        self.draw_frame(max(0, min(len(self.states) - 1, self.current_index + offset)))
        if self.animation is not None:
            self.animation.frame_seq = iter(range(self.current_index + 1, len(self.states)))
        self.fig.canvas.draw_idle()

    def toggle_pause(self, event: object = None) -> None:
        """Pause/resume replay; neither reference progress nor robot states are recomputed."""
        animation = self.animate()
        self.paused = not self.paused
        if self.paused:
            animation.event_source.stop()
        else:
            animation.event_source.start()
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def save_gif(self, path: str | Path) -> None:
        """Export the same computed-state replay using Pillow."""
        from matplotlib.animation import PillowWriter

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.animate().save(
            str(path), writer=PillowWriter(fps=1000 / self.config.deployment.frame_interval_ms)
        )

    def run(self) -> None:
        """Open the interactive orbitable 3D animation."""
        import matplotlib.pyplot as plt

        self.animate()
        plt.show()

    def close(self) -> None:
        """Close this viewer only."""
        import matplotlib.pyplot as plt

        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        plt.close(self.fig)
