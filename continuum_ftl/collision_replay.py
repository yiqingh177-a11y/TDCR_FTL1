"""Stage 3A collision overlays on stored states; no checks during playback."""

from pathlib import Path

import numpy as np

from .deployment_viewer import DeploymentViewer
from .models import CollisionStatus, CollisionTimeline, ProjectConfig


class CollisionReplay(DeploymentViewer):
    """Reuse replay controls; color the complete colliding section red, margin amber."""

    def __init__(self, config: ProjectConfig, timeline: CollisionTimeline) -> None:
        from matplotlib.widgets import Button

        self.timeline = timeline
        self.first_collision = next(
            (
                i
                for i, f in enumerate(timeline.frames)
                if f.collision.status == CollisionStatus.COLLISION
            ),
            None,
        )
        super().__init__(config, tuple(f.state for f in timeline.frames))
        self.fig.suptitle("Stage 3A: saved FTL motion / static collision post-processing")
        self.status.set_fontsize(8)
        self.axes.set_position([0.10, 0.22, 0.83, 0.57])
        self.warning.set_position((0.10, 0.12))
        self.target_marker.set_label("Saved final target")
        self.context_note.set_text(
            "Whole robot checked; red = collision, amber = margin. Discrete frames only."
        )
        self.context_note.set_fontsize(7)
        self.replay_button.ax.set_position([0.04, 0.055, 0.16, 0.05])
        self.pause_button.ax.set_position([0.23, 0.055, 0.16, 0.05])
        self.previous_button = Button(self.fig.add_axes([0.42, 0.055, 0.16, 0.05]), "Previous")
        self.next_button = Button(self.fig.add_axes([0.61, 0.055, 0.16, 0.05]), "Next")
        self.first_button = Button(self.fig.add_axes([0.80, 0.055, 0.16, 0.05]), "First hit")
        self.previous_button.on_clicked(lambda event: self.step_frame(-1))
        self.next_button.on_clicked(lambda event: self.step_frame(1))
        self.first_button.on_clicked(self.jump_to_first_collision)
        if self.first_collision is None:
            self.first_button.label.set_text("No hit")
            self.first_button.set_active(False)
        if len(timeline.reference_points_m):
            self.axes.plot(
                *timeline.reference_points_m.T,
                "--",
                color=config.deployment.reference_path_color,
                label="Saved desired path",
                zorder=2,
            )
        self.axes.plot([], [], color="#d62728", linewidth=4, label="Colliding section")
        self.axes.plot([], [], color="#db8b00", linewidth=4, label="Safety margin only")
        # Include the full passive segment, rather than limiting the view to FTL exposure.
        points = np.vstack([f.state.forward.centerline_m for f in timeline.frames])
        bounds = config.environment.workspace_bounds_m
        padding = config.deployment.plot_padding_m
        self.axes.set(
            xlim=(
                min(points[:, 0].min(), bounds[0, 0]) - padding,
                max(points[:, 0].max(), bounds[0, 1]) + padding,
            ),
            ylim=(
                min(points[:, 1].min(), bounds[1, 0]) - padding,
                max(points[:, 1].max(), bounds[1, 1]) + padding,
            ),
        )
        self.axes.get_legend().remove()
        self.fig.legend(
            *self.axes.get_legend_handles_labels(),
            loc="upper left",
            bbox_to_anchor=(0.005, 0.78),
            fontsize=6,
        )
        self.draw_frame(0)

    def draw_frame(self, index: int) -> tuple:
        """Display precomputed collision classifications; restore colors on clear frames."""
        artists = super().draw_frame(index)
        frame = self.timeline.frames[index]
        result = frame.collision
        hits = {c.section_index for c in result.contacts if c.clearance_m <= 0}
        near = {c.section_index for c in result.contacts if c.clearance_m > 0}
        for i, line in enumerate(self.lines):
            line.set_color(
                "#d62728"
                if i in hits
                else "#db8b00"
                if i in near
                else self.settings.section_colors[i]
            )
            line.set_linewidth(self.config.visualization.centerline_width + (3 if i in hits else 1))
        clearance = result.minimum_clearance_m
        gap = "N/A (no obstacles)" if clearance is None else f"{clearance * 1000:+.4f} mm"
        names = ", ".join(frame.state.section_names[i] for i in sorted(hits)) or "none"
        self.status.set_text(
            f"Checked frame {index + 1}/{len(self.timeline.frames)} | "
            f"source frame {frame.source_frame_index + 1} | step_index={frame.source_step_index}\n"
            f"Phase={frame.phase} | d={frame.state.distance_m * 1000:.2f} mm | "
            f"reference s={frame.desired_reference_progress_m * 1000:.2f} mm\n"
            f"Static status: {result.status.value} | colliding sections: {names}\n"
            f"Clearance lower bound: {gap}\n"
            f"Replay t={frame.replay_time_s:.2f} s (display time; physical time unavailable)"
        )
        self.status.set_color("#c62828" if hits else "#9a6500" if near else "#222222")
        if self.first_collision is None:
            self.warning.set_text("No geometric collision detected in checked frames.")
        else:
            first = self.timeline.frames[self.first_collision]
            prefix = "FIRST COLLISION HERE | " if index == self.first_collision else ""
            self.warning.set_text(
                f"{prefix}First collision: source frame {first.source_frame_index + 1}, "
                f"d={first.state.distance_m * 1000:.2f} mm"
            )
        return artists

    def jump_to_first_collision(self, event: object = None) -> None:
        """Pause at the first checked geometric collision, if any."""
        if self.first_collision is not None:
            self.step_frame(self.first_collision - self.current_index)


def save_clearance_plot(
    timeline: CollisionTimeline, config: ProjectConfig, path: str | Path
) -> None:
    """Plot physical clearance bound in saved order versus feed d and reference s."""
    import matplotlib.pyplot as plt

    clearance = np.array(
        [
            np.nan
            if f.collision.minimum_clearance_m is None
            else f.collision.minimum_clearance_m * 1000
            for f in timeline.frames
        ]
    )
    hits = np.array([f.collision.status == CollisionStatus.COLLISION for f in timeline.frames])
    near = np.array(
        [f.collision.status == CollisionStatus.SAFETY_MARGIN_VIOLATION for f in timeline.frames]
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    try:
        for ax, x, label in (
            (axes[0], [f.state.distance_m * 1000 for f in timeline.frames], "Deployment d (mm)"),
            (
                axes[1],
                [f.desired_reference_progress_m * 1000 for f in timeline.frames],
                "Desired reference progress (mm; approach = 0)",
            ),
        ):
            x = np.array(x)
            ax.plot(x, clearance, label="Signed clearance lower bound")
            ax.scatter(x[hits], clearance[hits], color="#d62728", s=12, label="Collision")
            ax.scatter(x[near], clearance[near], color="#db8b00", s=12, label="Margin only")
            ax.axhline(0, color="#c62828", linestyle="--", label="Contact")
            ax.axhline(
                config.environment.safety_margin_m * 1000,
                color="#db8b00",
                linestyle=":",
                label="Safety margin",
            )
            if hits.any():
                first = int(np.flatnonzero(hits)[0])
                ax.annotate(
                    f"First hit: source frame {timeline.frames[first].source_frame_index + 1}",
                    (x[first], clearance[first]),
                    xytext=(-15, 15),
                    textcoords="offset points",
                    ha="right",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                    arrowprops={"arrowstyle": "->"},
                    fontsize=8,
                )
            if np.isnan(clearance).all():
                ax.text(
                    0.5,
                    0.5,
                    "No obstacles: clearance undefined",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.set(xlabel=label, ylabel="Clearance (mm)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Stage 3A: full-body discrete-frame clearance / unchanged saved motion")
        finite = clearance[np.isfinite(clearance)]
        if len(finite):
            axes[1].set_ylim(
                min(float(finite.min()) - 0.2, -0.5),
                max(config.environment.safety_margin_m * 2000, 1.0),
            )
            axes[1].set_ylabel("Clearance (mm; contact detail)")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
