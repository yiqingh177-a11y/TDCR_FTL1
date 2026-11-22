"""Static summary of computed accepted FTL shapes, reusing the deployment viewer."""

from pathlib import Path

import numpy as np

from .deployment_viewer import DeploymentViewer
from .models import FTLConfig, FTLFrameEvaluation, FTLResult, ProjectConfig, ReferencePath


def save_ftl_summary(
    config: ProjectConfig, reference: ReferencePath, result: FTLResult, path: str | Path
) -> None:
    """Save real accepted intermediate shapes, final accepted shape and tip history.

    Rejected geometry is shown separately in the replay's retained failure frame.
    This function never performs FK or optimization.
    """
    viewer = DeploymentViewer(
        config,
        tuple(step.state for step in result.steps),
        reference_path=reference,
        exposures=tuple(step.exposure for step in result.steps),
        ftl_steps=result.steps,
        rollout_failure_reason=result.failure_reason,
    )
    try:
        accepted_indices = [
            i
            for i, step in enumerate(result.steps)
            if step.accepted and step.phase in ("FTL", "reshape")
        ]
        if not accepted_indices:
            accepted_indices = [i for i, step in enumerate(result.steps) if step.accepted]
        selected = np.unique(
            np.linspace(
                0,
                len(accepted_indices) - 1,
                min(len(accepted_indices), (config.ftl or FTLConfig()).summary_shape_count),
                dtype=int,
            )
        )
        for i in selected[:-1]:
            for curve, color in zip(
                result.steps[accepted_indices[i]].state.forward.section_curves,
                config.deployment.section_colors,
                strict=True,
            ):
                viewer.axes.plot(
                    *curve.points_m.T,
                    color=color,
                    alpha=0.25,
                    linewidth=config.visualization.centerline_width,
                )
        viewer.draw_frame(accepted_indices[-1])
        for axes in viewer.fig.axes[1:]:
            axes.set_visible(False)
        viewer.fig.suptitle("Stage 2C: accepted shapes / actual tip history")
        if result.reference_completed:
            error_mm = (
                np.linalg.norm(
                    result.steps[-1].state.forward.tip_position_m - config.target.position_m
                )
                * 1000
            )
            viewer.warning.set_text(
                f"Reference sequence completed | final target error = {error_mm:.3f} mm"
            )
        else:
            viewer.warning.set_text("Stopped: " + str(result.failure_reason))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        viewer.fig.savefig(path)
    finally:
        viewer.close()


def save_body_reference_comparison(
    config: ProjectConfig,
    reference: ReferencePath,
    runs: dict[str, FTLResult],
    evaluations: dict[str, tuple[FTLFrameEvaluation, ...]],
    output_directory: str | Path,
) -> None:
    """Save two-mode last-accepted shapes and common offline errors, metres to mm.

    Failed candidates never substitute for last accepted bodies in these plots.
    Histories, PCC bodies and per-frame errors are computed data, not reference
    copies. Both shape panels share equal scales and the same world bounds.
    """
    import matplotlib.pyplot as plt

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels = {"tip_history": "Actual tip history", "previous_body": "Previous body (Palmer-style)"}
    colors = {"tip_history": "#167f89", "previous_body": "#b06125"}
    last_steps = {
        mode: next(s for s in reversed(run.steps) if s.accepted) for mode, run in runs.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    all_points = [reference.points_m]
    for ax, (mode, run) in zip(axes, runs.items(), strict=True):
        step = last_steps[mode]
        ax.plot(*reference.points_m.T * 1000, "--", color="#8054a4", label="Fixed reference")
        if len(step.historical_tip_path_m):
            ax.plot(
                *step.historical_tip_path_m.T * 1000, color=colors[mode], label="Actual tip history"
            )
        for index, curve in enumerate(step.state.forward.section_curves[1:]):
            ax.plot(
                *curve.points_m.T * 1000,
                color="#333333",
                lw=2,
                label="Actual active PCC body" if index == 0 else None,
            )
            all_points.append(curve.points_m)
        ax.scatter(
            *step.state.forward.section_endpoints_m[1:].T * 1000,
            fc="white",
            ec="#444444",
            s=22,
            zorder=4,
        )
        ax.scatter(
            *reference.target_position_m * 1000,
            marker="*",
            color="#159947",
            s=90,
            label="Final target",
            zorder=5,
        )
        ax.axhline(config.deployment.environment_entry_z_m * 1000, color="#999999", lw=0.8)
        ax.set_title(
            f"{labels[mode]}\nLast accepted s = {step.reference_progress_m * 1000:.2f} mm\n"
            f"{'Target reached' if run.success else 'Stopped: ' + str(run.failure_reason)}",
            fontsize=9,
        )
        ax.set(xlabel="World Y [mm]", ylabel="World Z [mm]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    bounds = np.vstack(all_points) * 1000
    padding = config.deployment.plot_padding_m * 1000
    axes[0].set_xlim(bounds[:, 0].min() - padding, bounds[:, 0].max() + padding)
    axes[0].set_ylim(bounds[:, 1].min() - padding, bounds[:, 1].max() + padding)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Same robot, target path and solver settings; only body reference differs")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output / "final_shapes.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, field, title in zip(
        axes,
        ("body_to_reference_max_m", "body_to_history_max_m", "history_to_reference_max_m"),
        ("Body to fixed reference", "Body to actual tip history", "Tip history to fixed reference"),
        strict=True,
    ):
        for mode, frames in evaluations.items():
            ax.plot(
                [f.reference_progress_m * 1000 for f in frames],
                [getattr(f, field) * 1000 for f in frames],
                color=colors[mode],
                label=labels[mode],
            )
        ax.set(ylabel="Maximum deviation [mm]", title=title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("Actual-tip projected reference progress [mm]; accepted FTL frames only")
    fig.suptitle("Identical independent geometric evaluation (native objective errors excluded)")
    fig.tight_layout()
    fig.savefig(output / "independent_errors.png", dpi=160)
    plt.close(fig)
