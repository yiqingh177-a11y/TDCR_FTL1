"""Plot stored S1 frames only; no optimizer calls or height-monotonicity assumption."""

import json
from pathlib import Path

import matplotlib
import numpy as np


def make_s1_figures(result: dict, folder: Path, interval_ms: int) -> None:
    """Save the sampled sweep, reference overlay, error plots and actual-state GIF."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection

    rows = [r for r in result["steps"] if r["accepted"]]
    bodies = [np.asarray(r["body_yz_m"]) * 1000 for r in rows]
    reference = np.asarray(result["reference_yz_m"]) * 1000
    # Adjacent saved shapes are joined by small ruled patches, not a convex
    # hull. Normalized exposed arc matches endpoints for visualization only.
    resampled = []
    for body in bodies:
        arcs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(body, axis=0), axis=1))]
        grid = np.linspace(0, arcs[-1], 801)
        resampled.append(np.column_stack([np.interp(grid, arcs, body[:, k]) for k in (0, 1)]))
    patches = []
    for old, new in zip(resampled[:-1], resampled[1:], strict=True):
        patches.extend(np.stack((old[:-1], old[1:], new[1:], new[:-1]), axis=1))
    for with_reference, filename in ((False, "overlay.png"), (True, "reference_body_overlay.png")):
        fig, ax = plt.subplots(figsize=(4.5, 7), layout="constrained")
        ax.add_collection(PolyCollection(patches, facecolor="red", edgecolor="none"))
        if with_reference:
            ax.plot(*reference.T, "k--", lw=1, label="Fixed reference")
            ax.plot(*bodies[-1].T, color="#20578c", lw=1, label="Final accepted body")
            ax.legend(fontsize=7)
        ax.set(
            xlim=(-50, 200),
            ylim=(0, 400),
            xlabel="World Y (mm)",
            ylabel="World Z (mm)",
            title="S1: sampled centreline sweep",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        fig.savefig(folder / filename, dpi=200)
        plt.close(fig)
    for filename, metrics in (
        (
            "body_deviation.png",
            [
                ("max_body_to_reference_m", "Max body/fixed reference"),
                ("max_body_to_previous_m", "Max body/previous body"),
            ],
        ),
        ("tip_error.png", [("tip_error_m", "Actual tip/desired waypoint")]),
    ):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for key, label in metrics:
            ax.plot(
                [r["reference_progress_m"] * 1000 for r in rows],
                [r[key] * 1000 for r in rows],
                label=label,
            )
        ax.set(xlabel="Desired reference progress (mm)", ylabel="Error (mm)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(folder / filename, dpi=150)
        plt.close(fig)
    frames = [result["initial_state"], *result["steps"]]
    fig, ax = plt.subplots(figsize=(6, 8), layout="constrained")
    ax.plot(*reference.T, "k--", lw=1, label="Fixed reference")
    (line,) = ax.plot([], [], color="#20578c", lw=2, label="Actual exposed body")
    (tip,) = ax.plot([], [], "ro", ms=4)
    (waypoint,) = ax.plot([], [], "gx", ms=5)
    (trace,) = ax.plot([], [], color="#dd852e", lw=1, label="Actual tip history")
    (junctions,) = ax.plot([], [], "o", color="gray", ms=3)
    ax.set(xlim=(-50, 200), ylim=(-10, 410), xlabel="World Y (mm)", ylabel="World Z (mm)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)

    def draw(i: int) -> tuple:
        row = frames[i]
        body = np.asarray(row["body_yz_m"]).reshape(-1, 2) * 1000
        line.set_data(*body.T)
        line.set_color("#20578c" if row["accepted"] else "red")
        tip.set_data(*np.asarray(row["actual_tip_yz_m"])[:, None] * 1000)
        waypoint.set_data(*np.asarray(row.get("desired_tip_yz_m", [0, 0]))[:, None] * 1000)
        exposed = np.asarray(row["section_endpoints_yz_m"]) * 1000
        exposed = exposed[exposed[:, 1] >= 0]
        junctions.set_data(*exposed.T)
        history = np.asarray([r["actual_tip_yz_m"] for r in frames[: i + 1] if r["accepted"]])
        trace.set_data(*history.T * 1000)
        ax.set_title(
            f"S1 actual FTL | frame {i}/{len(frames) - 1}\n"
            f"tip error={row.get('tip_error_m', 0) * 1000:.6g} mm | "
            f"body/fixed={row.get('max_body_to_reference_m', 0) * 1000:.3f} mm",
            fontsize=10,
        )
        return line, tip, waypoint, trace, junctions

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=interval_ms)
    draw(len(frames) - 1)
    fig.savefig(folder / "animation_last_frame.png", dpi=150)
    animation.save(folder / "motion.gif", writer=PillowWriter(fps=1000 / interval_ms), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    output = Path("outputs/palmer_s1_experiment3")
    data = json.loads((output / "result.json").read_text())
    config = json.loads((output / "config_snapshot.json").read_text())
    make_s1_figures(data, output, config["frame_interval_ms"])
