"""Independent spatial Stage 2A/2B/2C entry point; external obstacles excluded."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import load_config
from .environment_exposure import exposure_report
from .ftl_evaluation import evaluate_ftl_rollout, summarize_ftl_evaluations
from .ftl_rollout import ftl_report
from .models import FTLConfig, FTLResult, ProjectConfig, ReferencePath, RobotConfig
from .reference_path import reference_path_report
from .spatial_ftl_rollout import generate_spatial_deployment_states, run_spatial_ftl
from .spatial_geometry import extract_spatial_exposure
from .spatial_reference import load_spatial_case, reference_planarity


def spatial_report(config: ProjectConfig, reference: ReferencePath, result: FTLResult) -> dict:
    """Shared result schema plus explicit 3D control/plane diagnostics, in SI units."""
    report = ftl_report(result, reference)
    planarity = reference_planarity(reference.points_m, config.deployment.geometry_tolerance_m)
    angle = np.deg2rad(config.spatial_reference.plane_rotation_deg)
    normal = np.array([np.cos(angle), np.sin(angle), 0.0])
    origin = reference.points_m[0]
    if np.max(np.abs((reference.points_m - origin) @ normal)) > planarity["tolerance_m"]:
        normal = np.array(planarity["best_fit_plane_normal"])
        origin = np.array(planarity["best_fit_plane_origin_m"])
    for row, step in zip(report["steps"], result.steps, strict=True):
        row["maximum_distance_from_reference_plane_m"] = (
            float(np.max(np.abs((step.state.forward.centerline_m - origin) @ normal)))
            if planarity["is_planar"]
            else None
        )
    report.update(
        {
            "dimension": 3,
            "stage": "Spatial Stage 2C: true incremental SQP FTL",
            "plane_rotation_deg": config.spatial_reference.plane_rotation_deg,
            "rotation_axis": "world +Z, right-hand rule",
            "feed_direction": [0, 0, 1],
            "configuration_parameterization": "local integrated bend_x, bend_y; radians",
            "solver_variables": "delta_d plus two local bend increments per eligible section",
            "reference_plane_constraint": False,
            "reference_planarity": planarity,
            "external_obstacles": [],
            "maximum_distance_from_reference_plane_m": (
                max(r["maximum_distance_from_reference_plane_m"] for r in report["steps"])
                if planarity["is_planar"]
                else None
            ),
        }
    )
    return report


def main() -> None:
    """Compute independent spatial states, export SI diagnostics, show orbitable replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_3d.json"))
    parser.add_argument("--case", default="staircase")
    parser.add_argument("--stage", choices=("deployment", "reference", "ftl"), default="ftl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    args = parser.parse_args()
    folder = args.output_dir or Path("outputs/runs_3d") / (
        args.case if args.stage == "ftl" else f"{args.case}_{args.stage}"
    )
    if any(
        (folder / name).resolve() == args.config.resolve()
        for name in (
            "config_snapshot.json",
            "result.json",
            "summary.png",
            "errors.png",
            "metrics.csv",
            "motion.gif",
        )
    ):
        parser.error("Output files must not overwrite the source config")
    raw = args.config.read_bytes()
    config = load_config(args.config)
    if not isinstance(config.robot, RobotConfig) or config.environment.obstacles:
        parser.error("Use a 3D robot and an empty external obstacle list")
    config, reference = load_spatial_case(config, args.case)
    print(
        f"Computing 3D {args.case} / {args.stage}; true spatial bends; no plane constraint...",
        flush=True,
    )
    result = None
    if args.stage == "ftl":
        result = run_spatial_ftl(config, reference)
        report = spatial_report(config, reference, result)
        states = tuple(step.state for step in result.steps)
        exposures = tuple(step.exposure for step in result.steps)
        evaluations = evaluate_ftl_rollout(config, reference, result)
        report["independent_evaluation"] = summarize_ftl_evaluations(evaluations)
    else:
        states = generate_spatial_deployment_states(config)
        ftl_settings = config.ftl or FTLConfig()
        exposures = tuple(
            extract_spatial_exposure(
                state.geometry,
                config.deployment.environment_entry_z_m,
                ftl_settings.validation_body_spacing_m,
                config.deployment.geometry_tolerance_m,
            )
            for state in states
        )
        evaluations = ()
        report = {
            "dimension": 3,
            "stage": args.stage,
            "reference_path": reference_path_report(reference),
            "states": [
                {
                    "d_m": state.distance_m,
                    "local_bends_rad": state.applied_bends_rad.tolist(),
                    "centerline_m": state.forward.centerline_m.tolist(),
                    "exposure": exposure_report(exposure),
                }
                for state, exposure in zip(states, exposures, strict=True)
            ],
        }
    report["case"] = args.case
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config_snapshot.json").write_bytes(raw)
    (folder / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if evaluations:
        rows = [asdict(row) for row in evaluations]
        with (folder / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                in (
                    "stage",
                    "case",
                    "success",
                    "accepted_ftl_steps",
                    "failure_reason",
                    "maximum_body_error_m",
                    "maximum_distance_from_reference_plane_m",
                    "last_accepted_target_error_m",
                    "computation_time_s",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .spatial_viewer import SpatialViewer

    viewer = SpatialViewer(
        config,
        states,
        reference if args.stage != "deployment" else None,
        exposures if args.stage != "deployment" else None,
        result.steps if result is not None else None,
        result.failure_reason if result else None,
    )
    try:
        last = max(i for i in range(len(states)) if result is None or result.steps[i].accepted)
        viewer.draw_frame(last)
        viewer.fig.savefig(folder / "summary.png", dpi=130)
        if args.save_gif:
            viewer.save_gif(folder / "motion.gif")
        if evaluations:
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
            s = [row.desired_reference_progress_m * 1000 for row in evaluations]
            axes[0].plot(
                s, [row.tip_error_m * 1000 for row in evaluations], label="Actual tip to waypoint"
            )
            axes[0].set(ylabel="Tip error (mm)")
            axes[1].plot(
                s,
                [row.native_body_error_max_m * 1000 for row in evaluations],
                label="Native body error",
            )
            axes[1].plot(
                s,
                [row.body_to_reference_max_m * 1000 for row in evaluations],
                label="Body to fixed path",
            )
            axes[1].set(ylabel="Body error (mm)", xlabel="Desired reference arc length (mm)")
            for ax in axes:
                ax.legend()
                ax.grid(alpha=0.3)
            fig.suptitle("Spatial Stage 2C: actual computed-state errors")
            fig.savefig(folder / "errors.png", dpi=150)
            plt.close(fig)
        viewer.draw_frame(0)
        if not args.no_show:
            viewer.run()
    finally:
        viewer.close()
    print(f"Saved spatial results to {folder}", flush=True)


if __name__ == "__main__":
    main()
