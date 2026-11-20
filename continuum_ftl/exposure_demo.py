"""Stage 2B visible reference and actual deployment/exposure replay, NOT FTL."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .deployment import deployment_report, generate_deployment_states
from .environment_exposure import exposure_report, extract_environment_exposure
from .models import ExposureStatus
from .reference_path import load_reference_path, reference_path_report


def main() -> None:
    """Reuse ONE Stage 1 report and unchanged Stage 2A states; open a GUI by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_2d.json"))
    parser.add_argument("--ik-result", type=Path, default=Path("outputs/stage1_reference_2d.json"))
    parser.add_argument(
        "--output", type=Path, help="Optional complete geometry/reference/exposure JSON"
    )
    parser.add_argument("--save-gif", type=Path, help="Optional replay GIF")
    parser.add_argument("--save-snapshot", type=Path, help="Optional final frame image")
    parser.add_argument(
        "--no-show", action="store_true", help="Export/test only; default opens GUI"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config.deployment is None:
        parser.error("Add Stage 2A deployment settings to the 2D configuration")
    if not args.ik_result.is_file():
        parser.error(
            "Export one existing Stage 1 result first: python -m continuum_ftl "
            f'--config "{args.config}" --headless --output "{args.ik_result}"'
        )
    reference = load_reference_path(config, args.ik_result)
    states = generate_deployment_states(config)
    exposures = tuple(
        extract_environment_exposure(
            state,
            config.deployment.environment_entry_z_m,
            max_arc_spacing_m=config.sampling.max_arc_spacing_m,
            tolerance_m=config.deployment.geometry_tolerance_m,
        )
        for state in states
    )
    report = deployment_report(states)
    report.update(
        {
            "stage": "2B: reference path and actual deployment exposure, NOT FTL",
            "reference_source": str(args.ik_result),
            "reference_path": reference_path_report(reference),
            "feed_exit_m": config.deployment.feed_exit_m.tolist(),
            "environment_entry_z_m": config.deployment.environment_entry_z_m,
            "arc_coordinate_conventions": {
                "reference_s": "From fixed reference entry to final IK tip, m",
                "exposed_s": "From actual entry to actual tip, m; resets each frame",
                "material_s": "From complete physical proximal end, m",
                "deployment_d": "Material advanced beyond feed exit, m; NOT reference tip progress",
            },
        }
    )
    for state_report, exposed in zip(report["states"], exposures, strict=True):
        state_report["exposure"] = exposure_report(exposed)
    summary = {
        "stage": report["stage"],
        "state_count": len(states),
        "reference_length_m": float(reference.arc_lengths_m[-1]),
        "reference_tip_error_m": reference.tip_error_m,
        "exposure_status_counts": {
            status.value: sum(exposed.status == status for exposed in exposures)
            for status in ExposureStatus
        },
        "final_exposure": {
            key: report["states"][-1]["exposure"][key]
            for key in ("exposed_length_m", "section_exposed_lengths_m", "entry_position_m")
        },
        "collision_checked": False,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.no_show and not (args.save_gif or args.save_snapshot):
        return
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    from .deployment_viewer import DeploymentViewer

    viewer = DeploymentViewer(config, states, reference_path=reference, exposures=exposures)
    try:
        if args.save_gif:
            viewer.save_gif(args.save_gif)
        if args.save_snapshot:
            args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.draw_frame(len(states) - 1)
            viewer.fig.savefig(args.save_snapshot)
        viewer.draw_frame(0)
        if not args.no_show:
            viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
