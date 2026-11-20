"""Stage 2A visible manual deployment: python -m continuum_ftl.deployment_demo."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .deployment import deployment_report, generate_deployment_states


def main() -> None:
    """Load shared JSON, generate complete geometry and open 2D replay by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_2d.json"))
    parser.add_argument("--output", type=Path, help="Optional JSON physical state sequence")
    parser.add_argument("--save-gif", type=Path, help="Optional replay GIF")
    parser.add_argument("--save-snapshot", type=Path, help="Optional final frame image")
    parser.add_argument(
        "--no-show", action="store_true", help="Export/test only; normal demo opens GUI"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    states = generate_deployment_states(config)
    report = deployment_report(states)
    report["feed_exit_m"] = config.deployment.feed_exit_m.tolist()
    report["environment_entry_z_m"] = config.deployment.environment_entry_z_m
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "stage",
                    "section_order",
                    "section_lengths_m",
                    "total_physical_length_m",
                    "release_thresholds_m",
                    "state_count",
                    "collision_checked",
                )
            }
        )
    )
    if args.no_show and not (args.save_gif or args.save_snapshot):
        return
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    from .deployment_viewer import DeploymentViewer

    viewer = DeploymentViewer(config, states)
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
