"""Stage 2C computed single-path FTL: python -m continuum_ftl.ftl_demo."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import load_config
from .ftl_rollout import ftl_report, ftl_summary, run_single_path_ftl
from .models import FTLConfig, ProjectConfig, ReferencePath, TargetConfig
from .planar_kinematics import tip_pose_2d
from .reference_path import extract_reference_path, load_reference_path


def load_ftl_case(
    config: ProjectConfig, case: str, fixtures_path: str | Path, ik_result_path: str | Path
) -> tuple[ProjectConfig, ReferencePath]:
    """Staircase uses one saved target-reaching shape; collision flags are ignored.

    Test-case targets are computed in memory; never rewrite the user's scene.
    Fixture q uses original C/B/A ordering. No IK search is run by this demo.
    """
    if case == "staircase":
        return config, load_reference_path(config, ik_result_path, require_collision_free=False)
    fixtures = json.loads(Path(fixtures_path).read_text(encoding="utf-8-sig"))
    if case not in ("straight", "single-bend", "s-bend"):
        raise ValueError(f"Unknown FTL case: {case}")
    q = np.asarray(fixtures[case], dtype=float)
    tip, _ = tip_pose_2d(config.robot, q)
    case_config = replace(
        config,
        target=TargetConfig(tip, config.target.tolerance_m),
        environment=replace(config.environment, obstacles=()),
    )
    return case_config, extract_reference_path(case_config, q)


def main() -> None:
    """Compute actual SQP states, report metrics, then open pause/replay/step viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_2d.json"))
    parser.add_argument("--ik-result", type=Path, default=Path("outputs/stage1_reference_2d.json"))
    parser.add_argument("--body-reference-mode", choices=("tip_history", "previous_body"))
    parser.add_argument(
        "--max-reshape-attempts", type=int, help="Fixed-waypoint trials; 0 disables"
    )
    parser.add_argument(
        "--case", choices=("straight", "single-bend", "s-bend", "staircase"), default="staircase"
    )
    parser.add_argument("--fixtures", type=Path, default=Path("configs/ftl_validation.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ftl_2d.json"))
    parser.add_argument("--save-gif", type=Path)
    parser.add_argument("--save-snapshot", type=Path)
    parser.add_argument("--save-summary", type=Path, default=Path("outputs/ftl_summary.png"))
    parser.add_argument(
        "--no-show", action="store_true", help="Export/testing only; default is visible replay"
    )
    args = parser.parse_args()
    config, reference = load_ftl_case(
        load_config(args.config), args.case, args.fixtures, args.ik_result
    )
    if args.body_reference_mode is not None:
        config = replace(
            config,
            ftl=replace(config.ftl or FTLConfig(), body_reference_mode=args.body_reference_mode),
        )
    if args.max_reshape_attempts is not None:
        if args.max_reshape_attempts < 0:
            parser.error("--max-reshape-attempts must be >= 0")
        config = replace(
            config,
            ftl=replace(config.ftl or FTLConfig(), max_reshape_attempts=args.max_reshape_attempts),
        )
    progress_mode = config.ftl.reference_progress_mode if config.ftl else "actual_tip"
    print(
        f"Computing {args.case}: reference mode={progress_mode}; "
        f"body reference={(config.ftl or FTLConfig()).body_reference_mode}; "
        "obstacle collisions OFF; obstacles are visual only...",
        flush=True,
    )
    result = run_single_path_ftl(config, reference)
    summary = {"case": args.case, **ftl_summary(result, reference)}
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"case": args.case, **ftl_report(result, reference)}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    from .deployment_viewer import DeploymentViewer
    from .ftl_visualization import save_ftl_summary

    save_ftl_summary(config, reference, result, args.save_summary)
    viewer = DeploymentViewer(
        config,
        tuple(step.state for step in result.steps),
        reference_path=reference,
        exposures=tuple(step.exposure for step in result.steps),
        ftl_steps=result.steps,
        rollout_failure_reason=result.failure_reason,
    )
    try:
        if args.save_gif:
            viewer.save_gif(args.save_gif)
        if args.save_snapshot:
            args.save_snapshot.parent.mkdir(parents=True, exist_ok=True)
            viewer.draw_frame(len(result.steps) - 1)
            viewer.fig.savefig(args.save_snapshot)
        viewer.draw_frame(0)
        if not args.no_show:
            viewer.run()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
