"""Stage 3A post-process saved FTL states; never invoke optimization."""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .collision_postprocess import analyze_saved_ftl, collision_timeline_report
from .config import load_config
from .models import TargetConfig


def main() -> None:
    """Export static collision diagnostics and open an unchanged-motion replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_2d.json"))
    parser.add_argument("--input", type=Path, default=Path("outputs/runs/staircase_03/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3a/staircase"))
    parser.add_argument(
        "--save-gif", action="store_true", help="Save the full replay as motion.gif"
    )
    parser.add_argument("--no-show", action="store_true", help="Export without opening a GUI")
    args = parser.parse_args()
    output = args.output_dir
    targets = [
        output / name
        for name in (
            "collision_report.json",
            "frames.csv",
            "config_snapshot.json",
            "clearance.png",
            "first_collision.png",
            "first_unsafe.png",
            "first_frame.png",
            "motion.gif",
        )
    ]
    if any(p.resolve() in (args.input.resolve(), args.config.resolve()) for p in targets):
        parser.error("Output files must not overwrite the input FTL result or configuration")
    config_raw = args.config.read_bytes()
    config = load_config(args.config)
    print(
        f"Stage 3A: checking complete saved robot against "
        f"{len(config.environment.obstacles)} obstacles; SQP OFF...",
        flush=True,
    )
    timeline = analyze_saved_ftl(config, args.input)
    report = collision_timeline_report(timeline, config)
    report["config_sha256"] = hashlib.sha256(config_raw).hexdigest()
    report["config_path"] = str(args.config.resolve())
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.json").write_bytes(config_raw)
    (output / "collision_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    rows = [
        {k: ";".join(v) if isinstance(v, list) else v for k, v in frame.items() if k != "contacts"}
        for frame in report["frames"]
    ]
    with (output / "frames.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("frames",)}, indent=2, allow_nan=False
        ),
        flush=True,
    )
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    from .collision_replay import CollisionReplay, save_clearance_plot

    config = replace(
        config, target=TargetConfig(timeline.target_position_m, config.target.tolerance_m)
    )
    save_clearance_plot(timeline, config, output / "clearance.png")
    viewer = CollisionReplay(config, timeline)
    try:
        event = report["first_collision"] or report["first_unsafe_frame"]
        name = (
            "first_collision"
            if report["first_collision"]
            else "first_unsafe"
            if event
            else "first_frame"
        )
        viewer.draw_frame(event["frame_number"] - 1 if event else 0)
        viewer.fig.savefig(output / f"{name}.png", dpi=150)
        if args.save_gif:
            viewer.save_gif(output / "motion.gif")
        viewer.draw_frame(0)
        if not args.no_show:
            viewer.run()
    finally:
        viewer.close()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != timeline.source_sha256:
        raise RuntimeError("Input FTL file changed during post-processing")
    print(f"Saved Stage 3A results to {output}", flush=True)


if __name__ == "__main__":
    main()
