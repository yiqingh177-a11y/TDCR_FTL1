"""Controlled four-case body-reference A/B experiment; no IK search or tuning."""

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy

from .config import load_config
from .ftl_demo import load_ftl_case
from .ftl_evaluation import evaluate_ftl_rollout, summarize_ftl_evaluations
from .ftl_rollout import ftl_report, ftl_summary, run_single_path_ftl
from .ftl_visualization import save_body_reference_comparison

CASES = ("straight", "single-bend", "s-bend", "staircase")
MODES = ("tip_history", "previous_body")


def _format_mm(value: float | None) -> str:
    return f"{value * 1000:.6f}" if value is not None else "N/A"


def main() -> None:
    """Run paired settings, export full actual states, common metrics and figures.

    Config controls all numerical and geometric settings. CLI reference step is
    an optional shared override for the experiment (e.g. requested 1 mm); it does
    not edit the source config or independently tune either method. Each run
    gets the same initial state, fixed path, solver and stopping conditions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/robot_2d.json"))
    parser.add_argument("--ik-result", type=Path, default=Path("outputs/stage1_reference_2d.json"))
    parser.add_argument("--fixtures", type=Path, default=Path("configs/ftl_validation.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/body_reference_comparison")
    )
    parser.add_argument("--reference-step-m", type=float)
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if args.reference_step_m is not None:
        data.setdefault("ftl", {})["reference_step_m"] = args.reference_step_m
    # Save an isolated complete config and validate it with the existing loader.
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "experiment_config.json"
    if snapshot.resolve() == args.config.resolve():
        config = load_config(args.config)
        if args.reference_step_m is not None:
            parser.error("Use a separate output directory when overriding reference step")
    else:
        snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")
        config = load_config(snapshot)
    if config.ftl is None or config.deployment is None:
        parser.error("This experiment requires FTL and deployment configuration")
    if config.ftl.reference_progress_mode != "scheduled":
        parser.error("This controlled experiment requires reference_progress_mode=scheduled")
    if snapshot.resolve() != args.config.resolve():
        # Include resolved defaults so a future default change cannot silently
        # change the meaning of this archived experiment configuration.
        data["ftl"] = asdict(config.ftl)
        snapshot.write_text(json.dumps(data, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    versions = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__}
    (output / "software_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    summaries = []
    for case in CASES:
        cfg, reference = load_ftl_case(config, case, args.fixtures, args.ik_result)
        runs, evaluations = {}, {}
        for mode in MODES:
            print(f"Computing {case} / {mode}...", flush=True)
            selected = replace(cfg, ftl=replace(cfg.ftl, body_reference_mode=mode))
            result = run_single_path_ftl(selected, reference)
            frames = evaluate_ftl_rollout(selected, reference, result)
            destination = output / case
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"{mode}.json").write_text(
                json.dumps(ftl_report(result, reference), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            frame_rows = [asdict(frame) for frame in frames]
            if frame_rows:
                with (destination / f"{mode}_metrics.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=list(frame_rows[0]))
                    writer.writeheader()
                    writer.writerows(frame_rows)
            summary = {
                "case": case,
                **ftl_summary(result, reference),
                "rejected_ftl_steps": sum(
                    s.phase == "FTL" and not s.accepted for s in result.steps
                ),
                "independent_evaluation": summarize_ftl_evaluations(frames),
            }
            summaries.append(summary)
            runs[mode], evaluations[mode] = result, frames
            print(
                json.dumps(
                    {
                        k: summary[k]
                        for k in (
                            "case",
                            "body_reference_mode",
                            "success",
                            "accepted_ftl_steps",
                            "failure_reason",
                            "computation_time_s",
                        )
                    }
                ),
                flush=True,
            )
        save_body_reference_comparison(cfg, reference, runs, evaluations, output / case)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# Body-reference A/B results",
        "",
        "Only the body reference differs within each case. Distances below are mm; "
        "independent metrics use accepted FTL frames only. Final means last accepted, "
        "including when the run failed. Native objective errors are separate in JSON.",
        "",
        "| Case | Mode | Endpoint success | Accepted / attempted | Final s (mm) | Failure | "
        "Final target error | Final body→reference | Final body→history | "
        "Worst body→reference | Mean frame-max body→reference | Solve time (s) |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        metrics = row["independent_evaluation"]
        fixed, history = metrics["body_to_reference"], metrics["body_to_history"]
        lines.append(
            f"| {row['case']} | {row['body_reference_mode']} | {row['success']} | "
            f"{row['accepted_ftl_steps']} / {row['attempted_ftl_steps']} | "
            f"{_format_mm(row['final_reference_progress_m'])} | "
            f"{row['failure_reason']} | "
            f"{_format_mm(row['last_accepted_target_error_m'])} | "
            f"{_format_mm(fixed['final_maximum_m']) if fixed else 'N/A'} | "
            f"{_format_mm(history['final_maximum_m']) if history else 'N/A'} | "
            f"{_format_mm(fixed['maximum_over_accepted_frames_m']) if fixed else 'N/A'} | "
            f"{_format_mm(fixed['mean_of_frame_maxima_m']) if fixed else 'N/A'} | "
            f"{row['computation_time_s']:.3f} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
