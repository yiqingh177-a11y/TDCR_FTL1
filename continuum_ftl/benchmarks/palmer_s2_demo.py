"""Experiment 2: one S2 run using the unchanged isolated S3 numerical engine."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .palmer_s3 import S3Settings, load_settings, reference_geometry, validate_reference
from .palmer_s3_demo import make_figures
from .palmer_s3_solver import run_s3


def validate_s2(config: Path, s3_folder: Path) -> tuple[S3Settings, dict]:
    """Validate sole profile change and independently check S2 endpoints/orientation."""
    settings = load_settings(config)
    original = json.loads((s3_folder / "config_snapshot.json").read_text(encoding="utf-8"))
    current = asdict(settings)
    original.pop("reference_bends_rad")
    current.pop("reference_bends_rad")
    if original != current:
        raise ValueError("Only reference_bends_rad may differ from the saved S3 settings")
    expected_bends = np.array([0.0, np.pi / 4, -np.pi / 4, 0.0])
    if not np.array_equal(settings.reference_bends_rad, expected_bends):
        raise ValueError("Experiment 2 accepts only S2 [0, pi/4, -pi/4, 0]")
    validation = validate_reference(settings)
    length, angle = settings.section_length_m, np.pi / 4
    y = length / angle * (1 - np.cos(angle))
    z = length / angle * np.sin(angle)
    expected = np.array(
        [[0, 0], [0, length], [y, length + z], [2 * y, length + 2 * z], [2 * y, 2 * length + 2 * z]]
    )
    curve = reference_geometry(settings)
    if np.max(np.abs(expected - curve.junction_positions_m)) > 1e-14:
        raise ArithmeticError("Independent closed-form S2 endpoints disagree with FK")
    rotation = np.eye(2)
    for bend in expected_bends:
        rotation = rotation @ np.array(
            [[np.cos(bend), np.sin(bend)], [-np.sin(bend), np.cos(bend)]]
        )
    if abs(curve.junction_angles_rad[-1]) > 1e-14 or np.max(abs(rotation - np.eye(2))) > 1e-14:
        raise ArithmeticError("S2 final orientation must be world +Z")
    validation.update(
        independent_closed_form_endpoints_yz_m=expected.tolist(),
        final_tip_angle_from_world_plus_z_rad=float(curve.junction_angles_rad[-1]),
        final_tip_tangent_yz=[0.0, 1.0],
        independent_tip_rotation_matrix=rotation.tolist(),
        only_changed_setting="reference_bends_rad",
        validated_before_optimization=True,
    )
    return settings, validation


def write_comparison(folder: Path, result: dict, s3: dict) -> None:
    """Compare unchanged metrics; paper constants are used only here, after solving."""
    rows = []
    for label, key, paper in (
        ("Maximum body-to-reference deviation", "max_body_to_reference_m", 5.577),
        ("Average frame-maximum body deviation*", "mean_of_frame_max_body_to_reference_m", 1.935),
        ("Maximum tip waypoint error", "max_tip_error_m", 0.660),
        ("Average tip waypoint error", "mean_tip_error_m", 0.162),
        ("Maximum body-to-previous deviation", "max_body_to_previous_m", None),
        ("Average spatial/body deviation*", "mean_of_frame_mean_body_to_reference_m", None),
    ):
        rows.append(
            {
                "metric": label,
                "units": "mm",
                "palmer_s2": paper,
                "reconstructed_s2": result[key] * 1000 if result[key] is not None else None,
                "existing_s3": s3[key] * 1000 if s3[key] is not None else None,
            }
        )
    for label, key in (
        ("Accepted steps", "accepted_steps"),
        ("Rejected steps", "rejected_steps"),
        ("Total step computation", "total_step_time_s"),
        ("Mean step computation", "mean_step_time_s"),
    ):
        rows.append(
            {
                "metric": label,
                "units": "steps" if "steps" in key else "s",
                "palmer_s2": None,
                "reconstructed_s2": result[key],
                "existing_s3": s3[key],
            }
        )
    (folder / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = ["| Metric | Palmer S2 | This S2 | Existing S3 |", "|---|---:|---:|---:|"]
    for row in rows:

        def value(number, unit=row["units"]):
            return "N/A" if number is None else f"{number:.7g} {unit}"

        table.append(
            f"| {row['metric']} | {value(row['palmer_s2'])} | "
            f"{value(row['reconstructed_s2'])} | {value(row['existing_s3'])} |"
        )
    peak = result["max_body_to_reference_m"]
    ratio = peak / s3["max_body_to_reference_m"] if peak is not None else None
    report = [
        "# Palmer 2014 Experiment 2: S2 only",
        "",
        f"Completed: **{result['completed']}**. Accepted {result['accepted_steps']}, "
        f"rejected {result['rejected_steps']}. Failure: `{result['failure_reason']}`.",
        "",
        *table,
        "",
        f"Maximum body deviation S2/S3 ratio: {ratio}; increase: "
        f"{(ratio - 1) * 100 if ratio is not None else None}%.",
        "",
        "* The paper does not fully specify its average path-deviation reduction. "
        "Our main average is the unweighted mean of each accepted frame's maximum "
        "body-to-fixed-reference distance. The spatial-mean alternative is also listed. "
        "Neither is claimed to be exactly the paper's metric.",
        "",
        "## What changed",
        "",
        "Only the input profile changed: [0, pi/6, -pi/6, 0] to "
        "[0, pi/4, -pi/4, 0]. Four fixed 100 mm sections, desired 2 mm reference "
        "steps, [0,2] mm variable feed, +/-0.064 rad bend increments, pi/2 absolute "
        "limits, previous_body reference, matching/sampling and SQP settings are identical.",
        "The unchanged function `palmer_s3_solver.run_s3` is the numerical engine. "
        "Its historical name is retained to preserve its source. S2 identity is assigned "
        "only to output metadata; figure labels read this metadata. "
        "No objective or metric changes.",
        "",
        "## Reference validation before solving",
        "",
        f"Endpoints YZ (m): `{result['geometry_validation']['section_endpoints_yz_m']}`.",
        "Closed-form equal/opposite circular-arc endpoints and an independent product "
        "of rotation matrices were checked before optimization. Final tangent is [0,1] "
        "in YZ, final angle 0 rad (+Z). Total reference arc length is 0.400 m.",
        "",
        "## Interpretation and reconstruction limits",
        "",
        "This is one run, with no tuning, retry or S3 rerun. If both cases completed, "
        "the maximum-deviation comparison covers the same full 400 mm. If S2 stopped "
        "early, its maximum covers only the accepted prefix and must not be treated "
        "as a full-path comparison.",
        "The ideal entrance is exactly the S3 reconstruction: first-entry introduction, "
        "new sections straight at introduction, partially introduced sections may bend; "
        "no physical guide, passive section or research-robot three-zone model.",
        "The comparison path is frozen previous accepted body plus the desired next tip. "
        "Desired arc step, actual tip displacement and optimized feed are recorded separately.",
        "Maximum fixed-reference body deviation is independently sampled, not a "
        "continuous supremum certificate. Timing includes each solve and fine validation; "
        "offline fixed-reference evaluation and plotting are excluded from step time. "
        "No numeric equivalence with the paper's LabView SQP implementation is asserted.",
        "",
        f"Accepted reference progress: {result['accepted_reference_progress_m']} m. "
        f"Final tip YZ: `{result['final_tip_yz_m']}` m. "
        f"Final q: `{result['final_configuration_rad']}` rad.",
        "",
        "## Files and replay",
        "",
        "`result.json`: actual configurations, body curves and diagnostics. "
        "`metrics.csv`: per-step controls/errors/time. `reference_validation_before_run.json`: "
        "independent geometry validation. `comparison.csv/json`: numeric comparison. "
        "`overlay.png`, `body_deviation.png`, `tip_error.png`, `s2_vs_s3.png`, "
        "`motion.gif`: plots and animation from computed states only.",
        "",
        "```cmd",
        ".venv\\Scripts\\python.exe -m continuum_ftl.benchmarks.palmer_s2_demo "
        "--replay outputs/palmer_s2_experiment2/result.json",
        "```",
        "",
        "Published source: Palmer, Cobos-Guzman & Axinte (2014), Table 1, p.1483, "
        "https://doi.org/10.1016/j.robot.2014.05.013 .",
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    """Compute S2 once, or replay only; refuse to overwrite a saved result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks/palmer_s2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/palmer_s2_experiment2"))
    parser.add_argument("--s3-dir", type=Path, default=Path("outputs/palmer_s3_experiment1"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    if args.replay:
        folder = args.replay.parent
        result = json.loads(args.replay.read_text(encoding="utf-8"))
        if result.get("profile") != "S2":
            parser.error("Replay requires an S2 result")
        settings = load_settings(folder / "config_snapshot.json")
    else:
        folder = args.output_dir
        if (folder / "result.json").exists():
            parser.error("S2 result exists; use --replay, not another optimization")
        settings, validation = validate_s2(args.config, args.s3_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "reference_validation_before_run.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        (folder / "config_snapshot.json").write_text(
            json.dumps(asdict(settings), indent=2), encoding="utf-8"
        )

        def progress(row, count):
            if row["step_index"] % 20 == 0 or not row["accepted"]:
                print(
                    f"S2 {row['step_index']}/{count}: accepted={row['accepted']}; "
                    f"{row['failure_reason'] or ''}",
                    flush=True,
                )

        result = run_s3(settings, progress)
        result.update(profile="S2", benchmark="Palmer 2014 S2 only", geometry_validation=validation)
        (folder / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        excluded = {
            "body_yz_m",
            "material_arcs_m",
            "comparison_path_yz_m",
            "section_endpoints_yz_m",
        }
        rows = [
            {key: value for key, value in row.items() if key not in excluded}
            for row in result["steps"]
        ]
        with (folder / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        s3 = json.loads((args.s3_dir / "result.json").read_text(encoding="utf-8"))
        write_comparison(folder, result, s3)
    make_figures(result, folder, settings, args.save_gif, not args.no_show)
    import matplotlib.pyplot as plt

    s3 = json.loads((args.s3_dir / "result.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for name, run in (("S2 (45 degrees)", result), ("Existing S3 (30 degrees)", s3)):
        rows = [row for row in run["steps"] if row["accepted"]]
        ax.plot(
            [row["reference_progress_m"] * 1000 for row in rows],
            [row["max_body_to_reference_m"] * 1000 for row in rows],
            label=name,
        )
    ax.set(xlabel="Desired reference progress (mm)", ylabel="Max body to fixed reference (mm)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(folder / "s2_vs_s3.png", dpi=150)
    plt.close(fig)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "completed",
                    "accepted_steps",
                    "rejected_steps",
                    "failure_reason",
                    "max_body_to_reference_m",
                    "mean_of_frame_max_body_to_reference_m",
                    "max_tip_error_m",
                    "mean_tip_error_m",
                    "total_step_time_s",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"Saved S2: {folder}", flush=True)


if __name__ == "__main__":
    main()
