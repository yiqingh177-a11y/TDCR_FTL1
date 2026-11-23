"""Controlled S3 sampling study: three runs, unchanged solver and fine evaluation."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..ftl_evaluation import point_to_polyline_distances
from ..ftl_matching import match_ordered_polyline, polyline_arcs
from ..planar_geometry import evaluate_curve
from .palmer_s3 import S3Settings, geometry, load_settings
from .palmer_s3_solver import exposed_samples, run_s3

BASELINE = Path("outputs/palmer_s3_experiment1")
STUDY = Path("outputs/palmer_s3_sampling_sensitivity")
SPACINGS = (0.005, 0.002, 0.001)


def write_json(path: Path, value: object) -> None:
    """Write finite SI result data into this study's dedicated output directory."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write parallel diagnostic rows, preserving actual numerical values."""
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_preservation(root: Path) -> None:
    """Check protected sources/configs/tests and historical experiment byte hashes."""
    manifest = json.loads((root / "preservation_manifest.json").read_text(encoding="utf-8"))
    changed = [
        name
        for name, digest in manifest.items()
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")


def changed_settings(baseline: S3Settings, spacing_m: float) -> S3Settings:
    """Only the optimization spacing changes; all other settings must remain equal."""
    settings = replace(baseline, optimization_spacing_m=spacing_m)
    original, current = asdict(baseline), asdict(settings)
    original.pop("optimization_spacing_m")
    current.pop("optimization_spacing_m")
    if original != current:
        raise ValueError("Study permits only optimization_spacing_m to change")
    return settings


def run_experiments(root: Path) -> None:
    """Run 5/2/1 mm once each, sequentially; never retry or change a failed run."""
    baseline = load_settings(BASELINE / "config_snapshot.json")
    if baseline.optimization_spacing_m != 0.02:
        raise ValueError("Expected preserved 20 mm S3 baseline")
    for spacing in SPACINGS:
        folder = root / f"{round(spacing * 1000)}mm"
        if (folder / "attempt.json").exists() or (folder / "result.json").exists():
            raise FileExistsError(f"Run already attempted: {folder}; use --analyse-only")
    plan = {
        "case": "S3 only",
        "baseline": str(BASELINE),
        "optimization_spacings_m": list(SPACINGS),
        "validation_spacing_m": baseline.validation_spacing_m,
        "comparison_spacing_m": baseline.comparison_spacing_m,
        "only_changed_setting": "optimization_spacing_m",
        "solver_source_sha256": hashlib.sha256(
            Path("continuum_ftl/benchmarks/palmer_s3_solver.py").read_bytes()
        ).hexdigest(),
        "sequential_independent_initial_states": True,
        "retry": False,
        "automatically_adopt_a_spacing": False,
        "baseline_not_rerun": True,
    }
    write_json(root / "experiment_plan.json", plan)
    for spacing in SPACINGS:
        folder = root / f"{round(spacing * 1000)}mm"
        folder.mkdir(parents=True, exist_ok=True)
        settings = changed_settings(baseline, spacing)
        write_json(folder / "config_snapshot.json", asdict(settings))
        write_json(folder / "attempt.json", {"attempt": 1, "spacing_m": spacing})
        started = perf_counter()
        with (folder / "progress.jsonl").open("w", encoding="utf-8") as log:

            def progress(row: dict, count: int, started=started, spacing=spacing) -> None:
                record = {
                    key: row[key]
                    for key in (
                        "step_index",
                        "accepted",
                        "failure_reason",
                        "tip_error_m",
                        "max_body_to_previous_m",
                        "solver_iterations",
                        "computation_time_s",
                    )
                }
                record["elapsed_s"] = perf_counter() - started
                log.write(json.dumps(record) + "\n")
                log.flush()
                if row["step_index"] % 10 == 0 or not row["accepted"]:
                    print(
                        f"S3 {spacing * 1000:g} mm: {row['step_index']}/{count}, "
                        f"accepted={row['accepted']}, elapsed={record['elapsed_s']:.1f}s "
                        f"{row['failure_reason'] or ''}",
                        flush=True,
                    )

            result = run_s3(settings, progress)
        result.update(profile="S3", optimization_spacing_m=spacing)
        write_json(folder / "result.json", result)
        excluded = {
            "body_yz_m",
            "material_arcs_m",
            "comparison_path_yz_m",
            "section_endpoints_yz_m",
        }
        write_csv(
            folder / "metrics.csv",
            [
                {key: value for key, value in row.items() if key not in excluded}
                for row in result["steps"]
            ],
        )
        print(
            f"Saved {folder}: completed={result['completed']}; "
            f"accepted={result['accepted_steps']}; reason={result['failure_reason']}",
            flush=True,
        )
        verify_preservation(root)


def sampling_diagnostics(settings: S3Settings, result: dict) -> list[dict]:
    """Re-evaluate returned states at optimization/fine grids, no new optimization.

    Common reference/history and unchanged local ordered matching are used for
    both grids. Near-tip means last 5 mm of material, not nearest spatial ball.
    Also check the exact 2 mm-behind-tip point independent of grid alignment.
    """
    rows = []
    for step in result["steps"]:
        curve = geometry(settings, np.array(step["configuration_rad"]), step["feed_m"])
        comparison = np.array(step["comparison_path_yz_m"])
        end = float(np.sum(curve.section_lengths_m))
        length = polyline_arcs(comparison)[-1]
        row = {
            "step_index": step["step_index"],
            "accepted": step["accepted"],
            "reference_progress_m": step["reference_progress_m"],
        }
        if not step["body_yz_m"]:
            row.update(
                optimization_body_error_m=None,
                fine_body_error_m=None,
                fine_minus_optimization_m=None,
                optimization_near_tip_5mm_m=None,
                fine_near_tip_5mm_m=None,
                exact_2mm_behind_tip_error_m=None,
                fine_peak_distance_behind_tip_m=None,
            )
            rows.append(row)
            continue
        grids = {}
        for name, spacing in (
            ("optimization", settings.optimization_spacing_m),
            ("fine", settings.validation_spacing_m),
        ):
            points, arcs = exposed_samples(settings, curve, spacing)
            match = match_ordered_polyline(
                points, comparison, length - (end - arcs), settings.matching_window_m
            )
            grids[name] = (points, arcs, match)
            row[name + "_body_error_m"] = float(np.max(match.distances_m))
            row[name + "_near_tip_5mm_m"] = float(
                np.max(match.distances_m[(end - arcs) <= 0.005 + 1e-12])
            )
        fine_points, fine_arcs, fine_match = grids["fine"]
        if not np.allclose(fine_points, step["body_yz_m"], atol=1e-14, rtol=0):
            raise ArithmeticError("Stored fine body geometry differs from recomputation")
        if abs(row["fine_body_error_m"] - step["max_body_to_previous_m"]) > 1e-13:
            raise ArithmeticError("Stored body-to-previous metric changed")
        row["fine_minus_optimization_m"] = (
            row["fine_body_error_m"] - row["optimization_body_error_m"]
        )
        row["fine_peak_distance_behind_tip_m"] = float(
            end - fine_arcs[np.argmax(fine_match.distances_m)]
        )
        arc = max(float(fine_arcs[0]), end - 0.002)
        point = evaluate_curve(curve, np.array([arc]))
        match = match_ordered_polyline(
            point, comparison, np.array([length - (end - arc)]), settings.matching_window_m
        )
        row["exact_2mm_behind_tip_error_m"] = float(match.distances_m[0])
        rows.append(row)
    return rows


def sequence_difference(settings: S3Settings, result: dict, baseline: dict) -> dict:
    """Compare accepted states at equal scheduled progress and material coordinates."""
    current = [row for row in result["steps"] if row["accepted"]]
    old = [row for row in baseline["steps"] if row["accepted"]]
    count = min(len(current), len(old))
    if not count:
        return {"common_accepted_steps": 0}
    angle_differences, feed_differences, tip_differences, body_differences = [], [], [], []
    material = np.array(baseline["reference_arc_lengths_m"])
    for row, original in zip(current[:count], old[:count], strict=True):
        if abs(row["reference_progress_m"] - original["reference_progress_m"]) > 1e-13:
            raise ValueError("State sequence progress does not align with baseline")
        q = np.array(row["configuration_rad"])
        old_q = np.array(original["configuration_rad"])
        angle_differences.append(q - old_q)
        feed_differences.append(row["feed_m"] - original["feed_m"])
        tip_differences.append(
            np.linalg.norm(np.array(row["actual_tip_yz_m"]) - original["actual_tip_yz_m"])
        )
        curve = geometry(settings, q, row["feed_m"])
        old_curve = geometry(settings, old_q, original["feed_m"])
        body_differences.append(
            float(
                np.max(
                    np.linalg.norm(
                        evaluate_curve(curve, material) - evaluate_curve(old_curve, material),
                        axis=1,
                    )
                )
            )
        )
    angles = np.array(angle_differences)
    return {
        "common_accepted_steps": count,
        "max_section_angle_difference_deg": float(np.rad2deg(np.max(abs(angles)))),
        "rms_section_angle_difference_deg": float(np.rad2deg(np.sqrt(np.mean(angles**2)))),
        "max_accumulated_feed_difference_mm": float(np.max(np.abs(feed_differences)) * 1000),
        "max_tip_position_difference_mm": float(np.max(tip_differences) * 1000),
        "max_full_body_world_position_difference_mm": float(np.max(body_differences) * 1000),
    }


def summary_row(
    label: str, result: dict, diagnostics: list[dict], count: int | None = None
) -> dict:
    """Identical metric aggregation on accepted states; optional common prefix."""
    steps = [row for row in result["steps"] if row["accepted"]]
    diag = [row for row in diagnostics if row["accepted"]]
    if count is not None:
        steps, diag = steps[:count], diag[:count]

    def maximum(key: str, values: list[dict] = steps):
        return max((row[key] for row in values), default=None)

    def mean(key: str):
        return float(np.mean([row[key] for row in steps])) if steps else None

    return {
        "optimization_spacing_mm": float(label),
        "completed": result["completed"],
        "accepted_steps": result["accepted_steps"],
        "evaluated_accepted_steps": len(steps),
        "failure_reason": result["failure_reason"],
        "body_to_reference_max_mm": (maximum("max_body_to_reference_m") or 0) * 1000
        if steps
        else None,
        "body_to_reference_mean_frame_max_mm": mean("max_body_to_reference_m") * 1000
        if steps
        else None,
        "body_to_reference_mean_frame_spatial_mean_mm": mean("mean_body_to_reference_m") * 1000
        if steps
        else None,
        "body_to_previous_fine_max_mm": maximum("max_body_to_previous_m") * 1000 if steps else None,
        "tip_error_max_mm": maximum("tip_error_m") * 1000 if steps else None,
        "tip_error_mean_mm": mean("tip_error_m") * 1000 if steps else None,
        "optimization_body_error_max_mm": maximum("optimization_body_error_m", diag) * 1000
        if steps
        else None,
        "fine_minus_optimization_max_mm": maximum("fine_minus_optimization_m", diag) * 1000
        if steps
        else None,
        "near_tip_fine_max_mm": maximum("fine_near_tip_5mm_m", diag) * 1000 if steps else None,
        "near_tip_optimization_max_mm": maximum("optimization_near_tip_5mm_m", diag) * 1000
        if steps
        else None,
        "exact_2mm_behind_tip_max_mm": maximum("exact_2mm_behind_tip_error_m", diag) * 1000
        if steps
        else None,
        "total_step_time_s": result["total_step_time_s"],
        "mean_step_time_s": result["mean_step_time_s"],
        "rollout_wall_time_s": result["total_rollout_time_s"],
        "sqp_iterations": sum(row["solver_iterations"] for row in result["steps"]),
    }


def analyse(root: Path) -> None:
    """Compare only saved S3 runs, using the same fine geometry/matching definitions."""
    baseline_settings = load_settings(BASELINE / "config_snapshot.json")
    baseline = json.loads((BASELINE / "result.json").read_text(encoding="utf-8"))
    results, settings_map = {"20": baseline}, {"20": baseline_settings}
    for spacing in SPACINGS:
        label = str(round(spacing * 1000))
        folder = root / (label + "mm")
        settings = load_settings(folder / "config_snapshot.json")
        if asdict(settings) != asdict(changed_settings(baseline_settings, spacing)):
            raise ValueError(f"Unexpected setting changes in {folder}")
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if not np.allclose(
            result["reference_yz_m"], baseline["reference_yz_m"], atol=1e-14, rtol=0
        ):
            raise ArithmeticError("Reference geometry changed")
        results[label], settings_map[label] = result, settings
    all_diagnostics, summaries, differences = {}, [], {}
    common_count = min(run["accepted_steps"] for run in results.values())
    for label, result in results.items():
        print(f"Diagnosing saved {label} mm states...", flush=True)
        diagnostics = sampling_diagnostics(settings_map[label], result)
        all_diagnostics[label] = diagnostics
        folder = root / ("baseline_20mm_analysis" if label == "20" else label + "mm")
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "sampling_diagnostics.csv", diagnostics)
        summary = summary_row(label, result, diagnostics)
        summaries.append(summary)
        write_json(folder / "summary.json", summary)
        differences[label] = sequence_difference(settings_map[label], result, baseline)
        write_json(folder / "sequence_difference_vs_baseline.json", differences[label])
        # Independent unrestricted point-to-polyline verification of fixed-reference
        # statistics on first, peak and final accepted frames; no change in metric.
        accepted = [row for row in result["steps"] if row["accepted"]]
        if accepted:
            for row in (
                accepted[0],
                max(accepted, key=lambda r: r["max_body_to_reference_m"]),
                accepted[-1],
            ):
                distances = point_to_polyline_distances(
                    np.array(row["body_yz_m"]), np.array(baseline["reference_yz_m"])
                )
                if abs(float(np.max(distances)) - row["max_body_to_reference_m"]) > 1e-13:
                    raise ArithmeticError("Fixed-reference maximum evaluation changed")
                if abs(float(np.mean(distances)) - row["mean_body_to_reference_m"]) > 1e-13:
                    raise ArithmeticError("Fixed-reference mean evaluation changed")
    common = [
        summary_row(label, result, all_diagnostics[label], common_count)
        for label, result in results.items()
    ]
    write_json(root / "comparison.json", summaries)
    write_csv(root / "comparison.csv", summaries)
    write_json(root / "common_prefix_comparison.json", common)
    write_csv(root / "common_prefix_comparison.csv", common)
    write_json(root / "sequence_comparison.json", differences)
    make_plots(root, results, all_diagnostics)
    write_reports(root, results, summaries, common, differences)
    verify_preservation(root)
    write_json(
        root / "verification.json",
        {
            "original_files_unchanged": True,
            "only_optimization_spacing_changed": True,
            "unchanged_fine_evaluation_m": baseline_settings.validation_spacing_m,
            "reference_and_target_geometry_unchanged": True,
            "fine_body_to_previous_recomputed_all_returned_valid_frames": True,
            "fixed_reference_metrics_cross_checked_first_peak_final": True,
            "global_common_accepted_steps": common_count,
            "no_s1_or_s2_run": True,
            "baseline_not_rerun": True,
            "no_new_default_adopted": True,
        },
    )
    print(json.dumps(summaries, indent=2), flush=True)


def make_plots(root: Path, results: dict, diagnostics: dict) -> None:
    """Static diagnostics, all from stored states; no extra optimization or animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"20": "#555555", "5": "#1680b8", "2": "#ed8b23", "1": "#23935a"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    qfig, qaxes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for label, result in results.items():
        rows = [row for row in result["steps"] if row["accepted"]]
        ds = [row for row in diagnostics[label] if row["accepted"]]
        s = [row["reference_progress_m"] * 1000 for row in rows]
        for ax, key in zip(
            axes.flat,
            ("max_body_to_reference_m", "max_body_to_previous_m", "tip_error_m"),
            strict=False,
        ):
            ax.plot(s, [row[key] * 1000 for row in rows], label=label + " mm", color=colors[label])
        axes[1, 1].plot(
            s,
            [row["fine_minus_optimization_m"] * 1000 for row in ds],
            label=label + " mm",
            color=colors[label],
        )
        for i, ax in enumerate(qaxes.flat):
            ax.plot(
                s,
                [np.rad2deg(row["configuration_rad"][i]) for row in rows],
                label=label + " mm",
                color=colors[label],
            )
    for ax, title in zip(
        axes.flat,
        (
            "Body to fixed reference: fine maximum",
            "Body to previous: fine maximum",
            "Actual tip to waypoint",
            "Fine body max minus optimization-grid max",
        ),
        strict=True,
    ):
        ax.set(xlabel="Scheduled progress (mm)", ylabel="Error (mm)", title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for i, ax in enumerate(qaxes.flat):
        ax.set(
            xlabel="Scheduled progress (mm)",
            ylabel="Signed bend (degrees)",
            title=f"Section {i + 1}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(root / "metric_comparison.png", dpi=150)
    plt.close(fig)
    qfig.savefig(root / "configuration_sequences.png", dpi=150)
    plt.close(qfig)
    for label in results:
        folder = root / ("baseline_20mm_analysis" if label == "20" else label + "mm")
        ds = [row for row in diagnostics[label] if row["accepted"]]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
        s = [row["reference_progress_m"] * 1000 for row in ds]
        for key, legend in (
            ("optimization_body_error_m", "Optimization-grid body max"),
            ("fine_body_error_m", "Fine body max"),
        ):
            axes[0].plot(s, [row[key] * 1000 for row in ds], label=legend)
        for key, legend in (
            ("optimization_near_tip_5mm_m", "Optimization grid: last 5 mm"),
            ("fine_near_tip_5mm_m", "Fine grid: last 5 mm"),
            ("exact_2mm_behind_tip_error_m", "Exact point 2 mm behind tip"),
        ):
            axes[1].plot(s, [row[key] * 1000 for row in ds], label=legend)
        for ax in axes:
            ax.set(xlabel="Scheduled progress (mm)", ylabel="Previous-body error (mm)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
        fig.suptitle(f"S3 optimization sampling: {label} mm")
        fig.savefig(folder / "sampling_gap_and_near_tip.png", dpi=150)
        plt.close(fig)


def write_reports(
    root: Path, results: dict, summaries: list[dict], common: list[dict], differences: dict
) -> None:
    """Document whole-run/common-prefix metrics, timing scope and unchanged definitions."""
    columns = (
        ("Body/fixed max mm", "body_to_reference_max_mm"),
        ("Mean frame-max body/fixed mm", "body_to_reference_mean_frame_max_mm"),
        ("Mean spatial body/fixed mm", "body_to_reference_mean_frame_spatial_mean_mm"),
        ("Body/previous fine max mm", "body_to_previous_fine_max_mm"),
        ("Tip max mm", "tip_error_max_mm"),
        ("Tip mean mm", "tip_error_mean_mm"),
        ("Max fine-minus-optimization mm", "fine_minus_optimization_max_mm"),
        ("Step computation total s", "total_step_time_s"),
    )

    def table(rows):
        lines = [
            "| Grid mm | Completed | Accepted | " + " | ".join(x[0] for x in columns) + " |",
            "|" + "---|" * (3 + len(columns)),
        ]
        for row in rows:
            values = [f"{row[key]:.7g}" if row[key] is not None else "N/A" for _, key in columns]
            lines.append(
                f"| {row['optimization_spacing_mm']:g} | {row['completed']} | "
                f"{row['accepted_steps']} | " + " | ".join(values) + " |"
            )
        return lines

    report = [
        "# S3 controlled optimization-sampling sensitivity",
        "",
        (
            "The historical 20 mm baseline was read only, not rerun. Three new runs "
            "started independently from the same straight state. Only "
            "optimization_spacing_m changed. No tuning, retries, squared-distance "
            "objective, S1/S2 runs, or default adoption."
        ),
        "",
        "## Whole accepted rollout",
        "",
        *table(summaries),
        "",
        (
            "If completion lengths differ, these whole-run extrema do not cover the same "
            "interval; use the common-prefix comparison below."
        ),
        "",
        f"## Common prefix: {common[0]['evaluated_accepted_steps']} accepted steps",
        "",
        *table(common),
        "",
        (
            "The accepted column is the total accepted run count; "
            "evaluated_accepted_steps in the CSV/JSON gives the common-prefix count used "
            "for this second table."
        ),
        "",
        "## Evaluation and causal controls",
        "",
        (
            "Validation remains 0.25 mm, previous-body comparison remains 1 mm, reference"
            " increment 2 mm, and previous_body matching, unsquared objective, tip weight"
            " 10, SQP options, bounds, first-entry rules and feed model are unchanged. "
            "The original run_s3 function is called without patches. Each snapshot is "
            "checked against the baseline for exactly one changed setting."
        ),
        (
            "Every saved valid trial's fine body-to-previous geometry and error is "
            "recomputed and checked. First/peak/final fixed-reference metrics are "
            "independently checked against the unchanged baseline reference. Fine error "
            "aggregation is unchanged: maximum, mean of frame maxima, and mean of frame "
            "spatial means are separate quantities."
        ),
        (
            "The optimization-grid body error is reconstructed at the returned state with"
            " the exact original coarse-grid objective formula and frozen comparison "
            "path. It is not a newly logged per-iteration optimizer trace. Fine and "
            "near-tip errors use the same comparison path. Last 5 mm and the exact point "
            "2 mm behind the tip are measured in material arc length."
        ),
        (
            "Changing histories/configurations can change the next previous-body "
            "reference: this is an outcome of the sampling intervention, not a change to "
            "its definition. The fixed-reference metric supplies the common spatial "
            "reference across all runs."
        ),
        "",
        "## Configuration changes vs baseline",
        "",
        (
            "Compared at equal scheduled progress over each run's common accepted prefix."
            " Body displacements use identical material coordinates over the full "
            "fixed-length 400 mm chain; feed and tip changes are also reported "
            "separately. These are deterministic effect sizes, not statistical "
            "significance tests."
        ),
        "",
        "```json",
        json.dumps(differences, indent=2),
        "```",
        "",
        "## Timing and failure interpretation",
        "",
        (
            "Step time includes SLSQP and original fine validation, excluding later "
            "independent fixed-reference metrics/plots. Wall rollout time is also saved. "
            "The baseline time is historical, not a contemporaneous repeated timing "
            "benchmark. Denser objectives increase function-evaluation work. Failed "
            "candidates remain in result.json; their errors are not included in "
            "accepted-trajectory aggregates. No failure is retried or bypassed."
        ),
        "",
        (
            "Data: comparison.csv/json, common_prefix_comparison.csv/json, "
            "sequence_comparison.json; metric_comparison.png and "
            "configuration_sequences.png. Each run directory contains snapshots, "
            "original-schema result.json and metrics.csv, progress.jsonl, "
            "sampling_diagnostics.csv, summary.json, and sampling_gap_and_near_tip.png."
        ),
        "",
        "Rebuild diagnostics without optimization:",
        "```cmd",
        ".venv\\Scripts\\python.exe -m continuum_ftl.benchmarks.palmer_s3_sampling --analyse-only",
        "```",
        "",
        (
            "Original defaults and historical results remain unchanged. No spacing is "
            "automatically selected."
        ),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    for row in summaries:
        label = str(round(row["optimization_spacing_mm"]))
        folder = root / ("baseline_20mm_analysis" if label == "20" else label + "mm")
        result = results[label]
        (folder / "REPORT.md").write_text(
            f"# S3 {label} mm optimization sampling\n\n"
            + "\n".join(table([row]))
            + f"\n\nFailure: `{result['failure_reason']}`.\n\n"
            f"Final q (rad): `{result['final_configuration_rad']}`. "
            f"Final tip (m): `{result['final_tip_yz_m']}`.\n\n"
            + (
                "Fine validation is unchanged at 0.25 mm; see ../REPORT.md for common-prefix "
                "comparisons and interpretation.\n"
            ),
            encoding="utf-8",
        )


def main() -> None:
    """Execute exactly the declared S3 study, or analyse saved results only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    if not (args.output_dir / "preservation_manifest.json").exists():
        parser.error("Prepare a preservation manifest before running this study")
    verify_preservation(args.output_dir)
    if not args.analyse_only:
        run_experiments(args.output_dir)
    analyse(args.output_dir)


if __name__ == "__main__":
    main()
