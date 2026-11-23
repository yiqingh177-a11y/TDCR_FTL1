"""Experiment 3: S1 once, with the unchanged original S2/S3 numerical engine."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np

from .palmer_s1_viewer import make_s1_figures
from .palmer_s3 import load_settings, reference_geometry, validate_reference
from .palmer_s3_solver import run_s3


def write_json(path: Path, value: object) -> None:
    """Save finite diagnostics as UTF-8 JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    """Validate geometry, run S1 once, then render saved states without retries."""
    matplotlib.use("Agg")
    folder = Path("outputs/palmer_s1_experiment3")
    if folder.exists():
        raise FileExistsError("S1 output exists; refusing a repeated run")
    settings = load_settings(Path("configs/benchmarks/palmer_s1.json"))
    previous = {}
    for profile, dirname in (("S3", "palmer_s3_experiment1"), ("S2", "palmer_s2_experiment2")):
        source = Path("outputs") / dirname
        baseline = json.loads((source / "config_snapshot.json").read_text())
        current = asdict(settings)
        baseline.pop("reference_bends_rad")
        current.pop("reference_bends_rad")
        if baseline != current:
            raise ValueError("Only the reference profile may change")
        previous[profile] = json.loads((source / "result.json").read_text())
    if settings.reference_bends_rad != [0, np.pi / 2, -np.pi / 2, 0]:
        raise ValueError("This experiment accepts S1 only")
    validation = validate_reference(settings)
    radius = 0.1 / (np.pi / 2)
    expected = np.array(
        [
            [0, 0],
            [0, 0.1],
            [radius, 0.1 + radius],
            [2 * radius, 0.1 + 2 * radius],
            [2 * radius, 0.2 + 2 * radius],
        ]
    )
    curve = reference_geometry(settings)
    residual = float(np.max(abs(curve.junction_positions_m - expected)))
    rotation = np.eye(2)
    for bend in settings.reference_bends_rad:
        rotation = rotation @ np.array(
            [[np.cos(bend), np.sin(bend)], [-np.sin(bend), np.cos(bend)]]
        )
    if residual > 1e-14 or np.max(abs(rotation - np.eye(2))) > 1e-14:
        raise ArithmeticError("Independent quarter-circle/rotation check failed")
    validation.update(
        independent_quarter_circle_endpoints_yz_m=expected.tolist(),
        independent_endpoint_residual_m=residual,
        final_tip_angle_rad=float(curve.junction_angles_rad[-1]),
        final_tip_tangent_yz=[0.0, 1.0],
        independent_rotation_matrix=rotation.tolist(),
        validated_before_optimization=True,
    )
    # Preserve both previous outputs and all pre-existing project source/config.
    paths = [
        p
        for root in ("continuum_ftl", "configs", "tests")
        for p in Path(root).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    paths += [
        p
        for root in ("outputs/palmer_s2_experiment2", "outputs/palmer_s3_experiment1")
        for p in Path(root).rglob("*")
        if p.is_file()
    ]
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    folder.mkdir()
    write_json(folder / "preservation_manifest.json", manifest)
    write_json(folder / "reference_validation_before_run.json", validation)
    write_json(folder / "config_snapshot.json", asdict(settings))
    print("Geometry validated before optimization:", expected.tolist(), flush=True)

    def progress(row: dict, count: int) -> None:
        if row["step_index"] % 10 == 0 or not row["accepted"]:
            print(
                f"S1 {row['step_index']}/{count}: accepted={row['accepted']} "
                f"{row['failure_reason'] or ''}",
                flush=True,
            )

    result = run_s3(settings, progress)
    result.update(profile="S1", benchmark="Palmer 2014 S1 only", geometry_validation=validation)
    write_json(folder / "result.json", result)
    excluded = {"body_yz_m", "material_arcs_m", "comparison_path_yz_m", "section_endpoints_yz_m"}
    rows = [{k: v for k, v in row.items() if k not in excluded} for row in result["steps"]]
    with (folder / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [
        ("Maximum path deviation", "max_body_to_reference_m", 19.500),
        ("Average frame-maximum path deviation*", "mean_of_frame_max_body_to_reference_m", 7.170),
        ("Maximum tip waypoint error", "max_tip_error_m", 1.290),
        ("Average tip waypoint error", "mean_tip_error_m", 0.475),
    ]
    comparison = [
        {
            "metric": label,
            "palmer_s1_mm": paper,
            "reconstructed_s1_mm": result[key] * 1000 if result[key] is not None else None,
        }
        for label, key, paper in metrics
    ]
    write_json(folder / "comparison.json", comparison)
    with (folder / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    trend = {
        name: {
            "completed": run["completed"],
            "accepted_steps": run["accepted_steps"],
            "maximum_body_deviation_mm": run["max_body_to_reference_m"] * 1000,
        }
        for name, run in {**previous, "S1": result}.items()
    }
    write_json(folder / "curvature_trend.json", trend)
    report = [
        "Palmer S1: one unchanged-engine run",
        "",
        f"Completed={result['completed']}; accepted={result['accepted_steps']}/200; "
        f"failure={result['failure_reason']}",
        f"Step computation time={result['total_step_time_s']:.3f} s.",
        "",
        "Metric | Palmer S1 (mm) | Reconstructed S1 (mm)",
    ]
    report += [
        f"{r['metric']} | {r['palmer_s1_mm']} | {r['reconstructed_s1_mm']}" for r in comparison
    ]
    report += [
        "",
        "Existing S3 / S2 and this S1:",
        json.dumps(trend, indent=2),
        "",
        "* Average is the mean of accepted frame maxima, not a verified exact match to "
        "the paper's reduction.",
        "Failed runs cover only their accepted prefix; do not infer a full-path trend "
        "from that prefix.",
        "Tip error is actual FK tip to scheduled waypoint. Fixed-reference body error is "
        "distinct from the native previous-body objective.",
        "Only reference_bends_rad changed. Optimization spacing remains the original 20 "
        "mm; fine evaluation remains 0.25 mm.",
        "No sensitivity-study settings were adopted. No parameter tuning, retries or "
        "other benchmark runs.",
        "Reference/body overlay shows all accepted centreline frames as a red envelope; "
        "animation includes any failed candidate.",
    ]
    (folder / "REPORT.txt").write_text("\n".join(report), encoding="utf-8")
    print(
        "RESULT",
        json.dumps(
            {
                "comparison": comparison,
                "trend": trend,
                "failure": result["failure_reason"],
                "time_s": result["total_step_time_s"],
            }
        ),
        flush=True,
    )
    make_s1_figures(result, folder, settings.frame_interval_ms)
    changed = [
        p
        for p, digest in manifest.items()
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest
    ]
    write_json(folder / "preservation_check.json", {"checked": len(manifest), "changed": changed})
    if changed:
        raise RuntimeError(f"Original files changed: {changed}")
    print("S1 saved; original files unchanged.", flush=True)


if __name__ == "__main__":
    main()
