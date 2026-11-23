"""Run/replay the isolated Palmer S3 reconstruction; no other cases or collisions."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .palmer_overlay import save_red_overlay
from .palmer_s3 import S3Settings, load_settings
from .palmer_s3_solver import run_s3


def comparison_rows(result: dict) -> list[tuple[str, float, float | None, str]]:
    """Paper values live only in reporting, never in solver objectives or acceptance."""

    def mm(key):
        return result[key] * 1000 if result[key] is not None else None

    return [
        ("Max path deviation", 3.674, mm("max_body_to_reference_m"), "mm"),
        ("Average path deviation*", 1.225, mm("mean_of_frame_max_body_to_reference_m"), "mm"),
        ("Max tip error", 0.540, mm("max_tip_error_m"), "mm"),
        ("Average tip error", 0.129, mm("mean_tip_error_m"), "mm"),
        ("Total computation time**", 10.75, result["total_step_time_s"], "s"),
        ("Average time per optimization step**", 0.053, result["mean_step_time_s"], "s"),
    ]


def write_report(folder: Path, result: dict, settings: S3Settings) -> None:
    """Write compact Chinese report with explicit metric/deployment reconstruction caveats."""
    table = ["| Metric | Palmer S3 | Reconstructed S3 |", "|---|---:|---:|"]
    for label, published, actual, unit in comparison_rows(result):
        value = "N/A" if actual is None else f"{actual:.6g} {unit}"
        table.append(f"| {label} | {published:g} {unit} | {value} |")
    report = f"""# Palmer 2014 S3：独立重建实验

本实验只运行 S3，没有运行其他 profile，没有修改原三段机器人、Stage 2C 默认值或旧实验。
完成：**{result["completed"]}**；接受 {result["accepted_steps"]} 步，
拒绝 {result["rejected_steps"]} 步；
接受的参考弧长进度 {result["accepted_reference_progress_m"] * 1000:.6f}/400 mm。
停止原因：`{result["failure_reason"]}`。

{chr(10).join(table)}

* 论文未明确说明 Avg dev 的时空归约方式。本表重建值采用“每帧身体到固定参考的最大距离，
再在接受帧间取算术平均”。
此外，“每帧空间平均距离，
再在帧间平均”结果为 {result["mean_of_frame_mean_body_to_reference_m"]} m。
两者都不是已确认与论文完全同义的指标。原文 Fig.8 是全部姿态叠加后的轨迹包络，
不能由此推定其精确测量程序。

** 重建时间为各步 SQP、入口几何和高分辨率验收的总时间，
不包括独立固定参考距离评价、绘图、GIF 编码和配置读取。
含独立固定参考评价的完整 rollout 时间：{result["total_rollout_time_s"]:.6f} s。
原文使用 LabView SQP、i7-960 3.2 GHz；本实验为当前电脑的 Python/SciPy SLSQP，
不能直接用运行时间评判算法优劣。

## 已核对的原文与几何

来源：[Palmer, Cobos-Guzman & Axinte (2014)](https://doi.org/10.1016/j.robot.2014.05.013)，
Sections 3–5、Fig.8、Table 1（印刷页 1483）。核对了本地原文的表格图像和全文文本。
表格 S3 给出 `[0, pi/6, -pi/6, 0]`；正文式(2)使用前一段末端建立下一段坐标系。
但 Table 1 未单独标注该向量的物理近端/远端排序，Fig.8 图注也没有消除所有符号歧义。
本实验明确采用用户指定的近端到远端顺序、YZ 平面、正弯曲朝 +Y，参考基座 (0,0)、初始切线 +Z。

四段均 100 mm，主动总长 400 mm，无被动段；参考弯曲向量保持原值，不作为求解的终端角约束。
独立圆弧积分与共享 PCC FK 端点的最大差为
{result["geometry_validation"]["independent_circle_integral_residual_m"]:.6g} m。
参考端点 YZ（m）：`{result["geometry_validation"]["section_endpoints_yz_m"]}`。

## 经用户确认的理想化入口假设

- 入口为 Z=0 的理想界面，初始基座 (0,-0.400) m，初始 q 全零，tip 在 (0,0)，暴露长度为零。
- 段长始终完整 100 mm；无 20 cm 被动段、无 -0.08 m Feed、无三分区或整段释放规则。
- 初始远端第4段已在入口且角度为零。其他段的**远端端点第一次到达 Z=0 后**记录为已引入；
该帧仍为零角，下一步才参与优化。
- 未引入的近端链一直保持直线，
因此首次引入对应累计 feed 约 0、100、200、300 mm（顺序4、3、2、1）。
  程序按真实 FK 端点检查，不按期望参考进度判定引入。
- 已部分进入的段允许弯曲。入口不是实体导向孔，允许实际中心线与入口平面的交点 Y 不为零；
不声称满足真实 Feed 机构约束。
- 完整四段 PCC 用于 FK。身体匹配评价只取入口以上、连通至 tip 的曲线；
这是本 benchmark 的单入口定义，未调用原项目环境暴露实现。
  多次穿越入口、tip 退到入口以下视为几何失败。
- 引入标志只在接受之后更新；失败候选不进入参照历史。遇到求解/几何/控制越界失败立即停止，
不重试、不调参。
- 终止于 200 个 2 mm 参考推进步骤全部接受，或首个失败；`completed` 表示完成预定进度，
tip 偏差另行如实报告，不用论文误差值作门槛。

## 目标、对应和数值设置

主参照仅为 `previous_body`：冻结上一接受状态在入口以上的身体中心线，连接至新期望 tip，
期间不更新。
目标 `J = 10*tip_error + max_body_to_previous`；两项均为欧氏距离（m）。
输入增量按 2 mm/0.064 rad 归一化，J 除以 {settings.objective_scale_m} m 改善条件数。
调用已有 `palmer_objective`、`match_ordered_polyline`、PCC FK 和精确入口交点求解，
不调用原研究机器人的 deployment/FTL 优化器。
因为原 `solve_ftl_step` 强耦合原释放模型，
本 benchmark 保留相同的 SLSQP 机制并单独编排变量/引入逻辑。

候选身体按 {settings.optimization_spacing_m * 1000:g} mm 采样，
旧身体参照 {settings.comparison_spacing_m * 1000:g} mm，
验收和固定参考评价 {settings.validation_spacing_m * 1000:g} mm，包含精确段边界与入口。
按距 tip 的剩余弧长估计对应位置，
在 ±{settings.matching_window_m * 1000:g} mm 窗口内做点到线段投影，近端到远端对应不后退。
与原文 Fig.5 离散索引梯度搜索相比，这是局部有序的连续线段投影重建，并非原算法的逐行复刻。
独立固定参考指标是无局部窗口的点到折线距离；所有“最大值”均为采样估计，
不是连续曲线最大偏差的数学证明。

期望参考弧长每步固定 2 mm；**实际 tip 位移和优化 feed 是另外两项**，逐步保存在 metrics.csv。
本次 feed 允许
{settings.min_feed_increment_m * 1000:g}～{settings.max_feed_increment_m * 1000:g} mm，
是优化变量，不是强制每步 2 mm。
原文 Section 3.2.3 给出 δb 可正可负 ±2 mm；本次按任务指定采用非负 feed。
原文没有完整给出 S3 每步 δp、初始暴露长度和引入细节；因此本实验属于声明了假设的 S3 重建，
不能称为严格数值复现。

## 如何解读差异

已验证：参考几何、固定段长、控制/绝对弯曲边界、零角引入、冻结参照和实际 FK 状态。
数值差异可能来自入口重建、期望弧长与实际位移的区别、匹配窗口/采样/投影、
非光滑 max 与欧氏范数目标、不同 SQP 终止条件及路径偏差统计方式。
这些是解释候选，**本次单次实验不能确定各因素的贡献，也不能把结果差异本身当成 solver bug**。
若出现失败，原始 solver_status/message、迭代数及失败候选均保留在 result.json；
不将失败候选混入接受状态统计。

## 文件与复跑

- `result.json`：全部真实候选/接受状态、曲线、引入状态、控制量和诊断。
- `metrics.csv`：逐步路径进度、实际 tip 位移、feed、两种身体误差、tip 误差、求解状态及耗时。
- `overlay.png`：参考、实际 tip 历史、若干接受形状；`body_deviation.png` / `tip_error.png`：
误差曲线。
- `motion.gif`：真实计算状态动画；`config_snapshot.json`：此次配置快照。
- 配置入口 `configs/benchmarks/palmer_s3.json`；求解 `palmer_s3_solver.py:solve_step/run_s3`；
几何 `palmer_s3.py:geometry/validate_reference`。

在项目根目录运行 `.venv\\Scripts\\python.exe -m continuum_ftl.benchmarks.palmer_s3_demo`，
默认计算并显示交互动画；
`--replay outputs/palmer_s3_experiment1/result.json` 只回放已保存状态，不重新优化。
"""
    (folder / "REPORT.md").write_text(report, encoding="utf-8")


def make_figures(
    result: dict, folder: Path, settings: S3Settings, save_gif: bool, show: bool
) -> None:
    """Plot/animate only stored FK states, including the first failed candidate if any."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.widgets import Button

    profile = result.get("profile", "S3")
    accepted = [row for row in result["steps"] if row["accepted"]]
    reference = np.array(result["reference_yz_m"]) * 1000
    save_red_overlay(result, folder / "overlay.png")
    progress = [row["reference_progress_m"] * 1000 for row in accepted]
    for filename, fields, title in (
        (
            "body_deviation",
            [
                ("max_body_to_reference_m", "Max body to fixed reference"),
                ("mean_body_to_reference_m", "Mean body to fixed reference"),
                ("max_body_to_previous_m", "Max body to previous-body comparison"),
            ],
            f"{profile}: distinct body deviation metrics",
        ),
        (
            "tip_error",
            [("tip_error_m", "Actual tip to current waypoint")],
            f"{profile}: actual waypoint error",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        for field, label in fields:
            ax.plot(progress, [row[field] * 1000 for row in accepted], label=label)
        ax.set(xlabel="Desired reference progress (mm)", ylabel="Error (mm)", title=title)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(folder / (filename + ".png"), dpi=140)
        plt.close(fig)

    frames = [result["initial_state"], *result["steps"]]
    fig, ax = plt.subplots(figsize=(7, 8))
    fig.subplots_adjust(bottom=0.14, top=0.80)
    ax.plot(*reference.T, "--", color="#713b8f", label=f"Fixed {profile} reference")
    ax.axhline(0, color="gray", ls=":", label="Ideal entrance Z=0")
    (body,) = ax.plot([], [], lw=2.5, color="#20578c", label="Computed exposed body")
    (tip,) = ax.plot([], [], "o", color="#c42e2e", label="Actual tip")
    (waypoint,) = ax.plot([], [], "x", color="green", label="Desired waypoint")
    (junctions,) = ax.plot([], [], "o", color="#555555", ms=4)
    (trace,) = ax.plot([], [], color="#d96d28", lw=1, label="Accepted tip history")
    (outside,) = ax.plot([], [], color="#bcbcbc", lw=1, label="Full body below entrance")
    ax.set(xlabel="World Y (mm)", ylabel="World Z (mm)", xlim=(-80, 100), ylim=(-410, 420))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    fig.legend(
        *ax.get_legend_handles_labels(),
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.80),
    )
    status = fig.text(0.05, 0.97, "", va="top", family="monospace", fontsize=8)
    control = {"index": 0, "paused": False}

    def draw(index):
        row = frames[index]
        control["index"] = index
        points = np.asarray(row["body_yz_m"]).reshape(-1, 2) * 1000
        body.set_data(*points.T)
        body.set_color("#20578c" if row["accepted"] else "#c42e2e")
        tip.set_data(*np.array(row["actual_tip_yz_m"])[:, None] * 1000)
        waypoint.set_data(*np.array(row.get("desired_tip_yz_m", [0, 0]))[:, None] * 1000)
        endpoints = np.array(row["section_endpoints_yz_m"]) * 1000
        junctions.set_data(*endpoints.T)
        # Unintroduced sections are straight. Below-interface full PCC geometry
        # is regenerated from stored q/feed only, never optimized in the viewer.
        from ..planar_geometry import evaluate_curve, sample_interval
        from .palmer_s3 import geometry

        curve = geometry(settings, np.array(row["configuration_rad"]), row["feed_m"])
        full = (
            evaluate_curve(
                curve,
                sample_interval(
                    curve, 0, 4 * settings.section_length_m, settings.validation_spacing_m
                ),
            )
            * 1000
        )
        outside.set_data(*np.where((full[:, 1] < 0)[:, None], full, np.nan).T)
        historical = (
            np.array(
                [frame["actual_tip_yz_m"] for frame in frames[: index + 1] if frame["accepted"]]
            )
            * 1000
        )
        trace.set_data(*historical.T)
        deviation = row.get("max_body_to_reference_m")
        deviation_mm = deviation * 1000 if deviation is not None else float("nan")
        status.set_text(
            f"Palmer {profile} reconstruction | frame {index + 1}/{len(frames)}\n"
            f"Reference s={row['reference_progress_m'] * 1000:.2f} mm | "
            f"feed={row['feed_m'] * 1000:.2f} mm\n"
            f"Introduced (proximal to distal): {row['introduced']}\n"
            f"Tip error={row.get('tip_error_m', 0) * 1000:.4g} mm | "
            f"max body/fixed={deviation_mm:.4g} mm\n"
            f"{row.get('failure_reason') or 'ACCEPTED'}"
        )
        return body, tip, waypoint, junctions, trace, outside, status

    animation = FuncAnimation(
        fig,
        draw,
        frames=len(frames),
        init_func=lambda: draw(0),
        interval=settings.frame_interval_ms,
        repeat=False,
    )
    replay = Button(fig.add_axes([0.18, 0.04, 0.25, 0.05]), "Replay")
    pause = Button(fig.add_axes([0.55, 0.04, 0.25, 0.05]), "Pause / Resume")

    def restart(_):
        animation.frame_seq = animation.new_frame_seq()
        control["paused"] = False
        draw(0)
        animation.event_source.start()

    def toggle(_):
        control["paused"] = not control["paused"]
        if control["paused"]:
            animation.event_source.stop()
        else:
            animation.event_source.start()

    replay.on_clicked(restart)
    pause.on_clicked(toggle)
    draw(len(frames) - 1)
    fig.savefig(folder / "animation_last_frame.png", dpi=130, bbox_inches="tight")
    if save_gif:
        animation.save(
            folder / "motion.gif",
            writer=PillowWriter(fps=max(1, round(1000 / settings.frame_interval_ms))),
        )
    if show:
        draw(0)
        plt.show()
    else:
        animation._draw_was_started = True
    plt.close(fig)


def main() -> None:
    """Compute one S3 reconstruction, or replay an existing S3 result without optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmarks/palmer_s3.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/palmer_s3_experiment1"))
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    if args.replay:
        folder = args.replay.parent
        settings = load_settings(folder / "config_snapshot.json")
        result = json.loads(args.replay.read_text(encoding="utf-8"))
        if result.get("benchmark") != "Palmer 2014 S3 only":
            parser.error("Replay accepts only a saved Palmer S3 benchmark")
    else:
        folder = args.output_dir
        if (folder / "result.json").exists():
            parser.error("Result already exists; use --replay or a new --output-dir")
        folder.mkdir(parents=True, exist_ok=True)
        settings = load_settings(args.config)
        (folder / "config_snapshot.json").write_text(
            json.dumps(asdict(settings), indent=2), encoding="utf-8"
        )

        def progress(row, count):
            if row["step_index"] % 20 == 0 or not row["accepted"]:
                print(
                    f"S3 step {row['step_index']}/{count}: accepted={row['accepted']}, "
                    f"tip_error={row['tip_error_m'] * 1000:.4g} mm, "
                    f"{row['failure_reason'] or ''}",
                    flush=True,
                )

        result = run_s3(settings, progress)
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
        write_report(folder, result, settings)
        manifest = folder / "original_files_manifest.json"
        if manifest.exists():
            original = json.loads(manifest.read_text(encoding="utf-8"))
            changed = [
                path
                for path, digest in original.items()
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            ]
            if changed:
                raise RuntimeError(f"Original source/config files changed: {changed}")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in ("steps", "reference_yz_m", "reference_arc_lengths_m", "initial_state")
            },
            indent=2,
        ),
        flush=True,
    )
    make_figures(result, folder, settings, args.save_gif, not args.no_show)
    print(f"Saved S3 results: {folder}", flush=True)


if __name__ == "__main__":
    main()
