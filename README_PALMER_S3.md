# Palmer 2014：独立 S3 重建实验

只运行四段、每段 100 mm 的平面 S3，不使用原 Stage 2C 的三段机器人或送入规则。
本实验使用 `previous_body`，参考形状为 `[0, pi/6, -pi/6, 0]`，近端到远端，正弯曲朝 +Y。

已经计算的结果及完整假设说明见
[实验报告](outputs/palmer_s3_experiment1/REPORT.md)。

## 直接看已计算的动画

在 `D:\Code\rr\TDCR_FTL1` 中运行：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.benchmarks.palmer_s3_demo --replay outputs/palmer_s3_experiment1/result.json
```

这条命令只回放已有真实 FK 状态，不重新优化。窗口提供 Replay 和 Pause / Resume。
`motion.gif` 也可以直接打开。

## 重新计算一次 S3

```cmd
.venv\Scripts\python.exe -m continuum_ftl.benchmarks.palmer_s3_demo --output-dir outputs/palmer_s3_new_run --save-gif
```

为保护已有实验，程序拒绝覆盖已经存在的 `result.json`；请使用新的输出目录。
加 `--no-show` 则仅导出图像和结果。

## 配置位置

所有 S3 配置位于 **`configs/benchmarks/palmer_s3.json`**，不是 `robot_2d.json` 或 `robot_3d.json`。

| 参数 | 单位与作用 |
|---|---|
| `section_length_m` | 每个完整主动段长度，默认 0.100 m，固定四段 |
| `reference_bends_rad` | 固定参考形状，近端到远端的四个有符号弯曲角 |
| `max_bend_rad` | 每段绝对弯曲上限，默认 π/2 rad |
| `desired_path_step_m` | 每步期望参考弧长增量，默认 0.002 m |
| `min_feed_increment_m / max_feed_increment_m` | 本次重建采用每步 0～0.002 m 的可优化送入量 |
| `max_bend_increment_rad` | 每步每段弯曲增量绝对值上限 0.064 rad |
| `tip_weight` | 目标 `tip_weight × tip_error + max_body_to_previous` 的 tip 权重，默认 10 |
| `optimization_spacing_m` | 优化时身体采样间距，默认 0.020 m |
| `comparison_spacing_m` | 上一接受身体参照的采样/加密间距，默认 0.001 m |
| `validation_spacing_m` | 独立验收及固定参考距离评价间距，默认 0.00025 m |
| `matching_window_m` | 局部有序对应的弧长搜索半窗口，默认 0.020 m |
| `objective_scale_m` | 目标函数除以此数值改善条件数，默认 0.001 m |
| `max_iterations / solver_tolerance / finite_difference_step` | SLSQP 迭代预算、ftol、有限差分 eps |
| `acceptance_tolerance` | 独立控制边界验收的数值容差；相应变量分别使用 m / rad |
| `geometry_tolerance_m` | 入口交点与新段引入的几何容差 |
| `frame_interval_ms` | 动画每帧显示间隔，不影响数值轨迹 |

原始结果保持这些初始重建值；没有为了贴合论文结果而自动调参。

## 模块

- `continuum_ftl/benchmarks/palmer_s3.py`：独立配置、完整四段 PCC 几何和独立圆弧积分验证。
- `palmer_s3_solver.py`：理想入口、逐段引入、冻结上一身体参照、SLSQP 和逐步诊断。
- `palmer_s3_demo.py`：单次 S3 入口、报告、图像和真实状态回放。

共享稳定的 PCC FK、入口交点算法、局部有序距离匹配和独立点到折线距离函数。
原 Stage 2C 的 deployment、优化器、配置与显示代码不改动。

## 评价口径

`max_body_to_reference_m` 是当前暴露身体到固定 S3 参考曲线的采样最大距离。
`max_body_to_previous_m` 是当前身体对上一步身体加新目标的局部有序匹配偏差。
两者分别保存，不能混为同一指标。
报告同时列出“各帧最大距离的平均值”和“各帧空间平均距离的平均值”，因为论文未充分说明 Avg dev 的归约方式。

每行 `metrics.csv` 分别记录 `desired_path_increment_m`、`actual_tip_displacement_m` 和 `feed_increment_m`。
它们不是同一个量。此实验无碰撞检测，也不把论文给出的误差作为验收门槛。
