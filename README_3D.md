# 独立三维版本：Stage 2A–2C

二维入口与参数保留。三维版本使用独立的 **`configs/robot_3d.json`**，不修改二维参数。
显示为 Matplotlib 三维动画，支持鼠标旋转视角。

## 直接运行

在 `D:\Code\rr\TDCR_FTL1` 下运行三维阶梯参考路径的 FTL：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d
```

先计算完整三维 SQP 状态序列，再打开动画。窗口中的 Replay 只重播已有状态。
可以用 Pause、Previous、Next 检查逐帧状态；改参数后需要重新启动程序。

选择其他原有参考轨迹：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --case s-bend
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --case single-bend
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --case straight
```

### 新增：非共面的空间 S 轨迹

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --case s-bend-3d
```

原来的 `s-bend` 仍是旋转后的平面 S 轨迹。新增的 `s-bend-3d` 改变各段的局部弯曲方向，
形成非共面的空间曲线；不是更换相机角度。两者都使用真正的三维 FTL 求解。
新轨迹保持 C/B/A 段长 0.08/0.06/0.04 m、总弧长 0.18 m，以及原 S 轨迹的弯曲量
0.9/1.3/0.8 rad。被动段、Feed 出口、送入规则、求解权重和误差设置均保持原值。

在 **`configs/robot_3d.json` → `spatial_reference.case_bends_xy_rad.s-bend-3d`** 中修改形状：

```json
"s-bend-3d": [
  [0.0, 0.9],
  [0.9192388155425119, -0.9192388155425119],
  [-0.273616114660535, 0.7517540966287268]
]
```

三行依次为 C/B/A，每行是该段局部坐标中的 `[bend_x, bend_y]`，单位 rad。
`theta = sqrt(bend_x² + bend_y²)`；`phi = atan2(bend_y, bend_x)`。
这组参数对应未整体旋转前的局部弯曲方向 90°、−45°、110°。
随后仍按 `plane_rotation_deg` 整体绕世界 Z 轴旋转 45°，但空间曲线本身没有共同平面。
后段的位置和方向由前段末端位姿连续串接。每行的模不能超过相应 `max_bend_rad`。

该 case 的 target 自动取所配置曲线的末端，修改形状时一同更新；参考构型不作为 FTL 的强制终端角度。
输出默认在 `outputs/runs_3d/s-bend-3d/`；加 `--save-gif` 可导出 `motion.gif`。
`result.json` 中的 `reference_planarity` 记录最佳拟合平面偏差和非共面判断。
此时 `maximum_distance_from_reference_plane_m` 为 `null`，因为不存在单一参考平面；
FTL 的 tip/body error 仍按三维距离正常计算、记录和显示。

Stage 2A 手动送入/释放演示，与 Stage 2B 参考路径/环境暴露演示：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --stage deployment
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d --stage reference
```

这两个阶段只展示几何与参考，不执行 FTL。默认 `--stage ftl` 才是实际三维 SQP。
`--save-gif` 可额外保存 GIF，`--no-show` 仅导出，`--output-dir 路径` 可指定独立输出目录。
默认输出为 `outputs/runs_3d/<case>/`，手动阶段则为 `<case>_deployment` 或 `<case>_reference`。
不覆盖二维演示的默认输出。

## 当前设置与参数位置

以下参数都在 **`configs/robot_3d.json`** 中。

| 内容 | 参数位置 | 当前值/含义 |
|---|---|---|
| 主动段长度 | `robot.sections[].length_m` | C/B/A = 0.08/0.06/0.04 m |
| 主动段半径 | `robot.sections[].radius_m` | 各 0.004 m，即直径 8 mm |
| 各段最大总弯曲角 | `robot.sections[].max_bend_rad` | 各 π/2；约束合成弯曲量，而非两个分量各自 90° |
| 被动段 | `deployment.passive_length_m / passive_radius_m` | 0.20/0.004 m，始终为直段 |
| Feed 出口 | `deployment.feed_exit_m` | 世界 XYZ `[0,0,-0.08]` m |
| 环境入口 | `deployment.environment_entry_z_m` | 世界 Z=0 m |
| 参考平面的旋转 | `spatial_reference.plane_rotation_deg` | 绕世界 +Z 按右手规则旋转 +45° |
| 原目标坐标 | `spatial_reference.target_yz_m` | 原 YZ `[0.1409612134,0.075]` m，自动转成世界 XYZ |
| 四条参考曲线的弯曲形状 | `spatial_reference.case_bends_rad` | 保留原 straight/single-bend/s-bend/staircase 的有符号弯曲角 |
| 最终位置容差 | `target.tolerance_m` | 0.001 m；不另填 `target.position_m`，避免两个目标来源 |
| δb 下界/上界 | `ftl.min_feed_increment_m / max_feed_increment_m` | -0.002/+0.002 m |
| 每步最大弯曲变化 | `ftl.max_bend_increment_rad` | 每段二维弯曲增量的模不超过 0.128 rad |
| 参考推进步长 | `ftl.reference_step_m` | 0.001 m |
| 误差权重 | `ftl.tip_weight` | 当前 10，目标为 tip_weight × tip_error + max_body_error |
| 身体参照 | `ftl.body_reference_mode` | `tip_history` 或 `previous_body`，均使用 XYZ 距离 |
| 误差是否作为停止门槛 | `ftl.enforce_error_limits` | false，保留误差记录；几何与求解检查仍执行 |
| 身体/参照采样 | `ftl.*_spacing_m` | 与二维含义相同，使用物理弧长采样 |
| 求解预算与差分 | `ftl.max_iterations / solver_tolerance / finite_difference_step` | 三维 SQP 的数值设置 |
| 手动演示弯曲目标 | `deployment.bend_targets_rad` | `(n,2)` 局部弯曲分量，仅用于 Stage 2A/B 手动几何演示 |
| 外部障碍物 | `environment.obstacles` | 空列表，本轮不建立外部障碍物 |
| 相机视角 | `visualization.elevation_deg / azimuth_deg` | 仅改变观察方向，不旋转模型或轨迹 |

`robot.base_position_m` 是参考形状/Stage 1 的世界基座位置；实际送入时的位置仍由 Feed 出口和 d 计算。
`robot.initial_configuration_rad` 延续原 Stage 1 三维 IK 的 `(theta,phi)` 初猜格式；
FTL 从直线入场开始，优化变量和日志使用 `(bend_x,bend_y)`，不使用这组 IK 初猜决定初始形状。

## 参考路径如何旋转

原曲线位于世界 YZ 平面，将 `[y,z]` 嵌入为 `[0,y,z]`，然后进行 Z 轴刚体旋转：

```text
X = -sin(45°) × y
Y =  cos(45°) × y
Z = z
```

因此弧长、各段长度、有符号弯曲序列及沿路径的前进方向均保持，送入方向仍为世界 +Z。
改变 `plane_rotation_deg` 时，参考路径和目标一起旋转，相机参数独立。
本次使用原目标构型的 PCC 弯曲参数生成参考，不重新找另一组 IK 形状，以免改变原轨迹。
参考构型的弯曲角仍不是最终机器人的强制终端约束。

## 真正三维求解的含义

每段使用两个局部**积分弯曲分量** `bend_x、bend_y`，单位 rad，表示朝局部 X/Y 方向弯曲：

```text
theta = sqrt(bend_x² + bend_y²)
kappa_x = bend_x / section_length
kappa_y = bend_y / section_length
```

它们不是普通刚性关节的姿态角。没有独立扭转自由度。
所有段释放后，SQP 同时优化 **δd + 3×2 个弯曲增量，共 7 个变量**。
段的绝对弯曲上限和单步变化上限都是二维圆盘约束，方向任意。

参考路径是旋转后的平面曲线，但**机器人不受参考平面约束**。目标函数使用三维欧氏距离，
FK 通过三维齐次变换逐段相乘。每一步均由三维求解器计算，未调用二维 FTL 求解后再旋转显示。
日志和窗口记录 `maximum_distance_from_reference_plane_m`，便于观察实际形状的离面偏差。

三分区、完整主动段、20 cm 直被动段、整段出 Feed 后才允许弯曲、部分段的环境暴露均保留。
暴露域通过空间 PCC 圆弧与世界 Z=0 的连续几何交点确定，不是按采样点正负筛选。
FTL 身体误差只评价进入环境的主动部分，被动段仍参与完整 FK。

## 输出与回放

- `result.json`：全部实际状态、XYZ 中心线与节点位姿、两分量弯曲、送入量、每步误差与求解状态。
- `metrics.csv`：独立的身体到固定参考路径/实际 tip 历史的几何距离。
- `summary.png`：最后接受状态的三维图。
- `errors.png`：tip/body error 随参考进度的曲线。
- `motion.gif`：开启 `--save-gif` 时生成的真实状态回放。
- `config_snapshot.json`：本次原始配置；`effective_ftl_settings` 在结果中保留完整求解参数。

求解失败仍保留失败候选并停止，失败状态不进入接受历史。
到达 target 只表示本次位置判定成功，不意味着身体误差为零。
本轮没有三维碰撞检查，也不启用二维的可选“保持 waypoint 整形”实验。
紫色 Feed 示意块与入口网格均为坐标/送入示意，不是外部实体障碍物。

## 模块结构与验证

- `spatial_geometry.py`：稳定的空间 PCC、连续材料坐标求值、入口交点和暴露域。
- `spatial_deployment.py`：完整物理链、三维节点 pose、Feed 送入与整段释放。
- `spatial_reference.py`：原有四条参考形状的刚体旋转，保持解析曲线与弧长。
- `spatial_ftl_optimizer.py`：真正三维的增量 SQP 和独立的物理约束验收。
- `spatial_ftl_rollout.py`：入场、参考推进、身体参照与状态序列管理。
- `spatial_viewer.py`：读取实际三维状态的交互回放，支持鼠标转视角。
- `ftl_demo_3d.py`：独立三维命令入口、结果导出与阶段选择。

配置、数据结构、弧长采样、距离匹配与报告代码做了兼容 2D/3D 的扩展。
二维部署、FK、FTL 优化器及原有入口继续使用原实现。
测试覆盖原有版本回归、旋转等距性、非共面 FK、平面外目标的真实求解、零曲率、
整段释放/回撤、三维圆盘约束、入口重入、失败处理和三维动画导出。

本次四种轨迹的实测结果、三维 GIF 和验证记录：
[三维 Stage 2A–2C 结果](outputs/spatial_stage2c_2026-11-22/RESULTS.md)。
