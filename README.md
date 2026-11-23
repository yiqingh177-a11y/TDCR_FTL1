# TDCR_FTL1 / continuum_ftl

**输出目录看不清时，先读 [outputs 导航指南](outputs/README.md)。**

**二维参数修改 [configs/robot_2d.json](configs/robot_2d.json)；独立三维版本修改
[configs/robot_3d.json](configs/robot_3d.json)。**

三维 Stage 2A–2C：`.venv\Scripts\python.exe -m continuum_ftl.ftl_demo_3d`。
运行、参数和方法见 [三维版本指南](README_3D.md)；现有二维入口继续保留。

二维和三维 FTL 窗口、汇总图现在均显示红色身体运动扫掠区域，随播放进度累积；
Replay 从头重画。三维使用半透明红色面，只统计已接受状态中进入环境的身体中心线。
该图是离散帧间的显示近似，不含机器人半径，也不替代碰撞检查。

已有结果无需重新求解即可生成红色图（在结果旁保存为 `red_sweep.png`）：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.sweep_visualization outputs/runs/staircase_03/result.json
.venv\Scripts\python.exe -m continuum_ftl.sweep_visualization outputs/spatial_s_bend_2026-11-23/result.json
```

**Stage 3A：已有 FTL 轨迹的完整机器人静态碰撞后处理**，运行方式、结果与计数定义见
[Stage 3A 指南](README_STAGE3A.md)。它不重新优化或修改原轨迹。

- `continuum_ftl/config.py`：读取和校验配置的代码，不是日常填写参数的地方。
- `outputs/*.json`：运行结果和日志，不是参数文件；修改它们不会改变下一次仿真。
- `configs/ftl_validation.json`：只用于 `straight / single-bend / s-bend` 三个测试用例的参考弯曲角。
- `configs/example.json`：原有 Stage 1 的 3D 配置，与当前 2D FTL 配置分开。

## 运行当前 FTL

在 Windows CMD 中执行：

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --config configs/robot_2d.json
```

改完参数后关闭旧窗口，重新运行。Replay 只重播旧窗口里已经计算好的状态，不重新读取配置。

当前使用 `scheduled`：参考路点按步推进，实际送入与弯曲由优化器求解。
误差显示并记录，超限不停止；障碍物仅显示，不参与 FTL 优化或停止判定。
到达 target 不代表整个身体精确跟随路径。

## 参数修改速查

以下 JSON 路径都在 **`configs/robot_2d.json`** 中。
世界坐标数组按 `[Y, Z]` 排列：Y 水平、Z 向上；向下移动就是减小 Z。
长度使用 **m**，角度使用 **rad**，回放间隔使用 **ms**。

### 机器人、target 与障碍物

| 想修改什么 | JSON 参数路径 | 当前值 / 写法 |
|---|---|---|
| C / B / A 段长度 | `robot.sections[0/1/2].length_m` | 0.08 / 0.06 / 0.04 m，近端到远端 |
| 每段半径 | `robot.sections[i].radius_m` | 0.004 m；直径 = 2 × 半径，目前为 8 mm |
| 每段最大弯曲角 | `robot.sections[i].max_bend_rad` | 1.5707963268，即允许正负 90° |
| 被动段长度、半径 | `deployment.passive_length_m / passive_radius_m` | 0.20 / 0.004 m，被动段始终直线 |
| 目标位置 | `target.position_m` | `[0.1409612134, 0.075]`，世界 `[Y,Z]` |
| 最终到达判定容差 | `target.tolerance_m` | 0.001 m = 1 mm，用于最终到达结果 |
| 障碍物位置和大小 | `environment.obstacles` | 每个矩形的 `min_m=[Y_min,Z_min]`、`max_m=[Y_max,Z_max]` |
| 图的坐标显示范围 | `environment.workspace_bounds_m` | `[[Y_min,Y_max],[Z_min,Z_max]]`，不是物理墙 |
| Stage 1 静态安全间隙 | `environment.safety_margin_m` | 0.001 m；当前纯 FTL 不使用此项 |

**移动紫色出口请改下面的 `deployment` 参数，不是 `robot.base_position_m`。**
`robot.base_position_m / base_angle_rad` 是 Stage 1 和参考路径的基座位姿；
Stage 2A/2B/2C 的实际送入位置由出口、总长和送入量计算。

FTL 回放画的是中心线。`radius_m` 不会自动变成线条粗细；显示线宽另见下表。

### 紫色送入机构和出口

紫色半透明矩形是送入机构，紫色小方块是出口；紫色虚线是参考路径。

| 想修改什么 | JSON 参数路径 | 当前值 |
|---|---|---|
| 出口位置 | `deployment.feed_exit_m` | `[0, -0.06]` m |
| 机构矩形中心 | `deployment.feed_block_center_m` | `[0, -0.115]` m |
| 机构宽度和高度 | `deployment.feed_block_size_m` | `[0.035, 0.11]` m |
| 环境入口高度 | `deployment.environment_entry_z_m` | `0` m |
| 机构透明度 | `deployment.feed_opacity` | `0.25`，0 全透明、1 不透明 |
| 机构颜色 | `deployment.feed_color` | `"#b279a2"` |

这次已将出口从 Z=-0.05 m 下移 **1 cm** 到 Z=-0.06 m，机构同步下移。
环境入口和 target 不动，因此出口到环境入口现在相距 **6 cm**。

修改时，出口必须始终对齐矩形的上边缘中心：

```text
feed_block_center_y = feed_exit_y
feed_block_center_z = feed_exit_z - feed_block_size_z / 2
```

例如，在当前基础上再下移 2 cm，只需同时改为：

```json
"feed_exit_m": [0, -0.08],
"feed_block_center_m": [0, -0.135]
```

保留机构高度 0.11 m。不要只移动矩形或只移动出口，否则配置校验会提示不对齐。
出口高度会改变真实送入几何；释放阈值仍由段长计算，当前 A/B/C 为 0.04/0.10/0.18 m。
机构宽度还需要不小于机器人直径。
若改变出口的 Y 或环境入口的 Z，需要同步调整 Stage 1 参考起点并重新导出，
使它与直线送入到入口时的位置一致。本次仅下移出口，不改入口和参考路径。

### FTL 优化和误差

固定 waypoint、有限次整形的可选试验默认关闭，参数与运行指令见
[整形试验说明](README_RESHAPE.md)。

| 想修改什么 | JSON 参数路径 | 当前值 / 作用 |
|---|---|---|
| 参考目标每步前移多少 | `ftl.reference_step_m` | 0.001 m = 1 mm |
| 每步最多回撤多少 | `ftl.min_feed_increment_m` | -0.002 m；负值允许回撤，设为 0 禁止回撤 |
| 每步最多送入多少 | `ftl.max_feed_increment_m` | 0.002 m；必须大于 0，实际增量由优化器选择 |
| 每段每步最多弯多少 | `ftl.max_bend_increment_rad` | 0.128 rad，约 7.33° |
| 参考路点推进方式 | `ftl.reference_progress_mode` | `"scheduled"` 按参考路点推进；`"actual_tip"` 按实际 tip 投影进度推进，无前进时停止 |
| 是否因误差超限停止 | `ftl.enforce_error_limits` | `false` 只显示、记录误差；`true` 启用以下两个误差阈值 |
| 单步 tip 误差阈值 | `ftl.tip_tolerance_m` | 0.0005 m = 0.5 mm，只有开启误差限制才用于拒绝步骤 |
| 身体误差阈值 | `ftl.body_tolerance_m` | 0.005 m = 5 mm，同上 |
| 优化时身体采样间距 | `ftl.optimization_body_spacing_m` | 最大 0.005 m |
| 独立验收的身体采样间距 | `ftl.validation_body_spacing_m` | 最大 0.001 m，需要不粗于优化网格 |
| 比较轨迹采样间距 | `ftl.comparison_spacing_m` | 最大 0.00025 m，需要细于优化身体网格 |
| 路径局部匹配窗口 | `ftl.matching_window_m` | 0.02 m，限制匹配的路径进度邻域 |
| FTL 身体跟随对象 | `ftl.body_reference_mode` | `"tip_history"`（默认，实际末端历史）或 `"previous_body"`（上一步已接受身体）；[A/B 实验说明](README_BODY_REFERENCE.md) |
| 每步 SLSQP 最大迭代数 | `ftl.max_iterations` | 当前配置 500；代码默认 150，可配置，非物理限制 |
| SLSQP 收敛容差 | `ftl.solver_tolerance` | 1e-7，针对缩放后的目标函数 |
| 数值梯度差分步长 | `ftl.finite_difference_step` | 1.4901161193847656e-8，无量纲，取 SciPy 默认量级；不是机器人运动步长 |
| 最多尝试多少 FTL 步 | `ftl.max_steps` | 当前配置 500；代码默认 250，不含直线接近阶段 |

**δb 的上下界都在 `ftl` 中修改，无需改 Python：**

```json
"min_feed_increment_m": -0.002,
"max_feed_increment_m": 0.002
```

这表示每次 FTL 求解允许 **−2～+2 mm**：负值让被动段底端沿 −Z 回撤，正值沿 +Z 送入。
例如把下界改成 `-0.0005`，就只允许每步最多回撤 0.5 mm；改为 `0` 恢复只送入。
省略新字段的旧配置仍默认下界为 0。上下界必须满足 `min <= 0 < max`，单位为 m。
优化器可以选择零增量；开放负值不代表每一步都必须回撤。

实际范围还会被物理送入距离 `0 <= d <= 总物理长度` 裁剪。
回撤到某段释放阈值时，该段必须拉直；重新进入机构后锁为直段，不能突然折回。
这仍是逐帧几何验证，不是连续时间运动保证。
入场前的直线 approach 仍按正向上限送入；参考 waypoint 的推进方式由
`reference_progress_mode / reference_step_m` 单独控制，开放负 δb 不会自动暂停参考路径。
日志中 `steps[].feed_increment_m` 保留正负号，`effective_ftl_settings` 保存本次上下界。

未显式写在 JSON 里的可选参数，也可以在对应块中新增同名字段。
例如在 `ftl` 中加入 `"tip_weight": 10.0` 可设置 tip 误差权重；
加入 `"minimum_progress_m": 0.000001` 可设置 `actual_tip` 模式的最小进度。
默认值定义在 `models.py`，但日常调整仍通过 JSON 覆盖，无需改 Python 文件。

### 回放速度、颜色，以及只用于手动演示的参数

| 参数路径 | 作用 / 生效位置 |
|---|---|
| `deployment.frame_interval_ms` | 当前 60 ms/帧；越大回放越慢，不是真实机器人速度约束 |
| `deployment.section_colors` | Stage 2A/2B/2C 的 Passive/C/B/A 四种颜色，依次排列 |
| `deployment.visible_passive_length_m` | 被动段的可视范围；完整物理被动段仍保留 |
| `deployment.plot_padding_m` | 坐标范围外的显示留白 |
| `visualization.centerline_width` | 中心线显示线宽，不改变半径或运动学 |
| `visualization.obstacle_color / obstacle_alpha` | 障碍物颜色 / 不透明度，仅改变显示 |
| `visualization.figure_size_inches` | 窗口尺寸 |
| `visualization.section_colors` | Stage 1 的机器人配色；Stage 2 系列使用上面的 `deployment.section_colors` |
| `deployment.end_distance_m / distance_step_m` | **仅 Stage 2A/2B 手动演示**的送入终点 / 步长；不控制 Stage 2C FTL |
| `deployment.bend_targets_rad / bend_ramp_distances_m` | **仅 Stage 2A/2B 手动演示**的目标弯曲角 / 过渡距离；不控制 FTL 解 |
| `solver.*` | **Stage 1 IK** 设置；FTL 求解器参数在 `ftl.*` |
| `ik_search.*` | **Stage 1 多起点 IK 搜索**设置，不控制 FTL 步数 |

还可在 `deployment` 中新增 `"reference_path_color": "#7b3294"` 改参考虚线颜色，
或 `"exposed_line_width_factor": 2.2` 改入场部分的加粗倍数。
在 `ftl` 中新增 `"history_color": "#158c91"` 可改实际 tip 历史轨迹颜色。

## 改完后是否需要重算参考路径

| 修改内容 | 当前 Stage 2C FTL 要做什么 |
|---|---|
| 出口位置、机构尺寸、回放样式、FTL 参数 | 重新运行 FTL 即可，不用重新算 Stage 1 参考构型 |
| robot 的长度、半径、角度上限、Stage 1 基座位姿或 target | 先重新导出 Stage 1 结果，再运行 FTL |
| 仅修改障碍物 | 当前纯 FTL 重新运行即可；障碍物不参与求解 |
| 修改任意参数后想看新运动 | 关闭旧窗口后重新运行；旧的输出文件不会自行更新 |

重新导出参考构型：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless --output outputs/stage1_reference_2d.json
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --config configs/robot_2d.json
```

Stage 2B 使用严格的参考结果检查；修改障碍物、安全间隙或采样配置后，
使用 Stage 2B 时也需要重新导出匹配的 Stage 1 结果。

JSON 不支持注释。90° 用 `1.5707963268` rad；1 cm 用 `0.01` m。
使用 `--case straight / single-bend / s-bend` 时，target 会由测试用例的 FK 在内存中生成；
要使用 `target.position_m` 中的目标，请使用默认 `staircase` 用例。

## 其他入口与说明

| 功能 | 命令 | 说明 |
|---|---|---|
| Stage 1：2D IK 和静态碰撞 | `.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json` | [Stage 1 文档](README_STAGE1.md) |
| Stage 1：原有 3D | `.venv\Scripts\python.exe -m continuum_ftl --config configs/example.json` | 使用独立的 3D 配置 |
| Stage 2A：手动送入与释放 | `.venv\Scripts\python.exe -m continuum_ftl.deployment_demo` | [Stage 2A 文档](README_STAGE2A.md) |
| Stage 2B：参考路径与入场几何 | `.venv\Scripts\python.exe -m continuum_ftl.exposure_demo` | [Stage 2B 文档](README_STAGE2B.md) |
| Stage 2C：数值 FTL | `.venv\Scripts\python.exe -m continuum_ftl.ftl_demo` | [FTL 方法与限制](README_STAGE2C.md) |

FTL 默认保存 `outputs/ftl_2d.json` 和 `outputs/ftl_summary.png`。
日志中的 `tip_error_m`、`body_following_error_max_m`、`planned_path_deviation_max_m`
分别是局部路点误差、实际历史跟随误差、固定参考路径偏差，单位都是 m。
当前 FTL 不检查碰撞，也未实现 RRT、力学或真实绳索。

验证命令：

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```
