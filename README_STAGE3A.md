# Stage 3A：离散帧静态碰撞后处理

读取已保存的 FTL 结果，逐帧检查完整机器人与现有障碍物。
**不运行 IK、SQP 或 FTL，不修改 q、送入量、参考路径或原始结果文件。**

## 运行

在 `D:\Code\rr\TDCR_FTL1` 下运行，检查已有阶梯记录并打开回放：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.collision_demo --input outputs/runs/staircase_03/result.json --output-dir outputs/stage3a_2026-11-22/staircase
```

检查 S 弯：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.collision_demo --input outputs/runs/s-bend02/result.json --output-dir outputs/stage3a_2026-11-22/s-bend
```

追加 `--save-gif` 将全部回放帧导出为输出目录中的 `motion.gif`。
追加 `--no-show` 只导出报告、CSV 和静态图，不打开窗口。
`--config configs/robot_2d.json` 可指定与原结果匹配的机器人配置，默认就是该文件。

输出文件可以重新生成；源 FTL JSON 和源配置不会被覆盖。

## 参数在哪里改

沿用 **`configs/robot_2d.json`**：

| 参数 | Stage 3A 的用途 |
|---|---|
| `robot.sections[].length_m / radius_m` | 原机器人各主动段的物理长度和半径 |
| `deployment.passive_length_m / passive_radius_m` | 完整被动段的长度和半径 |
| `deployment.feed_exit_m` | 从保存的送入距离还原原来的世界位姿 |
| `environment.obstacles` | 所有需要检测的实体障碍物，包含任意 Z<0 区域的障碍物 |
| `environment.safety_margin_m` | 安全间隙；当前 1 mm，与实际碰撞分别统计 |
| `sampling.max_arc_spacing_m` | 碰撞几何的最大弧长采样间距；当前 1 mm |
| `deployment.frame_interval_ms` | 回放间隔；不作为真实机器人的运动时间 |
| `visualization.*` | 图与环境的显示设置 |

FTL 的权重、优化预算、身体误差网格和 exposure 设置不会影响本次几何判碰。
改变障碍物后，可以直接重新检查同一条保存的轨迹，无须重新优化。
改变机器人长度或 Feed 出口位置后，旧轨迹通常不再匹配；程序会根据保存的
tip 和 section junctions 校验位姿，不匹配就报错。
旧 FTL 文件没有保存半径，必须提供原来的半径配置；本次加载的完整配置会随报告存档。

## 检测范围与方法

- 默认检查所有已接受的物理状态：包括 41 个 approach 帧及 180 个 FTL 帧；
  若文件含已接受的 `reshape` 帧，也一起检查。数量以实际输入为准。
- 原始文件中未接受的失败候选没有被执行，默认排除并统计数量。
- 全部 **Passive → C → B → A** 管状几何均检查，包含机器人半径。
  不以 Z=0、FTL exposure 或 FTL 身体误差采样区裁剪机器人。
- 因此 Transition Zone、Feed 附近或被动段所在位置如有配置的实体障碍物，同样能够检出。
- 紫色 Feed 矩形仍是示意图，不自动变成实心障碍物。要研究机构边界，应把实际的侧壁、
  通道边界等加入 `environment.obstacles`，而不是把整个示意块当成填实的矩形。
- 从保存的 q、d 重建同一 PCC 形状，按碰撞间距重新采样。复用原静态检查器的
  圆/矩形几何和逐弦 capsule，增加 PCC 子弧 sagitta 膨胀以保守包围弯曲管段。
- 这是离散帧环境碰撞检查，不检查帧间扫掠，也不检测机器人自身碰撞。

## Clearance、穿透和计数的含义

报告的 `minimum_clearance_lower_bound_m` 是整条机器人对所有障碍物的有符号间隙下界，
已包含机器人半径和子弧误差膨胀，但**未减去安全间隙**：

- `clearance <= 0`：接触或重叠，保守判为 collision。
- `0 < clearance <= safety_margin`：仅安全间隙不足，显示为 amber，不计入几何碰撞帧。
- `clearance > safety_margin`：clear。
- 无障碍物时 clearance 为 `null`，不是伪造为零或无穷大。

保守包围可能在非常接近边界处给出误报。采样越细，sagitta 膨胀越小。
有符号模式将机器舍入量级的近零数值视作接触，避免浮点相切误差影响分类。

原来的无符号“线段到实心矩形距离”进入矩形内部后会饱和为零，不能描述深重叠。
Stage 3A 增加了有符号线段/矩形距离，以
`max(0, -minimum_clearance)` 报告 **局部重叠深度的保守估计**。
它不是材料压缩量、接触力，也不是让整个机器人脱离障碍物所需的最小整体平移。
原 Stage 1 默认判碰接口和原有指标保持不变；此有符号模式只在 Stage 3A 显式启用。

碰撞次数分别报告：

- `collision_frame_count`：有几何碰撞的帧数。
- `collision_episode_count`：离散状态中，从无碰撞进入碰撞的连续区间数。
- `colliding_pair_frame_count`：section–obstacle 配对累计出现的碰撞帧数。
- `section_collision_frame_counts`：每个 section 各有多少碰撞帧。
- `safety_margin_only_frame_count`：只有间隙不足、没有实际重叠的帧数。

这些区间是离散帧事件统计，不是物理撞击次数，也不能排除两个 clear 帧之间发生短暂碰撞。

## 首次碰撞时刻

`first_collision` 同时给出：

- `source_frame_number`：原动画中从 1 开始的帧号，包含 approach。
- `source_step_index`：原 JSON 的零起始 step_index。
- `deployment_distance_m` 和 `desired_reference_progress_m`：首次检出时的送入距离与路点进度。
- `replay_time_s`：按回放间隔得到的显示时间。
- `physical_time_s: null`：原文件没有物理时间戳，不能用求解耗时或显示时间冒充实际碰撞时间。

这是“首次在保存帧中检测到”的碰撞，不是连续运动中的精确首次接触时刻。

## 回放与输出

动画保留完整原构型和帧顺序，不因碰撞停止或重新优化：

- red：碰撞 section；整段高亮。
- amber：只有安全间隙不足的 section。
- 窗口标注首次碰撞源帧号，`First hit` 可以直接跳到该帧。
- `Replay / Pause / Previous / Next` 控制实际已保存的状态。
- 图中线条表示中心线；判碰几何包含实际半径。

每个输出目录中：

| 文件 | 内容 |
|---|---|
| `collision_report.json` | 总体统计、首次事件、每帧 section–obstacle 碰撞及源文件 SHA-256 |
| `frames.csv` | 每帧 clearance、估计穿透、碰撞 section、原帧号和推进位置 |
| `clearance.png` | 随送入距离 d 的完整曲线，以及随参考进度 s 的接触附近放大图；标注首次碰撞 |
| `first_collision.png` | 首次碰撞高亮截图；无碰撞时改为首个间隙不足帧或第一帧 |
| `motion.gif` | 使用 `--save-gif` 时导出的完整回放 |
| `config_snapshot.json` | 本次实际使用的机器人尺寸、障碍物和采样设置 |

## 代码结构

- `collision_postprocess.py`：读取与校验保存状态、重建同一形状、逐帧检查、统计报告。
- `collision.py`：复用现有静态几何算法，增加可选有符号矩形距离，不依赖可视化。
- `collision_replay.py`：只读取已检查的数据高亮和绘图，不在动画回调内检查碰撞。
- `collision_demo.py`：命令入口与文件导出。
- `models.py`：逐帧结果和时间序列的数据结构。

FTL rollout、SQP 目标函数和运动约束没有接入任何碰撞限制。

本次两条已有轨迹的结果和可直接查看的 GIF：
[Stage 3A 实测结果](outputs/stage3a_2026-11-22/RESULTS.md)。
