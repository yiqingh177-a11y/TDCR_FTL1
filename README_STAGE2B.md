# Stage 2B — Reference Path and Environment Exposure

本阶段增加固定参考路径和实际机器人入场部分的几何提取；保留 Stage 1 和 Stage 2A。
不实现 FTL、SQP、路径匹配、碰撞检查或新的多候选 IK。动画仍是原有手动送入。
没有修改 `configs/robot_2d.json` 的任何内容，包括你设置的 target、障碍物和送入参数。

## 运行

Windows CMD，项目根目录：

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.exposure_demo
```

已为当前场景导出 `outputs/stage1_reference_2d.json`，因此可以直接运行。
所有参数仍集中在 `configs/robot_2d.json`，无需新依赖或 Isaac Sim。

首次使用、在另一台电脑上运行，或修改机器人、target、障碍物、安全余量、采样后，
先用原有 Stage 1 命令导出一个新的成功结果：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless --output outputs/stage1_reference_2d.json
.venv\Scripts\python.exe -m continuum_ftl.exposure_demo
```

第一条命令属于原有 Stage 1，会按其现有配置搜索并静态验证最终构型。
第二条才是 Stage 2B：只读取一个成功解，不调用 IK 或碰撞检查。
Stage 1 报告现在附带 `geometry_context`，加载时会核对场景，防止旧解被用于改过的障碍物。
旧报告缺少该字段时会要求重新导出；搜索失败或碰撞的诊断构型不能成为参考路径。
只有显示设置、手动送入参数或求解初值变化时，不需要重做最终构型验证。

导出完整数据、动画及最终帧：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.exposure_demo --output outputs/exposure_2d.json --save-gif outputs/exposure_2d.gif --save-snapshot outputs/exposure_2d.png
```

正常会打开交互窗口。仅文件导出时加 `--no-show`。
`--ik-result` 可指定另一个匹配场景的 Stage 1 报告，`--config` 可指定其他 2D 配置。

## 图中显示什么

- 淡紫色半透明矩形是 feed block；出口固定为 `(0,-0.05)` m。
- 水平虚线为环境入口 `z=0`，两条边界之间的 5 cm 不属于参考路径或评价区域。
- 灰色 Passive、蓝色 C、橙色 B、绿色 A 保持原来的实际几何和身份。
- **同色粗线**表示实际进入环境的连续部分；未进入部分保持普通线宽。
- 紫色虚线是从最终 IK 解提取的**固定参考 tip 路径**。实际机器人没有被贴到它上面。
- 顶部显示 d、tip、feed 释放标志、当前总入场长度和各物理段入场长度。
- `Pause/Resume` 暂停/继续观察，`Replay` 重播同一组已存储状态。

被动段可以按 Stage 2A 的显示设置部分裁出画面，但它的全部几何仍在状态中。
多次穿越入口、切触入口或一段沿入口平面延伸时，显示红色歧义诊断，不画一个猜测的入场域。
障碍物仅作本动画背景；本阶段的实际运动可能经过障碍物，不表示已经实现无碰撞 FTL。

## 三个不同概念

1. **Feed release**：C/B/A 完全离开出口后才允许弯曲；阈值仍为 180/100/40 mm。
2. **Environment exposure**：真实 PCC 曲线穿过 `z=0` 后，从入口交点到当前 tip 的连续部分。
   可以只包含某段的一部分，不必等整段进入。
3. **未来 FTL 评价域**：采用上述真实入场部分，但本阶段尚未计算匹配、body error 或优化目标。

Passive/C/B/A 的物理长度仍为 0.20/0.08/0.06/0.04 m，总长始终 0.38 m。
一个 B 段即使仅有末端 0.02 m 入场，仍是完整 0.06 m 的同一条 PCC 圆弧。
入口仅截取评价数据，不拆分物理段、不改变曲率、不增加自由度。

## 几何和弧长约定

所有坐标为原有世界 `(Y,Z)`，单位 m；角度 rad，近端到远端排列。

| 符号/数据 | 原点及含义 |
|---|---|
| `d` | 从 tip 初到 feed exit 开始计的送出材料长度，沿用 Stage 2A |
| `material_arc_lengths_m` | 从当前完整机器人的物理近端沿中心线累计的材料弧长 |
| `entry_material_arc_m` | 环境入口交点在上述材料坐标中的位置，每帧可能变化 |
| exposure `arc_lengths_m` | 从**当前实际入口交点**到实际 tip，从零开始 |
| reference `arc_lengths_m` | 从**固定参考路径入口**到最终 IK tip，从零开始 |

这些坐标的含义不同。没有令 d 等于参考路径 tip 进度，也没有把入场弧长固定为 `d-0.05`。
弯曲时，世界坐标的 5 cm 高度差与曲线上 5 cm 长度并不相等。
当前入场长度由精确交点计算 `total_material_length - entry_material_arc_m`。

对每条 PCC 圆弧先找 `dz/ds=0` 的转折点，分成高度单调区间，
再在区间内对**解析 PCC 位置函数**作括区间求根。该标量求根不是 IK/SQP。
这可以发现同一条粗采样圆弧内部的多次交点，不依赖已画出的折线。
切点、共面段及多个交点都返回歧义，不拼接互不连续的 `z>=0` 点集。
交点判定的空间分辨率使用 `deployment.geometry_tolerance_m`（当前 1e-9 m）。

单个向前入口且 tip 已在环境侧时，按真实材料弧长采样入口到 tip 的区间，
补入每个物理段的精确边界；段内均匀采样，间距不超过 `sampling.max_arc_spacing_m`。
每个样本都有段索引，内部接头约定归远端段。每段长度由弧长区间求交计算，
总和等于总入场长度。这里检测的是中心线入场，未用机器人半径扩大入口。
tip 正好向前到达入口时，入场长度为零，保留入口/tip 这个点。
没有进入时返回空数组；歧义时长度为 `None`，不会伪装成零。

参考路径保持最终 IK 中心线的精确起点和末端，并支持按弧长调用精确 PCC 插值。
末端就是求解所得的 tip，不会强行吸附到 target 制造不属于 PCC 的一小段。
目前最终 tip 的 target 残差约 4.10e-8 m，低于配置的 1 mm 容差，残差会明确保留在报告中。

## API 和文件职责

```python
from continuum_ftl.config import load_config
from continuum_ftl.deployment import generate_deployment_states
from continuum_ftl.environment_exposure import extract_environment_exposure
from continuum_ftl.reference_path import load_reference_path, interpolate_reference_path
import numpy as np

config = load_config("configs/robot_2d.json")
reference = load_reference_path(config, "outputs/stage1_reference_2d.json")
point = interpolate_reference_path(reference, np.array([0.03]))  # 世界 YZ, shape (1,2)
states = generate_deployment_states(config)  # 原有 Stage 2A 状态
exposure = extract_environment_exposure(
    states[70],
    config.deployment.environment_entry_z_m,
    max_arc_spacing_m=config.sampling.max_arc_spacing_m,
    tolerance_m=config.deployment.geometry_tolerance_m,
)
```

- `planar_kinematics.py`：原有 FK 保留；公开复用其稳定的局部 PCC 点公式。
- `planar_geometry.py`：连续曲线取点、保留边界的弧长采样、入口交点计算。
- `reference_path.py`：读取一个已验证解、检查场景、提取/插值固定参考路径。
- `environment_exposure.py`：从实际部署状态提取入场区间、每段长度和诊断。
- `deployment_viewer.py`：扩展原有 viewer，播放同一套状态；不重新解算形状。
- `exposure_demo.py`：连接输入、几何计算、报告与回放。
- `models.py`：集中新增结构化几何、参考路径和入场结果。

报告包含参考路径及连续插值所需的 PCC 段位姿；每个状态保留完整实际几何，
另存 entry 交点、材料弧长、入口相对弧长、段身份、各段长度和诊断。
显示配色仍在 `deployment.section_colors`；可选新增
`deployment.reference_path_color`（默认 `#7b3294`）和
`deployment.exposed_line_width_factor`（默认 2.2），无需修改现有 JSON。

## 验证结果

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

新增 15 项确定性测试；合计 87 项通过，包括原有 Stage 1、Stage 2A 与 3D 测试。
覆盖解析圆弧插值、精确端点与段边界、近零曲率、未入场、首次入场、
直线/弯曲弧长、B 段部分入场、粗采样仍能找交点、多次穿越及切触/共面诊断、
各段长度求和、场景过期拒绝、无求解器调用和实际 GIF 回放。

当前手动调度 131 帧：26 帧未入场、105 帧有有效入场区间、0 帧歧义。
末帧入场 0.21 m：Passive 0.03、C 0.08、B 0.06、A 0.04 m。
修改前快照在忽略上传的 `outputs/stage2b_before/`。
本阶段止于几何准备，不包含 Stage 2C 的匹配和 FTL 优化。
