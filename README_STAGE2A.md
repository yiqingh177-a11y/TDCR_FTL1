# Stage 2A — Feed-in and Deployment Kinematics

本阶段只实现完整物理机器人的送入几何、释放状态和可重播的二维动画。
不调用 IK、FTL、SQP 或碰撞检查。障碍物和 target 在图中仅作背景，未被改动。

**改参数请到 [configs/robot_2d.json](configs/robot_2d.json)，不要改 `config.py` 或 `outputs` 日志。**
具体文件位置、参数名、单位和生效阶段见 [README 参数修改速查](README.md#参数修改速查)。

## 运行

在 Windows CMD、项目根目录执行：

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.deployment_demo
```

默认打开动画，Replay 从头重播，Pause/Resume 暂停和继续。
无需安装新依赖。原有 `python -m continuum_ftl` 仍然是 Stage 1 IK 功能。

导出同一组动画帧、完整几何 JSON 和最终帧：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.deployment_demo --output outputs/deployment_2d.json --save-gif outputs/deployment_2d.gif --save-snapshot outputs/deployment_2d.png
```

仅用于测试或文件导出时可以附加 `--no-show`，正常演示不需要该选项。

## 几何与释放约定

世界横轴 Y、纵轴 Z。当前 feed exit 为 `(0,-0.06)` m，环境入口为 `z=0` m。
两者相距 0.06 m，不是同一条边界；这些位置由 `deployment` 参数控制。
半透明 feed block 的上边缘中心与出口重合，仅用于显示理想轴向送入机构。

| 物理顺序（近端到远端） | 长度 | 可以弯曲的条件 |
|---|---:|---|
| Passive | 0.20 m | 始终直线，无额外弯曲自由度 |
| C | 0.08 m | d >= 0.18 m |
| B | 0.06 m | d >= 0.10 m |
| A，包含 tip | 0.04 m | d >= 0.04 m |

总物理弧长始终为 **0.38 m**。d 从 tip 刚到 feed exit 时计零，表示送出出口的材料长度。
近端边界为：

```text
y_base = feed_exit_y = 0
z_base = feed_exit_z - total_physical_length + d
       = -0.06 - 0.38 + d
```

在 d=0 时，物理近端为 `(0,-0.44)`，被动段末端/主动 C 段起点为 `(0,-0.24)`，
tip 为 `(0,-0.06)`。全部主动弯曲角均为零。
Stage 1 的 robot.base_position/base_angle 保持不变；本演示采用上述独立部署边界与 +Z 送入方向。

先从近端画完整直线被动段，再在其末端调用原有 2D PCC，串联完整 C、B、A。
没有平移一条最终 IK 曲线，也没有把被动段假装成第四个可控 PCC 自由度。
只有三个主动角，任何未完全释放的角都被强制设为零。

释放阈值由各主动段到 tip 的累计物理长度计算；不依赖 z=0，也不依赖弯曲后的投影长度。
一个完整 PCC 段可以跨越 feed-to-entry 间隙和环境，不会因此被拆成独立主动片段。
这是当前演示的简化整段释放规则，不是所有绳驱连续体机器人的通用物理规律。

## 手动调度与参数

统一修改 `configs/robot_2d.json`：

- 主动长度、半径、角度限制：仍在 `robot.sections`，按 C/B/A 排列。
- 被动长度和显示半径：`deployment.passive_length_m/passive_radius_m`。
  被动显示半径 4 mm 是沿用主动段半径的演示选择，不是实测硬件参数。
- 出口、入口与矩形 feed block：`feed_exit_m`、`environment_entry_z_m`、`feed_block_*`。
- 演示送入范围和步长：`end_distance_m`、`distance_step_m`。
- 各段手动目标角与过渡距离：`bend_targets_rad`、`bend_ramp_distances_m`，按 C/B/A。
- 回放速度、颜色、被动段可视范围：`frame_interval_ms`、`section_colors`、
  `feed_color/feed_opacity`、`visible_passive_length_m`、`plot_padding_m`。

默认 d 从 0 到 0.26 m，步长不超过 0.002 m；释放事件和过渡结束点会精确插入。
每段在完全释放时仍从零角度开始，随后在 0.04 m 送入距离内用 smoothstep
`3*t^2-2*t^3` 连续变为手动目标角。目标为 C=+60°、B=-30°、A=+45°，
配置中使用 rad。值与一阶变化率在释放点连续，不会解锁即跳弯。
`frame_interval_ms` 只控制回放，不代表动力学仿真或硬件速度控制。

这些角度不是目标位置反解，也不是 FTL 调度；动画没有跟随参考路径或计算 body error。
穿过背景障碍物不触发停止，因为本阶段明确不做碰撞检查。
角度限制仍会检查。如果释放后的几何折回 feed exit 后方，会记录并在图中显示
`consistency_warnings`，不会偷偷平移或改弯曲角掩盖不一致。

## 实现与数据

- `deployment.py`：可复用 `compute_deployment()`、释放阈值、手动调度、状态生成和报告。
- `deployment_viewer.py`：只播放已生成的状态，不重新计算另一套几何。
- `deployment_demo.py`：独立入口，连接统一配置、几何、动画和可选导出。
- `models.py/config.py`：增加部署数据类型及可选 `deployment` 块校验；旧 3D 配置无需增加它。

`compute_deployment(robot, settings, d, requested_bends_rad, max_arc_spacing_m)` 返回：
近端和主动起点、请求/实际主动角、释放向量、释放阈值、完整中心线、
各物理段端点、切线角及几何一致性警告。向量均按近端到远端。

中心线按被动段和完整主动链分别采样，弧长单调，接头只保留一个全局点。
各段的弧长参数保持物理长度；弯曲后弦线折线的长度会略短，不能将其误认为机器人缩短。
被动段可以在图中部分裁掉，但状态和导出的 JSON 始终包含它的全部 0.20 m。

## 验证

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

新增测试覆盖物理顺序、总长、初始 tip、每个释放临界点、未释放强制直线、
feed 与环境入口的区别、整段跨 z=0、被动直线、位置/切线连接、调度连续性、
折回 feed 的不一致报告及 GIF 帧与状态数组一致。
原测试中一个写死 5 mm 的最小间隙断言改为配置的安全余量断言，避免改场景后误报。
未降低配置的安全间隙，也未修改碰撞算法。
带精确旧坐标的数值回归使用初版 5 cm 出口间隙作为固定测试样例；
可视化和当前场景集成测试仍加载实际 JSON，避免把用户可改的出口位置写死在回归断言中。

修改前快照保存在忽略上传的 `outputs/stage2a_before/`，其中包含原始场景 JSON。
本阶段到此为止，FTL 评价区域留待 Stage 2B。
