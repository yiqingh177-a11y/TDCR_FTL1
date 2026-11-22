# 固定 waypoint 的有限次整形试验

这是 Stage 2C 的可选试验，**默认关闭**，不修改 PCC 模型或原来的加权目标函数。
正常前进仍由现有 SQP 求解；当已接受状态的细网格最大身体误差超过
`ftl.body_tolerance_m` 时，可以暂停参考路点推进，尝试改善当前形状。

## 运行与参数

在项目根目录运行阶梯试验，保留原有运行文件：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --case staircase --max-reshape-attempts 3 --output outputs/runs/staircase_hold/result.json --save-summary outputs/runs/staircase_hold/summary.png
```

S 弯把 `--case staircase` 换为 `--case s-bend`，输出目录也换成 `s_bend_hold`。
`--max-reshape-attempts 0` 关闭试验，恢复原来的推进逻辑。
动画中只有实际接受的调整才成为新帧，显示 `Phase=reshape`；失败试探单独记录，不执行。
若没有整形候选通过验收，动画不会出现额外整形帧。

也可以在 **`configs/robot_2d.json` 的 `ftl` 块**添加：

```json
"max_reshape_attempts": 3,
"reshape_min_improvement_m": 0.00001
```

| 参数 | 作用 |
|---|---|
| `max_reshape_attempts` | 每个 waypoint 最多尝试几次整形；默认 0；非负整数。CLI 优先于 JSON |
| `reshape_min_improvement_m` | 每次接受至少改善多少身体误差；默认 0.00001 m = 0.01 mm |
| `body_tolerance_m` | 原有身体阈值，同时作为整形触发/结束阈值；当前 0.005 m |
| `tip_tolerance_m` | 整形候选的末端误差上限；当前 0.0005 m |
| `max_steps` | 原有正常前进求解次数预算，额外整形次数由每个路点的独立预算限制 |

本次实验在 `enforce_error_limits=false` 下运行：正常前进允许记录超限身体误差，
这样才能触发后续整形。若开启严格误差限制，正常前进可能先被拒绝，无法进入整形阶段。
不需要改 target、障碍物或 Stage 1 参考构型，也不用修改 `ftl_optimizer.py`。

## 每次如何处理

1. 正常前进求解被接受后，检查细网格身体误差是否超过阈值。
2. 固定该 waypoint 的参考弧长和目标坐标，同时冻结触发步所用的身体比较曲线。
   整个整形回合使用同一条曲线，避免仅更换参照就造成“误差降低”。
3. 以上一已接受真实形状为起点，采用零运动增量作为初猜，再调用同一个 SQP。
   权重、曲率自由度、单步送入和弯曲范围、释放和回撤检查均不变。
4. 只有求解与几何检查通过、末端仍在容差内、细网格身体误差至少下降指定量，才接受。
5. 达到身体阈值、预算用尽或第一次试探被拒绝，结束本轮整形。
   然后从最后已接受状态恢复正常 waypoint 推进。

这里“原地”表示 **waypoint 不前移**。底端仍可在原有 δb 范围内移动，
末端允许在容差内偏移；没有新增“底端固定”或“末端绝对不动”的等式约束。
整形仍使用 `tip_weight * tip_error + max_body_error`，没有更换优化目标或引入新的规划器。

**这不是强制身体误差必须达标的门控。** 预算结束后，即使身体误差仍较大，也恢复原推进规则。
整形求解失败会记录失败并丢弃该候选，保留当前状态；正常前进求解失败仍按原规则停止。
碰撞检查仍关闭；不增加连续时间运动保证。

## 结果在哪里

`result.json` 保留原有 `steps`，并增加：

- `attempted_reshape_steps`：所有整形试探数。
- `accepted_reshape_steps`：通过质量检查并实际进入回放的整形步数。
- `reshape_attempts[]`：包含每次试探的原始求解状态、完整候选形状、前后误差和原因。
- `steps[].phase = "reshape"`：真正执行的整形帧，目标参考弧长保持不变。

`RESHAPE_NO_BODY_IMPROVEMENT` 和 `RESHAPE_TIP_TOLERANCE_EXCEEDED` 表示额外质量检查未通过。
`SOLVER_FAILED: ...` 等原因保留原始失败信息。被拒绝候选不会追加到实际 tip 历史。
正常 FTL 步数和额外整形步数分开统计；独立几何评价包括所有已接受运动帧。

## 本次对照

实验与结果集中存放于
[`outputs/hold_reshape_2026-11-22`](outputs/hold_reshape_2026-11-22)。
`experiment_config.json` 是完整参数快照，不会随日常配置变化。
在项目根目录可以重新运行这份快照的两场景对照：

```cmd
.venv\Scripts\python.exe outputs/hold_reshape_2026-11-22/run_comparison.py
```

它会覆盖该实验目录中的对照结果，不覆盖 `outputs/runs` 中的已有记录。
两组仅改变 `max_reshape_attempts=0/3`。对照图同时显示原生身体误差、
独立的身体到固定参考路径距离，以及各次整形试探的实际改善量。
应同时检查所有实际运动帧的峰值与每个 waypoint 整形后的误差，不能隐藏整形前已经出现的偏差。
