# Stage 2C 身体参照 A/B 实验

**版本说明：** 2026-11-22 的初始 A/B 记录使用旧数值梯度差分步长 `1e-5`。
随后针对 status 9 的冻结失败步重放确认该扰动过大，默认已修正为 SciPy 的
`1.4901161193847656e-8`，详见 [根因报告](outputs/solver_status9_diagnosis_2026-11-22/ROOT_CAUSE.md)。
旧实验配置省略此字段；要严格重现旧失败，请在一份独立配置中显式设
`"finite_difference_step": 1e-5`。新运行请保留修正值，并使用新的输出目录。

只比较身体参照定义，不改三区域模型、物理段长、释放规则、环境内评价区、目标函数、SLSQP 或接受条件。没有碰撞或内力计算。

## 在哪里切换

修改 `configs/robot_2d.json` 中的一个字段：

```json
"body_reference_mode": "tip_history"
```

- `tip_history`：默认和原有行为。上一步已接受的实际 FK tip 历史＋下一 desired waypoint；属于本项目的有意改编。
- `previous_body`：上一步已接受状态的真实 PCC 环境内 active 身体＋下一 desired waypoint；是 Palmer-style 的上一身体参照。

两者都保留实际 tip 历史用于显示和独立评价。`reference_progress_mode` 独立控制推进方式，本次对照使用 `scheduled`。参考 IK 角度不是终态约束。

正常可视运行：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo
```

临时选模式、不编辑配置，并将新结果单独保存：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --body-reference-mode previous_body --output outputs/previous_body_demo.json --save-summary outputs/previous_body_summary.png
```

回放文字和 JSON 顶层均显示 `body_reference_mode`。`body_following_error_max_m` 是所选模式的**原生优化误差**，不能直接拿来判定哪种模式实际跟随更好。

## 一条命令运行完整对照

在 `TDCR_FTL1` 目录、现有虚拟环境中运行：

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_compare --reference-step-m 0.001 --output-dir outputs/body_reference_comparison
```

这会按顺序计算 straight、single-bend、s-bend、staircase，每例都跑两种模式。
`--reference-step-m 0.001` 是对**两组共同**采用的 1 mm 实验步长；不写回日常配置。
其它数值、几何和采样设置全部来自 `configs/robot_2d.json`，两组完全相同，不分别调参。
每次实验会在输出目录保存完整 `experiment_config.json` 和软件版本。

结果是离线科学对照图；想逐帧播放某组，使用上面的 `ftl_demo`，并通过 `--config` 指向实验配置、选择同一 mode。

每个场景输出：

- 两个模式各自的完整状态 JSON 和独立评价 CSV；失败候选保留在状态 JSON 中。
- `final_shapes.png`：两组最后已接受的真实 active PCC 身体、固定参考路径、实际 tip 历史和目标。
- `independent_errors.png`：两组使用相同评价方法得到的误差，横轴为实际 tip 投影参考进度。

根目录的 `summary.json` 和 `RESULTS.md` 汇总结果。失败时，图和独立指标使用**最后已接受**状态，不把失败候选或理想参考形状伪装为实际终态。

## previous_body 的几何与边界处理

1. 从上一已接受状态取得完整物理链及已有 Stage 2B 入口交点。
2. 使用同一段 PCC 公式，在 `comparison_spacing_m` 分辨率上重采样环境内 active 曲线；不是把旧的粗网格折线简单拉密，也不裁短物理段长。
3. 按近端到远端顺序保留精确入口、段边界和 tip，再接上一实际 tip 到下一 desired waypoint 的直线延伸。
4. 这条比较曲线在本次 SLSQP 调用期间固定。接受后下一步才换成新身体；失败则停止，失败形状不会成为下一条参照。

首次 tip 刚到入口时，旧暴露身体可以退化为一个点；此时比较路径就是入口点和新 waypoint 的连接。
新暴露的近端材料不要求在旧身体中有相同材料编号。原有 matcher 仍按 tip 侧弧长估计、±局部窗口和有序线段投影建立几何对应。
预期弧长为负时只把搜索范围裁到比较曲线的有效区间；旧入口端点是合法候选，仍计算真实欧氏距离，**不补假路径、不自动给零误差**。

被动段即使有部分中心线跨入环境也不参与 FTL 身体参照或评价，保持原项目的 active-only 规则。
释放权限仍按完整段出 feed 判断；已释放但还在过渡区的段仍可影响 tip 位姿。

## 两组共用的独立评价

`ftl_evaluation.py` 对所有已接受 FTL 帧统一计算：

1. 已接受实际 tip 历史 → 固定参考路径。
2. 当前环境内 active 身体 → 已接受实际 tip 历史。
3. 当前环境内 active 身体 → 固定参考路径。

使用无窗口、无单调限制的精确最近线段距离。身体在 `validation_body_spacing_m` 采样；固定解析 PCC 和查询用历史折线在 `comparison_spacing_m` 采样。单位为 m，图表转为 mm。
历史是实际 tip 点之间的直线连接，密化不改变该折线。
这些是有方向的采样曲线距离，不是连续时间／连续曲线的严格上界。

每个指标保留每帧最大值和采样点均值；汇总给出全运行最大值、逐帧最大值的平均、逐帧点均值的平均，以及最后已接受帧的最大值／均值。
计算时间是原 rollout 的求解时间，独立评价与绘图时间不混入。不同机器或系统负载下的时间不能当作严格性能基准。
若两组在不同进度失败，不能只比较各自终态误差就判优劣；应同时看终止位置及重叠进度区间。

## 实现位置

- `models.py` / `config.py`：可选模式、默认值与校验；旧配置缺字段仍为 `tip_history`。
- `ftl_matching.py::previous_body_comparison_path`：可复用、固定的上一身体参照构造。
- `ftl_rollout.py`：仅按模式选择比较路径，原优化器保持不变。
- `ftl_evaluation.py`：两组共用的独立几何指标。
- `ftl_compare.py`：四场景配对运行、结果和配置快照导出。
- `ftl_visualization.py` / `deployment_viewer.py`：对照图和模式显示。

测试命令：

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check continuum_ftl tests
.venv\Scripts\python.exe -m ruff format --check continuum_ftl tests
```
