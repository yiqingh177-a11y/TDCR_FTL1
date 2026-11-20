# continuum_ftl — Stage 1

独立的 **2D/3D PCC 运动学、位置 IK、最终构型静态碰撞检查** 项目。
代码放在 `TDCR_FTL1` 仓库内，Python 包名为 `continuum_ftl`；不依赖旧项目，
不使用 Isaac Sim、MuJoCo 或其他物理引擎。

当前阶段只回答：**在给定 IK 搜索预算内，是否找到一个满足弯曲约束、到达目标、
且与已配置障碍物保持安全间隙的最终形状？**
不计算进入该形状的运动路径，不证明导航无碰撞，不检查自碰撞。

## 运行

cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\activate
python -m continuum_ftl

需要 Python 3.11 或更高版本。请在本仓库根目录运行。
本次实现已在项目内创建 `.venv`，依赖通过 `pyproject.toml` 管理。
以后从 GitHub 克隆到其他电脑时，重新创建环境即可。



Windows CMD，新环境安装：

```cmd
cd /d D:\Code\rr\TDCR_FTL1
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

打开三维图：

```cmd
.venv\Scripts\python.exe -m continuum_ftl
```

仅计算，不加载 Matplotlib：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --headless --output outputs/result.json
```

不弹窗口，保存图与报告：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --headless --output outputs/result.json --save-figure outputs/result.png
```

指定其他配置：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/example.json
```

CLI 成功执行会返回退出码 0，包括有明确失败报告的 IK 结果。
无效配置和程序错误抛出带上下文的异常，不伪装成“目标不可达”。
集成其他程序时应读取结构化结果字段，不靠 CLI 退出码判断可达性。

## 目录与职责

```text
TDCR_FTL1/
├── pyproject.toml             依赖、包配置和 Ruff 设置
├── configs/example.json      原有 3D 示例配置，省略 dimension 时仍按 3D 加载
├── configs/robot_2d.json     2D YZ 示例，每次实验只加载一个完整 JSON
├── continuum_ftl/
│   ├── models.py             配置与结果 dataclass、状态枚举、坐标约定
│   ├── config.py             JSON 加载、值与维度检查
│   ├── kinematics.py         单段解析位姿、多段 FK、弧长采样
│   ├── planar_kinematics.py  有符号角的 YZ PCC，与原有 3D 公式独立
│   ├── inverse.py            位置目标函数、坐标转换、SciPy SLSQP 单次求解
│   ├── ik_search.py          可复现多起点搜索、去重、成功即停及搜索报告
│   ├── collision.py          整条中心线与球/轴对齐盒的静态几何检查
│   ├── evaluation.py         IK → FK → 碰撞 → 结构化报告
│   ├── visualization.py      Matplotlib 三维图；只负责显示
│   └── __main__.py           CLI、读取配置、调用流程、输出文件
└── tests/                    确定性的 unittest 数值与流程测试
```

依赖方向：`models` → `config/kinematics` → `inverse/collision` → `evaluation`。
CLI 负责连接流程和显示。核心数值模块不会导入可视化或旧项目。
数据类只表达配置和结果，没有规划器基类、插件注册或其他暂时用不上的抽象。

`README.md` 提供 2D/3D 快速入口，完整说明在本文件；占位文件 `1` 保留。
`.venv`、工具缓存和 `outputs` 被 Git 忽略；源码、配置、测试可正常上传。

## 配置与坐标系

`configs/example.json` 包含以下组：

| 组 | 内容 |
|---|---|
| `robot` | 逐段长度、半径、最大弯曲角；基座位姿；初始 q |
| `target` | 世界坐标目标点与位置容差 |
| `environment` | 显示范围、球/盒障碍物、碰撞安全间隙 |
| `solver` | SLSQP 迭代预算、目标函数停止阈值、弯曲约束数值容差 |
| `sampling` | 中心线最大弧长采样间距 |
| `visualization` | 图幅、视角、颜色、透明度、线宽、坐标轴长度及球面网格精度 |

段数直接由 `robot.sections` 的长度确定，不再保存一个可能与列表冲突的计数。
增加或删除 section 时，必须同步调整 `initial_configuration_rad` 的行数。

- 世界坐标系 +Z 向上，所有位置和障碍物都使用世界坐标，距离用 m。
- sections 从近端/基座到远端/末端排列。
- 每段局部未弯曲轴为 +Z，`phi=0` 向局部 +X 弯，`phi=pi/2` 向局部 +Y 弯。
- `q` 的形状为 `(n,2)`，每行 `[theta,phi]`，单位 rad。
  `theta` 范围为 `[0,max_bend_rad]`；`phi` 可为任意有限角度。
- `base_rpy_rad=[roll,pitch,yaw]`，旋转矩阵为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
  这是基座姿态约定，与连续体的弯曲自由度无关。
- tip orientation 为世界坐标下的 `(3,3)` 旋转矩阵，列向量表示末端局部坐标轴。
- box 的 `size_m` 为 XYZ **全尺寸**；球与机器人 section 都使用**半径**。
- `workspace_bounds_m` 仅设置显示范围，不会生成墙壁，也不是硬可达域。
  想限制机器人活动空间，需要显式添加障碍物。
- 可视化颜色使用 `#RRGGBB`；相机角度采用字段明确标注的 deg，仅影响显示。

当前 3D 例子为三段，长度 `[0.08,0.06,0.04]` m、半径 0.004 m、最大弯曲角 pi/2；
基座高度 0.03 m；目标 `[0.05,0.05,0.16]` m，位置容差 0.001 m。
**这些是演示参数，不代表真实 COBRA 尺寸或已经验证的硬件极限。**

配置加载时检查正数、有限值、数组形状、初始弯曲范围、唯一障碍物名称及未知字段。
配置数组只读；修改实验应编辑 JSON 并重新加载。

## PCC 与 IK 的实现

每段为固定弧长的空间圆弧，无独立扭转、轴向伸长或力学变形：

```text
T_world_tip = T_world_base @ T_section_1 @ ... @ T_section_n
```

第二段等后续段的局部形状连接在上一段末端的完整位姿上。
局部位置公式在非零曲率时等价于：

```text
x(s) = L/theta * (1-cos(theta*s/L)) * cos(phi)
y(s) = L/theta * (1-cos(theta*s/L)) * sin(phi)
z(s) = L/theta * sin(theta*s/L)
```

代码使用 sinc 恒等式避免 theta 近零时的相减和除零，伸直情况连续退化。

`ForwardResult` 提供精确解析末端、末端方向、段间端点、全局均匀弧长中心线，
以及用于碰撞的逐段采样曲线。全局中心线按总弧长均匀采样；逐段曲线另外保证
精确包含每个段的边界。这两个采样表示使用同一个解析位姿公式。

**长度指弧长，不是离散折线长度。** 弯曲后相邻采样点的弦长总和略短于真实长度；
随着采样加密，它应收敛到指定弧长，不能靠拉伸或缩短 section 修补这个差异。

IK 使用 SciPy 标准 `minimize(method="SLSQP")`，不实现自定义优化器。
数学目标函数 `position_objective` 与求解流程分开。内部坐标为
`[theta_1*cos(phi_1),theta_1*sin(phi_1), ...]`，避免伸直时 phi 不确定；
每段的圆盘不等式约束确保 theta 不超过该段上限。
对外输入输出保持 `(n,2)` 的 theta/phi。

目标函数使用与完整 PCC FK 相同的局部变换公式，只计算解析末端，不生成中心线；
`tip_pose` / `tip_pose_2d` 的结果与完整 FK 通过测试核对。
优化器允许试探不满足弯曲约束的中间值，
这些仅用于目标函数计算，最终解必须重新检查实际限制与末端误差。

单次 IK 接口仍只从给定初始 q 求解一次，不向目标函数加入障碍物惩罚。
多起点搜索由独立的 `ik_search.py` 组织，按明确预算重复调用它；随机种子来自配置。

## 静态碰撞与报告含义

每段全部中心线弦线扩张为 capsule，半径包括：

```text
robot section radius + maximum arc-to-chord deviation (sagitta)
```

圆弧每个子段的偏差采用解析公式计算，保证弯曲中心线管体被代理覆盖。
与 sphere 检查线段到球心距离；与 AABB 检查线段到实心盒的精确最短距离。
盒距离通过分段二次函数求最小值，没有在碰撞检查内再调用数值优化器。

`minimum_clearance_lower_bound_m` 是这个保守代理计算的间隙下界，
可能在边界附近保守地拒绝实际上未接触的形状；它不是穿透深度的精确测量。
无障碍物时字段为 `null`，表示没有定义最近障碍物距离。

| 字段/状态 | 含义 |
|---|---|
| `target_reached` | 满足弯曲限制且末端进入目标位置容差，与碰撞检查分开 |
| `optimizer_success` | 优化器停止状态，不等于目标到达 |
| `collision_status=clear` | 最终构型与所有障碍物的代理间隙严格大于安全余量 |
| `safety_margin_violation` | 没有代理接触，但间隙不足 |
| `collision` | 保守代理发生接触或重叠 |
| `static_target_feasible` | `target_reached` 且 `collision_free_with_margin` |
| `outside_length_envelope` | 目标距固定基座超过总长加容差，可直接排除 |
| `target_not_reached` | 优化器正常结束，但位置误差没有满足要求 |
| `optimization_failed` | 预算耗尽、约束违反或其他优化失败，不能证明不可达 |

若目标到达但碰撞，报告会保留 IK 解、碰撞障碍物、section 索引和间隙，
`static_target_feasible=false`。不能据此断言目标没有其他无碰撞解。
若失败迭代违反弯曲上限，用投影到边界的形状作诊断显示，并明确记录失败。

可视化显示中心线、段边界、最终 tip、目标及容差球、障碍物、基座坐标轴。
中心线线宽不代表机器人物理半径；半径由碰撞算法使用。
窗口标题明确标明仅检查最终形状，未检查运动路径。

## 测试与代码检查

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

独立几何测试包含：四分之一圆的圆心/半径和末端切线、基座旋转与 SciPy 旋转矩阵的
交叉核对、两段已知位姿的手算结果、全局均匀弧长采样、折线长度收敛、
零/极小曲率、任意段数、非法参数、IK 到达/不收敛/长度越界、
末端到达但本体碰撞、机器人半径、安全余量及粗弦采样之间的圆弧接触。

后续 FTL、RRT、插入、候选 IK 或新形状模型可以调用现有 FK/结果接口；
本阶段没有提前实现这些功能，也没有加入 tendon、材料或物理仿真。

## 2D 扩展：YZ 平面

### 运行与配置

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless
```

`configs/example.json` 及不带 `--config` 的原有命令继续使用 3D。
本次没有改动该示例中用户已经调整的尺寸、目标和障碍物。

当前 `robot_2d.json` 场景使用用户提供的阶梯通道：`L1`–`L7`、`R1`–`R7`、`T1`，
替换原来的两个演示障碍物。原始 `bounds_yz=[y_min,y_max,z_min,z_max]`
转换为 `min_m=[y_min,z_min]`、`max_m=[y_max,z_max]`。
目标为 `[0.1409612134,0.0882219046]` m，机器人三段长度仍为 `[0.08,0.06,0.04]` m，
半径均为 0.004 m。安全余量保持 0.001 m。

参考构型 `[90,-45,45]` deg 对应弧度 `[pi/2,-pi/4,pi/4]`，
经 FK 核对与给定目标一致，且通过最终静态碰撞检查。
这个参考构型不作为隐藏的 IK 初值；JSON 中初值仍为 `[0,0,0]`。
加 `--single-run` 复现原有行为时，从直线出发得到的构型会碰到 `L4` 和 `L5`，
报告 `target_reached=true`、`static_target_feasible=false`。
当前配置启用多起点搜索，默认会继续尝试不同初值，直到找到无碰撞最终构型或预算耗尽。

每次实验仍只有一个完整 JSON。顶层 `dimension` 为 `2` 或 `3`，省略即 `3`；
不能通过给 3D 数组截掉一列来隐式切换模式。

| 项目 | 2D | 3D（保持原接口） |
|---|---|---|
| 机器人类型 | `PlanarRobotConfig` | `RobotConfig` |
| 位置 | `[y,z]`，shape `(2,)` | `[x,y,z]`，shape `(3,)` |
| 基座姿态 | `base_angle_rad`，从 +Z 向 +Y 为正 | `base_rpy_rad` |
| q / `initial_configuration_rad` | `[theta_1,...,theta_n]`，shape `(n,)` | `[[theta_1,phi_1],...]`，shape `(n,2)` |
| 弯曲限制 | `abs(theta_i) <= max_bend_rad` | `0 <= theta_i <= max_bend_rad` |
| 障碍物 | `circle`、`rectangle` | `sphere`、`box` |
| tip orientation | 从 +Z 向 +Y 的未包裹累计角，rad | 世界旋转矩阵 `(3,3)` |
| 显示 | 横轴 y，纵轴 z，等比例 | 原有三维图 |

2D 不包含 phi，也不需要 3D 的相机 elevation/azimuth 或球面网格设置。
初始基座默认为 `[0,0]`、`base_angle_rad=0`，所以机器人沿 +Z 伸直。
`sections` 的长度仍确定段数，段顺序仍由基座到末端。
目标容差和全部尺寸单位为 m。

障碍物格式：

```json
{"kind":"circle","name":"circle_1","center_m":[-0.035,0.105],"radius_m":0.015}
```

```json
{"kind":"rectangle","name":"rect_1","min_m":[0.09,0.04],"max_m":[0.12,0.12]}
```

rectangle 的 `min_m` / `max_m` 为世界 `(y,z)` 下界和上界，各轴必须严格递增。
2D `workspace_bounds_m` 的两行分别是 y 与 z 的显示范围，仍不会自动生成墙壁。
`body_alpha` 控制图中物理半径管体的透明度，`obstacle_alpha` 控制障碍物显示；
这两个参数不影响碰撞判断。

`solver.method` 在 2D 可选 `SLSQP` 或 `L-BFGS-B`，默认 `SLSQP`。
3D 的非线性弯曲圆盘约束继续使用 `SLSQP`，不允许误选仅支持 box bounds 的方法。
原配置没有 method 时沿用 SLSQP；迭代预算、目标停止阈值、初值、采样精度和
安全间隙仍由配置传入，算法内不新增隐藏的实验参数。

### 数学与共享实现

2D 使用独立的解析 FK，局部位置为 `[(1-cos(k*s))/k, sin(k*s)/k]`，
以 signed sinc 写法处理零和近零曲率。平面姿态 beta 的旋转矩阵为：

```text
R_yz(beta) = [[ cos(beta), sin(beta)],
              [-sin(beta), cos(beta)]]
tip_tangent = [sin(beta), cos(beta)]
```

每段位置先按前一段累计姿态旋转再平移，tip angle 为 `base_angle + sum(theta)`。
全局中心线按相同的物理弧长 convention 均匀采样，没有重复的段边界点；
逐段曲线为碰撞检查保留精确边界，与 3D API 一致。

位置 IK 共用原有 SciPy 调用与失败报告流程，仅 2D 使用 signed q 和逐段正负边界，
不引入 phi。`position_objective_2d` 独立表达位置目标并调用同源的 2D 解析末端计算，
不加入障碍物惩罚；重复尝试由独立搜索层负责。

碰撞代码复用同一个点到线段距离和线段到轴对齐盒距离算法，在 2D 中分别用于
circle 和 rectangle；不是只检查中心线上的点。物理半径、安全间隙、弧到弦偏差扩张
及间隙下界的语义与 3D 一致。相切或刚好达到安全余量的情况保守拒绝。
图中半透明管体用米为单位表示真实半径，线宽不承担碰撞几何的作用。

`evaluate_target` 和 CLI 按机器人类型选择 FK 和图形，共用结构化结果。
报告新增 `dimension` / `position_coordinates`。2D 输出 `tip_orientation_angle_rad`；
3D 保留 `tip_orientation_matrix`，不混用角度与矩阵。
仍然分别报告 `target_reached`、`collision_status`、`static_target_feasible`，
目标达到但最终形状碰撞不等于目标没有其他无碰撞解。

### 2D/3D 一致性与回归

现有 3D 公式中 `phi=pi/2` 向 +Y 弯，因此测试采用固定映射：

```text
theta_3d = abs(theta_2d)
phi_3d = pi/2  if theta_2d >= 0 else -pi/2
```

默认比较使用零基座位姿；额外基座测试使用已知对应关系
`base_xyz=[0,y,z]`，`base_rpy=[-base_angle,0,0]`，不进行最佳拟合或任意坐标对齐。
长度、段数和采样弧长完全相同。

`test_pcc_2d.py` 覆盖直线、单段正弯、单段负弯、同向多段、S 弯、近零和最大弯曲，
对每个采样点比较 YZ 误差，并检查 3D 的 X 偏移、末端切线和段连接。
采用绝对容差 `1e-12 m`，用于容纳双精度三角函数与变换累积误差；
失败信息给出 `max_yz_error_m` 与 `max_abs_x_m`，不会自动对齐掩盖问题。

`test_planar_pipeline.py` 覆盖维度错误、非法障碍物、2D 两种求解器、正负 IK、
弯曲限制、长度排除、优化失败、沿本体的碰撞、相切、安全余量及无 Matplotlib 的 CLI。
原有测试保留；其中旧示例尺寸写死的地方改为独立测试几何或配置推导值，
避免修改演示尺寸使基础数值测试失去含义。

没有新增运行依赖，也没有实现 FTL、RRT、渐进插入或运动碰撞检查。

## 多起点无碰撞 IK 搜索

### 使用方式

当前 `robot_2d.json` 配置包含 `ik_search`，普通运行即启用搜索：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --search --headless --output outputs/staircase_search.json
```

没有 `ik_search` 的旧配置仍默认单次 IK；可用 `--search` 显式启用默认搜索设置。
`--single-run` 强制使用原流程。原有 `evaluate_target` 和 `solve_position_ik` 的单次语义不变。

可复用 API：

```python
import numpy as np
from continuum_ftl.config import load_config
from continuum_ftl.ik_search import find_collision_free_ik

config = load_config("configs/robot_2d.json")
result = find_collision_free_ik(config)  # 默认不提供参考构型
reference_result = find_collision_free_ik(config, np.deg2rad([90, -45, 45]))
print(result.success, result.attempts, result.computation_time)
```

显式 `settings=` 覆盖配置中的搜索设置，显式参考参数覆盖配置中的参考初值。
参考构型只用作第一轮初值，仍必须经过 IK、目标误差和准确静态碰撞检查。
实现不包含阶梯专用角度；参考角度只存在于测试、文档和用户可选配置中。

### 配置与顺序

```json
"ik_search": {
  "max_attempts": 64,
  "random_seed": 7,
  "max_seed_draws": 1024,
  "seed_distance_tolerance_rad": 0.001,
  "candidate_distance_tolerance_rad": 0.0001,
  "reference_configuration_rad": null
}
```

2D 参考值是 `(n,)` 的有符号 rad 向量；3D 是 `(n,2)` 的 theta/phi rad 数组。
例如可选阶梯参考为 `[1.5707963267948966,-0.7853981633974483,0.7853981633974483]`。
初值必须在各段限制内，否则明确报错。

顺序为：可选参考 → robot 中的初始 q → 固定随机种子产生的不同初值。
2D 从每段 signed angle 区间采样；3D 在每段二维弯曲圆盘中均匀采样。
找到第一个合格构型立即返回，不继续收集候选，也不声称得到最优解。
同一环境与依赖版本下，固定 random_seed 可复现初值序列和构型结果；运行时间不固定。

去重距离使用 rad：2D 为 signed q 的欧氏距离，3D 为
`[theta*cos(phi),theta*sin(phi),...]` 的欧氏距离，从而正确处理 phi 绕回及伸直时的方向退化。
近重复初值不调用 IK，不消耗 attempts；不同初值返回近重复候选时，
这次 IK 仍计入 attempts，但不再次检查同一个最终形状。
两种去重容差均可配置；有限候选去重可能保守丢弃邻近解，因此失败不是不可达证明。

`max_attempts` 约束实际 IK 调用数，包含失败或返回重复解的调用。
`max_seed_draws` 限制随机初值提案次数，防止去重容差过大时无穷等待。

### 接收与失败

只有 IK 已满足位置容差和弯曲限制的候选才进行完整中心线采样与本体碰撞检查。
最终接收时重新确认该形状的 tip 误差，并使用现有准确球/盒或圆/矩形检查器，
包括物理半径、圆弧偏差扩张和安全余量。没有引入近似快速检查来替代最终验证。

结果字段：

| 字段 | 含义 |
|---|---|
| `success` | 找到满足目标、弯曲限制、准确静态碰撞检查的解 |
| `configuration` | 成功解的 q，rad；失败为 null |
| `tip_error` | 成功解的误差，或失败搜索中最小诊断误差，m |
| `minimum_clearance` | 成功解的保守间隙下界，m；无障碍物或失败为 null |
| `attempts` | 实际执行的 IK 次数 |
| `computation_time` | 搜索函数耗时，s；不含绘图、导入和报告文件写入 |
| `failure_reason` | 成功为 null，否则 `NO_COLLISION_FREE_IK_FOUND` |
| `termination` | 找到解、IK 预算耗尽、随机提案耗尽或明确总长排除 |
| `attempt_history` | 各轮初值来源、q、误差、碰撞结果与重复标志 |

成功的最终形状报告放在 `accepted_final_configuration`。
失败时已检查的碰撞形状只作为 `diagnostic_rejected_configuration`，不会冒充成功解。
如果没有任何到达候选，就不生成最终形状图；若绘制失败诊断，图中明确标注 SEARCH FAILED。
所有结果仍标记 `motion_checked=false` 和 `self_collision_checked=false`。

测试新增：无障碍 2D/3D、同目标碰撞解与安全解、预算耗尽、参考优先、
随机复现、种子/候选去重、phi 等价、随机提案预算、未到达时不做完整碰撞检查，
以及解析 tip-only FK 与完整 FK 的一致性。未增加优化框架或运行依赖。
