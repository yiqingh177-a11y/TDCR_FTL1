# TDCR_FTL1 / continuum_ftl

独立的 2D/3D PCC 运动学、位置 IK 与最终构型静态碰撞检查，支持多起点无碰撞 IK 搜索。
目前仍为 Stage 1，不包含 FTL、RRT、插入或运动路径碰撞验证。

在项目根目录运行（Windows CMD）：

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json
```

原有 3D 模式：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/example.json
```

当前 2D 阶梯配置包含 `ik_search`，默认自动搜索无碰撞最终构型。
参考构型默认为 `null`，不会硬编码 `[90,-45,45]` 度作为解。
3D 可加 `--search` 使用同一个搜索层。加 `--single-run` 可复现原有单次 IK 流程：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/example.json --search
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --single-run
```

搜索报告包含 `success`、`configuration`、`tip_error`、`minimum_clearance`、
`attempts`、`computation_time` 和 `failure_reason`。
失败返回 `NO_COLLISION_FREE_IK_FOUND`，不等于目标数学上不可达。

不打开窗口，保存 2D 报告与图：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless --output outputs/result_2d.json --save-figure outputs/result_2d.png
```

2D 配置使用 `dimension: 2`，坐标为 `(y,z)`，每段只有一个可正可负的弯曲角。
3D 仍使用原来的 `(theta,phi)`；省略 `dimension` 继续按 3D 加载。
所有尺寸均为演示参数，不代表真实 COBRA 的验证尺寸。

安装、API、配置、算法、失败状态和测试命令见 [README_STAGE1.md](README_STAGE1.md)。

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```
