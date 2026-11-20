# TDCR_FTL1 / continuum_ftl

独立的 2D/3D PCC 运动学、位置 IK 与最终构型静态碰撞检查，支持多起点无碰撞 IK 搜索。
Stage 1、Stage 2A 功能保留；新增 Stage 2B 参考路径与实际环境暴露区域演示。
不包含 FTL 优化、RRT 或运动路径碰撞验证。

## Stage 2B：参考路径与入场部分

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.exposure_demo
```

读取 `outputs/stage1_reference_2d.json` 中一个已通过 Stage 1 静态验证的最终 IK 解。
紫色虚线是固定参考路径；各物理段保持原色，进入环境的实际部分用粗线突出。
可 Pause/Resume、Replay；显示每段入场长度及 feed 释放状态。
实际运动仍是 Stage 2A 手动送入，不做 FTL 或运动碰撞检查。

首次运行或修改 robot/target/obstacles 后，先用原有 Stage 1 命令重新导出匹配场景的结果：

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless --output outputs/stage1_reference_2d.json
```

参考路径、弧长定义、API、导出及验证说明见 [README_STAGE2B.md](README_STAGE2B.md)。

## Stage 2A：可视化送入演示

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.deployment_demo
```

显示固定出料口、环境入口、灰色被动段及彩色 C/B/A 段。
可点击 Replay 重播、Pause 暂停。三段全部从零角度开始，
A/B/C 分别在送入 40/100/180 mm 后解锁并按手动曲线平滑弯曲。
完整物理长度为 0.38 m；这不是 FTL，也不检查碰撞或追踪 target。
配置仍在 `configs/robot_2d.json`，新增参数位于 `deployment`。
详情与导出命令见 [README_STAGE2A.md](README_STAGE2A.md)。

## Stage 1：原有 IK 与静态碰撞

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
