# Stage 2C — Palmer-inspired single-path incremental SQP FTL

## 参数在哪里修改

**日常只改 [configs/robot_2d.json](configs/robot_2d.json)。** `config.py` 是读取和校验代码，
`outputs/*.json` 是运行日志。完整中文参数表、单位、修改示例和重算要求见
[README 参数修改速查](README.md#参数修改速查)。

| 想改什么 | 在 JSON 中找 | 当前值 |
|---|---|---|
| 紫色出口位置 | `deployment.feed_exit_m` | `[0,-0.08]` m |
| 紫色机构中心 | `deployment.feed_block_center_m` | `[0,-0.135]` m，需与出口同步移动 |
| target | `target.position_m` | `[Y,Z]`，修改后重新导出 Stage 1 参考构型 |
| 各段长度、半径、最大弯曲 | `robot.sections[0/1/2]` | 依次 C/B/A，长度 m、角度 rad |
| 参考步长 | `ftl.reference_step_m` | 0.001 m |
| δb 下界、上界 | `ftl.min_feed_increment_m / max_feed_increment_m` | -0.002 / +0.002 m |
| 误差是否限制运动 | `ftl.enforce_error_limits` | `false`：只显示、记录 |
| 路点推进方式 | `ftl.reference_progress_mode` | `"scheduled"` |
| 身体跟随参照 | `ftl.body_reference_mode` | `"tip_history"`（默认）或 `"previous_body"`；[A/B 实验](README_BODY_REFERENCE.md) |
| 回放快慢 | `deployment.frame_interval_ms` | 60 ms/帧，不是真实固定运动速度 |

当前出口 Z=-0.08 m，环境入口 Z=0，间隙为 8 cm；以配置文件为准。
下文旧停滞诊断及历史验证数字对应原来的 5 cm 间隙，不是参数修改后的固定结果。
最新结果请看重新运行生成的 `outputs/ftl_2d.json`。

This stage jointly optimizes feed advancement and active-section bending, using
the actual PCC body and its recorded tip history. It is not independent waypoint
IK. Stage 1, Stage 2A, Stage 2B and spatial FK/IK remain available.

## Pure FTL: obstacle collisions disabled

`run_single_path_ftl()` creates a local numerical configuration with an empty
obstacle list. The caller's original obstacles remain intact for display.
FTL has no collision constraint, collision penalty or stop-on-contact callback.
The viewer explicitly labels `Obstacle collisions: OFF`.

The FTL CLI calls `load_reference_path(..., require_collision_free=False)`:
only robot geometry and target provenance must match. Changed obstacles do not
invalidate this numerical FTL input. One saved target-reaching shape is accepted
even if its Stage 1 collision status was rejected; a failed search's ONE saved
target-reaching diagnostic can also be used. No new IK search is performed.
Robot/target mismatch and a configuration that misses the target remain errors.
Stage 1 collision checking and Stage 2B's default strict reference policy are unchanged.

Regression tests enclose the entire robot in a solid obstacle: the FTL states
and error metrics are identical to the obstacle-free run, and the numerical
optimizer receives no obstacles. The original obstacle still exists for display.

## Current mode: scheduled reference tracking, errors displayed

`ftl.reference_progress_mode="scheduled"` advances the desired reference
waypoint by 2 mm per optimization step, independently of actual tip progress.
The actual tip/body are still computed by joint feed/bend SLSQP and PCC FK.
Actual projected progress remains a separate measured diagnostic; it is never
relabelled as the desired progress. A low-progress step is flagged but does not
end this observation rollout. Every reference waypoint is still attempted in order.
This is prescribed-reference tracking for the current pure FTL experiment, not
the original leader-paced progress policy and not a fixed physical feed speed.

The original policy is preserved as `reference_progress_mode="actual_tip"`:
select the next waypoint from actual accepted tip progress, and stop if no
forward progress is made. Strict historical A/B/C/D regression tests explicitly
select this original mode.

The previous stop was reproduced at d=175.719 mm: B was at its 90-degree limit,
C was still locked before d=180 mm, and actual projected progress increased by
less than one micrometre. The trigger was `NO_FORWARD_REFERENCE_PROGRESS`.
The new schedule changes the sequence of optimization targets and allows the
computed robot to pass that earlier stall without changing lengths or bend limits.

`configs/robot_2d.json` now sets `ftl.enforce_error_limits` to **false**.
Tip/body error remains in the objective, on screen and in every diagnostic frame.
The tip/body thresholds no longer reject a step. Set this switch to `true` to
restore the original strict acceptance mode; existing tolerance values are retained.
Solver failure, invalid deployment/exposure, release and increment constraints,
and the step budget still stop a rollout. No-forward-progress stops only apply
in `actual_tip` mode; scheduled mode records `reference_progress_stalled` instead.

With thresholds disabled in the **previous actual-tip-paced mode**, the staircase test accepts 66 FTL steps and stops at
attempt 67 with `NO_FORWARD_REFERENCE_PROGRESS`. The last trial has tip waypoint
error 2.082 mm and maximum body error 12.701 mm. The last accepted tip is still
63.459 mm from the final target. This is a stalled rollout, not an error-limit stop.

In the recorded **scheduled-mode run before the outlet move** (5 cm gap), the same staircase reference completes
90 accepted FTL steps (116 frames including approach), with the actual tip
inside the configured target tolerance. The measured final target error is
approximately 2.66e-8 m and maximum body-following error is **0.016013 m**.
This is numerical kinematic tracking with substantial body deviation, not a
high-accuracy whole-body FTL or collision-free navigation claim. No synthetic
reference-aligned robot geometry is inserted to achieve completion.

Reports distinguish `reference_completed` from `target_reached`/`success`.
Finishing the desired waypoint sequence with a final residual is reported as
`reference_completed=true`, `success=false`, `failure_reason=null` in this mode;
the actual error is retained. Merely disabling thresholds never declares an
inaccurate final pose to have reached the target.

## Run the visible demo

Windows CMD, from the project root:

```cmd
cd /d D:\Code\rr\TDCR_FTL1
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo
```

The default uses ONE saved target-reaching Stage 1 shape in
`outputs/stage1_reference_2d.json`, the same reference as Stage 2B. Compute the
rollout first, then replay it. Replay, Pause/Resume, Previous and Next operate
on stored numerical states. A rejected trial remains visible as the final frame.
This demo does not switch paths or search for another final IK configuration.

If the Stage 1 input is missing or outdated, export it with the existing command:

```cmd
.venv\Scripts\python.exe -m continuum_ftl --config configs/robot_2d.json --headless --output outputs/stage1_reference_2d.json
```

The FTL demo itself never calls IK or collision checking. Its reference loading
checks robot/target provenance but ignores obstacle and collision provenance.

Default outputs are `outputs/ftl_2d.json` (complete diagnostics) and
`outputs/ftl_summary.png` (accepted intermediate shapes and actual tip history).
Export the actual computed animation and retained failure frame:

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --save-gif outputs/ftl_staircase.gif --save-snapshot outputs/ftl_failure.png
```

Add `--no-show` only for testing/file export. Other deterministic test paths:

```cmd
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --case straight --output outputs/straight.json --save-summary outputs/straight.png
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --case single-bend --output outputs/single_bend.json --save-summary outputs/single_bend.png
.venv\Scripts\python.exe -m continuum_ftl.ftl_demo --case s-bend --output outputs/s_bend.json --save-summary outputs/s_bend.png
```

Their known PCC reference configurations are in `configs/ftl_validation.json`.
Targets and obstacle-free test environments are constructed in memory using FK;
the user's staircase scene is not overwritten. All vectors use proximal-to-distal
**C/B/A** order, preserving the package convention. The mathematical variables
can equivalently be listed A/B/C; changing their list order changes no DOF.

## Method and adaptations

Optional fixed-waypoint reshaping is disabled by default. See
[the bounded reshaping experiment](README_RESHAPE.md) for its trigger, acceptance
rules, separate trial diagnostics and visible replay commands. It reuses this
same weighted objective and geometry, with no additional shape DOFs.

### 数值收敛修正：finite_difference_step

`ftl.finite_difference_step` 控制 SLSQP 在归一化控制变量上估计梯度的扰动，
不是 `reference_step_m` 或实际送入距离。原默认 `1e-5` 在末端误差接近零时过大，
实测导致六个保存的失败步在 500 次内不收敛。现默认采用 SciPy 的
`sqrt(machine epsilon) = 1.4901161193847656e-8`，同样的问题在 17–31 次内收敛。
权重、运动界限、误差接受和真正 solver failure 的拒绝规则未改变。
新日志的 `effective_ftl_settings` 会记录实际采用的全部 FTL 数值设置。

详细证据见 [status 9 根因报告](outputs/solver_status9_diagnosis_2026-11-22/ROOT_CAUSE.md)。
旧 A/B 实验使用原 `1e-5`；其结果保留，但严格复现旧试验时应显式指定该数值，
不能让旧配置中省略的字段继承新版本默认值。新 A/B 快照会保存解析后的完整 FTL 默认值。

Primary reference: D. Palmer, S. Cobos-Guzman and D. Axinte (2014),
[Real-time method for tip following navigation of continuum snake arm robots](https://doi.org/10.1016/j.robot.2014.05.013),
*Robotics and Autonomous Systems* 62, 1478–1485.
The initial implementation used the public preview and the Stage 2C specification;
the subsequent full-paper audit checked Eq. (3)-(4) and Figs. 4-7. The optional
`previous_body` experiment follows the previous accepted body's exposed active
curve; the default `tip_history` follows actual accepted tip history as requested
by this project. Both remain **Palmer-inspired**, preserving our deliberate
three-zone deployment model rather than claiming exact paper reproduction.
See [the controlled body-reference experiment](README_BODY_REFERENCE.md).

Principles retained: PCC geometry; joint feed/bend increments; a weighted tip/body
objective; coarse body versus dense path comparison; bounded warm-started SQP;
and sequential introduction of bending variables. Project-specific adaptations:
the fixed feed/environment separation, full-section release permission,
environment-only active-body evaluation, finite geometric validation and failure
thresholds. No material properties, forces or hardware controllers are inferred.

1. **Approach:** deploy the straight 0.38 m physical robot from d=0 to the
   configured feed-to-entry gap (currently d=0.08 m),
   using the existing Stage 2A FK. The tip then reaches the environment entry.
   This phase is not FTL. History starts at entry, not at the feed exit.
2. **Desired waypoint:** advance the fixed reference parameter from the previous
   accepted tip projection (`actual_tip`) or previous desired waypoint (`scheduled`)
   by at most `reference_step_m`, initially 2 mm.
   Reference progress is estimated by a local geometric projection; it is not d.
3. **Variables:** bounded increments in feed and relevant active bends. The
   previous accepted configuration and previous increment provide the warm start.
   The passive section is never a variable. Complete straight unreleased sections
   remain in physical FK; only released/potentially released sections receive
   optimization variables because they can influence the distal tip.
4. **Actual geometry:** every trial calls the existing `compute_deployment()`
   and `extract_environment_exposure()`. No point position is assigned from the
   desired path. Original physical lengths, junction poses and tangent continuity
   are retained.
5. **Comparison:** freeze the accepted actual tip history, add the intended
   straight extension from the old tip to the next desired waypoint, then sample
   this polyline more densely. That extension is a comparison target only.
6. **SQP:** minimize the unsmoothed objective

   `J = 10 * ||actual_tip - next_waypoint|| + max(body_to_comparison_distance)`.

   Physical body samples come only from the active body actually exposed above
   the entry plane. Transition/feed geometry affects FK but is not a body-error
   sample. Passive geometry remains visible and in exposure data but is excluded
   from the FTL body metric, including if it later enters the environment.
7. **Validation:** recompute the candidate using a finer body grid. Check
   geometry, unambiguous exposure, release permissions, increment limits, solver
   termination, and forward progress in `actual_tip` mode. Tip/body tolerance rejection applies only
   when `enforce_error_limits=true`.
   A successful solver flag alone is insufficient. Accept and update history,
   or retain the rejected trial and stop immediately.

No squared-error replacement, smoothing or epigraph substitution is used.
The solver receives dimensionless bounded controls and J divided by a positive
`objective_scale_m`; this changes numerical scaling, not the minimizer or the
reported physical objective. Max-distance and Euclidean norms remain nonsmooth.
SLSQP failures are reported with their original status/message.

## Ordered geometric matching

Distances are to **polyline segments**, not matching array indices. Sampling
density and body/path array lengths can differ. For each ordered body sample,
a tip-anchored arc-length estimate selects a local progress window. Exact
point-to-segment projection finds the nearest eligible location inside the
window. Correspondences cannot go backwards. Thus a remote loop outside the
window cannot win merely because it is spatially close.

The estimate selects a neighbourhood only; it does not force material arc length
to equal history length. Current implementation uses monotone greedy local
matching, not a global curve-registration solver. Strongly folded curves or
large accumulated deviations can therefore be rejected or fail this baseline.

Accepted trials also receive an independently matched maximum deviation from
the **fixed planned reference**. This metric is separate from following the
**actual history plus intended extension**, and is not silently substituted into
the objective. Dense reference chord coordinates are converted back to exact
reference arc coordinates when reporting progress.

Body mean is the arithmetic mean of sampled distances; maximum includes the
worst sample's world YZ position and material coordinate from the physical
proximal end. This is sampled geometric validation, not a continuous supremum
bound. Failed tips never enter accepted history.

## Release continuity and limits

Passive/C/B/A lengths remain 0.20/0.08/0.06/0.04 m. Feed exit remains
`(0,-0.08)` and environment entry remains `z=0`. A/B/C release at d=0.04/0.10/0.18 m.
An introduced variable starts with zero bend increment; full physical FK still
contains the straight section before it becomes a variable.

For a newly introduced section, the permitted bend is
limited by

`abs(theta_i) <= max_bend_increment * clip((d - release_i) / max_feed_increment, 0, 1)`.

Thus it remains straight at the exact release threshold and gains bend allowance
continuously over the next 2 mm of feed. This additional local ramp is our
deployment adaptation, not a universal property of the robot or Palmer's model.

Signed feed bounds are configured independently in `ftl`:

```json
"min_feed_increment_m": -0.002,
"max_feed_increment_m": 0.002
```

The current scene allows -2 to +2 mm per FTL step. Setting the lower bound to
`0` restores forward-only feed; omitted lower bounds also default to `0` for
legacy configurations. Require finite `min <= 0 < max`. Before optimization,
intersect these bounds with `-d <= delta_b <= total_physical_length - d`.
Controls are normalized by `max(abs(min), max)`, so asymmetric bounds retain
their actual physical meaning. Signed previous increments remain valid warm starts.

Release flags are recomputed from the current d, so retraction can lock a section
again. Keep the original introduction ramp for sections still in that phase.
For a mature section whose candidate range re-enters the release band, use
amplitude `abs(previous_bend_i) + max_bend_increment_rad` instead of the introduction cap,
and width `min(max_feed_increment_m, previous_d - release_i)`. If its bend still
exceeds the introduction envelope on the next step, retain this re-entry rule.
The width keeps the previous state feasible even inside the band; the allowance
tapers continuously to zero at the release threshold. Requested bends must
agree with actual applied bends: an unreleased section cannot silently discard
a nonzero request. Both feed and applied angular increments are independently
checked after SQP, and an inadmissible retraction is rejected without updating
accepted history. These are geometric continuity guards, not a physical force law.

Straight approach remains forward-only. Reference progress is still governed by
the existing scheduled/actual-tip rule; allowing negative feed does not introduce
waypoint holding or reverse reference progress. Trial geometry with invalid
environment exposure is still rejected, and no error threshold is changed here.
The rollout stores discrete validated configurations; it does not provide a
servo interpolation law or certify every intermediate instant between frames.

## Unified configuration

The existing scene file `configs/robot_2d.json` now has an optional `ftl` block.
Target, obstacle coordinates, physical dimensions, Stage 1 solver and Stage 2A
manual schedule were preserved. Remaining FTL defaults live in `FTLConfig` and
can also be overridden by adding their field names to the same block.

| Setting | Initial value | Meaning |
|---|---:|---|
| `enforce_error_limits` | false | Display/record errors; true restores strict rejection |
| `reference_progress_mode` | scheduled | Advance desired reference each step; actual_tip retains original stall stop |
| `reference_step_m` | 0.002 m | Next desired reference increment |
| `min_feed_increment_m` | -0.002 m in scene; 0 if omitted | Signed lower bound; 0 disables retraction |
| `max_feed_increment_m` | 0.002 m | Positive upper feed bound; also release-ramp width |
| `max_bend_increment_rad` | 0.064 rad | Per-section angular increment bound |
| absolute angle bound | `robot.sections[].max_bend_rad` | Existing signed physical bounds |
| `optimization_body_spacing_m` | 0.005 m | Coarse body comparison during SQP |
| `validation_body_spacing_m` | 0.001 m | Independent acceptance grid |
| `comparison_spacing_m` | 0.00025 m | Dense comparison/reference spacing |
| `matching_window_m` | 0.020 m | Local path-progress search half-width |
| `tip_tolerance_m` | 0.0005 m | Next-waypoint tolerance, active only with the switch on |
| `body_tolerance_m` | 0.005 m | Body tolerance, active only with the switch on |
| `max_iterations` / `solver_tolerance` | 150 / 1e-7 | SLSQP settings for scaled objective |
| `max_steps` | 250 | FTL attempt budget, excluding approach |

The 2 mm / 0.064 rad settings follow the initial baselines requested for this
stage; body/tip acceptance tolerances are explicit demonstration choices.
The FTL feed limit is the full physical length, not the old manual animation's
`deployment.end_distance_m`. Report/export/visualization time is excluded from
rollout computation time. Motion frame interval controls playback only.

## Code and API

- `ftl_matching.py`: history extension, dense sampling, ordered local segment
  matching, and the separate unsmoothed mathematical objective.
- `ftl_optimizer.py`: trial geometry/metrics, bounded incremental SLSQP, and
  independent candidate validation. Reuses Stage 2A/2B rather than duplicating FK.
- `ftl_rollout.py`: straight approach, waypoint sequencing, accepted history,
  stop-on-first-failure policy and complete diagnostics.
- `models.py` / `config.py`: typed settings/results and centralized validation.
- `deployment_viewer.py`: optional FTL overlays, actual stored-state replay,
  failure display and single-frame inspection. Older demos retain their modes.
- `ftl_visualization.py`: static summary of accepted intermediate shapes.
- `ftl_demo.py`: CLI and obstacle-free numerical fixtures; no IK/collision calls.

```python
from continuum_ftl.config import load_config
from continuum_ftl.reference_path import load_reference_path
from continuum_ftl.ftl_rollout import run_single_path_ftl, ftl_report

config = load_config("configs/robot_2d.json")
reference = load_reference_path(config, "outputs/stage1_reference_2d.json")
result = run_single_path_ftl(config, reference)
report = ftl_report(result, reference)
```

Each step stores the full actual geometry, release flags, exposed geometry and
length, d/feed increment, C/B/A bends and increments, actual/desired tip,
actual projected reference progress and desired progress, maximum/mean history
error, fixed-reference deviation, worst body point/material coordinate,
solver status/iterations/time, accepted flag, failure reason, frozen comparison
path and accepted historical tip path. Approach frames explicitly have no solver.

## Original ordered A/B/C/D validation (strict thresholds)

These baseline results use `enforce_error_limits=true` and the original 5 cm feed gap; they are retained as
strict-mode regression cases. First ordered run on this machine, metres converted to mm below. Wall times
depend on load; they are measurements, not real-time guarantees.

| Case | Result | Accepted / attempted FTL steps | Max body deviation, including rejected trial | Last trial waypoint error | Time |
|---|---|---:|---:|---:|---:|
| A: straight | Target reached | 91 / 91 | 0.065 mm | <0.001 mm | 32.09 s |
| B: single bend | Body tolerance exceeded | 62 / 63 | 5.192 mm | 0.353 mm | 18.53 s |
| C: S bend | Body tolerance exceeded | 56 / 57 | 5.221 mm | 0.188 mm | 20.46 s |
| D: staircase | Tip tolerance exceeded | 23 / 24 | 1.770 mm | 0.609 mm | 5.80 s |

Straight-case maximum deviation is approximately 65 micrometres from numerical
optimization/sampling. Curved cases change bending variables continuously within
the configured per-step limits. The S-bend reaches the second curvature region
before its recorded failure. Failure may reflect model/step limits, local SQP
behaviour, greedy correspondence or validation tolerance; this does not prove
the path or target mathematically impossible.

For D, the last **accepted** waypoint error is 0.437 mm and the last accepted
tip remains **129.22 mm from the final target**. The rejected trial error of
0.609 mm refers to its next local waypoint, not the final target. At failure B
is still locked: the available A-only shape cannot satisfy this step as computed.
This is an observed local failure, not a global infeasibility proof. An independent
default-demo rerun took about 1.26 s with the same numerical outcome.

The staircase animation has 26 straight-approach frames, 23 accepted FTL frames
and one rejected frame. The retained failed geometry is separate from the last
accepted robot/history. The summary figure contains accepted shapes only.

## Tests and scope

```cmd
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
```

New tests cover the exact two-term objective, unequal body/path sampling,
incremental extension, loop-resistant local windows, ordered A/B/C/D runs,
release constraints, actual geometry/metric recomputation, convergence without
geometric success, explicit solver failure, budget exhaustion, unchanged scene,
no IK/collision call during FTL, replay/GIF/failure retention and static summary.
Existing Stage 1, Stage 2A, Stage 2B and 3D tests are retained.
Regression also covers disabling both limits, keeping solver failure checks,
and reporting a completed reference with an inaccurate actual target position.
Tests also verify scheduled targets advance while a stationary numerical trial
stays physically stationary, actual-tip mode still stops, and the current
scheduled staircase run reaches the real FK target after passing the old stop.
Final regression: **112 tests passed** (25 FTL tests plus 87 existing tests),
Ruff lint/format checks passed, and `pip check` found no broken requirements.

No collision checking, RRT, alternative paths, forces, cable physics, passive
deformation, 3D FTL or online replanning are implemented. A static collision-free
reference does not certify the computed rollout. Pre-change snapshots are in
ignored `outputs/stage2c_before/`. This stage ends at Stage 2C.
