"""Configuration and result data. Distances are metres; model angles are radians.

World +Z is up. Local +Z is each section's unbent axis. Sections are ordered
base to tip. q has shape (section_count, 2), with columns (theta, phi): theta
is nonnegative, and phi runs from local +X towards local +Y. Base RPY means
Rz(yaw) @ Ry(pitch) @ Rx(roll), describing the base frame in the world frame.
PlanarRobotConfig instead uses signed q (n,), position (y,z), and a base angle
measured from +Z towards +Y. Its robot and obstacle arrays remain genuinely 2D.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SectionConfig:
    length_m: float
    radius_m: float
    max_bend_rad: float


@dataclass(frozen=True)
class RobotConfig:
    sections: tuple[SectionConfig, ...]
    base_position_m: FloatArray
    base_rpy_rad: FloatArray
    initial_configuration_rad: FloatArray

    @property
    def section_count(self) -> int:
        """Number of independently parameterized PCC sections."""
        return len(self.sections)


@dataclass(frozen=True)
class PlanarRobotConfig:
    """YZ robot: signed q (n,), base (y,z), base angle from +Z towards +Y (rad)."""

    sections: tuple[SectionConfig, ...]
    base_position_m: FloatArray
    base_angle_rad: float
    initial_configuration_rad: FloatArray

    @property
    def section_count(self) -> int:
        """Number of planar PCC sections, ordered from base to tip."""
        return len(self.sections)


@dataclass(frozen=True)
class TargetConfig:
    position_m: FloatArray
    tolerance_m: float


@dataclass(frozen=True)
class SphereObstacle:
    name: str
    center_m: FloatArray
    radius_m: float


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center_m: FloatArray
    size_m: FloatArray  # Full XYZ dimensions, not half extents.


Obstacle = SphereObstacle | BoxObstacle


@dataclass(frozen=True)
class CircleObstacle:
    """Circle in world YZ, center (2,) in metres."""

    name: str
    center_m: FloatArray
    radius_m: float


@dataclass(frozen=True)
class RectangleObstacle:
    """Axis-aligned world YZ rectangle, min/max bounds (2,) in metres."""

    name: str
    min_m: FloatArray
    max_m: FloatArray


@dataclass(frozen=True)
class EnvironmentConfig:
    workspace_bounds_m: FloatArray  # (dimension, 2), display bounds, not physical walls.
    safety_margin_m: float
    obstacles: tuple[Obstacle | CircleObstacle | RectangleObstacle, ...]


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int
    objective_tolerance: float
    bend_limit_tolerance_rad: float
    method: str = "SLSQP"


@dataclass(frozen=True)
class SamplingConfig:
    max_arc_spacing_m: float


@dataclass(frozen=True)
class IKSearchConfig:
    """Multi-start budgets and bend-space distances (rad); no motion planning.

    Reference q uses (n,) signed angles in 2D, or (n,2) theta/phi in 3D.
    max_seed_draws bounds random proposals, including duplicate rejections.
    """

    max_attempts: int = 64
    random_seed: int = 7
    max_seed_draws: int = 1024
    seed_distance_tolerance_rad: float = 1e-3
    candidate_distance_tolerance_rad: float = 1e-4
    reference_configuration_rad: FloatArray | None = None

    def __post_init__(self) -> None:
        for name in ("max_attempts", "max_seed_draws"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        for name in ("seed_distance_tolerance_rad", "candidate_distance_tolerance_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class VisualizationConfig:
    figure_size_inches: FloatArray
    elevation_deg: float
    azimuth_deg: float
    section_colors: tuple[str, ...]
    target_color: str
    obstacle_color: str
    obstacle_alpha: float
    centerline_width: float
    base_axis_length_m: float
    surface_samples: int


@dataclass(frozen=True)
class PlanarVisualizationConfig:
    """2D plot settings without 3D camera or surface parameters."""

    figure_size_inches: FloatArray
    section_colors: tuple[str, ...]
    target_color: str
    obstacle_color: str
    obstacle_alpha: float
    centerline_width: float
    base_axis_length_m: float
    body_alpha: float


@dataclass(frozen=True)
class DeploymentConfig:
    """Stage 2A fixed +Z feed settings; lengths m and signed angles rad.

    Vectors follow active section order. The bending schedule is manual, not FTL.
    """

    passive_length_m: float
    passive_radius_m: float
    active_section_names: tuple[str, ...]
    feed_exit_m: FloatArray
    environment_entry_z_m: float
    feed_block_center_m: FloatArray
    feed_block_size_m: FloatArray
    end_distance_m: float
    distance_step_m: float
    bend_targets_rad: FloatArray
    bend_ramp_distances_m: FloatArray
    frame_interval_ms: int
    visible_passive_length_m: float
    section_colors: tuple[str, ...]
    feed_color: str
    feed_opacity: float
    plot_padding_m: float
    geometry_tolerance_m: float
    reference_path_color: str = "#7b3294"
    exposed_line_width_factor: float = 2.2


@dataclass(frozen=True)
class ProjectConfig:
    description: str
    robot: RobotConfig | PlanarRobotConfig
    target: TargetConfig
    environment: EnvironmentConfig
    solver: SolverConfig
    sampling: SamplingConfig
    visualization: VisualizationConfig | PlanarVisualizationConfig
    ik_search: IKSearchConfig | None = None
    deployment: DeploymentConfig | None = None

    @property
    def dimension(self) -> int:
        """2 for signed planar PCC, 3 for theta/phi spatial PCC."""
        return 2 if isinstance(self.robot, PlanarRobotConfig) else 3


@dataclass(frozen=True)
class SectionCurve:
    """World points (K,d), local arc coordinates (K,), and arc-to-chord error bound."""

    points_m: FloatArray
    arc_lengths_m: FloatArray
    chord_deviation_bound_m: float


@dataclass(frozen=True)
class ForwardResult:
    """World FK: endpoints include the base, shape (n+1,d), d is 2 or 3.

    centerline_m is (M,d), uniformly sampled in global arc length. Per-section
    curves include exact boundaries and are sampled independently for collision.
    tip_orientation is a 3D rotation matrix (3,3), or the unwrapped 2D tangent
    angle from +Z towards +Y, in rad. No spatial phi is used for planar states.
    """

    tip_position_m: FloatArray
    tip_orientation: FloatArray | float  # 3D rotation matrix, or 2D tangent angle from +Z.
    section_endpoints_m: FloatArray
    centerline_m: FloatArray
    arc_lengths_m: FloatArray
    section_curves: tuple[SectionCurve, ...]


@dataclass(frozen=True)
class DeploymentState:
    """Complete physical chain at d (m). No passive bending DOF is introduced.

    Bend/release vectors have n active entries. forward includes full passive
    geometry, even if the viewer clips it. Boundary tangent angles (n+2,) rad.
    Samples are piecewise uniform in arc length and include the passive join.
    """

    distance_m: float
    proximal_position_m: FloatArray
    active_base_position_m: FloatArray
    requested_bends_rad: FloatArray
    applied_bends_rad: FloatArray
    released: NDArray[np.bool_]
    release_thresholds_m: FloatArray
    section_names: tuple[str, ...]
    section_lengths_m: FloatArray
    boundary_tangent_angles_rad: FloatArray
    forward: ForwardResult
    consistency_warnings: tuple[str, ...]


@dataclass(frozen=True)
class PlanarCurveGeometry:
    """Continuous physical PCC curve in world YZ, proximal to distal.

    n section lengths (m), n+1 exact junction positions (m) and tangent angles
    (rad). A straight passive section is represented by equal endpoint angles.
    Material arc length starts at the physical proximal end, independently of d.
    """

    section_names: tuple[str, ...]
    section_lengths_m: FloatArray
    junction_positions_m: FloatArray
    junction_angles_rad: FloatArray


class ExposureStatus(StrEnum):
    NOT_ENTERED = "not_entered"
    EXPOSED = "exposed"
    AMBIGUOUS = "ambiguous_entry_geometry"


@dataclass(frozen=True)
class EnvironmentExposure:
    """One continuous entry-to-tip domain, NOT a collision or FTL result.

    points (M,2) and arc vectors (M,) are empty before entry or on ambiguity.
    arc_lengths_m starts at entry; material_arc_lengths_m starts at the physical
    proximal end. Internal junction samples belong to the distal section.
    Unknown interval/lengths on ambiguity are None, never an invented zero.
    tip_entered only describes tip position relative to the entry plane.
    """

    status: ExposureStatus
    diagnostic: str
    tip_entered: bool
    entry_position_m: FloatArray | None
    entry_material_arc_m: float | None
    centerline_m: FloatArray
    arc_lengths_m: FloatArray
    material_arc_lengths_m: FloatArray
    exposed_length_m: float | None
    section_names: tuple[str, ...]
    section_exposed_lengths_m: FloatArray | None
    section_indices: NDArray[np.int64]
    tip_position_m: FloatArray
    intersection_material_arcs_m: FloatArray


@dataclass(frozen=True)
class ReferencePath:
    """Fixed desired tip curve; s=0 at environment entry, s=end at IK tip (m).

    The exact PCC geometry supports continuous arc-length interpolation.
    It is independent of deployment and actual exposed-body arc coordinates.
    section_boundary_arcs_m retains every junction inside the reference domain.
    The IK tip is retained exactly, rather than snapped to a target with residual.
    """

    geometry: PlanarCurveGeometry
    entry_material_arc_m: float
    points_m: FloatArray
    arc_lengths_m: FloatArray
    section_indices: NDArray[np.int64]
    section_boundary_arcs_m: FloatArray
    target_position_m: FloatArray
    tip_error_m: float


class IKStatus(StrEnum):
    TARGET_REACHED = "target_reached"
    OUTSIDE_LENGTH_ENVELOPE = "outside_length_envelope"
    TARGET_NOT_REACHED = "target_not_reached"
    OPTIMIZATION_FAILED = "optimization_failed"


@dataclass(frozen=True)
class IKResult:
    configuration_rad: FloatArray
    tip_position_m: FloatArray
    position_error_m: float
    status: IKStatus
    optimizer_success: bool
    message: str
    iterations: int
    function_evaluations: int

    @property
    def target_reached(self) -> bool:
        """Geometric target acceptance, independent of optimizer convergence."""
        return self.status == IKStatus.TARGET_REACHED


class CollisionStatus(StrEnum):
    CLEAR = "clear"
    SAFETY_MARGIN_VIOLATION = "safety_margin_violation"
    COLLISION = "collision"


@dataclass(frozen=True)
class CollisionContact:
    obstacle_name: str
    section_index: int  # Zero-based, proximal to distal.
    clearance_m: float


@dataclass(frozen=True)
class CollisionResult:
    """Clearance is a conservative lower bound; None means no obstacles exist."""

    status: CollisionStatus
    minimum_clearance_m: float | None
    contacts: tuple[CollisionContact, ...]

    @property
    def collision_free(self) -> bool:
        """Whether both geometric separation and the safety margin are satisfied."""
        return self.status == CollisionStatus.CLEAR


@dataclass(frozen=True)
class EvaluationResult:
    ik: IKResult
    forward: ForwardResult
    collision: CollisionResult

    @property
    def static_target_feasible(self) -> bool:
        """One target-reaching final shape is clear; this says nothing about a path."""
        return self.ik.target_reached and self.collision.collision_free


@dataclass(frozen=True)
class IKSearchAttempt:
    """One actual IK invocation; skipped seeds do not consume an attempt."""

    seed_source: str
    seed_configuration_rad: FloatArray
    configuration_rad: FloatArray
    ik_status: IKStatus
    tip_error_m: float
    collision_status: CollisionStatus | None
    duplicate_candidate: bool


@dataclass(frozen=True)
class IKSearchResult:
    """Static IK search outcome. Angles rad, lengths m, computation_time seconds.

    configuration is populated only on success. minimum_clearance is the
    accurate checker's conservative lower bound (None when no obstacles).
    On failure, diagnostic holds the closest target-reaching checked shape,
    if one exists; it must never be interpreted as an accepted solution.
    """

    success: bool
    configuration: FloatArray | None
    tip_error: float | None
    minimum_clearance: float | None
    attempts: int
    computation_time: float
    failure_reason: str | None
    termination: str
    history: tuple[IKSearchAttempt, ...]
    diagnostic: EvaluationResult | None
    skipped_duplicate_seeds: int
    skipped_duplicate_candidates: int
