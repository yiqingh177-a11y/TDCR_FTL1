"""Static 2D/3D tube-vs-round/axis-aligned-box checks, independent of rendering.

Per-chord capsules are inflated by each PCC subarc's sagitta, conservatively
enclosing the curved robot tube. Reported minimum clearance is a lower bound
for that tube; it can conservatively reject a near-boundary shape. No self-
collision or motion checking is implemented in Stage 1.
"""

import numpy as np

from .models import (
    BoxObstacle,
    CircleObstacle,
    CollisionContact,
    CollisionResult,
    CollisionStatus,
    EnvironmentConfig,
    FloatArray,
    ForwardResult,
    PlanarRobotConfig,
    RectangleObstacle,
    RobotConfig,
    SphereObstacle,
)


def segment_point_distance(start: FloatArray, end: FloatArray, point: FloatArray) -> float:
    """Distance (m) from point to closed segment; matching world (2,) or (3,) arrays."""
    _validate_vectors(start, end, point)
    direction = end - start
    squared_length = float(direction @ direction)
    fraction = (
        float(np.clip((point - start) @ direction / squared_length, 0.0, 1.0))
        if squared_length > 0
        else 0.0
    )
    return float(np.linalg.norm(start + fraction * direction - point))


def segment_box_distance(
    start: FloatArray, end: FloatArray, center: FloatArray, size: FloatArray
) -> float:
    """Exact distance (m) from a segment to a solid axis-aligned box.

    Inputs are matching world (2,) or (3,) arrays; size is full axis extents.
    Squared distance is quadratic between crossings of box/rectangle boundaries.
    Check each interval's
    endpoints and stationary point, avoiding a nested numerical optimizer.
    """
    _validate_vectors(start, end, center, size)
    if np.any(size <= 0):
        raise ValueError("Box/rectangle sizes must be positive")
    low, high = center - size / 2, center + size / 2
    direction = end - start
    knots = [0.0, 1.0]
    for axis in range(start.size):
        if direction[axis] != 0:
            for plane in (low[axis], high[axis]):
                fraction = (plane - start[axis]) / direction[axis]
                if 0 < fraction < 1:
                    knots.append(float(fraction))
    knots = sorted(set(knots))
    candidates = list(knots)
    for left, right in zip(knots[:-1], knots[1:], strict=True):
        midpoint = start + (left + right) / 2 * direction
        active = (midpoint < low) | (midpoint > high)
        boundary = np.where(midpoint < low, low, high)
        quadratic = float(direction[active] @ direction[active])
        if quadratic > 0:
            linear = float(direction[active] @ (start - boundary)[active])
            candidates.append(float(np.clip(-linear / quadratic, left, right)))
    points = start + np.array(candidates)[:, None] * direction
    offsets = np.maximum(np.maximum(low - points, points - high), 0.0)
    return float(np.min(np.linalg.norm(offsets, axis=1)))


def _validate_vectors(*vectors: FloatArray) -> None:
    shape = np.shape(vectors[0])
    if shape not in ((2,), (3,)) or any(
        np.shape(v) != shape or not np.isfinite(v).all() for v in vectors
    ):
        raise ValueError("Geometry requires matching finite world (2,) or (3,) arrays")


def check_static_collision(
    robot: RobotConfig | PlanarRobotConfig, forward: ForwardResult, environment: EnvironmentConfig
) -> CollisionResult:
    """Check every section against all obstacles, including radius and safety margin.

    Contacts use zero-based section indices. No obstacles -> clear with clearance
    None. Equality at contact or the safety margin is conservatively rejected.
    """
    if not np.isfinite(environment.safety_margin_m) or environment.safety_margin_m < 0:
        raise ValueError("Safety margin must be finite and nonnegative")
    dimension = 2 if isinstance(robot, PlanarRobotConfig) else 3
    if forward.tip_position_m.shape != (dimension,):
        raise ValueError("FK result dimension does not match robot")
    expected_types = (
        (CircleObstacle, RectangleObstacle) if dimension == 2 else (SphereObstacle, BoxObstacle)
    )
    if any(not isinstance(o, expected_types) for o in environment.obstacles):
        raise ValueError("Obstacle types must match the robot dimension")
    minimum = float("inf")
    contacts = []
    for index, (section, curve) in enumerate(
        zip(robot.sections, forward.section_curves, strict=True)
    ):
        radius = section.radius_m + curve.chord_deviation_bound_m
        for obstacle in environment.obstacles:
            clearance = float("inf")
            for start, end in zip(curve.points_m[:-1], curve.points_m[1:], strict=True):
                if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
                    gap = segment_point_distance(start, end, obstacle.center_m) - obstacle.radius_m
                elif isinstance(obstacle, BoxObstacle):
                    gap = segment_box_distance(start, end, obstacle.center_m, obstacle.size_m)
                elif isinstance(obstacle, RectangleObstacle):
                    center = (obstacle.min_m + obstacle.max_m) / 2
                    size = obstacle.max_m - obstacle.min_m
                    gap = segment_box_distance(start, end, center, size)
                else:
                    raise TypeError(f"Unsupported obstacle type: {type(obstacle).__name__}")
                clearance = min(clearance, gap - radius)
            minimum = min(minimum, clearance)
            if clearance <= environment.safety_margin_m:
                contacts.append(CollisionContact(obstacle.name, index, clearance))
    status = (
        CollisionStatus.COLLISION
        if minimum <= 0
        else CollisionStatus.SAFETY_MARGIN_VIOLATION
        if contacts
        else CollisionStatus.CLEAR
    )
    return CollisionResult(status, minimum if environment.obstacles else None, tuple(contacts))
