"""Stage 2B continuous actual-body exposure. No collision or FTL calculation."""

import numpy as np

from .models import (
    DeploymentState,
    EnvironmentExposure,
    ExposureStatus,
    PlanarCurveGeometry,
)
from .planar_geometry import (
    entry_intersections,
    evaluate_curve,
    material_boundaries,
    sample_interval,
    section_indices,
)


def extract_environment_exposure(
    deployment_state: DeploymentState,
    environment_entry_z_m: float,
    *,
    max_arc_spacing_m: float,
    tolerance_m: float,
) -> EnvironmentExposure:
    """Clip actual complete Stage 2A PCC geometry at the world entry plane (m).

    Reconstructs continuous geometry from the state's physical lengths, exact
    junction poses and applied bends, never from a desired reference path.
    The complete original state and section curvatures remain unchanged.
    """
    geometry = PlanarCurveGeometry(
        deployment_state.section_names,
        deployment_state.section_lengths_m,
        deployment_state.forward.section_endpoints_m,
        deployment_state.boundary_tangent_angles_rad,
    )
    return extract_curve_exposure(
        geometry,
        environment_entry_z_m,
        max_arc_spacing_m=max_arc_spacing_m,
        tolerance_m=tolerance_m,
    )


def extract_curve_exposure(
    geometry: PlanarCurveGeometry,
    environment_entry_z_m: float,
    *,
    max_arc_spacing_m: float,
    tolerance_m: float,
) -> EnvironmentExposure:
    """Single forward entry-to-tip interval for a continuous world YZ PCC curve.

    Lengths use physical material arc coordinates. Multiple intersections,
    tangency, plane-coincident arcs and a missing entry are explicit ambiguities.
    Tip on the plane with a forward tangent counts as entered with zero length.
    This classifies the centreline, not a radius-inflated robot cross section.
    """
    if not np.isfinite(max_arc_spacing_m) or max_arc_spacing_m <= 0:
        raise ValueError("Positive finite exposure sampling spacing required")
    intersections, coincident = entry_intersections(geometry, environment_entry_z_m, tolerance_m)
    tip = geometry.junction_positions_m[-1].copy()
    tip_entered = bool(tip[1] >= environment_entry_z_m - tolerance_m)
    count = len(geometry.section_names)

    def no_interval(status: ExposureStatus, diagnostic: str) -> EnvironmentExposure:
        empty = status == ExposureStatus.NOT_ENTERED
        return EnvironmentExposure(
            status,
            diagnostic,
            tip_entered,
            None,
            None,
            np.empty((0, 2)),
            np.empty(0),
            np.empty(0),
            0.0 if empty else None,
            geometry.section_names,
            np.zeros(count) if empty else None,
            np.empty(0, dtype=np.int64),
            tip,
            intersections,
        )

    if coincident or len(intersections) > 1:
        return no_interval(
            ExposureStatus.AMBIGUOUS,
            f"{len(intersections)} isolated entry-plane intersections; "
            f"plane-coincident section={coincident}. Single forward entry required.",
        )
    if not len(intersections):
        if not tip_entered:
            return no_interval(ExposureStatus.NOT_ENTERED, "Tip and entire curve are below entry.")
        return no_interval(ExposureStatus.AMBIGUOUS, "Curve is inside but has no entry crossing.")
    entry = float(intersections[0])
    boundaries = material_boundaries(geometry)
    index = int(section_indices(geometry, np.array([entry]))[0])
    angle = (
        geometry.junction_angles_rad[index]
        + (geometry.junction_angles_rad[index + 1] - geometry.junction_angles_rad[index])
        * (entry - boundaries[index])
        / geometry.section_lengths_m[index]
    )
    # A grazing contact or a backward crossing does not define the assumed domain.
    if np.cos(angle) <= np.sqrt(np.finfo(float).eps) or not tip_entered:
        return no_interval(
            ExposureStatus.AMBIGUOUS, "Entry is tangential or backward, not forward."
        )
    material_arcs = sample_interval(geometry, entry, boundaries[-1], max_arc_spacing_m)
    points = evaluate_curve(geometry, material_arcs)
    section_lengths = np.maximum(0.0, boundaries[1:] - np.maximum(boundaries[:-1], entry))
    return EnvironmentExposure(
        ExposureStatus.EXPOSED,
        "One continuous forward entry-to-tip interval.",
        tip_entered,
        points[0].copy(),
        entry,
        points,
        material_arcs - entry,
        material_arcs,
        float(boundaries[-1] - entry),
        geometry.section_names,
        section_lengths,
        section_indices(geometry, material_arcs),
        tip,
        intersections,
    )


def exposure_report(exposure: EnvironmentExposure) -> dict:
    """Serializable diagnostics; entry-relative and material arc coordinates are m."""
    return {
        "status": exposure.status.value,
        "diagnostic": exposure.diagnostic,
        "tip_entered": exposure.tip_entered,
        "entry_position_m": exposure.entry_position_m.tolist()
        if exposure.entry_position_m is not None
        else None,
        "entry_material_arc_m": exposure.entry_material_arc_m,
        "centerline_m": exposure.centerline_m.tolist(),
        "arc_lengths_from_entry_m": exposure.arc_lengths_m.tolist(),
        "material_arc_lengths_m": exposure.material_arc_lengths_m.tolist(),
        "exposed_length_m": exposure.exposed_length_m,
        "section_exposed_lengths_m": dict(
            zip(exposure.section_names, exposure.section_exposed_lengths_m.tolist(), strict=True)
        )
        if exposure.section_exposed_lengths_m is not None
        else None,
        "sample_section_indices": exposure.section_indices.tolist(),
        "sample_section_names": [exposure.section_names[i] for i in exposure.section_indices],
        "tip_position_m": exposure.tip_position_m.tolist(),
        "intersection_material_arcs_m": exposure.intersection_material_arcs_m.tolist(),
    }
