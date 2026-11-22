"""Stage 3A: full-robot static collision checks on saved accepted FTL states."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .collision import check_static_collision
from .deployment import compute_deployment
from .models import (
    CollisionStatus,
    CollisionTimeline,
    FrameCollisionResult,
    PlanarRobotConfig,
    ProjectConfig,
    SectionConfig,
)


def analyze_saved_ftl(config: ProjectConfig, source: str | Path) -> CollisionTimeline:
    """Read accepted frames, reconstruct identical PCC poses, check whole tubes.

    Reuses saved active q and feed d, with physical dimensions/radii and obstacle
    geometry from config. Validate reconstructed junctions and tip against saved
    world YZ poses; mismatched lengths/feed geometry are errors. Old FTL logs do
    not archive robot radii: the caller must provide the matching geometry config.
    Collision spacing comes from config.sampling.max_arc_spacing_m, with existing
    conservative subarc sagitta inflation. Passive and all active sections are
    checked everywhere, including feed and transition zones; no exposure clipping.
    No IK, SQP, FTL rollout, motion interpolation or collision response is invoked.
    Rejected trial frames are counted but not treated as executed robot states.
    """
    if not isinstance(config.robot, PlanarRobotConfig) or config.deployment is None:
        raise ValueError("Stage 3A requires a 2D robot with deployment settings")
    started = perf_counter()
    source = Path(source)
    raw = source.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Saved FTL result must be a JSON object")
    saved = data.get("steps")
    if not isinstance(saved, list) or not saved:
        raise ValueError("Saved FTL result must contain a nonempty steps list")
    robot = replace(
        config.robot,
        sections=(
            SectionConfig(
                config.deployment.passive_length_m, config.deployment.passive_radius_m, 0
            ),
            *config.robot.sections,
        ),
    )
    frames = []
    rejected = 0
    previous_index = -1
    tolerance = config.deployment.geometry_tolerance_m
    for source_index, step in enumerate(saved):
        if not isinstance(step, dict):
            raise ValueError("Each saved frame must be a JSON object")
        if type(step.get("accepted")) is not bool:
            raise ValueError(f"Source frame {source_index + 1}: accepted must be a boolean")
        if not step["accepted"]:
            rejected += 1
            continue
        step_index = step["step_index"]
        if type(step_index) is not int or step_index <= previous_index:
            raise ValueError("Accepted source step_index values must be strictly increasing")
        previous_index = step_index
        state = compute_deployment(
            config.robot,
            config.deployment,
            float(step["deployment_distance_m"]),
            np.asarray(step["configuration_rad"], dtype=float),
            config.sampling.max_arc_spacing_m,
        )
        for key, actual in (
            ("actual_tip_position_m", state.forward.tip_position_m),
            ("section_endpoints_m", state.forward.section_endpoints_m),
            ("configuration_rad", state.applied_bends_rad),
        ):
            recorded = np.asarray(step[key], dtype=float)
            if recorded.shape != actual.shape or not np.allclose(
                recorded, actual, atol=tolerance, rtol=0
            ):
                raise ValueError(
                    f"Source frame {source_index + 1}: {key} does not match the geometry config; "
                    "use the original lengths, feed exit and deployment geometry"
                )
        collision = check_static_collision(
            robot, state.forward, config.environment, signed_clearance=True
        )
        progress = float(step["desired_reference_progress_m"])
        if not np.isfinite(progress) or progress < 0:
            raise ValueError("Saved reference progress must be finite and nonnegative")
        frames.append(
            FrameCollisionResult(
                len(frames),
                source_index,
                step_index,
                str(step["phase"]),
                progress,
                len(frames) * config.deployment.frame_interval_ms / 1000,
                state,
                collision,
            )
        )
    if not frames:
        raise ValueError("No accepted states exist in the saved result")
    reference = data.get("reference_path", {})
    if not isinstance(reference, dict):
        raise ValueError("Saved reference_path must be a JSON object")
    points = np.asarray(reference.get("points_m", []), dtype=float)
    if points.size == 0:
        points = np.empty((0, 2))
    target = np.asarray(reference.get("target_position_m", config.target.position_m), dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or not np.isfinite(points).all()
        or target.shape != (2,)
        or not np.isfinite(target).all()
    ):
        raise ValueError("Reference path and target must contain finite world YZ positions")
    return CollisionTimeline(
        tuple(frames),
        str(source.resolve()),
        hashlib.sha256(raw).hexdigest(),
        len(saved),
        rejected,
        points,
        target,
        perf_counter() - started,
    )


def collision_frame_report(frame: FrameCollisionResult) -> dict:
    """One static test: physical clearance excludes the separately reported margin."""
    result = frame.collision
    clearance = result.minimum_clearance_m
    collision = result.status == CollisionStatus.COLLISION
    return {
        "frame_number": frame.frame_index + 1,
        "source_frame_number": frame.source_frame_index + 1,
        "source_step_index": frame.source_step_index,
        "phase": frame.phase,
        "deployment_distance_m": frame.state.distance_m,
        "desired_reference_progress_m": frame.desired_reference_progress_m,
        "replay_time_s": frame.replay_time_s,
        "physical_time_s": None,
        "status": result.status.value,
        "collision": collision,
        "minimum_clearance_lower_bound_m": clearance,
        "penetration_depth_estimate_m": max(0.0, -clearance) if clearance is not None else 0.0,
        "colliding_sections": sorted(
            {
                frame.state.section_names[c.section_index]
                for c in result.contacts
                if c.clearance_m <= 0
            }
        ),
        "margin_only_sections": sorted(
            {
                frame.state.section_names[c.section_index]
                for c in result.contacts
                if c.clearance_m > 0
            }
        ),
        "contacts": [
            {
                "obstacle": c.obstacle_name,
                "section_index": c.section_index,
                "section_name": frame.state.section_names[c.section_index],
                "clearance_m": c.clearance_m,
                "collision": c.clearance_m <= 0,
            }
            for c in result.contacts
        ],
    }


def collision_timeline_report(timeline: CollisionTimeline, config: ProjectConfig) -> dict:
    """Count colliding frames, episodes and pair-frame contacts without equating them."""
    frames = [collision_frame_report(frame) for frame in timeline.frames]
    hits = [frame for frame in frames if frame["collision"]]
    unsafe = [frame for frame in frames if frame["status"] != CollisionStatus.CLEAR]
    clearances = [
        f["minimum_clearance_lower_bound_m"]
        for f in frames
        if f["minimum_clearance_lower_bound_m"] is not None
    ]
    flags = [f["collision"] for f in frames]
    return {
        "stage": "3A: post-processing of discrete saved frames",
        "source_path": timeline.source_path,
        "source_sha256": timeline.source_sha256,
        "source_frame_count": timeline.source_frame_count,
        "checked_frame_count": len(frames),
        "checked_ftl_frames": sum(f["phase"] == "FTL" for f in frames),
        "checked_approach_frames": sum(f["phase"] == "approach" for f in frames),
        "checked_reshape_frames": sum(f["phase"] == "reshape" for f in frames),
        "excluded_rejected_frames": timeline.excluded_rejected_frames,
        "collision_frame_count": len(hits),
        "collision_episode_count": sum(
            flag and (i == 0 or not flags[i - 1]) for i, flag in enumerate(flags)
        ),
        "colliding_pair_frame_count": sum(c["collision"] for f in frames for c in f["contacts"]),
        "section_collision_frame_counts": {
            name: sum(name in f["colliding_sections"] for f in frames)
            for name in timeline.frames[0].state.section_names
        },
        "safety_margin_only_frame_count": len(unsafe) - len(hits),
        "unsafe_frame_count": len(unsafe),
        "first_collision": hits[0] if hits else None,
        "first_unsafe_frame": unsafe[0] if unsafe else None,
        "minimum_clearance_lower_bound_m": min(clearances) if clearances else None,
        "maximum_penetration_depth_estimate_m": max(
            f["penetration_depth_estimate_m"] for f in frames
        ),
        "safety_margin_m": config.environment.safety_margin_m,
        "computation_time_s": timeline.computation_time_s,
        "physical_time_available": False,
        "time_definition": (
            "replay_time_s = checked frame index * display interval; not physical time"
        ),
        "geometry_scope": (
            "entire Passive and active PCC tubes against all configured obstacles; no zone clipping"
        ),
        "section_names": list(timeline.frames[0].state.section_names),
        "section_radii_m": [
            config.deployment.passive_radius_m,
            *(s.radius_m for s in config.robot.sections),
        ],
        "collision_spacing_m": config.sampling.max_arc_spacing_m,
        "clearance_definition": (
            "signed local tube/obstacle gap lower bound including PCC sagitta, "
            "excluding safety margin"
        ),
        "penetration_definition": (
            "max(0, -minimum clearance): conservative local overlap-depth estimate, "
            "not force or whole-robot separation distance"
        ),
        "historical_radii_archived": False,
        "feed_block_is_collision_geometry": False,
        "continuous_motion_checked": False,
        "self_collision_checked": False,
        "optimization_invoked": False,
        "frames": frames,
    }
