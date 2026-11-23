"""Display-only sampled centreline sweeps; independent of planning and collision."""

import numpy as np


class CentrelineSweep:
    """Red patches between adjacent accepted exposed bodies in world metres.

    Curves are matched by normalized exposed arc length for rendering only.
    The surface approximates motion between saved frames, excludes robot radius,
    and does not certify swept-volume collision freedom. No reference points
    substitute for actual geometry. Replay displays only the elapsed prefix.
    """

    def __init__(
        self,
        axes,
        curves: tuple[np.ndarray, ...],
        accepted: tuple[bool, ...],
        *,
        dimension: int,
        samples: int = 129,
    ) -> None:
        from matplotlib.collections import PolyCollection
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if len(curves) != len(accepted) or dimension not in (2, 3) or samples < 2:
            raise ValueError(
                "Matching frame flags, dimension 2/3 and at least two samples required"
            )
        self.frames = []
        previous = None
        for curve, valid in zip(curves, accepted, strict=True):
            points = np.asarray(curve, dtype=float).reshape(-1, dimension)
            triangles = np.empty((0, 3, dimension))
            if not valid or not len(points):
                previous = None
            else:
                arcs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                keep = np.r_[True, np.diff(arcs) > 0]
                points, arcs = points[keep], arcs[keep]
                grid = np.linspace(0, arcs[-1], samples)
                current = np.column_stack(
                    [np.interp(grid, arcs, points[:, k]) for k in range(dimension)]
                )
                if previous is not None:
                    triangles = np.concatenate(
                        (
                            np.stack((previous[:-1], previous[1:], current[1:]), axis=1),
                            np.stack((previous[:-1], current[1:], current[:-1]), axis=1),
                        )
                    )
                previous = current
            self.frames.append(triangles)
        self.dimension = dimension
        if dimension == 2:
            self.artist = PolyCollection([], facecolor="red", edgecolor="none", zorder=1)
            axes.add_collection(self.artist)
        else:
            self.artist = Poly3DCollection([], facecolor="red", edgecolor="none", alpha=0.25)
            axes.add_collection3d(self.artist)

    def draw_frame(self, index: int) -> None:
        """Show accepted sweep through this frame, clearing future motion on replay."""
        if not 0 <= index < len(self.frames):
            raise IndexError("Sweep frame outside saved sequence")
        patches = np.concatenate(self.frames[: index + 1])
        self.artist.set_verts(patches)


def save_saved_sweep(result_path: str) -> None:
    """Render red_sweep.png beside a saved 2D/3D result, without solving again."""
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    source = Path(result_path)
    result = json.loads(source.read_text(encoding="utf-8"))
    reference = np.asarray(result["reference_path"]["points_m"])
    dimension = reference.shape[1]
    steps = result["steps"]
    curves = tuple(np.asarray((s.get("exposure") or {}).get("centerline_m", [])) for s in steps)
    accepted = tuple(s["accepted"] for s in steps)
    fig = plt.figure(figsize=(8, 8), layout="constrained")
    ax = fig.add_subplot(projection="3d" if dimension == 3 else None)
    sweep = CentrelineSweep(ax, curves, accepted, dimension=dimension)
    sweep.draw_frame(len(steps) - 1)
    ax.plot(*reference.T, "k--", lw=1, label="Fixed reference")
    last = max(i for i, s in enumerate(steps) if s["accepted"])
    body = curves[last].reshape(-1, dimension)
    if len(body):
        ax.plot(*body.T, color="#20578c", lw=2, label="Last accepted exposed body")
    history = np.asarray([s["actual_tip_position_m"] for s in steps if s["accepted"]])
    ax.plot(*history.T, color="#e69a30", lw=1, label="Actual tip history")
    points = np.vstack([reference, history, *[c.reshape(-1, dimension) for c in curves if c.size]])
    low, high = points.min(axis=0) - 0.01, points.max(axis=0) + 0.01
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    if dimension == 3:
        ax.set_zlim(low[2], high[2])
        ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="World Z (m)")
        ax.set_box_aspect(high - low, zoom=0.85)
    else:
        ax.set(xlabel="World Y (m)", ylabel="World Z (m)")
        ax.set_aspect("equal")
    ax.set_title(f"{result.get('case', 'FTL')}: accepted centreline sweep (red)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    destination = source.with_name("red_sweep.png")
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(
        description="Plot red sweeps from saved results; no optimization"
    )
    parser.add_argument("results", nargs="+", help="Saved 2D/3D result.json paths")
    for name in parser.parse_args().results:
        save_saved_sweep(name)
