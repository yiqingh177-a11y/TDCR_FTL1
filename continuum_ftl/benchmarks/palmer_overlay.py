"""Red centreline envelopes from stored accepted Palmer benchmark frames."""

from pathlib import Path

import numpy as np


def save_red_overlay(result: dict, destination: Path) -> None:
    """Fill the sampled motion envelope in world YZ (mm), without rerunning FK.

    These S-bend benchmark centrelines are monotone in Z. At each height, fill
    the lateral extent of the accepted frames. This is a discrete-frame
    centreline envelope, not robot thickness or a continuous collision proof.
    """
    import matplotlib.pyplot as plt

    bodies = [
        np.asarray(row["body_yz_m"], dtype=float) * 1000
        for row in result["steps"]
        if row["accepted"] and len(row["body_yz_m"]) >= 2
    ]
    if not bodies:
        raise ValueError("No accepted body frames to plot")
    if any(np.any(np.diff(body[:, 1]) < -1e-8) for body in bodies):
        raise ValueError("Height-envelope rendering requires Z-monotone bodies")
    heights = np.linspace(0, max(body[-1, 1] for body in bodies), 4001)
    left = np.full_like(heights, np.inf)
    right = np.full_like(heights, -np.inf)
    for body in bodies:
        valid = (heights >= body[0, 1] - 1e-8) & (heights <= body[-1, 1])
        lateral = np.interp(heights[valid], body[:, 1], body[:, 0])
        left[valid] = np.minimum(left[valid], lateral)
        right[valid] = np.maximum(right[valid], lateral)
    valid = np.isfinite(left) & np.isfinite(right)
    fig, ax = plt.subplots(figsize=(4.5, 7), layout="constrained")
    ax.fill_betweenx(heights, left, right, where=valid, color="red", linewidth=0)
    # Retain zero-width portions (e.g. the initial straight interval).
    ax.plot(left[valid], heights[valid], color="red", linewidth=0.6)
    ax.plot(right[valid], heights[valid], color="red", linewidth=0.6)
    ax.set(
        xlabel="World Y (mm)",
        ylabel="World Z (mm)",
        title=f"{result.get('profile', 'S3')}: swept centreline envelope",
        xlim=(-50, 200),
        ylim=(0, 400),
        xticks=np.arange(-50, 201, 50),
        yticks=np.arange(0, 401, 50),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=200)
    plt.close(fig)
