"""Command-line demo; load -> IK -> final-shape collision -> report/plot."""

import argparse
import json
import sys
from pathlib import Path

from .config import geometry_context, load_config
from .evaluation import evaluate_target, result_report


def main() -> None:
    """Run Stage 1 from project root; --headless does not import Matplotlib."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example.json"))
    parser.add_argument("--headless", action="store_true", help="Report without a GUI")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    parser.add_argument("--save-figure", type=Path, help="Optional PNG/PDF final-shape figure")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--search", action="store_true", help="Search for a collision-free final shape"
    )
    mode.add_argument("--single-run", action="store_true", help="Use the original one-IK workflow")
    args = parser.parse_args()
    config = load_config(args.config)
    use_search = args.search or (config.ik_search is not None and not args.single_run)
    if use_search:
        from .ik_search import find_collision_free_ik, search_result_report

        search = find_collision_free_ik(config)
        result = search.diagnostic
        report_data = search_result_report(search)
    else:
        result = evaluate_target(config)
        report_data = result_report(result)
    report_data["geometry_context"] = geometry_context(config)
    report = json.dumps(report_data, indent=2, allow_nan=False)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.save_figure or not args.headless:
        if result is None:
            print(
                "No target-reaching candidate found; no final configuration to plot.",
                file=sys.stderr,
            )
            return
        import matplotlib

        if args.headless:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .visualization import create_figure

        figure = create_figure(config, result)
        if use_search:
            title = (
                "Collision-free IK found"
                if search.success
                else "SEARCH FAILED — rejected diagnostic shape"
            )
            figure.suptitle(
                f"{title} | attempts={search.attempts} | time={search.computation_time:.3f} s"
            )
        if args.save_figure:
            args.save_figure.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save_figure)
        if not args.headless:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
