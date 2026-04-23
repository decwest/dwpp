#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from benchmark_compare_three_methods import (
    DEFAULT_MAX_SIM_STEPS,
    GOAL_REACH_TOLERANCE_DIST_OMNI,
    GOAL_REACH_TOLERANCE_HEADING,
    run_benchmark_for_path,
    write_overall_summary,
)
from path import append_heading_to_path


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "results" / "legacy_step_curves"
DEFAULT_LEGACY_MAX_SIM_STEPS = min(5000, DEFAULT_MAX_SIM_STEPS)


def build_legacy_step_curve(theta_rad: float, segment_length: float, points_per_segment: int) -> np.ndarray:
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    x1 = np.linspace(0.0, 1.0, points_per_segment)
    y1 = np.zeros_like(x1)

    x2 = np.linspace(1.0, 1.0 + segment_length * math.cos(theta_rad), points_per_segment)
    y2 = np.linspace(0.0, segment_length * math.sin(theta_rad), points_per_segment)

    x3 = np.linspace(1.0 + segment_length * math.cos(theta_rad), 4.0 + segment_length * math.cos(theta_rad), points_per_segment)
    y3 = np.full_like(x3, segment_length * math.sin(theta_rad))

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    return append_heading_to_path(np.c_[x, y])


def build_legacy_step_curve_paths(
    theta_deg_list: list[float],
    segment_length_list: list[float],
    points_per_segment: int,
) -> dict[str, np.ndarray]:
    paths: dict[str, np.ndarray] = {}
    for theta_deg in theta_deg_list:
        theta_rad = math.radians(theta_deg)
        theta_label = int(round(theta_deg))
        for segment_length in segment_length_list:
            length_label = int(segment_length) if float(segment_length).is_integer() else segment_length
            path_name = f"step_curve_theta{theta_label}_l{length_label}"
            paths[path_name] = build_legacy_step_curve(
                theta_rad=theta_rad,
                segment_length=float(segment_length),
                points_per_segment=points_per_segment,
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark updated DWPP on legacy 45/90/135-degree step curves")
    parser.add_argument("--angles-deg", nargs="*", type=float, default=[45.0, 90.0, 135.0])
    parser.add_argument("--segment-lengths", nargs="*", type=float, default=[2.0, 3.0, 4.0])
    parser.add_argument("--points-per-segment", type=int, default=100)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_REACH_TOLERANCE_DIST_OMNI)
    parser.add_argument(
        "--goal-heading-tolerance-deg",
        type=float,
        default=float(np.rad2deg(GOAL_REACH_TOLERANCE_HEADING)),
    )
    parser.add_argument("--max-sim-steps", type=int, default=DEFAULT_LEGACY_MAX_SIM_STEPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--path-names",
        nargs="*",
        default=None,
        help="Optional subset of legacy path names, e.g. step_curve_theta90_l3",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save comparison animation (MP4/GIF). Default: off",
    )
    parser.add_argument(
        "--save-dwpp-debug-animation",
        action="store_true",
        help="Save DWPP debug animation with look-ahead point and connecting arc.",
    )
    parser.add_argument(
        "--animation-max-seconds",
        type=float,
        default=None,
        help="Optional upper bound on animation duration in seconds.",
    )
    parser.add_argument(
        "--animation-frame-stride",
        type=int,
        default=1,
        help="Subsample animation frames by this stride.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    goal_tolerance_heading = float(np.deg2rad(args.goal_heading_tolerance_deg))

    paths = build_legacy_step_curve_paths(
        theta_deg_list=[float(theta_deg) for theta_deg in args.angles_deg],
        segment_length_list=[float(segment_length) for segment_length in args.segment_lengths],
        points_per_segment=int(args.points_per_segment),
    )

    if args.path_names:
        missing_path_names = [path_name for path_name in args.path_names if path_name not in paths]
        if missing_path_names:
            raise ValueError(f"Unknown path names: {missing_path_names}. Available: {list(paths.keys())}")
        paths = {path_name: paths[path_name] for path_name in args.path_names}

    overall_rows: list[dict[str, str | float]] = []
    for path_name, path in paths.items():
        rows = run_benchmark_for_path(
            path=path,
            path_name=path_name,
            output_root=output_root,
            goal_tolerance_dist=args.goal_tolerance,
            goal_tolerance_heading=goal_tolerance_heading,
            max_sim_steps=args.max_sim_steps,
            save_animation=args.save_animation,
            save_dwpp_debug_animation=args.save_dwpp_debug_animation,
            animation_max_seconds=args.animation_max_seconds,
            animation_frame_stride=args.animation_frame_stride,
        )
        for row in rows:
            row_with_path = {"Path": path_name}
            row_with_path.update(row)
            overall_rows.append(row_with_path)

    write_overall_summary(output_root, overall_rows)
    print(f"[INFO] Legacy-step benchmark finished. Output: {output_root}")


if __name__ == "__main__":
    main()
