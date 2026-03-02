#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from config import GOAL_REACH_TOLERANCE_DIST_OMNI, GOAL_REACH_TOLERANCE_HEADING
from path import (
    one_minus_cos_curve,
    right_angle_polyline_curve,
    right_angle_polyline_curve_last_segment_heading_minus_pi,
    straight_line_heading_step_curve,
)
from benchmark_compare_three_methods import (
    METHOD_SPECS,
    SimulationResult,
    calc_metrics,
    save_tracking_animation,
    save_tracking_plot_overlaid,
    save_tracking_plots_by_method,
    save_velocity_profiles_by_method,
    write_metrics_tables,
    write_overall_summary,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "results" / "compare_3_methods_from_nav2_csv"
DEFAULT_AUTO_PATH_ORDER = [
    "path3_right_angle_90_last_heading_minus_pi",
    "path4_one_minus_cos",
]
FILE_PREFIX_TO_METHOD_KEY = {
    "dwpp_nav2": "dwpp",
    "vpmin_nav2": "dwpp_omni_clip_min_l",
    "vpmax_nav2": "dwpp_omni_clip_max_l",
    "dwvp_nav2": "dwpp_omni",
}
AUTO_CSV_FILENAME_RE = re.compile(
    r"^(?P<prefix>[a-zA-Z0-9_]+)_(?P<date>\d{8}_\d{6})_(?P<nsec>\d{9})\.csv$"
)


def parse_run_spec(spec: str) -> tuple[str, str, Path]:
    # Format: <path_name>:<method_key>=<csv_path>
    if "=" not in spec:
        raise ValueError(f"Invalid --run format (missing '='): {spec}")
    left, raw_path = spec.split("=", 1)
    if ":" not in left:
        raise ValueError(f"Invalid --run format (missing ':'): {spec}")
    path_name, method_key = left.split(":", 1)
    if not path_name or not method_key or not raw_path:
        raise ValueError(f"Invalid --run format: {spec}")
    return path_name, method_key, Path(raw_path)


def discover_run_specs_from_directory(
    input_dir: Path,
    path_order: list[str],
    use_latest_runs: int | None,
) -> list[str]:
    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")

    required_method_keys = [m.key for m in METHOD_SPECS]
    method_entries: dict[str, list[tuple[str, int, Path]]] = {k: [] for k in required_method_keys}

    for csv_path in input_dir.glob("*.csv"):
        m = AUTO_CSV_FILENAME_RE.match(csv_path.name)
        if m is None:
            continue
        prefix = m.group("prefix")
        method_key = FILE_PREFIX_TO_METHOD_KEY.get(prefix)
        if method_key not in method_entries:
            continue
        date_token = m.group("date")
        nsec = int(m.group("nsec"))
        method_entries[method_key].append((date_token, nsec, csv_path))

    for method_key in required_method_keys:
        method_entries[method_key].sort(key=lambda x: (x[0], x[1]))

    empty_methods = [k for k, v in method_entries.items() if len(v) == 0]
    if empty_methods:
        raise ValueError(
            "No matching CSV files found for methods: "
            f"{empty_methods}. "
            "Expected file prefixes: "
            f"{sorted(FILE_PREFIX_TO_METHOD_KEY.keys())}"
        )

    available_complete_runs = min(len(v) for v in method_entries.values())
    target_runs = use_latest_runs if use_latest_runs is not None else len(path_order)

    if target_runs <= 0:
        raise ValueError("--use-latest-runs must be > 0")
    if len(path_order) != target_runs:
        raise ValueError(
            "Number of --path-order entries must match number of runs to import. "
            f"path_order={len(path_order)}, target_runs={target_runs}"
        )
    if target_runs > available_complete_runs:
        counts = {k: len(v) for k, v in method_entries.items()}
        raise ValueError(
            "Requested runs exceed available complete sets. "
            f"requested={target_runs}, available={available_complete_runs}, counts={counts}"
        )

    selected_per_method: dict[str, list[Path]] = {}
    for method_key in required_method_keys:
        selected = method_entries[method_key][-target_runs:]
        selected_per_method[method_key] = [entry[2] for entry in selected]

    run_specs: list[str] = []
    print("[INFO] Auto-discovered CSV mapping:")
    for run_idx, path_name in enumerate(path_order):
        for method_key in required_method_keys:
            csv_path = selected_per_method[method_key][run_idx]
            print(f"  - {path_name}:{method_key}={csv_path}")
            run_specs.append(f"{path_name}:{method_key}={csv_path}")

    return run_specs


def _find_first_float(row: dict[str, str], names: list[str], default: float = np.nan) -> float:
    for n in names:
        if n not in row:
            continue
        v = row[n]
        if v is None or v == "":
            continue
        return float(v)
    return default


def load_nav2_csv_result(csv_path: Path) -> SimulationResult:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    poses = []
    velocities_raw = []
    ref_velocities_raw = []
    break_flags = []
    times = []

    map_pose_valid_count = 0
    map_pose_total_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_x = _find_first_float(row, ["x", "pose_x"])
            local_y = _find_first_float(row, ["y", "pose_y"])
            local_yaw = _find_first_float(row, ["yaw", "pose_yaw"])
            map_x = _find_first_float(row, ["map_x"], default=np.nan)
            map_y = _find_first_float(row, ["map_y"], default=np.nan)
            map_yaw = _find_first_float(row, ["map_yaw"], default=np.nan)
            map_pose_valid = _find_first_float(row, ["map_pose_valid"], default=np.nan)

            has_map_pose = np.isfinite(map_x) and np.isfinite(map_y) and np.isfinite(map_yaw)
            if np.isfinite(map_pose_valid):
                map_pose_total_rows += 1
                has_map_pose = has_map_pose and bool(int(round(map_pose_valid)))
                if has_map_pose:
                    map_pose_valid_count += 1

            # Prefer map-frame pose when available. Fallback to legacy pose columns for old CSVs
            # or transient TF lookup failures.
            x = map_x if has_map_pose else local_x
            y = map_y if has_map_pose else local_y
            yaw = map_yaw if has_map_pose else local_yaw

            vx_real = _find_first_float(row, ["vx_real", "speed_vx"], default=np.nan)
            vy_real = _find_first_float(row, ["vy_real", "speed_vy"], default=np.nan)
            w_real = _find_first_float(row, ["w_real", "speed_w"], default=np.nan)
            if not np.isfinite(vx_real):
                vx_real = _find_first_float(row, ["v_real"], default=0.0)
            if not np.isfinite(vy_real):
                vy_real = 0.0
            if not np.isfinite(w_real):
                w_real = _find_first_float(row, ["w_real", "w_cmd", "feedback_w"], default=0.0)

            vx_cmd = _find_first_float(row, ["vx_cmd", "cmd_vx", "desired_vx"], default=np.nan)
            vy_cmd = _find_first_float(row, ["vy_cmd", "cmd_vy", "desired_vy"], default=np.nan)
            w_cmd = _find_first_float(row, ["w_cmd", "cmd_w", "desired_w"], default=np.nan)
            if not np.isfinite(vx_cmd):
                vx_cmd = _find_first_float(row, ["v_cmd"], default=0.0)
            if not np.isfinite(vy_cmd):
                vy_cmd = 0.0
            if not np.isfinite(w_cmd):
                w_cmd = _find_first_float(row, ["w_cmd"], default=0.0)

            violation_val = _find_first_float(
                row, ["velocity_violation", "constraints_violation"], default=0.0)
            violation = bool(int(round(violation_val)))

            if "t" in row and row["t"] not in (None, ""):
                t = float(row["t"])
            else:
                sec = _find_first_float(row, ["sec"], default=0.0)
                nsec = _find_first_float(row, ["nsec"], default=0.0)
                t = sec + 1e-9 * nsec

            poses.append([x, y, yaw])
            velocities_raw.append([vx_real, vy_real, w_real])
            ref_velocities_raw.append([vx_cmd, vy_cmd, w_cmd])
            break_flags.append([violation, violation, violation])
            times.append(t)

    if len(poses) == 0:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    if map_pose_total_rows > 0:
        if map_pose_valid_count == 0:
            print(
                f"[WARN] map_pose_valid exists but no valid map pose rows in {csv_path}. "
                "Falling back to legacy x/y/yaw."
            )
        elif map_pose_valid_count < map_pose_total_rows:
            print(
                f"[INFO] Using map pose columns for {csv_path} "
                f"(valid rows: {map_pose_valid_count}/{map_pose_total_rows}; "
                "invalid rows fallback to legacy x/y/yaw)."
            )

    poses_np = np.array(poses, dtype=float)
    velocities_raw_np = np.array(velocities_raw, dtype=float)
    ref_velocities_raw_np = np.array(ref_velocities_raw, dtype=float)
    break_flags_np = np.array(break_flags, dtype=bool)
    times_np = np.array(times, dtype=float)
    times_np = times_np - times_np[0]

    return SimulationResult(
        poses=poses_np,
        velocities_raw=velocities_raw_np,
        ref_velocities_raw=ref_velocities_raw_np,
        break_flags=break_flags_np,
        times=times_np,
    )


def get_reference_paths(args: argparse.Namespace) -> dict[str, np.ndarray]:
    one_minus_cos_num_points = args.one_minus_cos_num_points
    if one_minus_cos_num_points is None:
        one_minus_cos_num_points = 5 * args.points_per_segment + 1

    return {
        "path1_right_angle_90": right_angle_polyline_curve(
            segment_length=args.right_angle_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path2_straight_heading_step": straight_line_heading_step_curve(
            segment_length=args.heading_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path3_right_angle_90_last_heading_minus_pi": right_angle_polyline_curve_last_segment_heading_minus_pi(
            segment_length=args.right_angle_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path4_one_minus_cos": one_minus_cos_curve(
            amplitude=args.one_minus_cos_amplitude,
            length_x=args.one_minus_cos_length_x,
            num_points=one_minus_cos_num_points,
            cycles=args.one_minus_cos_cycles,
            resample_arclength=True,
        ),
    }


def get_figure_prefix(path_name: str) -> str:
    if path_name == "path3_right_angle_90_last_heading_minus_pi":
        return "exp1_"
    if path_name == "path4_one_minus_cos":
        return "exp2_"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the same benchmark figures as benchmark_compare_three_methods.py "
            "from Nav2 CSV logs. "
            "Use --run multiple times: <path_name>:<method_key>=<csv_path>"
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Format: <path_name>:<method_key>=<csv_path>",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Auto-discovery mode. Directory containing Nav2 CSV logs "
            "(e.g., dwpp_nav2_YYYYMMDD_HHMMSS_NNNNNNNNN.csv). "
            "If --run is omitted, run specs are generated from this directory."
        ),
    )
    parser.add_argument(
        "--path-order",
        nargs="+",
        default=DEFAULT_AUTO_PATH_ORDER,
        help=(
            "Path names assigned to the auto-discovered runs in chronological order. "
            "Default: path3_right_angle_90_last_heading_minus_pi path4_one_minus_cos"
        ),
    )
    parser.add_argument(
        "--use-latest-runs",
        type=int,
        default=None,
        help=(
            "Number of latest complete runs to import in auto-discovery mode. "
            "Default: len(--path-order)"
        ),
    )
    parser.add_argument("--right-angle-segment-length", type=float, default=1.0)
    parser.add_argument("--heading-segment-length", type=float, default=0.5)
    parser.add_argument("--points-per-segment", type=int, default=100)
    parser.add_argument("--one-minus-cos-amplitude", type=float, default=1.0)
    parser.add_argument("--one-minus-cos-length-x", type=float, default=2.0)
    parser.add_argument("--one-minus-cos-cycles", type=float, default=1.5)
    parser.add_argument("--one-minus-cos-num-points", type=int, default=None)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_REACH_TOLERANCE_DIST_OMNI)
    parser.add_argument(
        "--goal-heading-tolerance-deg",
        type=float,
        default=float(np.rad2deg(GOAL_REACH_TOLERANCE_HEADING)),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save tracking animation (MP4/GIF). Default: off",
    )
    args = parser.parse_args()

    if len(args.run) == 0:
        if args.input_dir is None:
            raise ValueError("At least one --run is required, or specify --input-dir for auto mode.")
        args.run = discover_run_specs_from_directory(
            input_dir=args.input_dir,
            path_order=list(args.path_order),
            use_latest_runs=args.use_latest_runs,
        )

    method_keys = {m.key for m in METHOD_SPECS}
    path_to_method_csv: dict[str, dict[str, Path]] = {}
    for spec in args.run:
        path_name, method_key, csv_path = parse_run_spec(spec)
        if method_key not in method_keys:
            raise ValueError(f"Unknown method key '{method_key}'. Valid: {sorted(method_keys)}")
        path_to_method_csv.setdefault(path_name, {})[method_key] = csv_path

    ref_paths = get_reference_paths(args)
    goal_tolerance_heading = float(np.deg2rad(args.goal_heading_tolerance_deg))

    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, str | float]] = []

    for path_name, method_to_csv in path_to_method_csv.items():
        if path_name not in ref_paths:
            raise ValueError(f"Unknown path name '{path_name}'. Valid: {sorted(ref_paths.keys())}")

        missing_keys = [m.key for m in METHOD_SPECS if m.key not in method_to_csv]
        if missing_keys:
            raise ValueError(
                f"Path '{path_name}' is missing CSVs for methods: {missing_keys}. "
                "Provide all methods to generate figures identical to benchmark_compare_three_methods.py."
            )

        path = ref_paths[path_name]
        path_dir = output_root / path_name
        path_dir.mkdir(parents=True, exist_ok=True)
        figure_prefix = get_figure_prefix(path_name)

        np.save(path_dir / "path.npy", path)

        results: dict[str, SimulationResult] = {}
        rows: list[dict[str, str | float]] = []

        for method_spec in METHOD_SPECS:
            csv_path = method_to_csv[method_spec.key]
            result = load_nav2_csv_result(csv_path)
            results[method_spec.key] = result

            np.save(path_dir / f"{method_spec.key}_poses.npy", result.poses)
            np.save(path_dir / f"{method_spec.key}_velocities.npy", result.velocities_raw)
            np.save(path_dir / f"{method_spec.key}_ref_velocities.npy", result.ref_velocities_raw)
            np.save(path_dir / f"{method_spec.key}_break_flags.npy", result.break_flags)
            np.save(path_dir / f"{method_spec.key}_times.npy", result.times)

            metrics = calc_metrics(
                path,
                result,
                goal_tolerance_dist=args.goal_tolerance,
                goal_tolerance_heading=goal_tolerance_heading,
            )
            row = {"Method": method_spec.label}
            row.update(metrics)
            rows.append(row)

        write_metrics_tables(path_dir, rows, figure_prefix=figure_prefix)
        save_tracking_plots_by_method(
            path=path,
            method_specs=METHOD_SPECS,
            results=results,
            output_dir=path_dir,
            file_prefix=figure_prefix,
        )
        save_tracking_plot_overlaid(
            path=path,
            method_specs=METHOD_SPECS,
            results=results,
            output_path=path_dir / f"{figure_prefix}tracking_poses.png",
        )
        save_velocity_profiles_by_method(
            method_specs=METHOD_SPECS,
            results=results,
            output_dir=path_dir,
            file_prefix=figure_prefix,
        )
        if args.save_animation:
            save_tracking_animation(
                path=path,
                method_specs=METHOD_SPECS,
                results=results,
                output_dir=path_dir,
            )

        for row in rows:
            row_with_path = {"Path": path_name}
            row_with_path.update(row)
            overall_rows.append(row_with_path)

    write_overall_summary(output_root, overall_rows)
    print(f"[INFO] Benchmark-from-CSV finished. Output: {output_root}")


if __name__ == "__main__":
    main()
