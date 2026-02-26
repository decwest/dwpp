from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
import numpy as np
from scipy.spatial.distance import cdist

from config import DT, GOAL_REACH_TOLERANCE_DIST_OMNI, VX_MAX, VX_MIN, VY_MAX, VY_MIN, W_MAX, W_MIN
from path import right_angle_polyline_curve, straight_line_heading_step_curve
from pure_pursuit import pure_pursuit
from robot import forward_simulation_differential, forward_simulation_omnidirectional


DEFAULT_MAX_SIM_STEPS = 20000
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "compare_3_methods"

plt.rcParams["font.size"] = 12  # 全体のフォントサイズが変更されます。
plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.weight"] = "normal"  # フォントの太さを細字に設定
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
# ハッチ線の太さを細く
mpl.rcParams['hatch.linewidth'] = 0.5

@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    is_omni: bool
    color: str


@dataclass
class SimulationResult:
    poses: np.ndarray
    velocities_raw: np.ndarray
    ref_velocities_raw: np.ndarray
    break_flags: np.ndarray
    times: np.ndarray


METHOD_SPECS = [
    MethodSpec("dwpp", "DWPP for Diff-drive", False, "tab:blue"),
    MethodSpec("dwpp_omni_clip", "VP for Omni", True, "tab:orange"),
    MethodSpec("dwpp_omni", "DWVP for Omni", True, "tab:green"),
]


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_path_headings(path: np.ndarray) -> np.ndarray:
    if path.shape[1] >= 3:
        return path[:, 2]
    if len(path) == 1:
        return np.array([0.0])
    diffs = np.diff(path[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.concatenate([headings, [headings[-1]]])


def convert_velocity_to_vx_vy_w(spec: MethodSpec, velocities_raw: np.ndarray) -> np.ndarray:
    if spec.is_omni:
        return velocities_raw

    v = velocities_raw[:, 0]
    w = velocities_raw[:, 1]
    out = np.zeros((len(velocities_raw), 3), dtype=float)
    out[:, 0] = v
    out[:, 2] = w
    return out


def simulate_single_method(
    path: np.ndarray,
    method_spec: MethodSpec,
    goal_tolerance: float,
    max_sim_steps: int
) -> SimulationResult:
    goal_xy = path[-1, :2]
    current_pose = np.array([0.0, 0.0, 0.0], dtype=float)
    current_velocity = np.zeros(3 if method_spec.is_omni else 2, dtype=float)

    poses = [current_pose.copy()]
    velocities_raw = [current_velocity.copy()]
    ref_velocities_raw = [current_velocity.copy()]
    break_flags = [[False, False, False] if method_spec.is_omni else [False, False]]
    times = [0.0]

    for _ in range(max_sim_steps):
        if float(np.linalg.norm(current_pose[:2] - goal_xy)) <= goal_tolerance:
            break

        next_velocity_ref, _, break_flag, _, _ = pure_pursuit(
            current_pose,
            current_velocity,
            path,
            method_spec.key
        )

        if method_spec.is_omni:
            next_pose, next_velocity = forward_simulation_omnidirectional(
                current_pose,
                current_velocity,
                next_velocity_ref
            )
        else:
            next_pose, next_velocity = forward_simulation_differential(
                current_pose,
                current_velocity,
                next_velocity_ref
            )

        poses.append(next_pose)
        velocities_raw.append(next_velocity)
        ref_velocities_raw.append(next_velocity_ref)
        break_flags.append(break_flag)
        times.append(times[-1] + DT)

        current_pose = next_pose
        current_velocity = next_velocity
    else:
        print(f"[WARN] Reached max_sim_steps for method={method_spec.key}")

    return SimulationResult(
        poses=np.array(poses, dtype=float),
        velocities_raw=np.array(velocities_raw, dtype=float),
        ref_velocities_raw=np.array(ref_velocities_raw, dtype=float),
        break_flags=np.array(break_flags, dtype=bool),
        times=np.array(times, dtype=float),
    )


def calc_metrics(path: np.ndarray, result: SimulationResult, goal_tolerance: float) -> dict[str, float]:
    path_xy = path[:, :2]
    path_headings = calc_path_headings(path)
    robot_xy = result.poses[:, :2]
    robot_headings = result.poses[:, 2]

    distance_matrix = cdist(robot_xy, path_xy, metric="euclidean")
    nearest_indices = np.argmin(distance_matrix, axis=1)
    pos_errors = distance_matrix[np.arange(len(robot_xy)), nearest_indices]

    ref_headings = path_headings[nearest_indices]
    heading_errors = np.abs(normalize_angle(robot_headings - ref_headings))

    flags = result.break_flags[1:] if len(result.break_flags) > 1 else result.break_flags
    violation_rate = float(np.mean(np.any(flags, axis=1)) * 100.0)

    goal_distances = np.linalg.norm(robot_xy - path[-1, :2], axis=1)
    reached_indices = np.where(goal_distances <= goal_tolerance)[0]
    travel_time = float(result.times[reached_indices[0]]) if len(reached_indices) > 0 else float("nan")

    return {
        "constraint_violation_rate_pct": violation_rate,
        "mean_position_error_m": float(np.mean(pos_errors)),
        "max_position_error_m": float(np.max(pos_errors)),
        "mean_heading_error_deg": float(np.rad2deg(np.mean(heading_errors))),
        "travel_time_s": travel_time,
    }


def format_value(value: float, digits: int = 4) -> str:
    if np.isfinite(value):
        return f"{value:.{digits}f}"
    return "N/A"


def write_metrics_tables(path_dir: Path, rows: list[dict[str, str | float]]) -> None:
    csv_path = path_dir / "metrics_table.csv"
    md_path = path_dir / "metrics_table.md"
    png_path = path_dir / "metrics_table.png"

    headers = [
        "Method",
        "Constraint Violation Rate [%]",
        "Mean Position Error [m]",
        "Max Position Error [m]",
        "Mean Heading Error [deg]",
        "Travel Time [s]",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["Method"],
                format_value(float(row["constraint_violation_rate_pct"])),
                format_value(float(row["mean_position_error_m"])),
                format_value(float(row["max_position_error_m"])),
                format_value(float(row["mean_heading_error_deg"])),
                format_value(float(row["travel_time_s"])),
            ])

    with open(md_path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                f"| {row['Method']} | "
                f"{format_value(float(row['constraint_violation_rate_pct']))} | "
                f"{format_value(float(row['mean_position_error_m']))} | "
                f"{format_value(float(row['max_position_error_m']))} | "
                f"{format_value(float(row['mean_heading_error_deg']))} | "
                f"{format_value(float(row['travel_time_s']))} |\n"
            )

    fig, ax = plt.subplots(figsize=(12, 1.8 + 0.5 * len(rows)))
    ax.axis("off")
    cell_text = []
    for row in rows:
        cell_text.append([
            row["Method"],
            format_value(float(row["constraint_violation_rate_pct"])),
            format_value(float(row["mean_position_error_m"])),
            format_value(float(row["max_position_error_m"])),
            format_value(float(row["mean_heading_error_deg"])),
            format_value(float(row["travel_time_s"])),
        ])
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def draw_path_heading_arrows(ax: plt.Axes, path: np.ndarray, color: str = "black") -> None:
    headings = calc_path_headings(path)
    n = len(path)
    if n == 0:
        return
    step = max(1, n // 30)
    indices = np.arange(0, n, step, dtype=int)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = 1.0 * max(0.1, 0.03 * diag)

    x = path[indices, 0]
    y = path[indices, 1]
    dx = arrow_length * np.cos(headings[indices])
    dy = arrow_length * np.sin(headings[indices])

    ax.quiver(
        x,
        y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        alpha=1.00,
        width=0.003,
        label="Path Heading"
    )


def set_equal_axis_with_min_span(
    ax: plt.Axes,
    xy_arrays: list[np.ndarray],
    min_span: float = 1.0,
    margin_ratio: float = 0.1
) -> None:
    all_xy = np.vstack(xy_arrays)
    x_min = float(np.min(all_xy[:, 0]))
    x_max = float(np.max(all_xy[:, 0]))
    y_min = float(np.min(all_xy[:, 1]))
    y_max = float(np.max(all_xy[:, 1]))

    x_span = max(x_max - x_min, min_span)
    y_span = max(y_max - y_min, min_span)

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    x_half = 0.5 * x_span * (1.0 + margin_ratio)
    y_half = 0.5 * y_span * (1.0 + margin_ratio)

    ax.set_xlim(x_center - x_half, x_center + x_half)
    ax.set_ylim(y_center - y_half, y_center + y_half)
    ax.set_aspect("equal")


def calc_tracking_layout(
    xy_arrays: list[np.ndarray],
    min_span_default: float = 1.0,
) -> tuple[tuple[float, float], float, float]:
    all_xy = np.vstack(xy_arrays)
    x_span_raw = float(np.max(all_xy[:, 0]) - np.min(all_xy[:, 0]))
    y_span_raw = float(np.max(all_xy[:, 1]) - np.min(all_xy[:, 1]))

    # 横長経路では最小スパンを下げて上下余白を詰める
    min_span_tight = max(0.20, 0.12 * max(x_span_raw, 1e-6))
    min_span = min(min_span_default, min_span_tight) if x_span_raw > y_span_raw else min_span_default

    x_span = max(x_span_raw, min_span)
    y_span = max(y_span_raw, min_span)
    data_aspect = x_span / max(y_span, 1e-6)

    if data_aspect >= 1.6:
        fig_height = 4.2
        fig_width = min(15.0, max(10.0, fig_height * data_aspect))
        margin_ratio = 0.10
    else:
        fig_width = 8.5
        fig_height = 8.5
        margin_ratio = 0.15

    return (fig_width, fig_height), min_span, margin_ratio


def save_tracking_plots_by_method(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.05, 0.04 * diag)
    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    fig_size, min_span, margin_ratio = calc_tracking_layout(xy_arrays, min_span_default=1.0)

    for spec in method_specs:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5, label="Reference Path")
        draw_path_heading_arrows(ax, path)

        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.8, label=spec.label)

        step = max(1, len(poses) // 25)
        indices = np.arange(0, len(poses), step, dtype=int)
        if indices[-1] != len(poses) - 1:
            indices = np.append(indices, len(poses) - 1)

        dx = pose_arrow_length * np.cos(poses[indices, 2])
        dy = pose_arrow_length * np.sin(poses[indices, 2])
        ax.quiver(
            poses[indices, 0],
            poses[indices, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=spec.color,
            alpha=0.65,
            width=0.0035,
        )

        set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
        ax.set_title(f"Tracking Pose Sequence ({spec.label})")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(2, min(4, len(labels))),
            frameon=True,
        )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
        fig.savefig(output_dir / f"tracking_poses_{spec.key}.png", dpi=200)
        plt.close(fig)


def save_tracking_plot_overlaid(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_path: Path
) -> None:
    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    fig_size, min_span, margin_ratio = calc_tracking_layout(xy_arrays, min_span_default=1.0)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.05, 0.04 * diag)
    for spec in method_specs:
        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.8, label=spec.label)

        step = max(1, len(poses) // 25)
        indices = np.arange(0, len(poses), step, dtype=int)
        if indices[-1] != len(poses) - 1:
            indices = np.append(indices, len(poses) - 1)

        dx = pose_arrow_length * np.cos(poses[indices, 2])
        dy = pose_arrow_length * np.sin(poses[indices, 2])
        ax.quiver(
            poses[indices, 0],
            poses[indices, 1],
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=spec.color,
            alpha=0.65,
            width=0.0035,
        )

    set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
    ax.set_title("Tracking Pose Sequence")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(2, min(5, len(labels))),
        frameon=True,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_velocity_profiles_by_method(
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    converted_cache = {}
    time_max = 0.0
    for spec in method_specs:
        result = results[spec.key]
        v_real = convert_velocity_to_vx_vy_w(spec, result.velocities_raw)
        v_ref = convert_velocity_to_vx_vy_w(spec, result.ref_velocities_raw)
        converted_cache[spec.key] = (result, v_real, v_ref)
        if len(result.times) > 0:
            time_max = max(time_max, float(result.times[-1]))

    axis_limits = [
        (VX_MIN, VX_MAX),
        (VY_MIN, VY_MAX),
        (W_MIN, W_MAX),
    ]
    y_limits = []
    for idx, (limit_min, limit_max) in enumerate(axis_limits):
        values_min = [limit_min]
        values_max = [limit_max]
        for spec in method_specs:
            _, v_real, v_ref = converted_cache[spec.key]
            values_min.extend([float(np.min(v_real[:, idx])), float(np.min(v_ref[:, idx]))])
            values_max.extend([float(np.max(v_real[:, idx])), float(np.max(v_ref[:, idx]))])
        y_min = min(values_min)
        y_max = max(values_max)
        span = max(y_max - y_min, 1e-6)
        margin = 0.08 * span
        y_limits.append((y_min - margin, y_max + margin))

    for spec in method_specs:
        result, v_real, v_ref = converted_cache[spec.key]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = [("vx [m/s]", 0), ("vy [m/s]", 1), ("omega [rad/s]", 2)]
        for ax, (ylabel, idx), (_, limit_max), y_lim in zip(axes, labels, axis_limits, y_limits):
            ax.plot(result.times, v_ref[:, idx], color="red", linewidth=1.4, label="Command")
            ax.plot(result.times, v_real[:, idx], color="blue", linewidth=1.4, label="Actual")
            ax.axhline(limit_max, color="black", linestyle="--", linewidth=1.1, label="Max Speed")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.0, time_max if time_max > 0.0 else 1.0)
            ax.set_ylim(*y_lim)
            ax.grid(True)

        axes[0].set_title(f"Velocity Profiles ({spec.label})")
        axes[-1].set_xlabel("Time [s]")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=True,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        fig.savefig(output_dir / f"velocity_profiles_{spec.key}.png", dpi=200)
        plt.close(fig)


def save_tracking_animation(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    set_equal_axis_with_min_span(ax, xy_arrays, min_span=1.0, margin_ratio=0.2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Path Tracking Animation")
    ax.grid(True)

    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.05, 0.05 * diag)

    trail_lines = {}
    pose_points = {}
    pose_arrows = {}
    for spec in method_specs:
        trail_line, = ax.plot([], [], color=spec.color, linewidth=2.0, label=spec.label)
        pose_point, = ax.plot([], [], marker="o", color=spec.color, markersize=5)
        pose_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=14, color=spec.color)
        pose_arrow.set_visible(False)
        ax.add_patch(pose_arrow)
        trail_lines[spec.key] = trail_line
        pose_points[spec.key] = pose_point
        pose_arrows[spec.key] = pose_arrow

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(2, min(5, len(labels))),
        frameon=True,
    )
    fig.subplots_adjust(top=0.84)

    max_frames = max(len(results[spec.key].poses) for spec in method_specs)

    def init():
        artists = []
        for spec in method_specs:
            trail_lines[spec.key].set_data([], [])
            pose_points[spec.key].set_data([], [])
            pose_arrows[spec.key].set_visible(False)
            artists.extend([trail_lines[spec.key], pose_points[spec.key], pose_arrows[spec.key]])
        return artists

    def update(frame_idx: int):
        artists = []
        for spec in method_specs:
            poses = results[spec.key].poses
            idx = min(frame_idx, len(poses) - 1)
            trail_lines[spec.key].set_data(poses[:idx + 1, 0], poses[:idx + 1, 1])
            pose_points[spec.key].set_data([poses[idx, 0]], [poses[idx, 1]])

            theta = poses[idx, 2]
            dx = arrow_length * np.cos(theta)
            dy = arrow_length * np.sin(theta)
            pose_arrows[spec.key].set_positions(
                (poses[idx, 0], poses[idx, 1]),
                (poses[idx, 0] + dx, poses[idx, 1] + dy)
            )
            pose_arrows[spec.key].set_visible(True)
            artists.extend([trail_lines[spec.key], pose_points[spec.key], pose_arrows[spec.key]])
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=max(1, int(round(1000.0 * DT))),
        blit=False,
        repeat=False,
    )

    fps = max(1, int(round(1.0 / DT)))
    mp4_path = output_dir / "tracking_comparison.mp4"
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps)
    except Exception:
        gif_path = output_dir / "tracking_comparison.gif"
        ani.save(gif_path, writer="pillow", fps=fps)

    plt.close(fig)


def run_benchmark_for_path(
    path: np.ndarray,
    path_name: str,
    output_root: Path,
    goal_tolerance: float,
    max_sim_steps: int
) -> list[dict[str, str | float]]:
    path_dir = output_root / path_name
    path_dir.mkdir(parents=True, exist_ok=True)

    np.save(path_dir / "path.npy", path)

    results: dict[str, SimulationResult] = {}
    rows: list[dict[str, str | float]] = []

    for spec in METHOD_SPECS:
        result = simulate_single_method(
            path=path,
            method_spec=spec,
            goal_tolerance=goal_tolerance,
            max_sim_steps=max_sim_steps,
        )
        results[spec.key] = result

        np.save(path_dir / f"{spec.key}_poses.npy", result.poses)
        np.save(path_dir / f"{spec.key}_velocities.npy", result.velocities_raw)
        np.save(path_dir / f"{spec.key}_ref_velocities.npy", result.ref_velocities_raw)
        np.save(path_dir / f"{spec.key}_break_flags.npy", result.break_flags)
        np.save(path_dir / f"{spec.key}_times.npy", result.times)

        metrics = calc_metrics(path, result, goal_tolerance)
        row = {"Method": spec.label}
        row.update(metrics)
        rows.append(row)

    write_metrics_tables(path_dir, rows)
    save_tracking_plots_by_method(
        path=path,
        method_specs=METHOD_SPECS,
        results=results,
        output_dir=path_dir,
    )
    save_tracking_plot_overlaid(
        path=path,
        method_specs=METHOD_SPECS,
        results=results,
        output_path=path_dir / "tracking_poses.png",
    )
    save_velocity_profiles_by_method(
        method_specs=METHOD_SPECS,
        results=results,
        output_dir=path_dir,
    )
    save_tracking_animation(
        path=path,
        method_specs=METHOD_SPECS,
        results=results,
        output_dir=path_dir,
    )

    return rows


def write_overall_summary(output_root: Path, overall_rows: list[dict[str, str | float]]) -> None:
    summary_csv = output_root / "summary_all_paths.csv"
    headers = [
        "Path",
        "Method",
        "Constraint Violation Rate [%]",
        "Mean Position Error [m]",
        "Max Position Error [m]",
        "Mean Heading Error [deg]",
        "Travel Time [s]",
    ]

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in overall_rows:
            writer.writerow([
                row["Path"],
                row["Method"],
                format_value(float(row["constraint_violation_rate_pct"])),
                format_value(float(row["mean_position_error_m"])),
                format_value(float(row["max_position_error_m"])),
                format_value(float(row["mean_heading_error_deg"])),
                format_value(float(row["travel_time_s"])),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark: DWPP vs VP/DWVP on omni paths")
    parser.add_argument("--right-angle-segment-length", type=float, default=1.0)
    parser.add_argument("--heading-segment-length", type=float, default=0.5)
    parser.add_argument("--points-per-segment", type=int, default=100)
    parser.add_argument("--goal-tolerance", type=float, default=GOAL_REACH_TOLERANCE_DIST_OMNI)
    parser.add_argument("--max-sim-steps", type=int, default=DEFAULT_MAX_SIM_STEPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "path1_right_angle_90": right_angle_polyline_curve(
            segment_length=args.right_angle_segment_length,
            points_per_segment=args.points_per_segment,
        ),
        "path2_straight_heading_step": straight_line_heading_step_curve(
            segment_length=args.heading_segment_length,
            points_per_segment=args.points_per_segment,
        ),
    }

    overall_rows: list[dict[str, str | float]] = []
    for path_name, path in paths.items():
        rows = run_benchmark_for_path(
            path=path,
            path_name=path_name,
            output_root=output_root,
            goal_tolerance=args.goal_tolerance,
            max_sim_steps=args.max_sim_steps,
        )
        for row in rows:
            row_with_path = {"Path": path_name}
            row_with_path.update(row)
            overall_rows.append(row_with_path)

    write_overall_summary(output_root, overall_rows)
    print(f"[INFO] Benchmark finished. Output: {output_root}")


if __name__ == "__main__":
    main()
