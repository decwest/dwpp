from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib as mpl
import numpy as np
from scipy.spatial.distance import cdist

from config import (
    A_MAX,
    AW_MAX,
    DT,
    GOAL_REACH_TOLERANCE_DIST_OMNI,
    GOAL_REACH_TOLERANCE_HEADING,
    V_MAX,
    V_MIN,
    VX_MAX,
    VX_MIN,
    VY_MAX,
    VY_MIN,
    W_MAX,
    W_MIN,
)
from path import (
    one_minus_cos_curve,
    right_angle_polyline_curve,
    right_angle_polyline_curve_last_segment_heading_minus_pi,
    straight_line_heading_step_curve,
)
from pure_pursuit import (
    DWPP_USE_REGULATED_VELOCITY,
    calc_dynamic_window_bounds,
    calc_regulated_translational_velocity,
    pure_pursuit,
)
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
    look_ahead_positions: np.ndarray | None = None
    curvatures: np.ndarray | None = None


METHOD_SPECS = [
    MethodSpec("dwpp_fixed", "DWPP", False, "tab:blue"),
    MethodSpec("dwpp", "DWPP with Auto Look-Ahead", False, "tab:red"),
]
DIFF_DRIVE_DWPP_METHOD_KEYS = {"dwpp_fixed", "dwpp"}
POST_GOAL_HEADING_KP_DIFF = 2.0


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def is_goal_reached(
    pose: np.ndarray,
    goal_pose: np.ndarray,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
) -> bool:
    pos_error = float(np.linalg.norm(pose[:2] - goal_pose[:2]))
    heading_error = float(abs(normalize_angle(pose[2] - goal_pose[2])))
    return (pos_error <= goal_tolerance_dist) and (heading_error <= goal_tolerance_heading)


def calc_goal_alignment_velocity_ref_diffdrive(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    goal_pose: np.ndarray,
) -> np.ndarray:
    heading_error = float(normalize_angle(goal_pose[2] - current_pose[2]))
    w_target = float(np.clip(POST_GOAL_HEADING_KP_DIFF * heading_error, W_MIN, W_MAX))

    # 加速度制約を満たす範囲で、v は 0、w は heading 誤差方向へ
    v_lower = max(float(current_velocity[0]) - A_MAX * DT, V_MIN)
    v_upper = min(float(current_velocity[0]) + A_MAX * DT, V_MAX)
    w_lower = max(float(current_velocity[1]) - AW_MAX * DT, W_MIN)
    w_upper = min(float(current_velocity[1]) + AW_MAX * DT, W_MAX)

    v_ref = float(np.clip(0.0, v_lower, v_upper))
    w_ref = float(np.clip(w_target, w_lower, w_upper))
    return np.array([v_ref, w_ref], dtype=float)


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
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
    max_sim_steps: int
) -> SimulationResult:
    goal_pose = path[-1, :3]
    current_pose = np.array([0.0, 0.0, 0.0], dtype=float)
    current_velocity = np.zeros(3 if method_spec.is_omni else 2, dtype=float)

    poses = [current_pose.copy()]
    velocities_raw = [current_velocity.copy()]
    ref_velocities_raw = [current_velocity.copy()]
    break_flags = [[False, False, False] if method_spec.is_omni else [False, False]]
    times = [0.0]
    look_ahead_positions = []
    curvatures = []

    for _ in range(max_sim_steps):
        if is_goal_reached(current_pose, goal_pose, goal_tolerance_dist, goal_tolerance_heading):
            break

        position_error_to_goal = float(np.linalg.norm(current_pose[:2] - goal_pose[:2]))
        if (method_spec.key in DIFF_DRIVE_DWPP_METHOD_KEYS) and (position_error_to_goal <= goal_tolerance_dist):
            next_velocity_ref = calc_goal_alignment_velocity_ref_diffdrive(
                current_pose=current_pose,
                current_velocity=current_velocity,
                goal_pose=goal_pose,
            )
            break_flag = [False, False]
            look_ahead_pos = goal_pose[:2].copy()
            curvature = 0.0
        else:
            next_velocity_ref, look_ahead_pos_raw, break_flag, curvature, _ = pure_pursuit(
                current_pose,
                current_velocity,
                path,
                method_spec.key
            )
            look_ahead_pos = np.array(look_ahead_pos_raw[:2], dtype=float)

        if not (
            np.all(np.isfinite(next_velocity_ref))
            and np.all(np.isfinite(look_ahead_pos))
            and np.isfinite(curvature)
        ):
            print(f"[WARN] Non-finite command detected for method={method_spec.key}; stopping simulation early")
            break

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

        if not (np.all(np.isfinite(next_pose)) and np.all(np.isfinite(next_velocity))):
            print(f"[WARN] Non-finite state detected for method={method_spec.key}; stopping simulation early")
            break

        poses.append(next_pose)
        velocities_raw.append(next_velocity)
        ref_velocities_raw.append(next_velocity_ref)
        break_flags.append(break_flag)
        times.append(times[-1] + DT)
        look_ahead_positions.append(np.array(look_ahead_pos, dtype=float))
        curvatures.append(float(curvature))

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
        look_ahead_positions=np.array(look_ahead_positions, dtype=float),
        curvatures=np.array(curvatures, dtype=float),
    )


def calc_metrics(
    path: np.ndarray,
    result: SimulationResult,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
) -> dict[str, float]:
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
    goal_heading_errors = np.abs(normalize_angle(robot_headings - path[-1, 2]))
    reached_indices = np.where(
        (goal_distances <= goal_tolerance_dist) & (goal_heading_errors <= goal_tolerance_heading)
    )[0]
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


def filter_finite_xy_arrays(xy_arrays: list[np.ndarray]) -> list[np.ndarray]:
    finite_arrays: list[np.ndarray] = []
    for xy in xy_arrays:
        arr = np.asarray(xy, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) == 0:
            continue
        finite_mask = np.all(np.isfinite(arr[:, :2]), axis=1)
        if np.any(finite_mask):
            finite_arrays.append(arr[finite_mask, :2])
    return finite_arrays


def finite_min(values: np.ndarray, default: float) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float(default)
    return float(np.min(finite_values))


def finite_max(values: np.ndarray, default: float) -> float:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return float(default)
    return float(np.max(finite_values))


def write_metrics_tables(path_dir: Path, rows: list[dict[str, str | float]], figure_prefix: str = "") -> None:
    csv_path = path_dir / "metrics_table.csv"
    md_path = path_dir / "metrics_table.md"
    png_path = path_dir / f"{figure_prefix}metrics_table.png"

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
    step = max(1, n // 24)
    indices = np.arange(0, n, step, dtype=int)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.24, 0.040 * diag)

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
        width=0.006,
        headwidth=5.0,
        headlength=6.0,
        headaxislength=5.0,
        label="Path Heading"
    )


def set_equal_axis_with_min_span(
    ax: plt.Axes,
    xy_arrays: list[np.ndarray],
    min_span: float = 1.0,
    margin_ratio: float = 0.1
) -> None:
    finite_xy_arrays = filter_finite_xy_arrays(xy_arrays)
    if len(finite_xy_arrays) == 0:
        half_span = 0.5 * max(float(min_span), 1e-6) * (1.0 + float(margin_ratio))
        ax.set_xlim(-half_span, half_span)
        ax.set_ylim(-half_span, half_span)
        ax.set_aspect("equal")
        return

    all_xy = np.vstack(finite_xy_arrays)
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
    finite_xy_arrays = filter_finite_xy_arrays(xy_arrays)
    if len(finite_xy_arrays) == 0:
        return (3.0, 3.0), float(min_span_default), 0.12

    all_xy = np.vstack(finite_xy_arrays)
    x_span_raw = float(np.max(all_xy[:, 0]) - np.min(all_xy[:, 0]))
    y_span_raw = float(np.max(all_xy[:, 1]) - np.min(all_xy[:, 1]))

    # 横長経路では最小スパンを下げて上下余白を詰める
    min_span_tight = max(0.20, 0.12 * max(x_span_raw, 1e-6))
    min_span = min(min_span_default, min_span_tight) if x_span_raw > y_span_raw else min_span_default

    x_span = max(x_span_raw, min_span)
    y_span = max(y_span_raw, min_span)
    data_aspect = x_span / max(y_span, 1e-6)

    if data_aspect >= 1.6:
        fig_height = 3.0
        fig_width = min(10.5, max(7.0, fig_height * data_aspect))
        margin_ratio = 0.08
    else:
        fig_width = 3.0
        fig_height = 3.0
        margin_ratio = 0.12

    return (fig_width, fig_height), min_span, margin_ratio


def build_animation_frame_indices(
    times: np.ndarray,
    max_duration_s: float | None,
    frame_stride: int,
) -> np.ndarray:
    if len(times) == 0:
        return np.array([0], dtype=int)

    stride = max(1, int(frame_stride))
    max_frame_idx = len(times) - 1
    if max_duration_s is not None:
        capped_duration = max(0.0, float(max_duration_s))
        max_frame_idx = int(np.searchsorted(times, capped_duration, side="right") - 1)
        max_frame_idx = max(0, min(max_frame_idx, len(times) - 1))

    frame_indices = np.arange(0, max_frame_idx + 1, stride, dtype=int)
    if frame_indices[-1] != max_frame_idx:
        frame_indices = np.append(frame_indices, max_frame_idx)
    return frame_indices


def calc_connecting_arc_points(
    current_pose: np.ndarray,
    look_ahead_pos: np.ndarray,
    curvature: float,
    num_points: int = 60,
) -> np.ndarray:
    start = np.array(current_pose[:2], dtype=float)
    target = np.array(look_ahead_pos[:2], dtype=float)
    if not np.all(np.isfinite(target)):
        return np.empty((0, 2), dtype=float)

    distance = float(np.linalg.norm(target - start))
    if distance <= 1e-9:
        return np.vstack([start, target])

    if (not np.isfinite(curvature)) or abs(curvature) <= 1e-6:
        return np.vstack([start, target])

    radius = 1.0 / curvature
    radius_abs = abs(radius)
    theta = float(current_pose[2])
    center = np.array([
        start[0] - np.sin(theta) * radius,
        start[1] + np.cos(theta) * radius,
    ], dtype=float)

    target_radius = float(np.linalg.norm(target - center))
    if abs(target_radius - radius_abs) > max(1e-3, 0.05 * radius_abs):
        return np.vstack([start, target])

    start_angle = float(np.arctan2(start[1] - center[1], start[0] - center[0]))
    target_angle = float(np.arctan2(target[1] - center[1], target[0] - center[0]))

    if curvature > 0.0:
        while target_angle < start_angle:
            target_angle += 2.0 * np.pi
    else:
        while target_angle > start_angle:
            target_angle -= 2.0 * np.pi

    angles = np.linspace(start_angle, target_angle, max(2, int(num_points)))
    arc_points = np.column_stack([
        center[0] + radius_abs * np.cos(angles),
        center[1] + radius_abs * np.sin(angles),
    ])
    arc_points[0] = start
    arc_points[-1] = target
    return arc_points


def calc_vw_line_segment_in_bounds(
    curvature: float,
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
) -> np.ndarray:
    if not np.isfinite(curvature):
        return np.empty((0, 2), dtype=float)

    if abs(curvature) <= 1e-12:
        if w_min <= 0.0 <= w_max:
            return np.array([[v_min, 0.0], [v_max, 0.0]], dtype=float)
        return np.empty((0, 2), dtype=float)

    candidates: list[tuple[float, float]] = []
    eps = 1e-12

    for v in (v_min, v_max):
        w = curvature * v
        if w_min - eps <= w <= w_max + eps:
            candidates.append((float(v), float(w)))

    for w in (w_min, w_max):
        v = w / curvature
        if v_min - eps <= v <= v_max + eps:
            candidates.append((float(v), float(w)))

    if len(candidates) == 0:
        return np.empty((0, 2), dtype=float)

    unique_candidates: list[tuple[float, float]] = []
    for point in candidates:
        if any(abs(point[0] - other[0]) <= 1e-9 and abs(point[1] - other[1]) <= 1e-9 for other in unique_candidates):
            continue
        unique_candidates.append(point)

    unique_candidates.sort(key=lambda point: (point[0], point[1]))
    if len(unique_candidates) == 1:
        return np.array([unique_candidates[0], unique_candidates[0]], dtype=float)

    return np.array([unique_candidates[0], unique_candidates[-1]], dtype=float)


def calc_constant_curvature_preview_arc(
    current_pose: np.ndarray,
    curvature: float,
    travel_distance: float,
    num_points: int = 60,
) -> np.ndarray:
    if (not np.isfinite(travel_distance)) or travel_distance <= 1e-9:
        return np.empty((0, 2), dtype=float)

    start = np.array(current_pose[:2], dtype=float)
    theta0 = float(current_pose[2])
    s = np.linspace(0.0, float(travel_distance), max(2, int(num_points)))

    if (not np.isfinite(curvature)) or abs(curvature) <= 1e-9:
        x = start[0] + s * np.cos(theta0)
        y = start[1] + s * np.sin(theta0)
        return np.column_stack([x, y])

    x = start[0] + (np.sin(theta0 + curvature * s) - np.sin(theta0)) / curvature
    y = start[1] + (-np.cos(theta0 + curvature * s) + np.cos(theta0)) / curvature
    return np.column_stack([x, y])


def calc_dynamic_window_curvature_range(
    dw_vmin: float,
    dw_vmax: float,
    dw_wmin: float,
    dw_wmax: float,
    v_eps: float = 1e-9,
) -> tuple[float, float]:
    if dw_vmax <= v_eps:
        return float("nan"), float("nan")

    if dw_wmin < -v_eps:
        kappa_min = float(dw_wmin / dw_vmin) if dw_vmin > v_eps else float("-inf")
    elif dw_wmin > v_eps:
        kappa_min = float(dw_wmin / dw_vmax)
    else:
        kappa_min = 0.0

    if dw_wmax > v_eps:
        kappa_max = float(dw_wmax / dw_vmin) if dw_vmin > v_eps else float("inf")
    elif dw_wmax < -v_eps:
        kappa_max = float(dw_wmax / dw_vmax)
    else:
        kappa_max = 0.0

    return kappa_min, kappa_max


def calc_visualized_curvature_for_preview(curvature: float, preview_distance: float) -> float:
    if np.isnan(curvature):
        return float("nan")

    min_radius = max(0.08, 0.18 * max(float(preview_distance), 1e-6))
    max_abs_curvature = 1.0 / min_radius

    if np.isposinf(curvature):
        return float(max_abs_curvature)
    if np.isneginf(curvature):
        return float(-max_abs_curvature)
    if not np.isfinite(curvature):
        return float("nan")

    return float(np.clip(curvature, -max_abs_curvature, max_abs_curvature))


def format_curvature_value(curvature: float) -> str:
    if np.isposinf(curvature):
        return "+inf"
    if np.isneginf(curvature):
        return "-inf"
    if not np.isfinite(curvature):
        return "N/A"
    return f"{float(curvature):.3f}"


def save_tracking_plots_by_method(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
    file_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.06, 0.055 * diag)
    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    fig_size, min_span, margin_ratio = calc_tracking_layout(xy_arrays, min_span_default=1.0)

    for spec in method_specs:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
        draw_path_heading_arrows(ax, path)

        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.0, label=spec.label)

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
            alpha=1.00,
            width=0.0055,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
        )

        set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.grid(True)
        plt.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}tracking_poses_{spec.key}.png", dpi=200)
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
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    pose_arrow_length = max(0.06, 0.055 * diag)
    for spec in method_specs:
        poses = results[spec.key].poses
        ax.plot(poses[:, 0], poses[:, 1], color=spec.color, linewidth=1.0, label=spec.label)

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
            alpha=1.00,
            width=0.0055,
            headwidth=4.5,
            headlength=5.5,
            headaxislength=4.5,
        )

    set_equal_axis_with_min_span(ax, xy_arrays, min_span=min_span, margin_ratio=margin_ratio)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_velocity_profiles_by_method(
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
    file_prefix: str = "",
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

    # 上段(vx/vy重ね描き)の縦軸
    values_min_v = [VX_MIN, VY_MIN]
    values_max_v = [VX_MAX, VY_MAX]
    for spec in method_specs:
        _, v_real, v_ref = converted_cache[spec.key]
        values_min_v.extend([
            finite_min(v_real[:, 0], VX_MIN), finite_min(v_ref[:, 0], VX_MIN),
            finite_min(v_real[:, 1], VY_MIN), finite_min(v_ref[:, 1], VY_MIN),
        ])
        values_max_v.extend([
            finite_max(v_real[:, 0], VX_MAX), finite_max(v_ref[:, 0], VX_MAX),
            finite_max(v_real[:, 1], VY_MAX), finite_max(v_ref[:, 1], VY_MAX),
        ])
    y_min_v = min(values_min_v)
    y_max_v = max(values_max_v)
    span_v = max(y_max_v - y_min_v, 1e-6)
    y_lim_v = (y_min_v - 0.08 * span_v, y_max_v + 0.08 * span_v)

    # 下段(omega)の縦軸は制約値近傍に固定
    omega_span = max(W_MAX - W_MIN, 1e-6)
    omega_margin = 0.08 * omega_span
    y_lim_w = (W_MIN - omega_margin, W_MAX + omega_margin)

    for spec in method_specs:
        result, v_real, v_ref = converted_cache[spec.key]

        fig, axes = plt.subplots(2, 1, figsize=(2.8, 2.8), sharex=True)

        # 上段: vx / vy を同一キャンバスに重ね描き（実線: vx, 破線: vy）
        axes[0].plot(result.times, v_ref[:, 0], color="red", linewidth=1.5)
        axes[0].plot(result.times, v_real[:, 0], color="blue", linewidth=1.5)
        axes[0].plot(result.times, v_ref[:, 1], color="red", linewidth=1.5, linestyle="--")
        axes[0].plot(result.times, v_real[:, 1], color="blue", linewidth=1.5, linestyle="--")
        axes[0].axhline(VX_MAX, color="black", linestyle="--", linewidth=0.8)
        if abs(VY_MAX - VX_MAX) > 1e-9:
            axes[0].axhline(VY_MAX, color="0.35", linestyle="--", linewidth=0.8)
        axes[0].set_ylabel(r"$v_x, v_y$ [m/s]")
        axes[0].set_xlim(0.0, time_max if time_max > 0.0 else 1.0)
        axes[0].set_ylim(*y_lim_v)
        axes[0].grid(True)

        # 下段: omega
        axes[1].plot(result.times, v_ref[:, 2], color="red", linewidth=1.5)
        axes[1].plot(result.times, v_real[:, 2], color="blue", linewidth=1.5)
        axes[1].axhline(W_MAX, color="black", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel(r"$\omega$ [rad/s]")
        axes[1].set_xlim(0.0, time_max if time_max > 0.0 else 1.0)
        axes[1].set_ylim(*y_lim_w)
        axes[1].grid(True)
        axes[1].set_xlabel("Time [s]")
        plt.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}velocity_profiles_{spec.key}.png", dpi=200)
        fig.savefig(output_dir / f"{file_prefix}velocity_profiles_{spec.key}.pdf", bbox_inches="tight")
        plt.close(fig)


def save_tracking_animation(
    path: np.ndarray,
    method_specs: list[MethodSpec],
    results: dict[str, SimulationResult],
    output_dir: Path,
    max_duration_s: float | None = None,
    frame_stride: int = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    xy_arrays = [path[:, :2]]
    for spec in method_specs:
        xy_arrays.append(results[spec.key].poses[:, :2])
    set_equal_axis_with_min_span(ax, xy_arrays, min_span=1.0, margin_ratio=0.2)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("Path Tracking Animation")
    ax.grid(True)

    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
    draw_path_heading_arrows(ax, path)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.05, 0.05 * diag)

    trail_lines = {}
    pose_points = {}
    pose_arrows = {}
    for spec in method_specs:
        trail_line, = ax.plot([], [], color=spec.color, linewidth=1.1, label=spec.label)
        pose_point, = ax.plot([], [], marker="o", color=spec.color, markersize=5)
        pose_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=10, color=spec.color, linewidth=1.0)
        pose_arrow.set_visible(False)
        ax.add_patch(pose_arrow)
        trail_lines[spec.key] = trail_line
        pose_points[spec.key] = pose_point
        pose_arrows[spec.key] = pose_arrow

    plt.tight_layout()

    frame_indices_by_method = {
        spec.key: build_animation_frame_indices(results[spec.key].times, max_duration_s, frame_stride)
        for spec in method_specs
    }
    max_frames = max(len(frame_indices_by_method[spec.key]) for spec in method_specs)

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
            frame_indices = frame_indices_by_method[spec.key]
            idx = int(frame_indices[min(frame_idx, len(frame_indices) - 1)])
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
        interval=max(1, int(round(1000.0 * DT * max(1, frame_stride)))),
        blit=False,
        repeat=False,
    )

    fps = max(1, int(round(1.0 / (DT * max(1, frame_stride)))))
    mp4_path = output_dir / "tracking_comparison.mp4"
    try:
        ani.save(mp4_path, writer="ffmpeg", fps=fps)
    except Exception:
        gif_path = output_dir / "tracking_comparison.gif"
        ani.save(gif_path, writer="pillow", fps=fps)

    plt.close(fig)


def save_method_debug_animation(
    path: np.ndarray,
    method_spec: MethodSpec,
    result: SimulationResult,
    output_dir: Path,
    max_duration_s: float | None = None,
    frame_stride: int = 1,
) -> None:
    fig = plt.figure(figsize=(15.0, 8.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_global = fig.add_subplot(grid[0, 0])
    ax_local = fig.add_subplot(grid[1, 0])
    ax_vw = fig.add_subplot(grid[:, 1])

    xy_arrays = [path[:, :2], result.poses[:, :2]]
    if result.look_ahead_positions is not None and len(result.look_ahead_positions) > 0:
        finite_mask = np.all(np.isfinite(result.look_ahead_positions), axis=1)
        if np.any(finite_mask):
            xy_arrays.append(result.look_ahead_positions[finite_mask])

    set_equal_axis_with_min_span(ax_global, xy_arrays, min_span=1.0, margin_ratio=0.2)
    ax_global.set_xlabel("$x$ [m]")
    ax_global.set_ylabel("$y$ [m]")
    ax_global.set_title(f"{method_spec.label} Debug Animation")
    ax_global.grid(True)
    ax_global.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")
    draw_path_heading_arrows(ax_global, path)

    ax_local.set_xlabel("$x$ [m]")
    ax_local.set_ylabel("$y$ [m]")
    ax_local.set_title("Local Zoom")
    ax_local.grid(True)
    ax_local.plot(path[:, 0], path[:, 1], "k--", linewidth=1.1, label="Reference Path")

    v_margin = 0.06 * max(V_MAX - V_MIN, 1e-6)
    w_margin = 0.06 * max(W_MAX - W_MIN, 1e-6)
    vw_vmin_axis = V_MIN - v_margin
    vw_vmax_axis = V_MAX + v_margin
    vw_wmin_axis = W_MIN - w_margin
    vw_wmax_axis = W_MAX + w_margin

    ax_vw.set_xlim(vw_vmin_axis, vw_vmax_axis)
    ax_vw.set_ylim(vw_wmin_axis, vw_wmax_axis)
    ax_vw.set_xlabel("$v$ [m/s]")
    ax_vw.set_ylabel("$\\omega$ [rad/s]")
    ax_vw.set_title("$v\\omega$ Plane")
    ax_vw.grid(True)
    ax_vw.axhline(0.0, color="0.7", linewidth=0.9)
    ax_vw.axvline(0.0, color="0.7", linewidth=0.9)

    span_x = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    span_y = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(span_x, span_y))
    arrow_length = max(0.05, 0.05 * diag)
    zoom_radius = max(0.4, 0.12 * diag)
    dw_kappa_preview_distance = max(0.25, 0.80 * zoom_radius)

    def create_debug_artists(ax: plt.Axes, show_labels: bool) -> dict[str, object]:
        trail_line, = ax.plot(
            [],
            [],
            color=method_spec.color,
            linewidth=1.4,
            label=method_spec.label if show_labels else None,
            zorder=2,
        )
        pose_point, = ax.plot(
            [],
            [],
            marker="o",
            color=method_spec.color,
            markersize=7,
            label="Robot" if show_labels else None,
            zorder=5,
        )
        look_ahead_point, = ax.plot(
            [],
            [],
            marker="o",
            color="crimson",
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=9,
            label="Look Ahead" if show_labels else None,
            zorder=7,
        )
        chord_line, = ax.plot(
            [],
            [],
            color="orange",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
            label="Look-Ahead Chord" if show_labels else None,
            zorder=4,
        )
        arc_line, = ax.plot(
            [],
            [],
            color="magenta",
            linewidth=2.0,
            alpha=0.9,
            label="Connecting Arc" if show_labels else None,
            zorder=6,
        )
        min_kappa_arc_line, = ax.plot(
            [],
            [],
            color="tab:green",
            linewidth=1.6,
            linestyle=":",
            alpha=0.95,
            label="DW $\\kappa_{min}$ Arc" if show_labels else None,
            zorder=3,
        )
        max_kappa_arc_line, = ax.plot(
            [],
            [],
            color="tab:cyan",
            linewidth=1.6,
            linestyle=":",
            alpha=0.95,
            label="DW $\\kappa_{max}$ Arc" if show_labels else None,
            zorder=3,
        )
        pose_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=12, color=method_spec.color, linewidth=1.2)
        pose_arrow.set_visible(False)
        ax.add_patch(pose_arrow)
        return {
            "trail_line": trail_line,
            "pose_point": pose_point,
            "look_ahead_point": look_ahead_point,
            "chord_line": chord_line,
            "arc_line": arc_line,
            "min_kappa_arc_line": min_kappa_arc_line,
            "max_kappa_arc_line": max_kappa_arc_line,
            "pose_arrow": pose_arrow,
        }

    global_artists = create_debug_artists(ax_global, show_labels=True)
    local_artists = create_debug_artists(ax_local, show_labels=False)
    dw_patch = Rectangle(
        (0.0, 0.0),
        0.0,
        0.0,
        fill=False,
        edgecolor="black",
        linewidth=1.5,
        linestyle="--",
        label="Dynamic Window",
    )
    dw_patch.set_visible(False)
    ax_vw.add_patch(dw_patch)
    vw_line, = ax_vw.plot(
        [],
        [],
        color="magenta",
        linewidth=2.0,
        label="$\\omega = \\kappa v$",
    )
    current_velocity_point, = ax_vw.plot(
        [],
        [],
        marker="o",
        color=method_spec.color,
        markersize=7,
        label="Current Velocity",
    )
    selected_velocity_point, = ax_vw.plot(
        [],
        [],
        marker="x",
        color="crimson",
        markersize=9,
        markeredgewidth=2.0,
        linestyle="None",
        label="Selected Command",
    )

    time_text = ax_global.text(0.02, 0.98, "", transform=ax_global.transAxes, ha="left", va="top")
    kappa_text = ax_local.text(0.02, 0.98, "", transform=ax_local.transAxes, ha="left", va="top")
    vw_text = ax_vw.text(0.02, 0.98, "", transform=ax_vw.transAxes, ha="left", va="top")
    ax_global.legend(loc="upper right")
    ax_vw.legend(loc="upper right")
    fig.tight_layout()

    frame_indices = build_animation_frame_indices(result.times, max_duration_s, frame_stride)
    look_ahead_positions = result.look_ahead_positions if result.look_ahead_positions is not None else np.empty((0, 2))
    curvatures = result.curvatures if result.curvatures is not None else np.empty((0,), dtype=float)

    def init():
        for artist_group in (global_artists, local_artists):
            artist_group["trail_line"].set_data([], [])
            artist_group["pose_point"].set_data([], [])
            artist_group["look_ahead_point"].set_data([], [])
            artist_group["chord_line"].set_data([], [])
            artist_group["arc_line"].set_data([], [])
            artist_group["min_kappa_arc_line"].set_data([], [])
            artist_group["max_kappa_arc_line"].set_data([], [])
            artist_group["pose_arrow"].set_visible(False)
        dw_patch.set_visible(False)
        vw_line.set_data([], [])
        current_velocity_point.set_data([], [])
        selected_velocity_point.set_data([], [])
        time_text.set_text("")
        kappa_text.set_text("")
        vw_text.set_text("")
        return [
            global_artists["trail_line"], global_artists["pose_point"], global_artists["look_ahead_point"],
            global_artists["chord_line"], global_artists["arc_line"],
            global_artists["min_kappa_arc_line"], global_artists["max_kappa_arc_line"], global_artists["pose_arrow"],
            local_artists["trail_line"], local_artists["pose_point"], local_artists["look_ahead_point"],
            local_artists["chord_line"], local_artists["arc_line"],
            local_artists["min_kappa_arc_line"], local_artists["max_kappa_arc_line"], local_artists["pose_arrow"],
            dw_patch, vw_line, current_velocity_point, selected_velocity_point,
            time_text, kappa_text, vw_text,
        ]

    def update(frame_number: int):
        idx = int(frame_indices[frame_number])
        current_pose = result.poses[idx]

        dx = arrow_length * np.cos(current_pose[2])
        dy = arrow_length * np.sin(current_pose[2])

        for artist_group in (global_artists, local_artists):
            artist_group["trail_line"].set_data(result.poses[:idx + 1, 0], result.poses[:idx + 1, 1])
            artist_group["pose_point"].set_data([current_pose[0]], [current_pose[1]])
            artist_group["pose_arrow"].set_positions(
                (current_pose[0], current_pose[1]),
                (current_pose[0] + dx, current_pose[1] + dy),
            )
            artist_group["pose_arrow"].set_visible(True)

        ax_local.set_xlim(current_pose[0] - zoom_radius, current_pose[0] + zoom_radius)
        ax_local.set_ylim(current_pose[1] - zoom_radius, current_pose[1] + zoom_radius)
        ax_local.set_aspect("equal")

        if idx < len(look_ahead_positions) and np.all(np.isfinite(look_ahead_positions[idx])):
            look_ahead_pos = look_ahead_positions[idx]
            curvature = float(curvatures[idx]) if idx < len(curvatures) else float("nan")
            arc_points = calc_connecting_arc_points(current_pose, look_ahead_pos, curvature)
            for artist_group in (global_artists, local_artists):
                artist_group["look_ahead_point"].set_data([look_ahead_pos[0]], [look_ahead_pos[1]])
                artist_group["chord_line"].set_data(
                    [current_pose[0], look_ahead_pos[0]],
                    [current_pose[1], look_ahead_pos[1]],
                )
                if len(arc_points) > 0:
                    artist_group["arc_line"].set_data(arc_points[:, 0], arc_points[:, 1])
                else:
                    artist_group["arc_line"].set_data([], [])
            kappa_text.set_text(f"$\\kappa$ = {curvature:.3f}")
        else:
            curvature = float("nan")
            for artist_group in (global_artists, local_artists):
                artist_group["look_ahead_point"].set_data([], [])
                artist_group["chord_line"].set_data([], [])
                artist_group["arc_line"].set_data([], [])
            kappa_text.set_text("$\\kappa$ = N/A")

        if idx < len(curvatures) and (idx + 1) < len(result.ref_velocities_raw):
            current_velocity = np.asarray(result.velocities_raw[idx], dtype=float)
            selected_velocity = np.asarray(result.ref_velocities_raw[idx + 1], dtype=float)
            dw_vmin, dw_vmax, dw_wmin, dw_wmax = calc_dynamic_window_bounds(current_velocity)
            if DWPP_USE_REGULATED_VELOCITY and np.isfinite(curvature):
                regulated_v = calc_regulated_translational_velocity(curvature)
            else:
                regulated_v = V_MAX
            dw_vmax = min(dw_vmax, max(dw_vmin, regulated_v))

            dw_patch.set_xy((dw_vmin, dw_wmin))
            dw_patch.set_width(max(dw_vmax - dw_vmin, 0.0))
            dw_patch.set_height(max(dw_wmax - dw_wmin, 0.0))
            dw_patch.set_visible(True)

            dw_kappa_min, dw_kappa_max = calc_dynamic_window_curvature_range(
                dw_vmin=dw_vmin,
                dw_vmax=dw_vmax,
                dw_wmin=dw_wmin,
                dw_wmax=dw_wmax,
            )
            min_dw_arc_points = calc_constant_curvature_preview_arc(
                current_pose=current_pose,
                curvature=calc_visualized_curvature_for_preview(dw_kappa_min, dw_kappa_preview_distance),
                travel_distance=dw_kappa_preview_distance,
            )
            max_dw_arc_points = calc_constant_curvature_preview_arc(
                current_pose=current_pose,
                curvature=calc_visualized_curvature_for_preview(dw_kappa_max, dw_kappa_preview_distance),
                travel_distance=dw_kappa_preview_distance,
            )
            for artist_group in (global_artists, local_artists):
                if len(min_dw_arc_points) > 0:
                    artist_group["min_kappa_arc_line"].set_data(min_dw_arc_points[:, 0], min_dw_arc_points[:, 1])
                else:
                    artist_group["min_kappa_arc_line"].set_data([], [])
                if len(max_dw_arc_points) > 0:
                    artist_group["max_kappa_arc_line"].set_data(max_dw_arc_points[:, 0], max_dw_arc_points[:, 1])
                else:
                    artist_group["max_kappa_arc_line"].set_data([], [])

            vw_segment = calc_vw_line_segment_in_bounds(
                curvature=curvature,
                v_min=vw_vmin_axis,
                v_max=vw_vmax_axis,
                w_min=vw_wmin_axis,
                w_max=vw_wmax_axis,
            )
            if len(vw_segment) > 0:
                vw_line.set_data(vw_segment[:, 0], vw_segment[:, 1])
            else:
                vw_line.set_data([], [])

            current_velocity_point.set_data([current_velocity[0]], [current_velocity[1]])
            selected_velocity_point.set_data([selected_velocity[0]], [selected_velocity[1]])
            vw_text.set_text(
                "\n".join([
                    f"DW: v=[{dw_vmin:.3f}, {dw_vmax:.3f}]",
                    f"    $\\omega$=[{dw_wmin:.3f}, {dw_wmax:.3f}]",
                    f"$\\kappa_{{DW}}$=[{format_curvature_value(dw_kappa_min)}, {format_curvature_value(dw_kappa_max)}]",
                    f"cmd=({selected_velocity[0]:.3f}, {selected_velocity[1]:.3f})",
                ])
            )
        else:
            dw_patch.set_visible(False)
            vw_line.set_data([], [])
            current_velocity_point.set_data([], [])
            selected_velocity_point.set_data([], [])
            for artist_group in (global_artists, local_artists):
                artist_group["min_kappa_arc_line"].set_data([], [])
                artist_group["max_kappa_arc_line"].set_data([], [])
            vw_text.set_text("")

        time_text.set_text(f"t = {result.times[idx]:.2f} s")
        return [
            global_artists["trail_line"], global_artists["pose_point"], global_artists["look_ahead_point"],
            global_artists["chord_line"], global_artists["arc_line"],
            global_artists["min_kappa_arc_line"], global_artists["max_kappa_arc_line"], global_artists["pose_arrow"],
            local_artists["trail_line"], local_artists["pose_point"], local_artists["look_ahead_point"],
            local_artists["chord_line"], local_artists["arc_line"],
            local_artists["min_kappa_arc_line"], local_artists["max_kappa_arc_line"], local_artists["pose_arrow"],
            dw_patch, vw_line, current_velocity_point, selected_velocity_point,
            time_text, kappa_text, vw_text,
        ]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=max(1, int(round(1000.0 * DT * max(1, frame_stride)))),
        blit=False,
        repeat=False,
    )

    fps = max(1, int(round(1.0 / (DT * max(1, frame_stride)))))
    output_path = output_dir / f"tracking_debug_{method_spec.key}.mp4"
    try:
        ani.save(output_path, writer="ffmpeg", fps=fps)
    except Exception:
        ani.save(output_path.with_suffix(".gif"), writer="pillow", fps=fps)

    plt.close(fig)


def run_benchmark_for_path(
    path: np.ndarray,
    path_name: str,
    output_root: Path,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
    max_sim_steps: int,
    save_animation: bool,
    save_dwpp_debug_animation: bool = False,
    animation_max_seconds: float | None = None,
    animation_frame_stride: int = 1,
) -> list[dict[str, str | float]]:
    path_dir = output_root / path_name
    path_dir.mkdir(parents=True, exist_ok=True)

    figure_prefix = ""
    if path_name == "path3_right_angle_90_last_heading_minus_pi":
        figure_prefix = "exp1_"
    elif path_name == "path4_one_minus_cos":
        figure_prefix = "exp2_"

    np.save(path_dir / "path.npy", path)

    results: dict[str, SimulationResult] = {}
    rows: list[dict[str, str | float]] = []

    for spec in METHOD_SPECS:
        result = simulate_single_method(
            path=path,
            method_spec=spec,
            goal_tolerance_dist=goal_tolerance_dist,
            goal_tolerance_heading=goal_tolerance_heading,
            max_sim_steps=max_sim_steps,
        )
        results[spec.key] = result

        np.save(path_dir / f"{spec.key}_poses.npy", result.poses)
        np.save(path_dir / f"{spec.key}_velocities.npy", result.velocities_raw)
        np.save(path_dir / f"{spec.key}_ref_velocities.npy", result.ref_velocities_raw)
        np.save(path_dir / f"{spec.key}_break_flags.npy", result.break_flags)
        np.save(path_dir / f"{spec.key}_times.npy", result.times)
        if result.look_ahead_positions is not None:
            np.save(path_dir / f"{spec.key}_look_ahead_positions.npy", result.look_ahead_positions)
        if result.curvatures is not None:
            np.save(path_dir / f"{spec.key}_curvatures.npy", result.curvatures)

        metrics = calc_metrics(
            path,
            result,
            goal_tolerance_dist=goal_tolerance_dist,
            goal_tolerance_heading=goal_tolerance_heading,
        )
        row = {"Method": spec.label}
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
    if save_animation:
        save_tracking_animation(
            path=path,
            method_specs=METHOD_SPECS,
            results=results,
            output_dir=path_dir,
            max_duration_s=animation_max_seconds,
            frame_stride=animation_frame_stride,
        )
    if save_dwpp_debug_animation:
        for spec in METHOD_SPECS:
            if spec.key != "dwpp":
                continue
            save_method_debug_animation(
                path=path,
                method_spec=spec,
                result=results[spec.key],
                output_dir=path_dir,
                max_duration_s=animation_max_seconds,
                frame_stride=animation_frame_stride,
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
    parser.add_argument("--max-sim-steps", type=int, default=DEFAULT_MAX_SIM_STEPS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--path-names",
        nargs="*",
        default=None,
        help="Optional subset of path names to run, e.g. path1_right_angle_90",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="Save tracking animation (MP4/GIF). Default: off",
    )
    parser.add_argument(
        "--save-dwpp-debug-animation",
        action="store_true",
        help="Save a DWPP debug animation with look-ahead point and connecting arc.",
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

    one_minus_cos_num_points = args.one_minus_cos_num_points
    if one_minus_cos_num_points is None:
        one_minus_cos_num_points = 5 * args.points_per_segment + 1

    paths = {
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
    print(f"[INFO] Benchmark finished. Output: {output_root}")


if __name__ == "__main__":
    main()
