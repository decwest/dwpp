"""Faithful NumPy-free ports of the Nav2 PP/RPP/DWPP control laws."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .simulator import SimulationConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Copied from dwpp_test_simulation/scripts/follow_path_test_gui_real.py,
# itself a validated port of dynamic_window_pure_pursuit_functions.hpp on the
# Decwest/navigation2 feature/revision_experiments_jazzy branch. Keep the
# signed max_decel convention and Eps values unchanged.
def _window_1d(last_vel, max_vel, min_vel, max_accel, max_decel, dt, eps):
    """C++ compute_window / evaluate lambda shared one-dimensional window."""
    if last_vel > eps:
        cand_max = last_vel + max_accel * dt
        cand_min = last_vel + max_decel * dt
    elif last_vel < -eps:
        cand_max = last_vel - max_decel * dt
        cand_min = last_vel - max_accel * dt
    else:
        cand_max = last_vel + max_accel * dt
        cand_min = last_vel - max_accel * dt
    return min(cand_max, max_vel), max(cand_min, min_vel)


def compute_dynamic_window(v_now, w_now, limits, dt):
    """Port of computeDynamicWindow (Eps=1e-3), before regulation."""
    eps = 1e-3
    v_max, v_min = _window_1d(
        v_now, limits["max_linear_vel"], limits["min_linear_vel"],
        limits["max_linear_accel"], limits["max_linear_decel"], dt, eps)
    w_max, w_min = _window_1d(
        w_now, limits["max_angular_vel"], limits["min_angular_vel"],
        limits["max_angular_accel"], limits["max_angular_decel"], dt, eps)
    return v_max, v_min, w_max, w_min


def evaluate_velocity_constraints(v_cmd, w_cmd, v_now, w_now, limits, dt):
    """Port of evaluateVelocityConstraints (Eps=1e-2 for branch and tolerance)."""
    eps = 1e-2

    def violated(cur, last, max_vel, min_vel, max_accel, max_decel):
        cand_max, cand_min = _window_1d(last, max_vel, min_vel, max_accel, max_decel, dt, eps)
        return cur > cand_max + eps or cur < cand_min - eps

    return violated(
        v_cmd, v_now, limits["max_linear_vel"], limits["min_linear_vel"],
        limits["max_linear_accel"], limits["max_linear_decel"],
    ) or violated(
        w_cmd, w_now, limits["max_angular_vel"], limits["min_angular_vel"],
        limits["max_angular_accel"], limits["max_angular_decel"],
    )


def calc_actual_velocity(v_cmd, w_cmd, v_now, w_now, limits, dt):
    """Port of recordData's calc_actual_velocity helper (Eps=1e-3)."""
    eps = 1e-3

    def clip(cur, last, max_vel, min_vel, max_accel, max_decel):
        cand_max, cand_min = _window_1d(last, max_vel, min_vel, max_accel, max_decel, dt, eps)
        if cur > cand_max + eps:
            return cand_max
        if cur < cand_min - eps:
            return cand_min
        return cur

    v_nav = clip(v_cmd, v_now, limits["max_linear_vel"], limits["min_linear_vel"],
                 limits["max_linear_accel"], limits["max_linear_decel"])
    w_nav = clip(w_cmd, w_now, limits["max_angular_vel"], limits["min_angular_vel"],
                 limits["max_angular_accel"], limits["max_angular_decel"])
    return v_nav, w_nav


def _transform_and_prune(
    path: "NDArray[np.float64]",
    pose: tuple[float, float, float],
    prune_index: int,
) -> tuple["NDArray[np.float64]", int]:
    """Prune through the closest pose and transform the remaining plan to base frame."""
    remaining = path[prune_index:]
    closest_offset = int(np.argmin(np.sum((remaining - pose[:2]) ** 2, axis=1)))
    prune_index += closest_offset
    remaining = path[prune_index:]
    dx = remaining[:, 0] - pose[0]
    dy = remaining[:, 1] - pose[1]
    c = math.cos(pose[2])
    s = math.sin(pose[2])
    transformed = np.column_stack((c * dx + s * dy, -s * dx + c * dy))
    return transformed, prune_index


def circle_segment_intersection(
    p1: "NDArray[np.float64]", p2: "NDArray[np.float64]", radius: float
) -> "NDArray[np.float64]":
    """Closed-form Nav2 circle/segment intersection in the robot frame."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx, dy = x2 - x1, y2 - y1
    dr2 = dx * dx + dy * dy
    if dr2 <= np.finfo(float).eps:
        return np.asarray(p2, dtype=float)
    determinant = x1 * y2 - x2 * y1
    dd = (x2 * x2 + y2 * y2) - (x1 * x1 + y1 * y1)
    radicand = max(0.0, radius * radius * dr2 - determinant * determinant)
    root = math.sqrt(radicand)
    direction = math.copysign(1.0, dd)
    return np.array(
        (
            (determinant * dy + direction * dx * root) / dr2,
            (-determinant * dx + direction * dy * root) / dr2,
        ),
        dtype=float,
    )


def get_lookahead_point(
    transformed_path: "NDArray[np.float64]", lookahead: float
) -> "NDArray[np.float64]":
    distances = np.hypot(transformed_path[:, 0], transformed_path[:, 1])
    candidates = np.flatnonzero(distances >= lookahead)
    if len(candidates) == 0:
        return transformed_path[-1].copy()
    index = int(candidates[0])
    if index > 0:
        return circle_segment_intersection(transformed_path[index - 1], transformed_path[index], lookahead)
    return transformed_path[index].copy()


def _path_length(path: "NDArray[np.float64]") -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))))


def regulated_linear_velocity(
    curvature: float,
    transformed_path: "NDArray[np.float64]",
    config: SimulationConfig,
) -> float:
    velocity = config.max_linear_velocity
    if config.use_curvature_regulation:
        if abs(curvature) > 0.0:
            radius = abs(1.0 / curvature)
            if radius < config.regulated_min_radius:
                velocity *= radius / config.regulated_min_radius
        velocity = max(velocity, config.regulated_min_speed)

    scale = 1.0
    if _path_length(transformed_path) < config.approach_scaling_distance:
        scale = math.hypot(float(transformed_path[-1, 0]), float(transformed_path[-1, 1])) / config.approach_scaling_distance
    velocity = min(velocity, max(velocity * scale, config.min_approach_velocity))
    return min(max(velocity, config.min_linear_velocity), config.max_linear_velocity)


def apply_regulation_to_window(
    window: tuple[float, float, float, float], regulated_velocity: float
) -> tuple[float, float, float, float]:
    v_hi, v_lo, w_hi, w_lo = window
    regulated_hi = min(v_hi, regulated_velocity)
    regulated_lo = max(v_lo, 0.0)
    if regulated_hi < regulated_lo:
        if regulated_lo > 0.0:
            regulated_hi = regulated_lo
        else:
            regulated_lo = regulated_hi
    return regulated_hi, regulated_lo, w_hi, w_lo


def optimal_velocity_in_window(
    window: tuple[float, float, float, float], curvature: float
) -> tuple[float, float]:
    """Faithful forward-motion port of computeOptimalVelocityWithinDynamicWindow."""
    v_hi, v_lo, w_hi, w_lo = window
    if abs(curvature) < 1e-3:
        if w_lo <= 0.0 <= w_hi:
            return v_hi, 0.0
        return v_hi, w_lo if abs(w_lo) <= abs(w_hi) else w_hi

    candidates = (
        (v_lo, curvature * v_lo),
        (v_hi, curvature * v_hi),
        (w_lo / curvature, w_lo),
        (w_hi / curvature, w_hi),
    )
    valid = [
        (v, w)
        for v, w in candidates
        if v_lo <= v <= v_hi and w_lo <= w <= w_hi
    ]
    if valid:
        return max(valid, key=lambda command: command[0])

    corners = ((v_lo, w_lo), (v_lo, w_hi), (v_hi, w_lo), (v_hi, w_hi))
    denominator = math.sqrt(curvature * curvature + 1.0)
    best = corners[0]
    closest = math.inf
    for corner in corners:
        distance = abs(curvature * corner[0] - corner[1]) / denominator
        if distance < closest or (abs(distance - closest) <= 1e-3 and corner[0] > best[0]):
            closest = distance
            best = corner
    return best


def compute_command(
    path: "NDArray[np.float64]",
    pose: tuple[float, float, float],
    previous_command: tuple[float, float],
    prune_index: int,
    config: SimulationConfig,
) -> tuple[tuple[float, float], int]:
    transformed_path, prune_index = _transform_and_prune(path, pose, prune_index)
    if config.use_velocity_scaled_lookahead:
        lookahead = min(
            max(abs(previous_command[0]) * config.lookahead_time, config.min_lookahead),
            config.max_lookahead,
        )
    else:
        lookahead = config.fixed_lookahead
    carrot = get_lookahead_point(transformed_path, lookahead)
    carrot_dist2 = float(np.dot(carrot, carrot))
    curvature = 2.0 * float(carrot[1]) / carrot_dist2 if carrot_dist2 > 0.001 else 0.0
    linear_velocity = regulated_linear_velocity(curvature, transformed_path, config)

    if not config.use_dynamic_window:
        return (linear_velocity, curvature * linear_velocity), prune_index

    window = compute_dynamic_window(
        previous_command[0], previous_command[1], config.limits_dict(), config.dt
    )
    window = apply_regulation_to_window(window, linear_velocity)
    return optimal_velocity_in_window(window, curvature), prune_index
