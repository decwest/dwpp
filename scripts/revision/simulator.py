"""Unicycle model and deterministic path-following trial simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulationConfig:
    """Frozen controller and robot settings used by the real-robot experiments."""

    dt: float = 1.0 / 30.0
    timeout: float = 120.0
    goal_tolerance: float = 0.10

    min_linear_velocity: float = 0.0
    max_linear_velocity: float = 0.5
    min_angular_velocity: float = -1.0
    max_angular_velocity: float = 1.0
    linear_acceleration: float = 0.5
    linear_deceleration: float = 0.5
    angular_acceleration: float = 1.0
    angular_deceleration: float = 1.0

    fixed_lookahead: float = 0.6
    use_velocity_scaled_lookahead: bool = True
    lookahead_time: float = 1.4
    min_lookahead: float = 0.3
    max_lookahead: float = 0.7

    use_curvature_regulation: bool = True
    regulated_min_radius: float = 0.9
    regulated_min_speed: float = 0.25
    min_approach_velocity: float = 0.05
    approach_scaling_distance: float = 0.6

    use_dynamic_window: bool = True
    controller_name: str = "DWPP"

    def limits_dict(self) -> dict[str, float]:
        """Return the signed-deceleration mapping used by the validated port."""
        return {
            "max_linear_vel": self.max_linear_velocity,
            "min_linear_vel": self.min_linear_velocity,
            "max_angular_vel": self.max_angular_velocity,
            "min_angular_vel": self.min_angular_velocity,
            "max_linear_accel": self.linear_acceleration,
            "max_linear_decel": -self.linear_deceleration,
            "max_angular_accel": self.angular_acceleration,
            "max_angular_decel": -self.angular_deceleration,
        }


DEFAULT_CONFIG = SimulationConfig()


@dataclass(frozen=True)
class TrialResult:
    path_name: str
    controller: str
    seed: int
    sigma_xy: float
    sigma_yaw: float
    reached_goal: bool
    steps: int
    travel_time: float
    mean_error: float
    max_error: float
    violation_ratio: float
    max_abs_curvature: float
    positions: "NDArray[np.float64]"
    commands: "NDArray[np.float64]"
    applied_commands: "NDArray[np.float64]"
    curvatures: "NDArray[np.float64]"

    def as_row(self) -> dict[str, object]:
        return {
            "path": self.path_name,
            "controller": self.controller,
            "seed": self.seed,
            "sigma_xy": self.sigma_xy,
            "sigma_yaw": self.sigma_yaw,
            "reached_goal": self.reached_goal,
            "steps": self.steps,
            "travel_time": self.travel_time,
            "mean_error": self.mean_error,
            "max_error": self.max_error,
            "violation_ratio": self.violation_ratio,
            "max_abs_curvature": self.max_abs_curvature,
        }


def controller_config(name: str, base: SimulationConfig = DEFAULT_CONFIG) -> SimulationConfig:
    """Build one of the four frozen experiment controller configurations."""
    normalized = name.upper()
    if normalized == "PP":
        return replace(
            base,
            controller_name="PP",
            use_velocity_scaled_lookahead=False,
            use_curvature_regulation=False,
            use_dynamic_window=False,
        )
    if normalized == "APP":
        return replace(
            base,
            controller_name="APP",
            use_velocity_scaled_lookahead=True,
            use_curvature_regulation=False,
            use_dynamic_window=False,
        )
    if normalized == "RPP":
        return replace(
            base,
            controller_name="RPP",
            use_velocity_scaled_lookahead=True,
            use_curvature_regulation=True,
            use_dynamic_window=False,
        )
    if normalized == "DWPP":
        return replace(
            base,
            controller_name="DWPP",
            use_velocity_scaled_lookahead=True,
            use_curvature_regulation=True,
            use_dynamic_window=True,
        )
    raise ValueError(f"Unknown controller: {name}")


def make_paths() -> dict[str, "NDArray[np.float64]"]:
    """Generate the three 900-point polylines used by ``analysis_lib.make_path``."""
    paths: dict[str, NDArray[np.float64]] = {}
    for name, theta in zip(
        ("PathA", "PathB", "PathC"),
        (math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0),
        strict=True,
    ):
        x1 = np.linspace(0.0, 3.0, 300)
        y1 = np.zeros_like(x1)
        x2 = np.linspace(3.0, 3.0 + 3.0 * math.cos(theta), 300)
        y2 = np.linspace(0.0, 3.0 * math.sin(theta), 300)
        x3 = np.linspace(3.0 + 3.0 * math.cos(theta), 6.0 + 3.0 * math.cos(theta), 300)
        y3 = np.full_like(x3, 3.0 * math.sin(theta))
        paths[name] = np.column_stack((np.concatenate((x1, x2, x3)), np.concatenate((y1, y2, y3))))
    return paths


def exact_arc_step(
    pose: tuple[float, float, float], v: float, w: float, dt: float
) -> tuple[float, float, float]:
    """Integrate a constant unicycle command exactly for one control period."""
    x, y, theta = pose
    if abs(w) > 1e-9:
        next_theta = theta + w * dt
        x += v / w * (math.sin(next_theta) - math.sin(theta))
        y += -v / w * (math.cos(next_theta) - math.cos(theta))
        theta = next_theta
    else:
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += w * dt
    return x, y, theta


def _tracking_errors(
    positions: "NDArray[np.float64]", path: "NDArray[np.float64]"
) -> "NDArray[np.float64]":
    # Trials contain at most 3601 samples. Chunking bounds temporary memory while
    # retaining the exact distance to all 900 discrete reference points.
    errors = np.empty(len(positions), dtype=float)
    for start in range(0, len(positions), 512):
        xy = positions[start : start + 512, :2]
        delta = xy[:, None, :] - path[None, :, :]
        errors[start : start + len(xy)] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return errors


def simulate_trial(
    path: "NDArray[np.float64]",
    path_name: str,
    config: SimulationConfig = DEFAULT_CONFIG,
    *,
    sigma_xy: float = 0.0,
    sigma_yaw: float = 0.0,
    seed: int = 0,
) -> TrialResult:
    """Run one deterministic trial, optionally adding independent pose noise."""
    from .controllers import calc_actual_velocity, compute_command, evaluate_velocity_constraints

    rng = np.random.default_rng(seed)
    true_pose = (0.0, 0.0, 0.0)
    previous_command = (0.0, 0.0)
    # Raw controller output of the previous cycle. The violation metric of the
    # paper (Sec. 4.1) compares consecutive RAW commands (the plugin logger uses
    # its own previous output as v_now), independent of the downstream clipping.
    previous_raw_command = (0.0, 0.0)
    prune_index = 0
    positions = [true_pose]
    commands: list[tuple[float, float]] = []
    applied_commands: list[tuple[float, float]] = []
    curvatures: list[float] = []
    violations = 0
    max_steps = int(math.ceil(config.timeout / config.dt))
    reached_goal = math.hypot(true_pose[0] - path[-1, 0], true_pose[1] - path[-1, 1]) <= config.goal_tolerance

    for _ in range(max_steps):
        if reached_goal:
            break
        observed_pose = (
            true_pose[0] + float(rng.normal(0.0, sigma_xy)),
            true_pose[1] + float(rng.normal(0.0, sigma_xy)),
            true_pose[2] + float(rng.normal(0.0, sigma_yaw)),
        )
        command, prune_index, curvature = compute_command(
            path, observed_pose, previous_command, prune_index, config
        )
        if evaluate_velocity_constraints(
            command[0],
            command[1],
            previous_raw_command[0],
            previous_raw_command[1],
            config.limits_dict(),
            config.dt,
        ):
            violations += 1
        previous_raw_command = command
        # The controller output is checked as v_cmd. The acceleration-limited
        # velocity accepted by the base is then applied without lag (A1). This
        # is the same deterministic projection used by the validated real-trial
        # logger and is required to represent the frozen robot limits.
        applied_command = calc_actual_velocity(
            command[0],
            command[1],
            previous_command[0],
            previous_command[1],
            config.limits_dict(),
            config.dt,
        )
        true_pose = exact_arc_step(true_pose, applied_command[0], applied_command[1], config.dt)
        previous_command = applied_command
        positions.append(true_pose)
        commands.append(command)
        applied_commands.append(applied_command)
        curvatures.append(curvature)
        reached_goal = math.hypot(true_pose[0] - path[-1, 0], true_pose[1] - path[-1, 1]) <= config.goal_tolerance

    positions_array = np.asarray(positions, dtype=float)
    commands_array = np.asarray(commands, dtype=float).reshape((-1, 2))
    applied_commands_array = np.asarray(applied_commands, dtype=float).reshape((-1, 2))
    curvatures_array = np.asarray(curvatures, dtype=float)
    errors = _tracking_errors(positions_array, path)
    steps = len(commands)
    return TrialResult(
        path_name=path_name,
        controller=config.controller_name,
        seed=seed,
        sigma_xy=sigma_xy,
        sigma_yaw=sigma_yaw,
        reached_goal=reached_goal,
        steps=steps,
        travel_time=steps * config.dt,
        mean_error=float(np.mean(errors)),
        max_error=float(np.max(errors)),
        violation_ratio=100.0 * violations / steps if steps else 0.0,
        max_abs_curvature=float(np.max(np.abs(curvatures_array))) if steps else 0.0,
        positions=positions_array,
        commands=commands_array,
        applied_commands=applied_commands_array,
        curvatures=curvatures_array,
    )
