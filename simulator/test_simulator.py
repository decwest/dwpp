#!/usr/bin/env python3
"""Minimal dependency-free assertions for the DWPP kinematic simulator."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.controllers import (
    _window_1d,
    calc_actual_velocity,
    compute_dynamic_window,
    evaluate_velocity_constraints,
)
from simulator.simulator import DEFAULT_CONFIG
from simulator.studies import run_smoke


def _reference_window(last_vel, max_vel, min_vel, max_accel, max_decel, dt, eps):
    """Verbatim arithmetic from follow_path_test_gui_real._window_1d."""
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


def _reference_dynamic(v_now, w_now, limits, dt):
    v_max, v_min = _reference_window(
        v_now,
        limits["max_linear_vel"],
        limits["min_linear_vel"],
        limits["max_linear_accel"],
        limits["max_linear_decel"],
        dt,
        1e-3,
    )
    w_max, w_min = _reference_window(
        w_now,
        limits["max_angular_vel"],
        limits["min_angular_vel"],
        limits["max_angular_accel"],
        limits["max_angular_decel"],
        dt,
        1e-3,
    )
    return v_max, v_min, w_max, w_min


def _reference_violation(v_cmd, w_cmd, v_now, w_now, limits, dt):
    def violated(current, previous, prefix):
        high, low = _reference_window(
            previous,
            limits[f"max_{prefix}_vel"],
            limits[f"min_{prefix}_vel"],
            limits[f"max_{prefix}_accel"],
            limits[f"max_{prefix}_decel"],
            dt,
            1e-2,
        )
        return current > high + 1e-2 or current < low - 1e-2

    return violated(v_cmd, v_now, "linear") or violated(w_cmd, w_now, "angular")


def _reference_actual(v_cmd, w_cmd, v_now, w_now, limits, dt):
    def clipped(current, previous, prefix):
        high, low = _reference_window(
            previous,
            limits[f"max_{prefix}_vel"],
            limits[f"min_{prefix}_vel"],
            limits[f"max_{prefix}_accel"],
            limits[f"max_{prefix}_decel"],
            dt,
            1e-3,
        )
        if current > high + 1e-3:
            return high
        if current < low - 1e-3:
            return low
        return current

    return clipped(v_cmd, v_now, "linear"), clipped(w_cmd, w_now, "angular")


def test_window_reference_random() -> None:
    rng = np.random.default_rng(20260810)
    limits = DEFAULT_CONFIG.limits_dict()
    for _ in range(2000):
        v_now = float(rng.uniform(-0.75, 0.75))
        w_now = float(rng.uniform(-1.5, 1.5))
        v_cmd = float(rng.uniform(-0.75, 0.75))
        w_cmd = float(rng.uniform(-1.5, 1.5))
        dt = float(rng.uniform(0.005, 0.1))
        assert compute_dynamic_window(v_now, w_now, limits, dt) == _reference_dynamic(v_now, w_now, limits, dt)
        assert evaluate_velocity_constraints(v_cmd, w_cmd, v_now, w_now, limits, dt) == _reference_violation(
            v_cmd, w_cmd, v_now, w_now, limits, dt
        )
        assert calc_actual_velocity(v_cmd, w_cmd, v_now, w_now, limits, dt) == _reference_actual(
            v_cmd, w_cmd, v_now, w_now, limits, dt
        )


def test_smoke() -> None:
    with __import__("tempfile").TemporaryDirectory(prefix="dwpp-simulator-test-") as directory:
        summary = run_smoke(Path(directory))
        assert summary["passed"]


def main() -> int:
    test_window_reference_random()
    print("PASS: dynamic-window and violation math matches the validated reference on 2000 random samples")
    test_smoke()
    print("PASS: smoke ordering and DWPP-zero-violation assertions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
