# DWPP: Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints

DWPP is a pure pursuit variant that computes velocity commands in the velocity space (the v–ω plane) and selects the command within the dynamic window closest to the path-tracking condition ω = κv, so that the commands always respect the velocity and acceleration constraints.

## 🟢 Nav2 implementations

- **Official Nav2 (ROS 2 Lyrical and later):** DWPP is integrated into the official Nav2 repository.
  👉 https://github.com/ros-navigation/navigation2
- **ROS 2 Humble / Jazzy:** a standalone Nav2 plugin is available.
  👉 https://github.com/Decwest/nav2_dynamic_window_pure_pursuit_controller

## Simulator quickstart

The kinematic simulator used for the simulation studies in the paper lives in [`simulator/`](simulator/). It is a self-contained Python package (no ROS required) that reproduces the paper's tables and figures deterministically.

```bash
# run all studies (outputs under results/simulation/)
PYTHONPATH=. uv run python simulator/run_studies.py --study all

# run a single study
PYTHONPATH=. uv run python simulator/run_studies.py --study smoke  # smoke|app|rpp|noise|lookahead|isotime

# run the minimal assertions
PYTHONPATH=. uv run python simulator/test_simulator.py
```

See [`simulator/README.md`](simulator/README.md) for details.

## Folder Structure

- `simulator/`
  Kinematic simulator and simulation studies used in the paper (see above).

- `scripts/`
  Simulation scripts for various pure pursuit methods from the conference-paper experiments. You can refer to these to understand how DWPP can be implemented.

- `results/`
  Result figures and tables used in the paper.

## Citation

If you use this code, please cite the following paper:

> Fumiya Ohnishi, Masaki Takahashi, “Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints”, Proceedings of the 19th International Conference on Intelligent Autonomous Systems, Genoa, Italy, 2025.
