# DWPP: Dynamic Window Pure Pursuit Considering Velocity and Acceleration Constraints

DWPP is a pure pursuit variant that computes velocity commands in the velocity space (the v–ω plane) and selects the command within the dynamic window closest to the path-tracking condition ω = κv, so that the commands always respect the velocity and acceleration constraints.

This repository accompanies the article

> Fumiya Ohnishi, Masaki Takahashi, "DWPP: Dynamic window pure pursuit considering velocity and acceleration constraints," *Robotics and Autonomous Systems*, 2026.

## 🟢 Nav2 implementations

- **Official Nav2 (ROS 2 Lyrical and later):** DWPP is integrated into the official Nav2 repository.
  👉 https://github.com/ros-navigation/navigation2
- **ROS 2 Humble / Jazzy:** a standalone Nav2 plugin is available.
  👉 https://github.com/Decwest/nav2_dynamic_window_pure_pursuit_controller

## Contents

- [`simulator/`](simulator/): kinematic simulator and the simulation studies of the article (Section 5). Outputs are under `results/simulation/`.
- [`real_robot/`](real_robot/): data of the real-robot experiments (Section 4) and the scripts that regenerate Figs. 7–11 and Tables 2–6.

## Reproducing the results

```bash
# simulation studies (smoke|app|rpp|noise|lookahead|isotime|all) -> results/simulation/
PYTHONPATH=. uv run python simulator/run_studies.py --study all

# real-robot figures and tables -> real_robot/analysis_out/
python3 real_robot/scripts/analyze_revision_experiments.py
```

See [`simulator/README.md`](simulator/README.md) and [`real_robot/README.md`](real_robot/README.md) for details.

## Citation

> Fumiya Ohnishi, Masaki Takahashi, "DWPP: Dynamic window pure pursuit considering velocity and acceleration constraints," *Robotics and Autonomous Systems*, 2026.

Conference version: Fumiya Ohnishi, Masaki Takahashi, "Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints," Proceedings of the 19th International Conference on Intelligent Autonomous Systems, Genoa, Italy, 2025.
