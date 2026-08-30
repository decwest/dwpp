# DWPP kinematic simulator

This self-contained package ports the Nav2 regulated-pure-pursuit and dynamic-window command laws used for the journal experiments. It generates deterministic CSV tables, LaTeX table material, and PDF/PNG figures under `results/simulation/`; it does not import ROS or alter the conference-paper simulator in `scripts/`.

Run every study from the repository root:

```bash
PYTHONPATH=. uv run python simulator/run_studies.py --study all
```

Run one study with `--study smoke|app|rpp|noise|lookahead|isotime`. The smoke study exits nonzero unless PathC has `PP > APP > RPP > DWPP` mean tracking error and DWPP has exactly zero constraint violations on every path.

Run the minimal reference-port and smoke assertions with:

```bash
PYTHONPATH=. uv run python simulator/test_simulator.py
```

The pose controller uses the previous base-accepted velocity. Controller output is evaluated as `v_cmd`; the validated `calc_actual_velocity` projection applies the configured acceleration and absolute limits, and that accepted velocity is applied by the exact unicycle arc step without actuator lag. Noise is sampled independently at each controller update with seeds 0--19. The reported lookahead practical lower bound is the smallest `L` with 100% goal completion whose mean tracking error is within 10% of the best fully successful setting at the same noise level.

Generated outputs are organized by study:

- `smoke/`: `smoke_table.{csv,tex}` for all three validation paths.
- `app/`: `app_table.{csv,tex}`, `app_tradeoff`, `app_violation_vs_lookahead_time`, and `app_trajectories` (figures are PDF and PNG).
- `rpp/`: `rpp_table.{csv,tex}`, `rpp_tradeoff`, `rpp_violation_vs_min_radius`, and `rpp_trajectories` (figures are PDF and PNG).
- `noise/`: `noise_table.{csv,tex}`, `noise_mean_error`, `noise_violation_ratio`, and `noise_trajectories` (figures are PDF and PNG).
- `lookahead/`: `lookahead_table.{csv,tex}`, `lookahead_practical_bounds.{csv,tex}`, `lookahead_tradeoff`, and `lookahead_trajectories` (figures are PDF and PNG). The twin-axis tradeoff replaces the former separate mean-error and travel-time figures.
- `isotime/`: `isotime_table.{csv,tex}` and `isotime_trajectories.{pdf,png}`.

The paper's simulation studies and their tables use PathC only; the smoke study intentionally retains PathA, PathB, and PathC as an internal validation sweep.
