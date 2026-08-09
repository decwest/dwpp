# DWPP journal-revision kinematic studies

This self-contained package ports the Nav2 regulated-pure-pursuit and dynamic-window command laws used for the journal experiments. It generates deterministic CSV tables, LaTeX `tabular` fragments, and PDF/PNG figures under `results/revision_sim/`; it does not import ROS or alter the conference-paper simulator in `scripts/`.

Run every study from the repository root:

```bash
uv run python scripts/revision/run_studies.py --study all
```

Run one study with `--study smoke|app|rpp|noise|lookahead|isotime`. The smoke study exits nonzero unless PathC has `PP > APP > RPP > DWPP` mean tracking error and DWPP has exactly zero constraint violations on every path.

Run the minimal reference-port and smoke assertions with:

```bash
uv run python scripts/revision/test_revision.py
```

The pose controller uses the previous base-accepted velocity. Controller output is evaluated as `v_cmd`; the validated `calc_actual_velocity` projection applies the frozen acceleration and absolute limits, and that accepted velocity is applied by the exact unicycle arc step without actuator lag. Noise is sampled independently at each controller update with seeds 0--19. The reported lookahead practical lower bound is the smallest `L` with 100% goal completion whose mean tracking error is within 10% of the best fully successful setting at the same noise level.
