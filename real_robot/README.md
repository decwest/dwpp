# Real-robot experiments

Data and scripts behind Section 4 of the article (Figs. 7–11, Tables 2–6).

## Layout

```
data/
├── exp1_path_tracking/<PathA|PathB|PathC>/<PP|APP|RPP|DWPP|MPPI>/
│   ├── *.csv                 five path-tracking trials, one row per 30 Hz control cycle
│   └── timing/*_timing.csv   per-cycle computation time
└── exp2_corridor/
    ├── <RPP|DWPP>/           five confined-corridor trials (*.csv) and their timing/
    └── map/                  occupancy grid (map.pgm, map.yaml) and reference path (fixed_plan.csv)
scripts/
├── analyze_revision_experiments.py   regenerates the figures and tables listed below
├── analysis_lib.py                   reference paths, metrics, and plotting helpers
└── estimate_amcl_noise.py            AMCL correction statistics quoted in Section 4.2
analysis_out/                         output of the scripts (ignored by git)
```

For the PP family the computation time was measured in separate runs; the logs of those runs sit next to their timing files in `timing/`. Figures rendered when the path-tracking data were acquired are kept alongside the data.

## Reproducing the figures and tables

Requires Python 3 with numpy, pandas, matplotlib, and scipy. From the repository root:

```bash
python3 real_robot/scripts/analyze_revision_experiments.py
```

Outputs are written to `real_robot/analysis_out/`.

| Output | Article |
|---|---|
| `velocity_violation_*`, `all_tracking_errors_*`, `max_tracking_errors_*`, `travel_time_*` | Tables 2–5 |
| `computation_time_table.*` | Computation times in Section 4.2 |
| `Path*_path_comparison.pdf`, `Path*_<Ctrl>_velocity.pdf` | Figs. 7–9 |
| `exp2_corridor_table.*` | Table 6 |
| `exp2_corridor_speed_*.pdf`, `exp2_corridor_vcmd_*.pdf` | Figs. 10–11 |

`python3 real_robot/scripts/estimate_amcl_noise.py` writes the AMCL statistics to `analysis_out/amcl_noise_stats.csv`.
