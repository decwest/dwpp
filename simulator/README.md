# DWPP kinematic simulator

A self-contained Python port (no ROS) of the Nav2 pure pursuit controllers and DWPP, used for the simulation studies of Section 5 of the article. Outputs are deterministic and go to `results/simulation/<study>/`.

From the repository root:

```bash
PYTHONPATH=. uv run python simulator/run_studies.py --study all    # or one of the studies below
PYTHONPATH=. uv run python simulator/test_simulator.py             # minimal assertions
```

| Study | Article |
|---|---|
| `app` | Table 7, Fig. 12 |
| `rpp` | Table 8, Fig. 13 |
| `isotime` | Table 9 |
| `noise` | Table 10, Fig. 15 |
| `lookahead` | Table 11, Fig. 16 |
| `smoke` | validation only: exits nonzero unless Path C gives `PP > APP > RPP > DWPP` in mean tracking error and DWPP has zero constraint violations on every path |

The studies of the article use Path C only; `smoke` runs Paths A–C.
