#!/usr/bin/env python3
"""実機ログから AMCL 補正ノイズ (σ_xy, σ_θ) を実験グループ別に推定する。

実① (path tracking) と実② (corridor) は AMCL パラメータが異なるため、
グループを分けて推定する。採用試行は analyze_revision_experiments.py の
既定と同一:

  exp1: data/exp1_path_tracking/<PathA-C>/<PP|APP|RPP|DWPP|MPPI>/*.csv
  exp2: data/exp2_corridor/<RPP|DWPP>/*.csv

方法: map→odom 変換 (map_odom_* 列) が変化した行を AMCL 更新イベントとして
検出し、変換の変化を同一行のオドメトリ姿勢 o で合成して「ロボット位置での
補正ジャンプ」を計算する:

    δp = (p_new + R(ψ_new)·o) − (p_old + R(ψ_old)·o)
    δψ = wrap(ψ_new − ψ_old)

map_base 列は odom_base 列と非同期サンプリングされており (同一行でも時刻ずれ
があり実運動が漏れ込む)、ポーズ増分差分ベースの推定には使えないことを確認済み。
上式は固定の o に対する変換差分なので実運動が混入せず、yaw 補正のレバーアーム
効果 (odom 原点まわりの回転が並進に化ける分) も正しくロボット位置での値になる。

出力: analysis_out/amcl_noise_stats.csv + コンソール要約

usage: python3 real_robot/scripts/estimate_amcl_noise.py   # from the repository root
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

PATH_LIST = ["PathA", "PathB", "PathC"]
FROZEN_CONTROLLERS = ["PP", "APP", "RPP", "DWPP"]

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def trial_innovations(csv_path: Path):
    """1試行分の AMCL 更新ジャンプ (δx, δy, δψ) と所要時間・更新数を返す。"""
    df = pd.read_csv(csv_path)
    cols = ["odom_base_x", "odom_base_y",
            "map_odom_x", "map_odom_y", "map_odom_yaw"]
    df = df.dropna(subset=cols)
    if len(df) < 2:
        raise ValueError("too few rows")

    t = df["sec"].to_numpy() + 1e-9 * df["nsec"].to_numpy()
    duration = float(t[-1] - t[0])

    o = df[["odom_base_x", "odom_base_y"]].to_numpy()
    tp = df[["map_odom_x", "map_odom_y"]].to_numpy()
    ty = df["map_odom_yaw"].to_numpy()

    # 変換が (量子化された 6 桁表現で) 変化した行 = AMCL 更新イベント
    changed = np.any(np.diff(tp, axis=0) != 0.0, axis=1) | (np.diff(ty) != 0.0)
    idx = np.flatnonzero(changed) + 1
    if idx.size == 0:
        return np.empty((0, 2)), np.empty(0), duration

    # 変換差分を同一行のオドメトリ位置 o で合成 → ロボット位置での補正ジャンプ
    c_new, s_new = np.cos(ty[idx]), np.sin(ty[idx])
    c_old, s_old = np.cos(ty[idx - 1]), np.sin(ty[idx - 1])
    ox, oy_ = o[idx, 0], o[idx, 1]
    dp = np.c_[
        (tp[idx, 0] + c_new * ox - s_new * oy_) - (tp[idx - 1, 0] + c_old * ox - s_old * oy_),
        (tp[idx, 1] + s_new * ox + c_new * oy_) - (tp[idx - 1, 1] + s_old * ox + c_old * oy_),
    ]
    dpsi = wrap_angle(ty[idx] - ty[idx - 1])
    return dp, dpsi, duration


def collect_group(name: str, csv_paths, verbose=True):
    """グループの全試行からジャンプ統計を集計して1行分の dict を返す。"""
    dps, dpsis = [], []
    total_duration = 0.0
    n_trials = 0
    for p in csv_paths:
        try:
            dp, dpsi, duration = trial_innovations(p)
        except Exception as exc:
            print(f"  WARN: {p}: {exc}")
            continue
        dps.append(dp)
        dpsis.append(dpsi)
        total_duration += duration
        n_trials += 1
    dp = np.vstack(dps) if dps else np.empty((0, 2))
    dpsi = np.concatenate(dpsis) if dpsis else np.empty(0)
    n_jump = len(dpsi)

    xy = dp.ravel()  # 等方仮定で x/y 成分をプールした per-axis 統計
    row = {
        "group": name,
        "n_trials": n_trials,
        "duration_s": round(total_duration, 1),
        "n_updates": n_jump,
        "update_rate_hz": n_jump / total_duration if total_duration else float("nan"),
        "sigma_xy_m": float(np.sqrt(np.mean(xy ** 2))) if n_jump else float("nan"),
        "sigma_theta_rad": float(np.sqrt(np.mean(dpsi ** 2))) if n_jump else float("nan"),
        "mean_xy_m": float(np.mean(xy)) if n_jump else float("nan"),
        "mean_theta_rad": float(np.mean(dpsi)) if n_jump else float("nan"),
        "p95_abs_dp_m": float(np.percentile(np.hypot(dp[:, 0], dp[:, 1]), 95)) if n_jump else float("nan"),
        "p95_abs_dpsi_rad": float(np.percentile(np.abs(dpsi), 95)) if n_jump else float("nan"),
    }
    row["ratio_theta_over_xy"] = (
        row["sigma_theta_rad"] / row["sigma_xy_m"] if row["sigma_xy_m"] else float("nan"))
    if verbose:
        print(f"  {name}: {n_trials} trials, {total_duration:.0f} s, "
              f"{n_jump} updates ({row['update_rate_hz']:.2f} Hz), "
              f"sigma_xy={row['sigma_xy_m']*1000:.2f} mm, "
              f"sigma_theta={row['sigma_theta_rad']*1000:.2f} mrad, "
              f"ratio={row['ratio_theta_over_xy']:.3f}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp1-dir", type=Path,
                        default=REPO_ROOT / "data" / "exp1_path_tracking")
    parser.add_argument("--exp2-dir", type=Path,
                        default=REPO_ROOT / "data" / "exp2_corridor")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis_out")
    args = parser.parse_args()

    frozen = [p for path in PATH_LIST for ctrl in FROZEN_CONTROLLERS
              for p in sorted((args.exp1_dir / path / ctrl).glob("*.csv"))]
    mppi = [p for path in PATH_LIST
            for p in sorted((args.exp1_dir / path / "MPPI").glob("*.csv"))]
    exp2 = [p for ctrl in ["RPP", "DWPP"]
            for p in sorted((args.exp2_dir / ctrl).glob("*.csv"))]

    print("== AMCL correction-noise estimate (per experiment group) ==")
    rows = [
        collect_group("exp1_frozen_ppfamily", frozen),
        collect_group("exp1_mppi", mppi),
        collect_group("exp1_all", frozen + mppi),
        collect_group("exp2_corridor", exp2),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "amcl_noise_stats.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
