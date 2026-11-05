#!/usr/bin/env python3
"""
DWPP RAS revision 実験の一括解析スクリプト。

これ1本を叩くだけで論文用の表(CSV + LaTeX断片)と図(PDF/PNG)が out_dir に出力される:

  python3 real_robot/scripts/analyze_revision_experiments.py   # from the repository root

出力先は既定で real_robot/analysis_out/ (gitignore 済み)。

セクション:
  exp1   : 実①  追従指標の統合表 (凍結データ PP/APP/RPP/DWPP + 新規 MPPI) と経路比較図・速度プロファイル図
  timing : 計算時間表 (exp1_path_tracking 配下の <Path>/<手法>/timing/*_timing.csv を
           手法別に集計、論文掲載用)
  exp2   : 実②  障害物環境 (Corridor) の RPP vs DWPP 比較表とキー図 (v_cmd と dynamic window 上下限)

データが無いセクションは警告してスキップする(実験前でも凍結データ部分だけ動く)。
集計は既存ノートブック dwpp_real_experiment_analysis.ipynb と同一のセマンティクス
(map_base_* を試行開始点で再アンカー、cdist 最近傍距離、mean/std は ddof=0)。
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
import analysis_lib as al  # noqa: E402

CONTROL_PERIOD_MS = 1000.0 / 30.0
FROZEN_CONTROLLERS = ["PP", "APP", "RPP", "DWPP"]
PATH_LIST = ["PathA", "PathB", "PathC"]
EXP2_CONTROLLERS = ["RPP", "DWPP"]
EXP2_PATH_LABEL = "Corridor"
EXP2_COLLISION_DIST = 0.25  # scan_min_dist がこれ未満なら衝突疑いフラグ
V_MAX = 0.50
W_MAX = 1.0

if "MPPI" not in al.color_dict:
    al.color_dict["MPPI"] = "purple"


def log(msg, summary_lines=None):
    print(msg)
    if summary_lines is not None:
        summary_lines.append(msg)


def corridor_reference_path(length=9.0, n_points=900):
    xs = np.linspace(0.0, length, n_points)
    return np.c_[xs, np.zeros_like(xs)]


def trial_metrics(csv_path: Path, reference: np.ndarray, anchor_to_start: bool = True) -> dict:
    """ノートブック cell 4 と同一の per-trial 指標 + 実②向け追加指標"""
    df = pd.read_csv(csv_path)

    # controller_server はゴール到達時にゼロ指令を発行し、GUIレコーダはそれも
    # 記録する(プラグイン内蔵CSVには存在しない行)。末尾の完全ゼロ指令行を
    # 除外して凍結データと意味論を揃える(急停止が違反として混入するのを防ぐ。
    # 検証済み: これで pp_time の DWPP 違反率は全経路 0.000% になり、凍結CSV
    # 60本には末尾ゼロ行が無いため凍結テーブルの再現には影響しない)。
    nonzero = df.index[(df["v_cmd"] != 0.0) | (df["w_cmd"] != 0.0)]
    if len(nonzero) and nonzero[-1] + 1 < len(df):
        df = df.loc[:nonzero[-1]]

    t = df["sec"].to_numpy() + df["nsec"].to_numpy() * 1e-9
    t = t - t[0]

    if anchor_to_start:
        # ロボット原点アンカーの経路 (実①の折れ線): 試行開始姿勢基準に変換
        x, y, _ = al.transform_pose_to_path_origin(
            df["map_base_x"].to_numpy(), df["map_base_y"].to_numpy(), df["map_base_yaw"].to_numpy(),
            df["map_base_x"].to_numpy()[0], df["map_base_y"].to_numpy()[0], df["map_base_yaw"].to_numpy()[0],
        )
    else:
        # map 座標系の固定経路 (実②の fixed_plan): map 姿勢をそのまま比較
        x = df["map_base_x"].to_numpy()
        y = df["map_base_y"].to_numpy()
    robot_path = np.vstack((x, y)).T
    tracking_errors = al.calc_tracking_error(robot_path, reference)

    metrics = {
        "csv": csv_path,
        "df": df,
        "t": t,
        "x": x,
        "y": y,
        "violation_rate": al.calc_violation_rate(df["velocity_violation"].to_numpy()),
        "tracking_errors": tracking_errors,
        "travel_time": al.measure_travel_time(t),
        "distance": float(np.sum(np.hypot(np.diff(x), np.diff(y)))),
    }
    if "scan_min_dist" in df.columns:
        scan = df["scan_min_dist"].to_numpy()
        scan = scan[np.isfinite(scan)]
        metrics["scan_min_dist_mean"] = float(scan.mean()) if scan.size else float("nan")
        metrics["scan_min_dist_min"] = float(scan.min()) if scan.size else float("nan")
    return metrics


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def write_table(df: pd.DataFrame, out_dir: Path, name: str, float_fmt="%.4f"):
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, float_format=float_fmt)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(df.to_latex(float_format=lambda v: float_fmt % v))
    print(f"  wrote {csv_path.name} / {tex_path.name}")


def collect_trials(data_dir: Path, reference, summary, anchor_to_start: bool = True):
    trials = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        try:
            trials.append(trial_metrics(csv_path, reference, anchor_to_start=anchor_to_start))
        except Exception as exc:
            log(f"  WARN: failed to read {csv_path}: {exc}", summary)
    return trials


# ---------------------------------------------------- paper-style figure helpers

# (a) 経路比較図のパネル構成: A/B は手法ごとに5枚横並び、C は PP family + MPPI の2枚
PATH_COMPARISON_LAYOUTS = {
    "PathA": [["PP"], ["APP"], ["RPP"], ["DWPP"], ["MPPI"]],
    "PathB": [["PP"], ["APP"], ["RPP"], ["DWPP"], ["MPPI"]],
    "PathC": [["PP", "APP", "RPP", "DWPP"], ["MPPI"]],
}


def plot_path_comparison_paper(path_name, all_data, controllers, out_dir):
    """論文 Fig 7-9(a) の経路比較図 (notebook cell 11 の描画規約を踏襲)。
    PATH_COMPARISON_LAYOUTS に従いパネル分割し、各パネルに該当手法の全試行と
    参照経路 (最前面) を描く。凡例は別ファイル path_comparison_label.png。"""
    layout = PATH_COMPARISON_LAYOUTS.get(path_name, [[c] for c in controllers])
    groups = [[c for c in g if c in controllers] for g in layout]
    groups = [g for g in groups if g]
    ref = al.reference_path[path_name]

    # 参照経路レンジから equal-aspect 前提の figsize を決める (高さ 4.2 in 基準)
    pad = 0.6
    x_range = float(ref[:, 0].max() - ref[:, 0].min()) + 2 * pad
    y_range = float(ref[:, 1].max() - ref[:, 1].min()) + 2 * pad
    height = 3.2
    panel_w = height * y_range / x_range
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(panel_w * len(groups) + 0.9, height + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, group in zip(axes[0], groups):
        for controller in group:
            for tr in all_data[path_name].get(controller, []):
                ax.plot(tr["y"], tr["x"], color=al.color_dict[controller], linewidth=0.5)
        ax.plot(ref[:, 1], ref[:, 0], "k--", linewidth=1, alpha=0.7)
        ax.set_title(group[0] if len(group) == 1 else "PP family")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    axes[0][0].invert_xaxis()  # sharex なので全パネルに適用される
    axes[0][0].set_ylabel("$x$ [m]")
    fig.supxlabel("$y$ [m]")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{path_name}_path_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_profile_paper(tr, path_name, controller, out_dir):
    """論文 Fig 7-9(b)-(e) と同一スタイル (notebook cell 13 準拠) の速度プロファイル。
    出荷済みの凍結図は cell 20 の font.size=20 が有効な状態で生成されているため、
    ここでも 20 に合わせる (page size ≈ 472x184 pt, ω目盛 = -1/0/1)。"""
    df_t = tr["df"]
    with plt.rc_context({"font.size": 20}):
        _plot_velocity_profile_paper_inner(tr, df_t, path_name, controller, out_dir)


def _plot_velocity_profile_paper_inner(tr, df_t, path_name, controller, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.plot(tr["t"], df_t["v_cmd"], "-", color="red", linewidth=1, alpha=0.8)
    ax1.plot(tr["t"], df_t["v_real"], "-", color="blue", linewidth=1)
    ax1.axhline(y=V_MAX, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel("$v$ [m/s]")
    ax1.set_yticks([0, 0.25, 0.50])
    ax1.grid(True, alpha=0.3)
    ax2.plot(tr["t"], df_t["w_cmd"], "-", color="red", linewidth=1, alpha=0.8)
    ax2.plot(tr["t"], df_t["w_real"], "-", color="blue", linewidth=1)
    ax2.axhline(y=W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax2.axhline(y=-W_MAX, color="black", linestyle="--", linewidth=2, alpha=0.7)
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("$\\omega$ [rad/s]")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{path_name}_{controller}_velocity.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_path_comparison_legend(controllers, out_dir):
    """Fig 7-9(a) の上に置く共有凡例 (path_comparison_label.png) を再生成する。"""
    handles = [Line2D([], [], color="black", linestyle="--", linewidth=3, label="Path")]
    handles += [Line2D([], [], color=al.color_dict[c], linewidth=3, label=c) for c in controllers]
    fig = plt.figure(figsize=(9, 0.5))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=True,
              edgecolor="black", handlelength=1.6, columnspacing=1.3, handletextpad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"path_comparison_label.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def load_occupancy_map(map_yaml: Path):
    """ROS の map.yaml + P5 PGM を numpy で読み、(img, extent) を返す。
    extent は map 座標系での (xmin, xmax, ymin, ymax)。PGM の行0が地図上端
    なので imshow は origin='upper' で描くこと。"""
    text = map_yaml.read_text()
    res = float(re.search(r"^resolution:\s*([\d.eE+-]+)", text, re.M).group(1))
    ox, oy = [float(v) for v in
              re.search(r"^origin:\s*\[([^\]]+)\]", text, re.M).group(1).split(",")[:2]]
    image = re.search(r"^image:\s*(\S+)", text, re.M).group(1)
    data = (map_yaml.parent / image).read_bytes()
    m = re.match(rb"P5\s+(?:#[^\n]*\s+)*(\d+)\s+(\d+)\s+(\d+)\s", data)
    w, h = int(m.group(1)), int(m.group(2))
    img = np.frombuffer(data, dtype=np.uint8, count=w * h, offset=m.end()).reshape(h, w)
    extent = (ox, ox + w * res, oy, oy + h * res)
    return img, extent


def draw_corridor_map(ax, img, extent, plan_xy):
    """corridor 図共通の地図背景 + 軸設定 (経路 x 範囲 +0.8 m マージンにクロップ)。"""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, origin="upper", extent=extent,
              interpolation="nearest", zorder=0)
    ax.set_xlim(plan_xy[:, 0].min() - 0.8, plan_xy[:, 0].max() + 0.8)
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_ylabel("$y$ [m]")


# ---------------------------------------------------------------- exp1

def run_exp1(exp1_dir: Path, out_dir: Path, summary):
    log("== exp1: tracking metrics (PP family + MPPI) ==", summary)

    all_data = defaultdict(dict)  # [path][controller] -> list of trial metrics
    controllers = list(FROZEN_CONTROLLERS)
    if any((exp1_dir / p / "MPPI").is_dir() for p in PATH_LIST):
        controllers.append("MPPI")
    else:
        log(f"  WARN: no MPPI data under {exp1_dir} - tables will cover the PP family only", summary)

    for path_name in PATH_LIST:
        reference = al.reference_path[path_name]
        for controller in controllers:
            trials = collect_trials(exp1_dir / path_name / controller, reference, summary)
            all_data[path_name][controller] = trials
            log(f"  {path_name}/{controller}: {len(trials)} trials", summary)

    # ---- aggregate tables (notebook cell 6/8 semantics)
    records = []
    for path_name in PATH_LIST:
        for controller in controllers:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            vv_m, vv_s = mean_std([tr["violation_rate"] for tr in trials])
            me_m, me_s = mean_std([tr["tracking_errors"].mean() for tr in trials])
            mx_m, mx_s = mean_std([tr["tracking_errors"].max() for tr in trials])
            tt_m, tt_s = mean_std([tr["travel_time"] for tr in trials])
            records.append({
                "path": path_name, "controller": controller,
                "velocity_violation_mean": vv_m, "velocity_violation_std": vv_s,
                "mean_tracking_errors_mean": me_m, "mean_tracking_errors_std": me_s,
                "max_tracking_errors_mean": mx_m, "max_tracking_errors_std": mx_s,
                "travel_time_mean": tt_m, "travel_time_std": tt_s,
                "n_trials": len(trials),
            })
    if not records:
        log("  WARN: no exp1 data at all - skipped", summary)
        return
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "exp1_per_condition_records.csv", index=False)

    col_order = [c for c in controllers]
    for value, name in [
        ("velocity_violation_mean", "velocity_violation_mean_table"),
        ("velocity_violation_std", "velocity_violation_std_table"),
        ("mean_tracking_errors_mean", "all_tracking_errors_mean_table"),
        ("mean_tracking_errors_std", "all_tracking_errors_std_table"),
        ("max_tracking_errors_mean", "max_tracking_errors_mean_table"),
        ("max_tracking_errors_std", "max_tracking_errors_std_table"),
        ("travel_time_mean", "travel_time_mean_table"),
        ("travel_time_std", "travel_time_std_table"),
    ]:
        table = df.pivot(index="path", columns="controller", values=value)
        table = table.reindex(columns=[c for c in col_order if c in table.columns])
        write_table(table, out_dir, name)

    # ---- figures (論文 Fig 7-9 スタイル: 全試行の経路比較 + 各手法の代表試行の速度プロファイル)
    rep_idx = 3  # ノートブック踏襲: 4番目の試行を代表に(足りなければ末尾)
    for path_name in PATH_LIST:
        plot_path_comparison_paper(path_name, all_data, controllers, out_dir)
        for controller in controllers:
            trials = all_data[path_name].get(controller, [])
            if not trials:
                continue
            tr = trials[min(rep_idx, len(trials) - 1)]
            plot_velocity_profile_paper(tr, path_name, controller, out_dir)
    make_path_comparison_legend(controllers, out_dir)
    log(f"  exp1 figures written to {out_dir}", summary)


# ---------------------------------------------------------------- timing

def run_timing(exp1_dir: Path, out_dir: Path, summary):
    """exp1_dir 配下の <Path>/<手法>/timing/*_timing.csv を全て集計する
    (MPPI は追従試行と同時に計測、PP 系は計時専用の再走行で計測。
    再走行の走行ログも同じ timing/ に置かれるが *_timing.csv のみ読む)"""
    log("== timing: per-controller computation time ==", summary)
    timing_files = sorted(exp1_dir.glob("*/*/timing/*_timing.csv"))
    log(f"  {exp1_dir.name}: {len(timing_files)} timing files", summary)
    if not timing_files:
        log(f"  WARN: no timing CSVs under {exp1_dir} - skipped", summary)
        return
    frames = []
    for f in timing_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = str(f.relative_to(DATA_ROOT)) if DATA_ROOT in f.parents else str(f)
            frames.append(df)
        except Exception as exc:
            log(f"  WARN: failed to read {f}: {exc}", summary)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)

    records = []
    for controller, g in df.groupby("controller_id"):
        ok = g[g["success"].astype(bool)] if "success" in g.columns else g
        ms = ok["compute_time_ms"].to_numpy(dtype=float)
        records.append({
            "controller": controller,
            "n_trials": g["source_file"].nunique(),
            "n_cycles": len(ms),
            "mean_ms": ms.mean(),
            "std_ms": ms.std(ddof=0),
            "max_ms": ms.max(),
            "p99_ms": float(np.percentile(ms, 99)),
            "mean_ratio_to_period": ms.mean() / CONTROL_PERIOD_MS,
        })
    table = pd.DataFrame(records).set_index("controller")
    # 論文の手法順に並べる
    order = [c for c in FROZEN_CONTROLLERS + ["MPPI"] if c in table.index]
    table = table.reindex(order + [c for c in table.index if c not in order])
    write_table(table, out_dir, "computation_time_table")
    log(f"  pooled {len(df)} cycles from {len(timing_files)} trials "
        f"(control period {CONTROL_PERIOD_MS:.2f} ms)", summary)


# ---------------------------------------------------------------- exp2

def run_exp2(exp2_dir: Path, out_dir: Path, summary, plan_csv: Path = None,
             map_yaml: Path = None, label: str = EXP2_PATH_LABEL,
             r_min: float = 1.5, d_prox: float = 0.7):
    """label は出力ファイルの stem (exp2_<label>) にのみ使う"""
    log("== exp2: obstacle corridor (RPP vs DWPP, proximity heuristic ON) ==", summary)
    base = exp2_dir

    if plan_csv is not None and Path(plan_csv).is_file():
        # 凍結済み map 座標系の参照経路 (実験で全試行に共通配信した fixed plan)
        plan = pd.read_csv(plan_csv)
        reference = np.c_[plan["x"].to_numpy(), plan["y"].to_numpy()]
        anchor = False
        log(f"  reference: fixed plan {plan_csv} ({len(reference)} pts, map frame)", summary)
    else:
        reference = corridor_reference_path()
        anchor = True
        log("  reference: straight 9 m corridor line (start-anchored)", summary)

    all_trials = {}
    for controller in EXP2_CONTROLLERS:
        trials = collect_trials(base / controller, reference, summary, anchor_to_start=anchor)
        all_trials[controller] = trials
        log(f"  {label}/{controller}: {len(trials)} trials", summary)
    if not any(all_trials.values()):
        log(f"  WARN: no exp2 data under {base} - skipped", summary)
        return

    # ---- per-trial + aggregate table (RPP論文 Table 2 準拠 + 制約違反率)
    per_trial_records = []
    agg_records = []
    for controller, trials in all_trials.items():
        if not trials:
            continue
        for i, tr in enumerate(trials):
            per_trial_records.append({
                "controller": controller, "trial": i,
                "time_s": tr["travel_time"],
                "distance_m": tr["distance"],
                "avg_speed_mps": tr["distance"] / tr["travel_time"] if tr["travel_time"] > 0 else float("nan"),
                "avg_dist_to_path_m": float(tr["tracking_errors"].mean()),
                "max_dist_to_path_m": float(tr["tracking_errors"].max()),
                "avg_dist_to_obstacle_m": tr.get("scan_min_dist_mean", float("nan")),
                "min_dist_to_obstacle_m": tr.get("scan_min_dist_min", float("nan")),
                "collision_flag": int(tr.get("scan_min_dist_min", float("inf")) < EXP2_COLLISION_DIST),
                "violation_rate_pct": tr["violation_rate"],
            })
        rec = {"controller": controller, "n_trials": len(trials)}
        for key, values in [
            ("time_s", [tr["travel_time"] for tr in trials]),
            ("distance_m", [tr["distance"] for tr in trials]),
            ("avg_speed_mps", [tr["distance"] / tr["travel_time"] for tr in trials]),
            ("avg_dist_to_path_m", [float(tr["tracking_errors"].mean()) for tr in trials]),
            ("max_dist_to_path_m", [float(tr["tracking_errors"].max()) for tr in trials]),
            ("avg_dist_to_obstacle_m", [tr.get("scan_min_dist_mean", float("nan")) for tr in trials]),
            ("min_dist_to_obstacle_m", [tr.get("scan_min_dist_min", float("nan")) for tr in trials]),
            ("violation_rate_pct", [tr["violation_rate"] for tr in trials]),
        ]:
            m, s = mean_std(values)
            rec[f"{key}_mean"] = m
            rec[f"{key}_std"] = s
        rec["collisions_total"] = sum(
            1 for tr in trials if tr.get("scan_min_dist_min", float("inf")) < EXP2_COLLISION_DIST
        )
        agg_records.append(rec)

    stem = f"exp2_{label.lower()}"
    pd.DataFrame(per_trial_records).to_csv(out_dir / f"{stem}_per_trial_records.csv", index=False)
    write_table(pd.DataFrame(agg_records).set_index("controller").T, out_dir, f"{stem}_table")
    log("  NOTE: collision_flag は scan_min_dist < "
        f"{EXP2_COLLISION_DIST} m の自動判定。実験ノートの手動カウントと突き合わせること", summary)

    # ---- figures
    rep_trials = {c: trials[min(3, len(trials) - 1)]
                  for c, trials in all_trials.items() if trials}

    # (1)(2) 地図ベースの図は map.yaml と fixed_plan がある場合のみ
    if map_yaml is not None and Path(map_yaml).is_file() and not anchor:
        img, extent = load_occupancy_map(Path(map_yaml))
        plot_corridor_map_plan(reference, img, extent, out_dir, stem)
        plot_corridor_speed_colored(rep_trials, reference, img, extent, out_dir, stem)
    else:
        log(f"  WARN: map yaml missing or no fixed plan ({map_yaml}) - "
            "map-based corridor figures skipped", summary)

    # (3) 速度指令・曲率指令・障害物距離の 2x3 パネル
    plot_corridor_vcmd_panels(rep_trials, out_dir, stem, r_min=r_min, d_prox=d_prox)
    log(f"  exp2 figures written to {out_dir}", summary)


def plot_corridor_map_plan(plan_xy, img, extent, out_dir, stem):
    """実②の設定図: 占有格子地図 + 大域経路 (fixed plan) + start/goal。"""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    draw_corridor_map(ax, img, extent, plan_xy)
    ax.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", linewidth=1.5, label="Reference path", zorder=2)
    ax.plot(plan_xy[0, 0], plan_xy[0, 1], "o", color="limegreen", mec="black", ms=8,
            zorder=4, label="Start")
    ax.plot(plan_xy[-1, 0], plan_xy[-1, 1], "*", color="red", mec="black", ms=12,
            zorder=4, label="Goal")
    ax.set_xlabel("$x$ [m]")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_map_plan.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_corridor_speed_colored(rep_trials, plan_xy, img, extent, out_dir, stem):
    """実②の追従軌跡を実現速度 v_real で色付けした図 (制御器ごとに1枚)。"""
    norm = plt.Normalize(0.0, V_MAX)
    for controller in EXP2_CONTROLLERS:
        if controller not in rep_trials:
            continue
        tr = rep_trials[controller]
        fig, ax = plt.subplots(figsize=(8, 2.4), layout="constrained")
        draw_corridor_map(ax, img, extent, plan_xy)
        ax.plot(plan_xy[:, 0], plan_xy[:, 1], "k--", linewidth=1, alpha=0.7, zorder=2)
        v = tr["df"]["v_real"].to_numpy(dtype=float)
        pts = np.column_stack([tr["x"], tr["y"]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_v = 0.5 * (v[:-1] + v[1:])
        ok = np.isfinite(seg_v) & np.isfinite(segs).all(axis=(1, 2))
        lc = LineCollection(segs[ok], cmap="turbo", norm=norm, linewidth=2.5,
                            capstyle="round", zorder=3)
        lc.set_array(seg_v[ok])
        ax.add_collection(lc)
        ax.set_xlabel("$x$ [m]")
        fig.colorbar(lc, ax=ax, orientation="vertical", fraction=0.03, pad=0.02,
                     label="Linear velocity [m/s]")
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}_speed_{controller.lower()}.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_corridor_vcmd_panels(rep_trials, out_dir, stem, r_min=1.5, d_prox=0.7):
    """実②のキー図 (制御器ごとにパネル別の3枚): (1) 速度指令(赤)+実現速度(青)、
    ヒューリスティック発動区間の指令値は原因パネルと同色で上描き
    (曲率=オレンジ、近接=green、両方発動時は縮小率が小さい方)。
    説明は共有凡例ストリップ {stem}_vcmd_legend.pdf に集約(パネル内凡例なし)
    (2) 曲率指令 kappa = w_cmd/v_cmd と閾値 ±1/R_min
    (3) 最小障害物距離と cost-scaling 距離 d_prox。横軸は時間。"""
    CURV_COLOR = "tab:orange"
    PROX_COLOR = "tab:green"
    for controller in EXP2_CONTROLLERS:
        if controller not in rep_trials:
            continue
        tr = rep_trials[controller]
        df_t, t = tr["df"], tr["t"]
        v_cmd = df_t["v_cmd"].to_numpy(dtype=float)
        w_cmd = df_t["w_cmd"].to_numpy(dtype=float)
        v_real = df_t["v_real"].to_numpy(dtype=float)
        dist = df_t["scan_min_dist"].to_numpy(dtype=float)

        kappa = np.where(np.abs(v_cmd) >= 0.05,
                         w_cmd / np.where(v_cmd == 0.0, np.nan, v_cmd), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.where(np.abs(kappa) > 1e-6, 1.0 / np.abs(kappa), np.inf)
        curv_scale = np.where(radius < r_min, radius / r_min, np.inf)
        prox_scale = np.where(dist < d_prox, dist / d_prox, np.inf)
        cause = np.zeros(len(t), dtype=int)
        cause[(curv_scale < np.inf) & (curv_scale <= prox_scale)] = 1
        cause[(prox_scale < np.inf) & (prox_scale < curv_scale)] = 2

        # (1) velocity profile: cause-colored halo bands under the lines
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        ax.plot(t, np.where(cause == 1, v_cmd, np.nan), color=CURV_COLOR,
                linewidth=6, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.plot(t, np.where(cause == 2, v_cmd, np.nan), color=PROX_COLOR,
                linewidth=6, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.plot(t, v_real, color="blue", linewidth=1.0, zorder=2)
        ax.plot(t, v_cmd, color="red", linewidth=1.2, zorder=3)
        ax.set_ylabel(r"$v$ [m/s]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}_vcmd_{controller.lower()}_vel.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

        # (2) commanded curvature profile
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        ax.plot(t, kappa, color=CURV_COLOR, linewidth=1.2)
        ax.axhline(y=1.0 / r_min, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=-1.0 / r_min, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(r"$\kappa$ [1/m]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}_vcmd_{controller.lower()}_curv.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

        # (3) obstacle distance profile
        fig, ax = plt.subplots(figsize=(2.6, 2.2))
        ax.plot(t, dist, color=PROX_COLOR, linewidth=1.2)
        ax.axhline(y=d_prox, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(r"$d_O$ [m]")
        ax.set_xlabel("Time [s]")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}_vcmd_{controller.lower()}_dist.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    # shared legend strip: line styles, heuristic bands, and threshold lines
    handles = [
        Line2D([0], [0], color="red", linewidth=1.5, label="Command velocity"),
        Line2D([0], [0], color="blue", linewidth=1.5, label="Actual velocity"),
        Line2D([0], [0], color=CURV_COLOR, linewidth=6, alpha=0.35,
               solid_capstyle="round", label="Curvature heuristic active"),
        Line2D([0], [0], color=PROX_COLOR, linewidth=6, alpha=0.35,
               solid_capstyle="round", label="Proximity heuristic active"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5,
               label=r"Thresholds ($\pm 1/R_{\min}$, $d_{\mathrm{prox}}$)"),
    ]
    fig = plt.figure(figsize=(7.6, 0.8))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3, frameon=True, fontsize=9,
              handlelength=1.8, columnspacing=1.2, handletextpad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_vcmd_legend.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp1-dir", type=Path, default=DATA_ROOT / "exp1_path_tracking",
                        help="実① 経路追従データ (<PathA-C>/<PP|APP|RPP|DWPP|MPPI>/*.csv、"
                             "計算時間は同じ階層の timing/*_timing.csv)")
    parser.add_argument("--exp2-dir", type=Path, default=DATA_ROOT / "exp2_corridor",
                        help="実② corridor データ (<RPP|DWPP>/*.csv と map/)")
    parser.add_argument("--exp2-plan", type=Path, default=None,
                        help="実②の凍結参照経路 CSV (map 座標系, columns x,y[,yaw]; "
                             "default: <exp2-dir>/map/fixed_plan.csv)。"
                             "見つからない場合は直線コリドー参照にフォールバック")
    parser.add_argument("--exp2-map", type=Path, default=None,
                        help="実②の占有格子地図 map.yaml (default: --exp2-plan と同ディレクトリの map.yaml)")
    parser.add_argument("--exp2-rmin", type=float, default=1.5,
                        help="実②の曲率ヒューリスティック閾値 R_min [m] (図の閾値線と原因色分けに使用)")
    parser.add_argument("--exp2-dprox", type=float, default=0.7,
                        help="実②の近接ヒューリスティック cost-scaling 距離 d_prox [m]")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="出力先 (default: real_robot/analysis_out)")
    parser.add_argument("--only", choices=["exp1", "exp2", "timing", "legends"], action="append",
                        help="指定セクションのみ実行(複数指定可)。legends は共有凡例のみ再生成")
    args = parser.parse_args()

    out_dir = args.out_dir or (REPO_ROOT / "analysis_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = args.only or ["exp1", "timing", "exp2"]

    summary = []
    log(f"exp1_dir : {args.exp1_dir}", summary)
    log(f"exp2_dir : {args.exp2_dir}", summary)
    log(f"out_dir  : {out_dir}", summary)

    exp2_plan = args.exp2_plan or args.exp2_dir / "map" / "fixed_plan.csv"
    exp2_map = args.exp2_map or exp2_plan.parent / "map.yaml"

    if "exp1" in sections:
        run_exp1(args.exp1_dir, out_dir, summary)
    if "timing" in sections:
        run_timing(args.exp1_dir, out_dir, summary)
    if "exp2" in sections:
        run_exp2(args.exp2_dir, out_dir, summary, plan_csv=exp2_plan, map_yaml=exp2_map,
                 r_min=args.exp2_rmin, d_prox=args.exp2_dprox)
    if "legends" in sections and "exp1" not in sections:
        make_path_comparison_legend(FROZEN_CONTROLLERS + ["MPPI"], out_dir)
        log("== legends: path_comparison_label regenerated ==", summary)

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nDone. Outputs in {out_dir} (see summary.txt)")


if __name__ == "__main__":
    main()
