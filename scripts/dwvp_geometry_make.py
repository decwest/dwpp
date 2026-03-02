#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWVP geometry figure (TikZ-like oblique projection) for IROS paper.

- Two panels: (a) ray intersects Dynamic Window (AABB) -> choose on intersection segment
             (b) ray does not intersect -> choose closest feasible point (AABB projection)

Outputs:
  - dwvp_geometry_a.pdf / dwvp_geometry_a.png
  - dwvp_geometry_b.pdf / dwvp_geometry_b.png

Notes:
  * This uses the same oblique basis as your TikZ:
      x={(1.05cm,0cm)}, y={(0.60cm,0.35cm)}, z={(0cm,1.05cm)}
    i.e., a linear projection from (v_x, v_y, ω) to 2D.
  * Matplotlib mathtext does NOT support \\bm by default, so the labels use \\mathbf.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl

plt.rcParams["font.size"] = 12  # 全体のフォントサイズが変更されます。
plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.weight"] = "normal"  # フォントの太さを細字に設定
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
# ハッチ線の太さを細く
mpl.rcParams['hatch.linewidth'] = 0.5

# ----------------------------
# 2D projector (TikZ-like)
# ----------------------------
def make_projector(
    x_vec: tuple[float, float] = (1.05, 0.0),
    y_vec: tuple[float, float] = (0.60, 0.35),
    z_vec: tuple[float, float] = (0.0, 1.05),
):
    """
    Return a function proj(p3) that maps a 3D point p3=[x,y,z] to 2D.
    """
    M = np.array([x_vec, y_vec, z_vec], dtype=float).T  # shape (2,3)

    def proj(p3: np.ndarray) -> np.ndarray:
        p3 = np.asarray(p3, dtype=float).reshape(3, 1)
        return (M @ p3).ravel()

    return proj


PROJ = make_projector()


# ----------------------------
# Geometry utilities
# ----------------------------
def ray_aabb_intersection_alpha(v: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, alpha_min: float = 0.0):
    """
    Ray: p(alpha) = alpha * v, alpha >= alpha_min
    AABB: [box_min, box_max]

    Return: (hit: bool, alpha_low: float|None, alpha_high: float|None)
    """
    v = np.asarray(v, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)

    a_low = float(alpha_min)
    a_high = float("inf")

    for i in range(3):
        if abs(v[i]) < 1e-12:
            # Ray parallel to slab; origin must be inside that slab
            if 0.0 < box_min[i] or 0.0 > box_max[i]:
                return False, None, None
            continue

        a1 = box_min[i] / v[i]
        a2 = box_max[i] / v[i]
        a_enter = min(a1, a2)
        a_exit = max(a1, a2)

        a_low = max(a_low, a_enter)
        a_high = min(a_high, a_exit)

        if a_low > a_high:
            return False, None, None

    return True, a_low, a_high


def closest_point_on_aabb_to_ray(v: np.ndarray, box_min: np.ndarray, box_max: np.ndarray):
    """
    Solve: min_{alpha>=0, x in box} ||x - alpha v||^2.

    Approach:
      Minimize g(alpha)=dist^2(point alpha v, AABB) over alpha>=0.
      g(alpha) is piecewise quadratic with breakpoints at alpha where alpha*v hits AABB bounds.

    Return dict:
      {
        "alpha": alpha_star,
        "x_box": closest feasible point on AABB,
        "dist2": squared distance
      }
    """
    v = np.asarray(v, dtype=float)
    box_min = np.asarray(box_min, dtype=float)
    box_max = np.asarray(box_max, dtype=float)

    def clamp(p):
        return np.minimum(np.maximum(p, box_min), box_max)

    # Breakpoints where coordinate hits a bound
    bps = [0.0]
    for i in range(3):
        if abs(v[i]) < 1e-12:
            continue
        for bound in (box_min[i], box_max[i]):
            a = bound / v[i]
            if a >= 0:
                bps.append(float(a))
    bps = sorted(set(bps))

    intervals = [(bps[k], bps[k + 1]) for k in range(len(bps) - 1)]
    intervals.append((bps[-1], float("inf")))

    best = {"dist2": float("inf"), "alpha": None, "x_box": None}

    for left, right in intervals:
        mid = left + 1.0 if np.isinf(right) else 0.5 * (left + right)
        p_mid = mid * v

        # Determine which coordinates are clamped in this interval
        status = []
        bounds = []
        for i in range(3):
            if p_mid[i] < box_min[i] - 1e-12:
                status.append("low")
                bounds.append(box_min[i])
            elif p_mid[i] > box_max[i] + 1e-12:
                status.append("high")
                bounds.append(box_max[i])
            else:
                status.append("in")
                bounds.append(None)

        # g(alpha)=sum (B - alpha*v_i)^2 over clamped dims -> quadratic
        a = b = c = 0.0
        for i in range(3):
            if status[i] == "in":
                continue
            B = bounds[i]
            vi = v[i]
            a += vi * vi
            b += -2.0 * B * vi
            c += B * B

        candidates = [left]
        if not np.isinf(right):
            candidates.append(right)

        if a > 1e-12:
            alpha0 = -b / (2.0 * a)
            if alpha0 < left:
                alpha0 = left
            if (not np.isinf(right)) and alpha0 > right:
                alpha0 = right
            candidates.append(alpha0)

        for alpha in candidates:
            if alpha < 0:
                continue
            p = alpha * v
            x = clamp(p)
            d = x - p
            dist2 = float(d @ d)
            if dist2 < best["dist2"] - 1e-12:
                best = {"dist2": dist2, "alpha": float(alpha), "x_box": x}

    return best


def aabb_corners(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    xmin, ymin, zmin = box_min
    xmax, ymax, zmax = box_max
    return np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=float,
    )


# ----------------------------
# Drawing utilities (2D)
# ----------------------------
def draw_arrow(ax, p0, p1, lw=1.2, ls="-", color="k", mutation_scale=12, zorder=3):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    arrow = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_segment(ax, p0, p1, lw=2.0, ls="-", color="k", zorder=2):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=lw, linestyle=ls, color=color, zorder=zorder)


def place_text(ax, p, s, dx=0.0, dy=0.0, fontsize=11, ha="left", va="center", bbox=True, zorder=5, alpha=0.85):
    p = np.asarray(p, dtype=float)
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=alpha) if bbox else None
    ax.text(p[0] + dx, p[1] + dy, s, fontsize=fontsize, ha=ha, va=va, bbox=bb, zorder=zorder)


def draw_aabb_2d(ax, box_min, box_max, lw=2.2):
    corners3 = aabb_corners(box_min, box_max)
    corners2 = np.array([PROJ(p) for p in corners3])

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for i, j in edges:
        draw_segment(ax, corners2[i], corners2[j], lw=lw, color="k")
    return corners2


def draw_panel(ax, box_min, box_max, vdes, current=None, panel_label="(a)", mode="auto"):
    """
    mode: 'intersection' / 'closest' / 'auto'
    """
    o2 = PROJ([0, 0, 0])

    # axes
    draw_arrow(ax, o2, PROJ([2, 0, 0]), lw=1.5, mutation_scale=12)
    draw_arrow(ax, o2, PROJ([0, 2, 0]), lw=1.5, mutation_scale=12)
    draw_arrow(ax, o2, PROJ([0, 0, 2]), lw=1.5, mutation_scale=12)

    place_text(ax, PROJ([2, 0, 0]), r"$v_x$", dx=0.10, dy=-0.08, fontsize=13, ha="left", va="top")
    place_text(ax, PROJ([0, 2, 0]), r"$v_y$", dx=0.06, dy=0.08, fontsize=13, ha="left", va="bottom")
    place_text(ax, PROJ([0, 0, 2]), r"$\omega$", dx=0.06, dy=0.08, fontsize=13, ha="left", va="bottom")

    # AABB
    draw_aabb_2d(ax, box_min, box_max, lw=2.2)

    # desired ray
    vdes = np.asarray(vdes, dtype=float)
    draw_arrow(ax, o2, PROJ(vdes), lw=1.4, mutation_scale=12)
    draw_segment(ax, o2, PROJ(1.6 * vdes), lw=1.1, ls="--", color="0.55", zorder=1)
    place_text(ax, PROJ(vdes), r"$\mathbf{v}^{\mathrm{des}}_t$", dx=0.12, dy=0.10, fontsize=11, ha="left", va="bottom")

    # current velocity
    if current is not None:
        pU = PROJ(current)
        ax.scatter([pU[0]], [pU[1]], s=55, color="k", zorder=4)
        place_text(ax, pU, r"$\mathbf{v}_{\mathrm{cmd},t}$", dx=0.12, dy=-0.03, fontsize=11, ha="left", va="center")

    # intersection or closest
    hit, a_low, a_high = ray_aabb_intersection_alpha(vdes, box_min, box_max, 0.0)
    mode_use = ("intersection" if hit else "closest") if mode == "auto" else mode

    if mode_use == "intersection" and hit:
        pmin3 = a_low * vdes
        pmax3 = a_high * vdes
        pmin2 = PROJ(pmin3)
        pmax2 = PROJ(pmax3)

        draw_segment(ax, pmin2, pmax2, lw=3.0, color="k", zorder=3)
        ax.scatter([pmin2[0]], [pmin2[1]], s=65, color="k", zorder=4)
        ax.scatter([pmax2[0]], [pmax2[1]], s=85, color="k", zorder=4)

        place_text(ax, pmin2, r"$\alpha_{\min}\,\mathbf{v}^{\mathrm{des}}_t$", dx=0.08, dy=-0.16, fontsize=11, ha="left", va="top")
        place_text(ax, pmax2, r"$\alpha_{\max}\,\mathbf{v}^{\mathrm{des}}_t$", dx=0.28, dy=0.10, fontsize=11, ha="left", va="bottom")
        place_text(ax, pmax2, r"$\mathbf{v}_{\mathrm{cmd},t+1}$", dx=0.28, dy=-0.08, fontsize=11, ha="left", va="top")

    elif mode_use == "closest":
        sol = closest_point_on_aabb_to_ray(vdes, box_min, box_max)
        x3 = sol["x_box"]
        x2 = PROJ(x3)

        ax.scatter([x2[0]], [x2[1]], s=120, color="k", zorder=4)
        place_text(ax, x2, r"$\mathbf{v}_{\mathrm{cmd},t+1}$", dx=0.16, dy=0.00, fontsize=11, ha="left", va="center")

        # show connector from ray-point to the closest feasible point
        p3 = sol["alpha"] * vdes
        draw_segment(ax, PROJ(p3), x2, lw=1.6, ls=":", color="0.55", zorder=2)

    # panel label
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=16, ha="left", va="top")

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main():
    # Paper-like fonts
    plt.rcParams.update(
        {
            "font.size": 11,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        }
    )

    # --- parameters (same as your TikZ example) ---
    vdes = np.array([1.0, 0.4, 0.2], dtype=float)

    # common limits (for consistent framing)
    pts = np.array([PROJ([0, 0, 0]), PROJ([2, 0, 0]), PROJ([0, 2, 0]), PROJ([0, 0, 2]), PROJ([1.6, 0.64, 0.32])])
    xmin, ymin = pts.min(axis=0) - np.array([0.4, 0.4])
    xmax, ymax = pts.max(axis=0) + np.array([0.75, 0.55])

    panel_specs = [
        {
            "suffix": "a",
            "panel_label": "(a)",
            "box_min": np.array([0.3, -0.1, -0.2], dtype=float),
            "box_max": np.array([1.2, 0.6, 0.4], dtype=float),
            "current": np.array([0.75, 0.25, 0.10], dtype=float),
            "mode": "intersection",
        },
        {
            "suffix": "b",
            "panel_label": "(b)",
            "box_min": np.array([0.3, 0.9, -0.2], dtype=float),
            "box_max": np.array([1.2, 1.6, 0.4], dtype=float),
            "current": None,
            "mode": "closest",
        },
    ]

    for spec in panel_specs:
        fig = plt.figure(figsize=(4.0, 3.3))
        ax = fig.add_axes([0.08, 0.08, 0.88, 0.88])
        draw_panel(
            ax,
            spec["box_min"],
            spec["box_max"],
            vdes,
            current=spec["current"],
            panel_label=spec["panel_label"],
            mode=spec["mode"],
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        fig.savefig(f"dwvp_geometry_{spec['suffix']}.pdf", bbox_inches="tight")
        fig.savefig(f"dwvp_geometry_{spec['suffix']}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
