"""Paper-oriented plotting helpers for the revision studies."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dwpp-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "PP": "#4477AA",
    "APP": "#228833",
    "RPP": "#EE9922",
    "DWPP": "#CC3311",
    "APP+DW": "#AA3377",
}
MARKERS = {"PP": "o", "APP": "s", "RPP": "^", "DWPP": "D", "APP+DW": "P"}
NOISE_COLORS = ("#4477AA", "#228833", "#AA3377")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "figure.figsize": (6.0, 4.0),
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fig.savefig(
        output_base.with_suffix(".pdf"),
        metadata={
            "Creator": "DWPP revision simulator",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        output_base.with_suffix(".png"),
        metadata={"Software": "DWPP revision simulator"},
    )
    plt.close(fig)


def _series_color(label: str, index: int) -> str:
    """Use paper controller colors, falling back to distinct series colors."""
    controller = label.split("@", maxsplit=1)[0]
    if controller in COLORS:
        return COLORS[controller]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return default_colors[index % len(default_colors)]


def plot_trajectories(
    reference_path: np.ndarray,
    panels: Sequence[tuple[str, Sequence[tuple[str, np.ndarray]]]],
    output_base: Path,
) -> None:
    """Plot true trajectories in the coordinate convention used by the paper."""
    if not panels:
        raise ValueError("At least one trajectory panel is required")

    configure_style()
    panel_count = len(panels)
    figsize = (6.5, 6.5) if panel_count == 1 else (5.2 * panel_count, 4.8)
    fig, axes = plt.subplots(1, panel_count, figsize=figsize, squeeze=False)
    for ax, (title, trajectories) in zip(axes[0], panels, strict=True):
        for index, (label, positions) in enumerate(trajectories):
            values = np.asarray(positions, dtype=float)
            ax.plot(
                values[:, 1],
                -values[:, 0],
                color=_series_color(label, index),
                label=label,
                linewidth=1.0,
            )
        ax.plot(
            reference_path[:, 1],
            -reference_path[:, 0],
            "k--",
            label="Reference Path",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend()
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_lookahead_tradeoff(
    groups: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float]]],
    output_base: Path,
    *,
    title: str,
) -> None:
    """Plot tracking error and travel time against lookahead on twin axes."""
    configure_style()
    fig, error_ax = plt.subplots()
    time_ax = error_ax.twinx()
    sigma_handles: list[Line2D] = []
    for index, (label, lookaheads, errors, times) in enumerate(groups):
        color = NOISE_COLORS[index % len(NOISE_COLORS)]
        error_ax.plot(
            lookaheads,
            errors,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1.5,
        )
        time_ax.plot(
            lookaheads,
            times,
            color=color,
            marker="^",
            linestyle="--",
            linewidth=1.5,
        )
        sigma_handles.append(Line2D([0], [0], color=color, linewidth=2.0, label=label))

    quantity_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label="Mean tracking error",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="^",
            linestyle="--",
            linewidth=1.5,
            label="Travel time",
        ),
    ]
    error_ax.set_xlabel(r"Lookahead distance $L$ [m]")
    error_ax.set_ylabel("Mean tracking error [m]")
    time_ax.set_ylabel("Travel time [s]")
    error_ax.set_title(title)
    error_ax.grid(True, alpha=0.3)
    error_ax.legend(handles=[*sigma_handles, *quantity_handles], loc="best")
    fig.tight_layout()
    save_figure(fig, output_base)


def _annotate_points(
    fig: plt.Figure,
    ax: plt.Axes,
    points: Sequence[tuple[float, float, float]],
    occupied_artists: Sequence[object] = (),
) -> None:
    """Place compact numeric labels greedily without label-label overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [artist.get_window_extent(renderer) for artist in occupied_artists]
    axes_box = ax.get_window_extent(renderer)
    offsets = (
        (4, 5), (4, 13), (4, -13), (-17, 5), (-17, 13), (-17, -13),
        (11, 20), (-24, 20), (11, -21), (-24, -21), (28, 5), (-36, 5),
        (28, 17), (-36, 17), (28, -17), (-36, -17),
    )
    for x, y, value in points:
        annotation = None
        for offset in offsets:
            candidate = ax.annotate(
                f"{value:.2g}",
                (x, y),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
            )
            fig.canvas.draw()
            box = candidate.get_window_extent(renderer).expanded(1.10, 1.20)
            inside = (
                box.x0 >= axes_box.x0
                and box.x1 <= axes_box.x1
                and box.y0 >= axes_box.y0
                and box.y1 <= axes_box.y1
            )
            if inside and not any(box.overlaps(previous) for previous in occupied):
                annotation = candidate
                occupied.append(box)
                break
            candidate.remove()
        if annotation is None:
            # Dense clusters may exhaust the preferred offsets. Preserve the
            # required annotation using a deterministic wider fallback.
            fallback = (42, 8 + 8 * (len(occupied) % 3))
            annotation = ax.annotate(
                f"{value:.2g}",
                (x, y),
                xytext=fallback,
                textcoords="offset points",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "linewidth": 0.35, "alpha": 0.6},
            )
            fig.canvas.draw()
            occupied.append(annotation.get_window_extent(renderer).expanded(1.10, 1.20))


def plot_tradeoff(
    groups: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float]]],
    output_base: Path,
    *,
    title: str,
    annotation_label: str,
) -> None:
    configure_style()
    fig, ax = plt.subplots()
    annotation_points: list[tuple[float, float, float]] = []
    for label, times, errors, annotation_values in groups:
        ax.plot(
            times,
            errors,
            marker=MARKERS.get(label, "o"),
            color=COLORS.get(label),
            linewidth=1.5,
            label=label,
        )
        annotation_points.extend(
            (float(x), float(y), float(value))
            for x, y, value in zip(times, errors, annotation_values, strict=True)
        )
    ax.set_xlabel("Travel time [s]")
    ax.set_ylabel("Mean tracking error [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    legend = ax.legend(title=annotation_label)
    _annotate_points(fig, ax, annotation_points, (legend,))
    save_figure(fig, output_base)


def plot_lines(
    groups: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    output_base: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    configure_style()
    fig, ax = plt.subplots()
    for label, xs, ys in groups:
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(label, "o"),
            color=COLORS.get(label),
            linewidth=1.5,
            label=label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_base)


def plot_bands(
    groups: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float]]],
    output_base: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    configure_style()
    fig, ax = plt.subplots()
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, (label, xs, means, stds) in enumerate(groups):
        x = np.asarray(xs, dtype=float)
        mean = np.asarray(means, dtype=float)
        std = np.asarray(stds, dtype=float)
        color = COLORS.get(label, default_colors[index % len(default_colors)])
        ax.plot(x, mean, marker=MARKERS.get(label, "o"), color=color, linewidth=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, output_base)
