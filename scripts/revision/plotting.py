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
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Plot true trajectories in the coordinate convention used by the paper.

    Returns the final (xlim, ylim) of the first panel so a later call can
    reuse the same plotting range.
    """
    if not panels:
        raise ValueError("At least one trajectory panel is required")

    configure_style()
    panel_count = len(panels)
    if figsize is None:
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
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if legend:
            ax.legend()
    fig.tight_layout()
    limits = (axes[0][0].get_xlim(), axes[0][0].get_ylim())
    save_figure(fig, output_base)
    return limits


def save_trajectory_legend(series_labels: Sequence[str], output_base: Path) -> None:
    """Shared horizontal legend strip for compact trajectory panels."""
    configure_style()
    handles = [
        Line2D([0], [0], color=_series_color(label, index), linewidth=1.5, label=label)
        for index, label in enumerate(series_labels)
    ]
    handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label="Reference Path")
    )
    with plt.rc_context({"font.size": 12, "legend.fontsize": 11}):
        fig = plt.figure(figsize=(7.2, 0.4))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=True,
                  handlelength=1.5, columnspacing=1.0, handletextpad=0.5)
        save_figure(fig, output_base)


def plot_lookahead_tradeoff(
    groups: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float]]],
    output_base: Path,
    *,
    title: str,
    xlabel: str = r"Lookahead distance $L$ [m]",
    colors: Sequence[str] = NOISE_COLORS,
) -> None:
    """Plot tracking error and travel time against the swept parameter on twin axes."""
    configure_style()
    fig, error_ax = plt.subplots()
    time_ax = error_ax.twinx()
    sigma_handles: list[Line2D] = []
    for index, (label, lookaheads, errors, times) in enumerate(groups):
        color = colors[index % len(colors)]
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
    error_ax.set_xlabel(xlabel)
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


def plot_velocity_profiles(
    panels: Sequence[tuple[str, float, np.ndarray, np.ndarray]],
    output_base: Path,
    *,
    v_max: float,
    w_max: float,
) -> None:
    """Per-controller v/omega time series for the 2x2 subfigure layout.

    Each panel entry is (label, dt, raw_commands Nx2, applied_commands Nx2)
    and produces one <output_base>_<label>.pdf with v on top and omega below;
    red shows the raw controller output, blue the executed command after the
    acceleration-limited projection, black dashed lines the velocity limits.
    The omega axis is clipped to +-1.5 rad/s as in the real-robot figures, so
    raw commands far outside the window run off the frame. A shared legend
    strip is written to <output_base>_legend.pdf.
    """
    configure_style()
    # Match the real-robot velocity-profile style (font 20, 7x3 in, v left and
    # omega right) so the subfigures render at the same visual size.
    for label, dt, raw, applied in panels:
        raw = np.asarray(raw, dtype=float)
        applied = np.asarray(applied, dtype=float)
        t = np.arange(len(raw)) * dt
        with plt.rc_context({"font.size": 20}):
            fig, (ax_v, ax_w) = plt.subplots(1, 2, figsize=(7, 3))
            ax_v.plot(t, raw[:, 0], "-", color="red", linewidth=1, alpha=0.8)
            ax_v.plot(t, applied[:, 0], "-", color="blue", linewidth=1)
            ax_v.axhline(y=v_max, color="black", linestyle="--", linewidth=1, alpha=0.7)
            ax_v.set_xlabel(r"$t$ [s]")
            ax_v.set_ylabel(r"$v$ [m/s]")
            ax_v.set_yticks([0, 0.25, 0.50])
            ax_v.grid(True, alpha=0.3)
            ax_w.plot(t, raw[:, 1], "-", color="red", linewidth=1, alpha=0.8)
            ax_w.plot(t, applied[:, 1], "-", color="blue", linewidth=1)
            ax_w.axhline(y=w_max, color="black", linestyle="--", linewidth=2, alpha=0.7)
            ax_w.axhline(y=-w_max, color="black", linestyle="--", linewidth=2, alpha=0.7)
            ax_w.set_xlabel(r"$t$ [s]")
            ax_w.set_ylabel(r"$\omega$ [rad/s]")
            ax_w.set_ylim(-1.5, 1.5)
            ax_w.set_yticks([-1, 0, 1])
            ax_w.grid(True, alpha=0.3)
            fig.tight_layout()
            save_figure(fig, output_base.parent / f"{output_base.name}_{label.lower()}")
    handles = [
        Line2D([0], [0], color="red", linewidth=1.5, alpha=0.8, label="Raw command"),
        Line2D([0], [0], color="blue", linewidth=1.5, label="Executed command"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Velocity limits"),
    ]
    with plt.rc_context({"font.size": 12, "legend.fontsize": 11}):
        fig = plt.figure(figsize=(6.4, 0.4))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.legend(handles=handles, loc="center", ncol=3, frameon=True,
                  handlelength=1.8, columnspacing=1.4, handletextpad=0.6)
        save_figure(fig, output_base.parent / f"{output_base.name}_legend")


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
