import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_VIDEO = Path(__file__).resolve().parent / "exp1_dwvp.MP4"
DEFAULT_OUT = Path(__file__).resolve().parent / "overlay_exp1_dwvp.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay robot snapshots from a movie into one image."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--interval-sec", type=float, default=1.5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.60,
        help="Blend alpha for robot overlays (0.0=fully transparent, 1.0=fully opaque).",
    )
    parser.add_argument("--diff-thresh", type=int, default=18)
    parser.add_argument("--min-area", type=float, default=3500.0)
    parser.add_argument("--bg-samples", type=int, default=31)
    parser.add_argument(
        "--max-track-jump-px",
        type=float,
        default=220.0,
        help="Max centroid jump between sampled frames before fallback to largest contour.",
    )
    return parser.parse_args()


def get_frame_count(cap: cv2.VideoCapture) -> int:
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames > 0:
        return n_frames

    # Fallback for containers that do not expose frame count.
    tmp = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        tmp += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return tmp


def read_frame_at(cap: cv2.VideoCapture, idx: int) -> tuple[bool, np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return cap.read()


def build_background_median(
    cap: cv2.VideoCapture,
    n_frames: int,
    n_samples: int,
) -> np.ndarray:
    sample_count = max(3, min(n_samples, n_frames))
    sample_idxs = np.unique(
        np.linspace(0, n_frames - 1, num=sample_count, dtype=np.int32)
    ).tolist()

    sampled_frames: list[np.ndarray] = []
    for idx in sample_idxs:
        ret, frame = read_frame_at(cap, int(idx))
        if ret:
            sampled_frames.append(frame)

    if not sampled_frames:
        raise RuntimeError("Cannot sample frames to build background.")

    stacked = np.stack(sampled_frames, axis=0)
    return np.median(stacked, axis=0).astype(np.uint8)


def contour_center(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w * 0.5, y + h * 0.5
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def extract_robot_mask(
    frame: np.ndarray,
    background_gray: np.ndarray,
    diff_thresh: int,
    min_area: float,
    prev_center: tuple[float, float] | None,
    max_track_jump_px: float,
    k_close: np.ndarray,
    k_open: np.ndarray,
    k_dil: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background_gray)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
    mask = cv2.dilate(mask, k_dil, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[np.ndarray, tuple[float, float], float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        center = contour_center(contour)
        candidates.append((contour, center, area))

    if not candidates:
        return np.zeros_like(mask), None

    if prev_center is None:
        selected = max(candidates, key=lambda x: x[2])
    else:
        selected = min(
            candidates,
            key=lambda x: np.hypot(x[1][0] - prev_center[0], x[1][1] - prev_center[1]),
        )
        jump = float(
            np.hypot(
                selected[1][0] - prev_center[0],
                selected[1][1] - prev_center[1],
            )
        )
        if jump > max_track_jump_px:
            selected = max(candidates, key=lambda x: x[2])

    out_mask = np.zeros_like(mask)
    cv2.drawContours(out_mask, [selected[0]], -1, 255, thickness=-1)
    out_mask = cv2.dilate(out_mask, k_dil, iterations=1)
    return out_mask, selected[1]


def main() -> None:
    args = parse_args()
    video = str(args.video)
    out = str(args.out)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(args.interval_sec * fps)))
    n_frames = get_frame_count(cap)

    ret, start = read_frame_at(cap, 0)
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    background = build_background_median(cap, n_frames, args.bg_samples)
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    k_close = np.ones((9, 9), np.uint8)
    k_open = np.ones((3, 3), np.uint8)
    k_dil = np.ones((5, 5), np.uint8)

    canvas = start.astype(np.float32)

    sample_idxs = list(range(0, n_frames, step))
    sample_idxs.reverse()

    prev_center: tuple[float, float] | None = None
    for idx in sample_idxs:
        ret, frame = read_frame_at(cap, idx)
        if not ret:
            continue

        mask, prev_center = extract_robot_mask(
            frame=frame,
            background_gray=background_gray,
            diff_thresh=args.diff_thresh,
            min_area=args.min_area,
            prev_center=prev_center,
            max_track_jump_px=args.max_track_jump_px,
            k_close=k_close,
            k_open=k_open,
            k_dil=k_dil,
        )
        if mask.sum() == 0:
            continue

        m = (mask > 0).astype(np.float32)[..., None]
        canvas = canvas * (1.0 - args.alpha * m) + frame.astype(np.float32) * (args.alpha * m)

    # Start pose is kept crisp by pasting with a full-weight mask computed
    # against the same background model used for all frames.
    start_mask, _ = extract_robot_mask(
        frame=start,
        background_gray=background_gray,
        diff_thresh=args.diff_thresh,
        min_area=args.min_area,
        prev_center=None,
        max_track_jump_px=args.max_track_jump_px,
        k_close=k_close,
        k_open=k_open,
        k_dil=k_dil,
    )
    if start_mask.sum() > 0:
        m0 = (start_mask > 0).astype(np.float32)[..., None]
        canvas = canvas * (1.0 - m0) + start.astype(np.float32) * m0

    cap.release()

    out_img = np.clip(canvas, 0, 255).astype(np.uint8)
    cv2.imwrite(out, out_img)
    print("saved:", out)
    print(
        "fps={:.2f}, interval={}s, step={} frames, n_frames={}, bg_samples={}".format(
            fps,
            args.interval_sec,
            step,
            n_frames,
            args.bg_samples,
        )
    )


if __name__ == "__main__":
    main()
