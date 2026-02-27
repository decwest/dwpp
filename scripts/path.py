import numpy as np
import math


def append_heading_to_path(path_xy: np.ndarray) -> np.ndarray:
    if len(path_xy) == 0:
        return np.empty((0, 3))
    if len(path_xy) == 1:
        return np.array([[path_xy[0, 0], path_xy[0, 1], 0.0]])

    diffs = np.diff(path_xy, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])

    return np.c_[path_xy, headings]

def sin_curves() -> list:
    paths = []
    
    x = np.linspace(0, 2*np.pi, 500)
    
    # y = a*sin(bx)
    a_list = [0.5, 1.0, 1.5]
    b_list = [1.0, 2.0 ,3.0]
    
    for a in a_list:
        for b in b_list:
            y = a * np.sin(b * x)
            path = append_heading_to_path(np.c_[x, y])
            paths.append(path)
            
    return paths

def step_curves() -> list:
    
    paths = []
    
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_list = [2.0, 3.0, 4.0]
    
    for theta in theta_list:
        for l in l_list:
            # section 1
            x1 = np.linspace(0, 1, 100)
            y1 = np.zeros_like(x1)
            
            # section 2
            x2 = np.linspace(1.0, 1.0+l*math.cos(theta), 100)
            y2 = np.linspace(0.0, l*math.sin(theta), 100)
            
            # section 3
            x3 = np.linspace(1.0+l*math.cos(theta), 4.0+l*math.cos(theta), 100)
            y3 = np.ones_like(x3) * l * math.sin(theta)
            
            x = np.concatenate([x1, x2, x3])
            y = np.concatenate([y1, y2, y3])
            path = append_heading_to_path(np.c_[x, y])
            paths.append(path)
    
    return paths


def straight_line_heading_step_curve(segment_length: float = 1.0, points_per_segment: int = 100) -> np.ndarray:
    """
    一直線上 (y=0) の経路に対し、姿勢のみを段階的に変える N x 3 経路を生成する。
    姿勢セグメント: [0, 90, 180, 270, 360] [deg]
    各セグメント長: segment_length [m]
    """
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    headings_deg = np.array([0.0, 90.0, 180.0, 270.0, 360.0], dtype=float)
    headings_rad = np.deg2rad(headings_deg)
    n_segments = len(headings_rad)

    # 5セグメント x 1m を想定（デフォルト）
    x = np.linspace(0.0, n_segments * segment_length, n_segments * points_per_segment + 1)
    y = np.zeros_like(x)

    theta = np.empty_like(x)
    for i, heading in enumerate(headings_rad):
        start = i * points_per_segment
        end = (i + 1) * points_per_segment
        theta[start:end] = heading
    theta[-1] = headings_rad[-1]

    return np.c_[x, y, theta]


def right_angle_polyline_curve(segment_length: float = 0.5, points_per_segment: int = 50) -> np.ndarray:
    """
    90度の折れ線経路 (N x 3) を生成する。
    1辺目: (0, 0) -> (segment_length, 0), 姿勢 0 [rad]
    2辺目: (segment_length, 0) -> (segment_length, segment_length), 姿勢 pi/2 [rad]
    """
    if points_per_segment <= 0:
        raise ValueError("points_per_segment must be > 0")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")

    x1 = np.linspace(0.0, segment_length, points_per_segment + 1)
    y1 = np.zeros_like(x1)
    theta1 = np.zeros_like(x1)

    x2 = np.full(points_per_segment + 1, segment_length)
    y2 = np.linspace(0.0, segment_length, points_per_segment + 1)
    theta2 = np.full(points_per_segment + 1, np.pi / 2.0)

    # 折れ点の重複を避けるため2区間目の先頭を除外
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    theta = np.concatenate([theta1, theta2[1:]])

    return np.c_[x, y, theta]


def right_angle_polyline_curve_last_segment_heading_minus_pi(
    segment_length: float = 0.5,
    points_per_segment: int = 50
) -> np.ndarray:
    """
    path1(90度折れ線)をベースに、最後の1点のみ姿勢角を -pi に設定した経路を生成する。
    それ以外の点の姿勢角は経路の接線方向に合わせる。
    """
    base_path = right_angle_polyline_curve(
        segment_length=segment_length,
        points_per_segment=points_per_segment,
    )
    path = append_heading_to_path(base_path[:, :2])
    path[-1, 2] = -np.pi

    return path


def _resample_xy_by_arclength(x: np.ndarray, y: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total_length = float(s[-1])

    if total_length <= 1e-12:
        return (
            np.linspace(float(x[0]), float(x[-1]), num_points),
            np.linspace(float(y[0]), float(y[-1]), num_points),
        )

    s_new = np.linspace(0.0, total_length, num_points)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new


def one_minus_cos_curve(
    amplitude: float = 1.0,
    length_x: float = 10.0,
    num_points: int = 200,
    cycles: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
    theta0: float = 0.0,
    resample_arclength: bool = True,
) -> np.ndarray:
    """
    y = A(1 - cos(k(x-x0))) 形状の経路を N x 3 (x, y, theta) で生成する。
    theta は接線方向（arctan2(dy, dx)）を用いる。
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    if length_x <= 0.0:
        raise ValueError("length_x must be > 0")

    a = float(amplitude)
    l = float(length_x)
    k = 2.0 * math.pi * float(cycles) / l

    x = np.linspace(float(x0), float(x0) + l, num_points, dtype=float)
    u = x - float(x0)
    y = float(y0) + a * (1.0 - np.cos(k * u))

    if abs(theta0) > 0.0:
        c = math.cos(theta0)
        s = math.sin(theta0)
        x_shift = x - float(x0)
        y_shift = y - float(y0)
        x = float(x0) + c * x_shift - s * y_shift
        y = float(y0) + s * x_shift + c * y_shift

    if resample_arclength:
        x, y = _resample_xy_by_arclength(x, y, num_points)

    dx = np.gradient(x)
    dy = np.gradient(y)
    theta = np.arctan2(dy, dx)

    return np.c_[x, y, theta]
