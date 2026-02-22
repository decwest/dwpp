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
