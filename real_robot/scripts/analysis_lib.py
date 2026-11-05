from __future__ import annotations  # Python 3.8 互換 (list[Path] 等の遅延評価)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math
from pathlib import Path
import os
from collections import defaultdict

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'stix'   # ← 重要
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

color_dict = {"PP": "blue", "APP": "green", "RPP": "orange", "DWPP": "red"}

# Make path
def make_path():
    """
    Function to generate reference paths
    Returns:
        (path_A, path_B, path_C)
    """
    paths = []
    # 生成するカーブの角度パターン
    theta_list = [np.pi/4, np.pi/2, 3*np.pi/4]
    l_segment = 3.0  # 直線区間の長さパラメータ

    for theta in theta_list:
        # 1. 直進 (0 -> 3m)
        x1 = np.linspace(0, 3, 300)
        y1 = np.zeros_like(x1)
        
        # 2. 斜め直線 (角度thetaで長さl)
        # Note: 実際には直線補間だが、ここでは簡易的に生成
        x2 = np.linspace(3.0, 3.0 + l_segment * math.cos(theta), 300)
        y2 = np.linspace(0.0, l_segment * math.sin(theta), 300)
        
        # 3. 終端直進 (さらに3m進む)
        x3 = np.linspace(
            3.0 + l_segment * math.cos(theta), 
            6.0 + l_segment * math.cos(theta), 300)
        y3 = np.ones_like(x3) * l_segment * math.sin(theta)

        # 結合
        xs = np.concatenate([x1, x2, x3])
        ys = np.concatenate([y1, y2, y3])
        
        path = np.c_[xs, ys]
        
        paths.append(path)
        

    # 展開して返す
    return (paths[0], paths[1], paths[2])

PathA, PathB, PathC = make_path()
reference_path = {
    "PathA": PathA,
    "PathB": PathB,
    "PathC": PathC
}

# Make iso path
def make_iso_path():
    """
    Function to generate reference paths
    Returns:
        (path_A, path_B, path_C)
    """
    
    Lu = 0.985
    
    # Path A: 5*Lu straight line
    x_A = np.linspace(0, 5*Lu, 500)
    y_A = np.zeros_like(x_A)
    path_A = np.c_[x_A, y_A]
    
    # Path B: Square with side length 5*Lu
    x_B1 = np.linspace(0, 5*Lu, 500)
    y_B1 = np.zeros_like(x_B1)
    y_B2 = np.linspace(0, -5*Lu, 500)
    x_B2 = np.ones_like(y_B2) * 5 * Lu
    x_B3 = np.linspace(5*Lu, 0, 500)
    y_B3 = np.ones_like(x_B3) * -5 * Lu
    y_B4 = np.linspace(-5*Lu, 0, 500)
    x_B4 = np.zeros_like(y_B4)
    x_B = np.concatenate([x_B1, x_B2, x_B3, x_B4])
    y_B = np.concatenate([y_B1, y_B2, y_B3, y_B4])
    path_B = np.c_[x_B, y_B]
    
    # Path C: Straight line + arc
    x_C1 = np.linspace(0, 5*Lu, 500)
    y_C1 = np.zeros_like(x_C1)
    theta_list = np.linspace(0, np.pi/2, 500)
    x_C2 = 5*Lu*(1+np.sin(theta_list))
    y_C2 = 5*Lu*(np.cos(theta_list)-1)
    x_C = np.concatenate([x_C1, x_C2])
    y_C = np.concatenate([y_C1, y_C2])
    path_C = np.c_[x_C, y_C]
    
    return path_A, path_B, path_C

iso_PathA, iso_PathB, iso_PathC = make_iso_path()
iso_reference_path = {
    "PathA": iso_PathA,
    "PathB": iso_PathB,
    "PathC": iso_PathC
}

# Calculate violation rate
def calc_violation_rate(violation_flags) -> float:
    violation_count = np.sum(violation_flags)
    total_count = len(violation_flags)
    violation_rate = violation_count / total_count * 100
    return violation_rate

from scipy.spatial.distance import cdist

# Calculate RMSE between robot path and reference path
def calc_tracking_error(robot_path: np.ndarray, path: np.ndarray) -> float:

    # Compute the distances between each point on the robot trajectory and each point on the path at once (using scipy's cdist)
    # distance_matrix has shape = (number of points in the robot trajectory, number of points in the path)
    distance_matrix = cdist(robot_path, path, metric='euclidean')

    # Take the minimum along the row direction to extract the minimum distance for each robot point
    min_distances = np.min(distance_matrix, axis=1)
    
    return min_distances

def calc_rmse(robot_path: np.ndarray, path: np.ndarray) -> float:

    tracking_errors = calc_tracking_error(robot_path, path)

    # Compute RMSE
    rmse = np.sqrt(np.mean(tracking_errors**2))
    
    return rmse

# Plot velocity profile
def plot_velocity_profile(t, v_real, w_real, v_cmd, w_cmd, path_name, controller_name, data_dir):
    # Create subplots for translational and rotational velocities (horizontal layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Plot translational velocity
    ax1.plot(t, v_cmd, '-', color='red', label='Reference', linewidth=1, alpha=0.8)
    ax1.plot(t, v_real, '-', color='blue', label='Actual', linewidth=1)
    # Add velocity limit line
    ax1.axhline(y=0.50, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Max Velocity')

    # Plot rotational velocity
    ax2.plot(t, w_cmd, '-', color='red', label='Reference', linewidth=1, alpha=0.8)
    # ax2.plot(timestamps, cmd_angular, '-', color='blue', label='Command', linewidth=3)
    ax2.plot(t, w_real, '-', color='blue', label='Actual', linewidth=1)
    # Add velocity limit lines
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Max Velocity')
    ax2.axhline(y=-1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Min Velocity')

    # Set labels and legends
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Linear Velocity [m/s]')
    # ax1.set_yticks([0, 0.10, 0.20])
    ax1.grid(True, alpha=0.3)
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Velocity [rad/s]')
    # ax2.set_ylim(-0.7, 0.7)
    # ax2.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax2.grid(True, alpha=0.3)
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filename = data_dir / f"{path_name}_{controller_name}_velocity.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# Plot dynamic window
def plot_dynamic_window(curvatures, vs, ws, next_vs, next_ws, dw_max_vs, dw_min_vs, dw_max_ws, dw_min_ws, actual_vs, actual_ws, v_regs, path_name, controller_name, base_dir=Path(__file__).resolve().parent.parent / 'analysis_out'):
    
    save_dir = base_dir/ "DynamicWindow"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, (curvature, v, w, next_v, next_w, dw_max_v, dw_min_v, dw_max_w, dw_min_w, actual_v, actual_w, v_reg) \
        in enumerate(zip(curvatures, vs, ws, next_vs, next_ws, dw_max_vs, dw_min_vs, dw_max_ws, dw_min_ws, actual_vs, actual_ws, v_regs)):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        
        # Dynamic Windowの範囲を四角形でプロット
        vertexes = [
            (dw_min_v, dw_min_w),
            (dw_min_v, dw_max_w),
            (dw_max_v, dw_max_w),
            (dw_max_v, dw_min_w)
        ]
        rectangle = patches.Polygon(vertexes, closed=True, fill=False, color='black', alpha=1.0, label='Dynamic Window')
        ax.add_patch(rectangle)
        
        # omega = curvature * v の直線をプロット
        v_list = np.linspace(-0.1, 0.6, 100)
        omega_line = curvature * v_list
        ax.plot(v_list, omega_line, 'g--', label='ω = k * v')
        
        # 現在の速度点
        ax.scatter(v, w, s=12, color='blue', label='Current Velocity')
        # 速度指令値の速度点
        ax.scatter(next_v, next_w, s=12, color='red', label='Command Velocity')
        # 制御後の速度点
        ax.scatter(actual_v, actual_w, s=12, color='magenta', label='Actual Velocity')
        
        # v_regの描画
        ax.axvline(x=v_reg, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Regulated Velocity')
        
        ax.set_xlabel('Linear Velocity [m/s]')
        ax.set_ylabel('Angular Velocity [rad/s]')
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.5, 1.6)
        ax.grid(True, alpha=0.3)
        # ax.set_aspect('equal')
        # plt.legend()
        plt.savefig(save_dir / f'{idx}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

# measure travel time
def measure_travel_time(timestamp):
    start_time = timestamp[0]
    end_time = timestamp[-1]
    travel_time = end_time - start_time
    return travel_time

# Plot path
def plot_path(csv_paths: list[Path], path_name: str, controller_names: list[str]):
    paths = []
    for csv_path in csv_paths:
        # read csv
        df = pd.read_csv(csv_path)
        data_dir = csv_path.parent
        
        # Extract relevant data
        map_base_x = df["map_base_x"].to_numpy()
        map_base_y = df["map_base_y"].to_numpy()
        map_base_yaw = df["map_base_yaw"].to_numpy()
        
        x, y, yaw = transform_pose_to_path_origin(map_base_x, map_base_y, map_base_yaw, map_base_x[0], map_base_y[0], map_base_yaw[0])
        path = np.vstack((x, y)).T
        paths.append(path)
    
    fig = plt.figure(figsize=(6.5,6.5))
    ax = fig.add_subplot(1, 1, 1)
    
    for path, controller_name in zip(paths, controller_names):
        x = path.T[0]
        y = path.T[1]
        # Plot robot trajectory (rotate: X->Y, Y->-X for X-up, Y-left coordinate system)
        ax.plot(y, [-x for x in x], color=color_dict[controller_name], label=controller_name, linewidth=1)

    ref_x = [point[0] for point in  reference_path[path_name]]
    ref_y = [point[1] for point in reference_path[path_name]]
    ax.plot(ref_y, [-x for x in ref_x], 'k--', label='Reference Path', linewidth=1, alpha=0.7)
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.legend()
    
    filepath = data_dir / 'path_comparison.png'

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

# Transform pose to path origin
def transform_pose_to_path_origin(x, y, yaw, path_origin_x, path_origin_y, path_origin_yaw):
    # Translate
    x_translated = x - path_origin_x
    y_translated = y - path_origin_y
    # Rotate
    cos_yaw = math.cos(-path_origin_yaw)
    sin_yaw = math.sin(-path_origin_yaw)
    x_rotated = x_translated * cos_yaw - y_translated * sin_yaw
    y_rotated = x_translated * sin_yaw + y_translated * cos_yaw
    return x_rotated, y_rotated, yaw - path_origin_yaw

def analyze_single_csv(csv_path: Path, path_name: str, controller_name: str):
    # read csv
    df = pd.read_csv(csv_path)
    data_dir = csv_path.parent
    
    # Extract relevant data
    timestamp = df["sec"].to_numpy() + df["nsec"].to_numpy() * 1e-9
    timestamp -= timestamp[0]  # Normalize to start from zero
    curvature = df["curvature"].to_numpy()
    v_now = df["v_now"].to_numpy()
    w_now = df["w_now"].to_numpy()
    v_cmd = df["v_cmd"].to_numpy()
    w_cmd = df["w_cmd"].to_numpy()
    actual_v = df["v_nav"].to_numpy()
    actual_w = df["w_nav"].to_numpy()
    v_real = df["v_real"].to_numpy()
    w_real = df["w_real"].to_numpy()
    dw_max_v = df["dw_v_max"].to_numpy()
    dw_max_w = df["dw_w_max"].to_numpy()
    dw_min_v = df["dw_v_min"].to_numpy()
    dw_min_w = df["dw_w_min"].to_numpy()
    v_reg = df["v_reg"].to_numpy()
    
    map_base_x = df["odom_base_x"].to_numpy()
    map_base_y = df["odom_base_y"].to_numpy()
    map_base_yaw = df["odom_base_yaw"].to_numpy()
    velocity_violation = df["velocity_violation"].to_numpy()
    
    x, y, yaw = transform_pose_to_path_origin(map_base_x, map_base_y, map_base_yaw, map_base_x[0], map_base_y[0], map_base_yaw[0])
    
    # calc velocity violation ratio
    violation_rate = calc_violation_rate(velocity_violation)
    print(f" Velocity violation rate: {violation_rate*100:.2f}%")
    
    # calc RMSE
    path = reference_path[path_name]
    robot_path = np.vstack((x, y)).T
    rmse = calc_rmse(robot_path, path)
    print(f" RMSE to {path_name} of {controller_name}: {rmse:.4f} m")
    
    # calc travel time
    travel_time = measure_travel_time(timestamp)
    print(f" Travel time of {controller_name}: {travel_time:.2f} s")
    
    # plot velocity profile
    plot_velocity_profile(timestamp, v_real, w_real, v_cmd, w_cmd, path_name, controller_name, data_dir)
    