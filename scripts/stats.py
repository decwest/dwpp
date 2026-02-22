from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_path_headings(path: np.ndarray) -> np.ndarray:
    # path が N x 3 の場合はそのまま姿勢を使用
    if path.shape[1] >= 3:
        return path[:, 2]

    # path が N x 2 の場合は接線方向を姿勢として近似
    if len(path) == 1:
        return np.array([0.0])

    diffs = np.diff(path[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])
    return headings

def calc_rmse(robot_poses_dict: defaultdict, path: np.ndarray) -> dict:
    """
    各手法で得られたロボット軌跡 (robot_poses) に対し、
    与えられた path の各点との最小距離(= Cross Track Error) を算出し、
    そのRMSEを表示する関数。
    
    Parameters
    ----------
    robot_poses_dict : defaultdict
        キーが method_name, 値がロボットの軌跡（Nx3 など）のリスト(または配列)
    path : np.ndarray
        パスを構成する2次元座標列 (Mx2 など)
    """
    rmse_dict = {}
    
    # path を (N,2) とみなしておく。必要に応じて座標次元を合わせる
    path_xy = path[:, :2]  # 念のためX,Yだけ切り出す

    for method_name in robot_poses_dict:
        # ロボット軌跡を取り出して np.ndarray 化し、X,Yのみ使う
        robot_poses = np.array(robot_poses_dict[method_name])
        robot_xy = robot_poses[:, :2]  # [x, y]だけを抽出

        # ロボット軌跡の各点とpath上の各点間の距離をまとめて計算 (scipyのcdistを使用)
        # distance_matrix は shape=(ロボット軌跡の点数, パスの点数)
        distance_matrix = cdist(robot_xy, path_xy, metric='euclidean')

        # 行方向(min)をとることで、各ロボット点に対する最小距離を取り出す
        min_distances = np.min(distance_matrix, axis=1)

        # RMSEを算出
        rmse = np.sqrt(np.mean(min_distances**2))
        
        rmse_dict[method_name] = rmse

        # 結果を表示 (method_name_dictがある場合)
        # print(f"Method: {method_name_dict[method_name]}, RMSE: {rmse:.6f}")

    return rmse_dict


def calc_heading_rmse(robot_poses_dict: defaultdict, path: np.ndarray) -> dict:
    """
    姿勢誤差(heading error)のみのRMSEを算出する。
    RMSE = sqrt(mean(e_theta^2))
    """
    heading_rmse_dict = {}

    path_xy = path[:, :2]
    path_headings = calc_path_headings(path)

    for method_name in robot_poses_dict:
        robot_poses = np.array(robot_poses_dict[method_name])
        robot_xy = robot_poses[:, :2]

        distance_matrix = cdist(robot_xy, path_xy, metric='euclidean')
        nearest_indices = np.argmin(distance_matrix, axis=1)

        # 姿勢誤差（最近傍パス点の姿勢との差）
        if robot_poses.shape[1] >= 3:
            robot_headings = robot_poses[:, 2]
        else:
            robot_headings = np.zeros(len(robot_poses))
        ref_headings = path_headings[nearest_indices]
        heading_errors = normalize_angle(robot_headings - ref_headings)

        heading_rmse = np.sqrt(np.mean(heading_errors**2))
        heading_rmse_dict[method_name] = float(heading_rmse)

    return heading_rmse_dict


def calc_rmse_with_heading(robot_poses_dict: defaultdict, path: np.ndarray) -> dict:
    # 後方互換のため残す（中身は角度誤差のみRMSE）
    return calc_heading_rmse(robot_poses_dict, path)

def calc_break_constraints_rate(break_constraints_flags_dict: dict) -> dict:
    
    break_constraints_rate_dict = {}
    
    for method_name in break_constraints_flags_dict:
        break_constraints_flags = np.array(break_constraints_flags_dict[method_name])
        break_constraints_rate = np.mean(break_constraints_flags, axis=0) * 100.0
        break_constraints_rate_dict[method_name] = break_constraints_rate
        # print(f"Method: {method_name_dict[method_name]}, constraints break ratio v: {break_constraints_rate[0]} %")
        # print(f"Method: {method_name_dict[method_name]}, constraints break ratio w: {break_constraints_rate[1]} %")
    
    return break_constraints_rate_dict
