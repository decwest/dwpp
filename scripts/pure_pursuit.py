import numpy as np
import math
from config import MIN_LOOK_AHEAD_DISTANCE, MAX_LOOK_AHEAD_DISTANCE, LOOK_AHEAD_TIME, V_MAX, V_MIN, W_MAX, W_MIN, \
    A_MAX, AW_MAX, VX_MAX, VX_MIN, VY_MAX, VY_MIN, AX_MAX, AY_MAX, DT, APPROACH_VELOCITY_SCALING_DIST, \
    MIN_APPROACH_LINEAR_VELOCITY, GOAL_TORELANCE_DIST, REGULATED_LINEAR_SCALING_MIN_RADIUS, \
    REGULATED_LINEAR_SCALING_MIN_SPEED, K

ACCEL_CONSTRAINT_EPS = 1e-10
DWPP_USE_REGULATED_VELOCITY = False

def pure_pursuit(current_pose: np.ndarray, current_velocity: np.ndarray, path: np.ndarray, method_name: str)\
    -> tuple[np.ndarray, np.ndarray, list[bool], float, float]:
    # calc index of current position
    current_idx = calc_index(current_pose, path)
    
    # calc distances between initial position and each position
    path_distances = calc_path_distances(path)

    if method_name in ["dwpp_omni", "dwpp_omni_clip", "dwpp_omni_clip_min_l", "dwpp_omni_clip_max_l"]:
        # calc look ahead distance (Adaptive Pure Pursuit)
        look_ahead_distance = calc_look_ahead_distance(current_velocity, method_name)
        # 全方位版: path は N x 3 ([x, y, theta]) を前提
        look_ahead_pose = calc_look_ahead_pose(current_idx, path, path_distances, look_ahead_distance)
        desired_velocity = calc_desired_velocity_vector_omnidirectional(current_pose, look_ahead_pose)
        is_accel = decide_accel_or_decel_omnidirectional(current_idx, path_distances, current_velocity)
        if method_name == "dwpp_omni":
            next_velocity_ref = calc_optimal_velocity_considering_dynamic_window_omnidirectional(
                current_velocity=current_velocity,
                desired_velocity=desired_velocity,
                is_accel=is_accel
            )
        else:
            # 比較手法: 理想ベクトルをそのまま速度指令値にする
            next_velocity_ref = desired_velocity.copy()
        break_constraints_flag = evaluate_accelaration_constraints_omnidirectional(current_velocity, next_velocity_ref)
        return next_velocity_ref, look_ahead_pose, break_constraints_flag, float("nan"), float("nan")

    if method_name == "dwpp":
        # DWPP は前方の複数点を候補とし、DW と交差する曲率の中から最大速度を与える候補を選ぶ
        is_accel = decide_accel_or_decel(current_idx, path_distances)
        next_velocity_ref, look_ahead_pos, curvature, regulated_v = calc_dwpp_velocity_with_auto_look_ahead(
            current_pose=current_pose,
            current_velocity=current_velocity,
            current_idx=current_idx,
            path=path,
            path_distances=path_distances,
            is_accel=is_accel,
            use_regulated_velocity=DWPP_USE_REGULATED_VELOCITY,
        )
    else:
        # 差動二輪版 (既存実装)
        look_ahead_distance = calc_look_ahead_distance(current_velocity, method_name)
        curvature, look_ahead_pos = calc_curvature_to_look_ahead_position(
            current_pose, current_idx, path, path_distances, look_ahead_distance
        )

        if method_name == "rpp":
            # calc regulated translational velocity (Regulated Pure Pursuit)
            regulated_v = calc_regulated_translational_velocity(curvature)
        elif method_name == "dwpp_fixed" and DWPP_USE_REGULATED_VELOCITY:
            regulated_v = calc_regulated_translational_velocity(curvature)
        else:
            regulated_v = V_MAX

        if method_name in ["pp", "app", "rpp"]:
            # calc translational velocity
            v_ref = calc_reference_translational_velocity(current_pose, path[-1])

            # regulate translational velocity
            if method_name == "rpp":
                v_ref = min(v_ref, regulated_v)

            # calc angular velocity
            w_ref = curvature * v_ref
            next_velocity_ref = np.array([v_ref, w_ref])

        else:
            # decide accel or decel
            is_accel = decide_accel_or_decel(current_idx, path_distances)
            # calc dynamic window and optimal next velocity
            next_velocity_ref = calc_optimal_velocity_considering_dynamic_window(
                current_velocity, regulated_v, curvature, is_accel
            )

    break_constraints_flag = evaluate_accelaration_constraints(current_velocity, next_velocity_ref)

    # debug用に、next_velocity_refと前方注視点の位置も返す
    return next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v

def evaluate_accelaration_constraints(current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> list[bool]:
    break_constraints_flag = [False, False]
    if current_velocity[0] - A_MAX * DT > next_velocity_ref[0] + ACCEL_CONSTRAINT_EPS or \
        current_velocity[0] + A_MAX * DT < next_velocity_ref[0] - ACCEL_CONSTRAINT_EPS:
        break_constraints_flag[0] = True
    if current_velocity[1] - AW_MAX * DT > next_velocity_ref[1] + ACCEL_CONSTRAINT_EPS or \
        current_velocity[1] + AW_MAX * DT < next_velocity_ref[1] - ACCEL_CONSTRAINT_EPS:
        break_constraints_flag[1] = True
    
    return break_constraints_flag

def calc_reference_translational_velocity(current_pose: np.ndarray, goal_pose: np.ndarray) -> float:
    v_ref = V_MAX
    
    # ゴールに近づいたら速度を制限する
    distance_to_goal = float(np.linalg.norm(goal_pose[:2] - current_pose[:2]))
    if distance_to_goal < APPROACH_VELOCITY_SCALING_DIST:
        v_ref = max(v_ref * distance_to_goal / APPROACH_VELOCITY_SCALING_DIST, MIN_APPROACH_LINEAR_VELOCITY)
    if distance_to_goal < GOAL_TORELANCE_DIST:
        v_ref = 0.0
    
    return v_ref

def calc_index(current_pose: np.ndarray, path: np.ndarray) -> np.intp:
    # current_pose: [x, y, theta]
    # ロボットの位置と、経路上の各位置間の距離を計算し、最も近い位置のインデックスを返す
    distances = np.linalg.norm(path[:, :2] - current_pose[:2], axis=1)
    idx = np.argmin(distances)
    
    return idx

def calc_path_distances(path: np.ndarray) -> np.ndarray:
    # 経路の距離の累積和を計算
    ## 点間の差を計算
    differences = np.diff(path[:, :2], axis=0)
    ## 各差のノルム（距離）を計算
    distances = np.linalg.norm(differences, axis=1)
    ## 累積距離を計算
    path_distances = np.concatenate(([0.0], np.cumsum(distances)))
    
    return path_distances


def calc_min_look_ahead_distance_from_velocity(current_velocity: np.ndarray, is_omni: bool = False) -> float:
    if is_omni:
        current_speed = float(np.linalg.norm(current_velocity[:2]))
    else:
        current_speed = abs(float(current_velocity[0]))
    return current_speed * DT

def calc_look_ahead_distance(current_velocity: np.ndarray, method_name: str) -> float:
    # calc look ahead distance
    if method_name == "dwpp_omni_clip_min_l":
        return max(
            MIN_LOOK_AHEAD_DISTANCE,
            calc_min_look_ahead_distance_from_velocity(current_velocity, is_omni=True),
        )
    if method_name == "dwpp_omni_clip_max_l":
        return MAX_LOOK_AHEAD_DISTANCE

    is_omni = method_name in ["dwpp_omni", "dwpp_omni_clip"]
    min_look_ahead_distance = calc_min_look_ahead_distance_from_velocity(current_velocity, is_omni=is_omni)
    max_look_ahead_distance = max(MAX_LOOK_AHEAD_DISTANCE, min_look_ahead_distance)

    if method_name in ["app", "rpp", "dwpp_fixed", "dwpp_wo_rpp", "dwpp_omni", "dwpp_omni_clip"]:
        if is_omni:
            current_speed = float(np.linalg.norm(current_velocity[:2]))
        else:
            current_speed = abs(float(current_velocity[0]))
        look_ahead_distance = LOOK_AHEAD_TIME * current_speed
        look_ahead_distance = float(np.clip(
            look_ahead_distance,
            min_look_ahead_distance,
            max_look_ahead_distance,
        ))
        # look_ahead_distance = MIN_LOOK_AHEAD_DISTANCE + (STATIC_LOOK_AHEAD_DISTANCE - MIN_LOOK_AHEAD_DISTANCE) / V_MAX * current_velocity[0]
    else:
        look_ahead_distance = max(MIN_LOOK_AHEAD_DISTANCE, min_look_ahead_distance)
        
    return look_ahead_distance

def calc_curvature_to_look_ahead_position(current_pose: np.ndarray, current_idx: np.intp, path: np.ndarray, path_distances: np.ndarray, look_ahead_distance: float ) -> tuple[float, np.ndarray]:
    # 前方注視点の位置を計算
    ## 現在位置の距離を取得
    current_distance = path_distances[current_idx]
    ## 前方注視点の距離を計算
    look_ahead_pos_distance = current_distance + look_ahead_distance
    ## 前方注視点のインデックスを取得
    look_ahead_idx = min(np.searchsorted(path_distances, look_ahead_pos_distance), len(path) - 1)
    ## 前方注視点の位置を取得
    look_ahead_pos = path[look_ahead_idx, :2]

    return calc_curvature_to_point(current_pose, look_ahead_pos), look_ahead_pos


def calc_curvature_to_point(current_pose: np.ndarray, target_pos: np.ndarray) -> float:
    # 任意の目標点に対し、現在姿勢から円弧で接続したときの曲率を計算
    look_ahead_angle = math.atan2(target_pos[1] - current_pose[1], target_pos[0] - current_pose[0]) - current_pose[2]
    distance = float(np.linalg.norm(target_pos - current_pose[:2]))
    if distance == 0.0:
        return 0.0
    return 2.0 * math.sin(look_ahead_angle) / distance

def calc_regulated_translational_velocity(curvature: float) -> float:
    # Curvature heuristics
    if curvature == 0.0:
        return V_MAX
    
    curvature_radius = 1.0 / abs(curvature)
    if curvature_radius <= REGULATED_LINEAR_SCALING_MIN_RADIUS:
        regulated_v = V_MAX * curvature_radius / REGULATED_LINEAR_SCALING_MIN_RADIUS
    else:
        regulated_v = V_MAX
    
    regulated_v = max(regulated_v, REGULATED_LINEAR_SCALING_MIN_SPEED)
    
    # Proximity heuristics is ommitted, this simulation does not include obstacles
    
    return regulated_v

def decide_accel_or_decel(current_idx: np.intp, path_distances: np.ndarray) -> bool:
    # 経路のゴールまでの距離を計算
    goal_distance = path_distances[-1] - path_distances[current_idx]
    
    # 制動距離を計算
    decel_distance = (V_MAX ** 2) / (2 * A_MAX)
    
    # ゴール距離が制動距離よりも長い場合、加速
    if goal_distance > decel_distance:
        return True
    else:
        return False


def calc_forward_point_indices(current_pose: np.ndarray, current_idx: np.intp, path: np.ndarray) -> np.ndarray:
    candidate_indices = np.arange(current_idx, len(path), dtype=np.intp)
    candidate_points = path[candidate_indices, :2]
    relative_points = candidate_points - current_pose[:2]
    distances = np.linalg.norm(relative_points, axis=1)

    c = math.cos(current_pose[2])
    s = math.sin(current_pose[2])
    body_x = c * relative_points[:, 0] + s * relative_points[:, 1]

    forward_mask = (body_x > 1e-9) & (distances > 1e-9)
    if np.any(forward_mask):
        return candidate_indices[forward_mask]

    nonzero_mask = distances > 1e-9
    return candidate_indices[nonzero_mask]


def apply_min_look_ahead_distance_to_indices(
    current_idx: np.intp,
    path_distances: np.ndarray,
    candidate_indices: np.ndarray,
    min_look_ahead_distance: float,
) -> np.ndarray:
    if len(candidate_indices) == 0:
        return candidate_indices

    current_distance = float(path_distances[current_idx])
    eligible_mask = path_distances[candidate_indices] >= (current_distance + min_look_ahead_distance - 1e-12)
    eligible_indices = candidate_indices[eligible_mask]
    return eligible_indices if len(eligible_indices) > 0 else candidate_indices


def calc_dynamic_window_bounds(current_velocity: np.ndarray) -> tuple[float, float, float, float]:
    dw_vmax = min(current_velocity[0] + A_MAX * DT, V_MAX)
    dw_vmin = max(current_velocity[0] - A_MAX * DT, V_MIN)
    dw_wmax = min(current_velocity[1] + AW_MAX * DT, W_MAX)
    dw_wmin = max(current_velocity[1] - AW_MAX * DT, W_MIN)
    return dw_vmin, dw_vmax, dw_wmin, dw_wmax


def calc_velocity_command_for_curvature(
    current_velocity: np.ndarray,
    regulated_v: float,
    curvature: float,
    prefer_high_speed: bool,
) -> tuple[np.ndarray, bool, float]:
    dw_vmin, dw_vmax, dw_wmin, dw_wmax = calc_dynamic_window_bounds(current_velocity)

    # regulated_v の考慮
    if dw_vmax > regulated_v:
        dw_vmax = max(dw_vmin, regulated_v)

    # Dynamic window と曲率直線の交点を計算
    velocity_candidates = [
        (dw_vmin, curvature * dw_vmin),
        (dw_vmax, curvature * dw_vmax),
    ]
    if curvature != 0.0:
        velocity_candidates.extend([
            (dw_wmin / curvature, dw_wmin),
            (dw_wmax / curvature, dw_wmax),
        ])

    valid_velocity_candidates = []
    for candidate in velocity_candidates:
        if dw_vmin <= candidate[0] <= dw_vmax and dw_wmin <= candidate[1] <= dw_wmax:
            valid_velocity_candidates.append(candidate)

    if len(valid_velocity_candidates) > 0:
        valid_velocity_candidates.sort(key=lambda candidate: candidate[0])
        selected_velocity = valid_velocity_candidates[-1] if prefer_high_speed else valid_velocity_candidates[0]
        return np.array(selected_velocity, dtype=float), True, 0.0

    # 交点がない場合は DW の頂点のうち曲率直線に最も近い点を採用する
    dw_coords = [
        (dw_vmin, dw_wmin),
        (dw_vmin, dw_wmax),
        (dw_vmax, dw_wmin),
        (dw_vmax, dw_wmax),
    ]
    distance_from_coords = []
    for point in dw_coords:
        dist = abs(curvature * point[0] - point[1]) / math.sqrt(curvature**2 + 1)
        distance_from_coords.append(dist)

    min_dist = min(distance_from_coords)
    min_dist_dw_coords = []
    for point, dist in zip(dw_coords, distance_from_coords):
        if dist == min_dist:
            min_dist_dw_coords.append(point)
    min_dist_dw_coords.sort(key=lambda point: point[0])
    selected_velocity = min_dist_dw_coords[-1] if prefer_high_speed else min_dist_dw_coords[0]
    return np.array(selected_velocity, dtype=float), False, min_dist


def calc_dwpp_velocity_with_auto_look_ahead(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    current_idx: np.intp,
    path: np.ndarray,
    path_distances: np.ndarray,
    is_accel: bool,
    use_regulated_velocity: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    forward_indices = calc_forward_point_indices(current_pose, current_idx, path)
    min_look_ahead_distance = calc_min_look_ahead_distance_from_velocity(current_velocity, is_omni=False)
    forward_indices = apply_min_look_ahead_distance_to_indices(
        current_idx=current_idx,
        path_distances=path_distances,
        candidate_indices=forward_indices,
        min_look_ahead_distance=min_look_ahead_distance,
    )

    # まれに候補が作れない場合は末端点を使って処理を継続する
    if len(forward_indices) == 0:
        fallback_idx = min(current_idx, len(path) - 1)
        forward_indices = np.array([fallback_idx], dtype=np.intp)

    last_candidate = None

    for candidate_idx in forward_indices:
        look_ahead_pos = path[candidate_idx, :2]
        curvature = calc_curvature_to_point(current_pose, look_ahead_pos)
        regulated_v = calc_regulated_translational_velocity(curvature) if use_regulated_velocity else V_MAX
        next_velocity_ref, has_intersection, distance_to_line = calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=regulated_v,
            curvature=curvature,
            prefer_high_speed=True,
        )

        candidate = {
            "velocity": next_velocity_ref,
            "look_ahead_pos": look_ahead_pos,
            "curvature": curvature,
            "regulated_v": regulated_v,
            "distance_to_line": distance_to_line,
            "path_distance": float(path_distances[candidate_idx]),
        }
        last_candidate = candidate

        if has_intersection:
            return (
                candidate["velocity"],
                candidate["look_ahead_pos"],
                float(candidate["curvature"]),
                float(candidate["regulated_v"]),
            )

    selected_candidate = last_candidate
    assert selected_candidate is not None
    return (
        selected_candidate["velocity"],
        selected_candidate["look_ahead_pos"],
        float(selected_candidate["curvature"]),
        float(selected_candidate["regulated_v"]),
    )

def calc_optimal_velocity_considering_dynamic_window(current_velocity: np.ndarray, regulated_v: float, curvature: float, is_accel: bool) -> np.ndarray:
    next_velocity, _, _ = calc_velocity_command_for_curvature(
        current_velocity=current_velocity,
        regulated_v=regulated_v,
        curvature=curvature,
        prefer_high_speed=is_accel,
    )
    return next_velocity


def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_path_theta(path: np.ndarray, idx: np.intp) -> float:
    if path.shape[1] >= 3:
        return float(path[idx, 2])

    # path が N x 2 の場合のフォールバック（接線角）
    if len(path) == 1:
        return 0.0
    prev_idx = max(idx - 1, 0)
    next_idx = min(idx + 1, len(path) - 1)
    delta = path[next_idx, :2] - path[prev_idx, :2]
    if float(np.linalg.norm(delta)) == 0.0:
        return 0.0
    return float(math.atan2(delta[1], delta[0]))


def calc_look_ahead_pose(current_idx: np.intp, path: np.ndarray, path_distances: np.ndarray, look_ahead_distance: float) -> np.ndarray:
    current_distance = path_distances[current_idx]
    look_ahead_pose_distance = current_distance + look_ahead_distance
    look_ahead_idx = min(np.searchsorted(path_distances, look_ahead_pose_distance), len(path) - 1)
    look_ahead_x = float(path[look_ahead_idx, 0])
    look_ahead_y = float(path[look_ahead_idx, 1])
    look_ahead_theta = calc_path_theta(path, look_ahead_idx)
    return np.array([look_ahead_x, look_ahead_y, look_ahead_theta])


def calc_desired_velocity_vector_omnidirectional(current_pose: np.ndarray, look_ahead_pose: np.ndarray) -> np.ndarray:
    # 位置誤差（world）
    dx = look_ahead_pose[0] - current_pose[0]
    dy = look_ahead_pose[1] - current_pose[1]
    
    # 距離
    distance = math.sqrt(dx**2 + dy**2)

    # 位置誤差をロボット座標へ変換
    c = math.cos(current_pose[2])
    s = math.sin(current_pose[2])
    ex_body = c * dx + s * dy
    ey_body = -s * dx + c * dy
    
    max_vel = np.sqrt(VX_MAX**2 + VY_MAX**2)
    e_theta = normalize_angle(look_ahead_pose[2] - current_pose[2])

    if max_vel <= 1e-9:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    if distance <= 1e-9:
        desired_w = float(np.clip(e_theta / max(DT, 1e-6), W_MIN, W_MAX))
        return np.array([0.0, 0.0, desired_w], dtype=float)
    
    # 並進方向の速度指令値の算出
    desired_vx = max_vel * ex_body / (distance + 1e-6)  # 0除算を避けるため、小さな値を加える
    desired_vy = max_vel * ey_body / (distance + 1e-6)

    # 位置偏差を補正するのに要する時間
    time_to_goal_position = distance / max_vel
    
    # 姿勢偏差を補正するのに設ける時間
    time_to_goal_orientation = max(K * time_to_goal_position, DT)
    
    # 回転速度の算出
    desired_w = float(np.clip(e_theta / time_to_goal_orientation, W_MIN, W_MAX))

    desired_velocity = np.array([desired_vx, desired_vy, desired_w], dtype=float)
    return np.nan_to_num(desired_velocity, nan=0.0, posinf=0.0, neginf=0.0)


def decide_accel_or_decel_omnidirectional(current_idx: np.intp, path_distances: np.ndarray, current_velocity: np.ndarray) -> bool:
    goal_distance = path_distances[-1] - path_distances[current_idx]
    speed = float(np.linalg.norm(current_velocity[:2]))
    # 軸別の加速度制約が異なる場合は、保守的に小さい方を使う
    trans_accel_limit = min(AX_MAX, AY_MAX)
    if trans_accel_limit <= 0.0:
        return False
    decel_distance = (speed ** 2) / (2.0 * trans_accel_limit)
    return goal_distance > decel_distance


def evaluate_accelaration_constraints_omnidirectional(current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> list[bool]:
    limits = np.array([AX_MAX, AY_MAX, AW_MAX]) * DT
    break_constraints = np.abs(next_velocity_ref - current_velocity) > (limits + ACCEL_CONSTRAINT_EPS)
    return [bool(break_constraints[0]), bool(break_constraints[1]), bool(break_constraints[2])]


def calc_dynamic_window_bounds_omnidirectional(current_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dw_min = np.array([
        max(current_velocity[0] - AX_MAX * DT, VX_MIN),
        max(current_velocity[1] - AY_MAX * DT, VY_MIN),
        max(current_velocity[2] - AW_MAX * DT, W_MIN),
    ])
    dw_max = np.array([
        min(current_velocity[0] + AX_MAX * DT, VX_MAX),
        min(current_velocity[1] + AY_MAX * DT, VY_MAX),
        min(current_velocity[2] + AW_MAX * DT, W_MAX),
    ])
    return dw_min, dw_max


def calc_clipped_velocity_omnidirectional_without_dynamic_window(
    current_velocity: np.ndarray,
    desired_velocity: np.ndarray
) -> np.ndarray:
    # 比較手法: 理想ベクトルをDW境界で成分ごとにクリッピング
    dw_min, dw_max = calc_dynamic_window_bounds_omnidirectional(current_velocity)
    return np.clip(desired_velocity, dw_min, dw_max)


def calc_optimal_velocity_considering_dynamic_window_omnidirectional(
    current_velocity: np.ndarray,
    desired_velocity: np.ndarray,
    is_accel: bool
) -> np.ndarray:
    # dynamic window を直方体として構築
    dw_min, dw_max = calc_dynamic_window_bounds_omnidirectional(current_velocity)
    desired_velocity = np.nan_to_num(desired_velocity, nan=0.0, posinf=0.0, neginf=0.0)

    if float(np.linalg.norm(desired_velocity)) == 0.0:
        return np.clip(np.zeros(3), dw_min, dw_max)

    intersects, alpha_min, alpha_max = calc_ray_box_intersection_alpha_range(desired_velocity, dw_min, dw_max)
    if intersects:
        alpha = alpha_max if is_accel else alpha_min
    else:
        alpha = calc_closest_alpha_to_box_from_ray(desired_velocity, dw_min, dw_max, prefer_large_alpha=is_accel)

    if not np.isfinite(alpha):
        return np.clip(np.zeros(3), dw_min, dw_max)

    return np.clip(np.nan_to_num(alpha * desired_velocity, nan=0.0, posinf=0.0, neginf=0.0), dw_min, dw_max)


def calc_ray_box_intersection_alpha_range(ray_direction: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> tuple[bool, float, float]:
    alpha_low = 0.0
    alpha_high = float("inf")
    eps = 1e-12

    for d, lower, upper in zip(ray_direction, box_min, box_max):
        if abs(d) < eps:
            # この軸は alpha に依存しない。0 が範囲外なら交差不可。
            if lower <= 0.0 <= upper:
                continue
            return False, 0.0, 0.0

        a1 = lower / d
        a2 = upper / d
        axis_low = min(a1, a2)
        axis_high = max(a1, a2)

        alpha_low = max(alpha_low, axis_low)
        alpha_high = min(alpha_high, axis_high)

        if alpha_low > alpha_high:
            return False, 0.0, 0.0

    if alpha_high < 0.0:
        return False, 0.0, 0.0

    alpha_low = max(alpha_low, 0.0)
    if alpha_low <= alpha_high:
        return True, alpha_low, alpha_high
    return False, 0.0, 0.0


def calc_closest_alpha_to_box_from_ray(
    ray_direction: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
    prefer_large_alpha: bool
) -> float:
    eps = 1e-12
    tol = 1e-10
    best_alpha = 0.0
    best_dist = float("inf")

    def eval_alpha(alpha: float):
        nonlocal best_alpha, best_dist
        point = alpha * ray_direction
        nearest = np.clip(point, box_min, box_max)
        dist = float(np.sum((point - nearest) ** 2))
        if dist + tol < best_dist:
            best_dist = dist
            best_alpha = alpha
            return
        if abs(dist - best_dist) <= tol:
            if prefer_large_alpha and alpha > best_alpha:
                best_alpha = alpha
            if (not prefer_large_alpha) and alpha < best_alpha:
                best_alpha = alpha

    # 区分点（各軸の範囲境界を ray が横切る alpha）
    breakpoints = [0.0]
    for d, lower, upper in zip(ray_direction, box_min, box_max):
        if abs(d) < eps:
            continue
        breakpoints.append(lower / d)
        breakpoints.append(upper / d)

    breakpoints = sorted(set(float(b) for b in breakpoints if np.isfinite(b) and b >= 0.0))
    if len(breakpoints) == 0:
        return 0.0

    for alpha in breakpoints:
        eval_alpha(alpha)

    def calc_interval_quadratic_coeff(a: float, b: float | None) -> tuple[float, float]:
        if b is None:
            probe = a + 1.0
        else:
            probe = 0.5 * (a + b)

        qa = 0.0
        qb = 0.0
        for d, lower, upper in zip(ray_direction, box_min, box_max):
            p = probe * d
            if p < lower - eps:
                qa += d ** 2
                qb += -2.0 * lower * d
            elif p > upper + eps:
                qa += d ** 2
                qb += -2.0 * upper * d
            # inside の場合は寄与ゼロ
        return qa, qb

    for i in range(len(breakpoints) - 1):
        left = breakpoints[i]
        right = breakpoints[i + 1]
        qa, qb = calc_interval_quadratic_coeff(left, right)
        if qa <= eps:
            continue
        alpha_star = -qb / (2.0 * qa)
        if left <= alpha_star <= right:
            eval_alpha(alpha_star)

    # 最終区間 [last, +inf)
    last = breakpoints[-1]
    qa, qb = calc_interval_quadratic_coeff(last, None)
    if qa > eps:
        alpha_star = -qb / (2.0 * qa)
        if alpha_star >= last:
            eval_alpha(alpha_star)

    return best_alpha
