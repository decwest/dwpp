import numpy as np
import math
from config import MIN_LOOK_AHEAD_DISTANCE, MAX_LOOK_AHEAD_DISTANCE, LOOK_AHEAD_TIME, V_MAX, V_MIN, W_MAX, W_MIN, \
    A_MAX, AW_MAX, VX_MAX, VX_MIN, VY_MAX, VY_MIN, AX_MAX, AY_MAX, DT, APPROACH_VELOCITY_SCALING_DIST, \
    MIN_APPROACH_LINEAR_VELOCITY, GOAL_TORELANCE_DIST, REGULATED_LINEAR_SCALING_MIN_RADIUS, \
    REGULATED_LINEAR_SCALING_MIN_SPEED, K

ACCEL_CONSTRAINT_EPS = 1e-10

def pure_pursuit(current_pose: np.ndarray, current_velocity: np.ndarray, path: np.ndarray, method_name: str)\
    -> tuple[np.ndarray, np.ndarray, list[bool], float, float]:
    # calc index of current position
    current_idx = calc_index(current_pose, path)
    
    # calc distances between initial position and each position
    path_distances = calc_path_distances(path)
    
    # calc look ahead distance (Adaptive Pure Pursuit)
    look_ahead_distance = calc_look_ahead_distance(current_velocity, method_name)

    if method_name in ["dwpp_omni", "dwpp_omni_clip", "dwpp_omni_clip_min_l", "dwpp_omni_clip_max_l"]:
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

    # 差動二輪版 (既存実装)
    curvature, look_ahead_pos = calc_curvature_to_look_ahead_position(current_pose, current_idx, path, path_distances, look_ahead_distance)

    if method_name in ["rpp", "dwpp"]:
        # calc regulated translational velocity (Regulated Pure Pursuit)
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
        next_velocity_ref = calc_optimal_velocity_considering_dynamic_window(current_velocity, regulated_v, curvature, is_accel)

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

def calc_look_ahead_distance(current_velocity: np.ndarray, method_name: str) -> float:
    # calc look ahead distance
    if method_name == "dwpp_omni_clip_min_l":
        return MIN_LOOK_AHEAD_DISTANCE
    if method_name == "dwpp_omni_clip_max_l":
        return MAX_LOOK_AHEAD_DISTANCE

    if method_name in ["app", "rpp", "dwpp", "dwpp_wo_rpp", "dwpp_omni", "dwpp_omni_clip"]:
        if method_name in ["dwpp_omni", "dwpp_omni_clip"]:
            current_speed = float(np.linalg.norm(current_velocity[:2]))
        else:
            current_speed = current_velocity[0]
        look_ahead_distance = LOOK_AHEAD_TIME * current_speed
        look_ahead_distance = min(max(look_ahead_distance, MIN_LOOK_AHEAD_DISTANCE), MAX_LOOK_AHEAD_DISTANCE)
        # look_ahead_distance = MIN_LOOK_AHEAD_DISTANCE + (STATIC_LOOK_AHEAD_DISTANCE - MIN_LOOK_AHEAD_DISTANCE) / V_MAX * current_velocity[0]
    else:
        look_ahead_distance = MIN_LOOK_AHEAD_DISTANCE
        
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
    
    # 曲率を計算
    ## 前方注視点に向けた角度を計算
    look_ahead_angle = (math.atan2(look_ahead_pos[1] - current_pose[1], look_ahead_pos[0] - current_pose[0]) - current_pose[2])
    ## 前方注視点までの距離を計算
    L = float(np.linalg.norm(look_ahead_pos - current_pose[:2]))
    if L == 0.0:
        return 0.0, look_ahead_pos
    ## 曲率を計算
    curvature = 2.0 * math.sin(look_ahead_angle) / L
    
    return curvature, look_ahead_pos

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

def calc_optimal_velocity_considering_dynamic_window(current_velocity: np.ndarray, regulated_v: float, curvature: float, is_accel: bool) -> np.ndarray:
    # dynamic windowを作る
    dw_vmax = min(current_velocity[0] + A_MAX * DT, V_MAX)
    dw_vmin = max(current_velocity[0] - A_MAX * DT, V_MIN)
    dw_wmax = min(current_velocity[1] + AW_MAX * DT, W_MAX)
    dw_wmin = max(current_velocity[1] - AW_MAX * DT, W_MIN)
    
    # regulated_vの考慮
    if dw_vmax > regulated_v:
        dw_vmax = max(dw_vmin, regulated_v)
        # print("Regulated v is considered.")
        # print(f"dw_vmax: {dw_vmax}")
    
    # Dynamic windowと曲率直線の交点を計算
    ## DWの4辺との交点を算出（解析解を代入）
    velocity_candidates = []
    p1 = (dw_vmin, curvature * dw_vmin)
    p2 = (dw_vmax, curvature * dw_vmax)
    velocity_candidates.append(p1)
    velocity_candidates.append(p2)
    if curvature != 0.0:
        p3 = (dw_wmin / curvature, dw_wmin)
        p4 = (dw_wmax / curvature, dw_wmax)
        velocity_candidates.append(p3)
        velocity_candidates.append(p4)
    ## 交点がDWの範囲内にあるか確認
    valid_velocity_candidates = []
    for v in velocity_candidates:
        if dw_vmin <= v[0] <= dw_vmax and dw_wmin <= v[1] <= dw_wmax:
            valid_velocity_candidates.append(v)
    
    # 最適な速度を計算
    ## 交点がある場合
    if len(valid_velocity_candidates) > 0:
        # 加速減速に基づいて目標速度を決める
        # 並進速度でソートする
        valid_velocity_candidates.sort(key=lambda x: x[0])
        # 加速する場合は、最も早い速度を選択
        if is_accel:
            next_velocity = valid_velocity_candidates[-1]
        # 減速する場合は、最も遅い速度を選択
        else:
            next_velocity = valid_velocity_candidates[0]
    
    ## 交点がない場合
    else:
        # 4頂点との距離を計算（距離関数は凸関数で、長方形は凸領域であることから、極値は4頂点上でとる）
        distance_from_coords = []
        dw_coords = [
            (dw_vmin, dw_wmin),
            (dw_vmin, dw_wmax),
            (dw_vmax, dw_wmin),
            (dw_vmax, dw_wmax),
        ]
        for p in dw_coords:
            # 曲率直線から頂点までの距離を算出
            dist = abs(curvature * p[0] - p[1]) / math.sqrt(curvature**2 + 1)
            distance_from_coords.append(dist)
        
        # 最短距離を見つける
        min_dist = min(distance_from_coords)
        # 最短距離となる速度候補のリストを作る
        min_dist_dw_coords = []
        for p, dist in zip(dw_coords, distance_from_coords):
            if dist == min_dist:
                min_dist_dw_coords.append(p)
        # 最短距離となる速度候補のリストを並進速度でソート
        min_dist_dw_coords.sort(key=lambda x: x[0])
        # 加速する場合は、最も早い速度を選択
        if is_accel:
            next_velocity = min_dist_dw_coords[-1]
        # 減速する場合は、最も遅い速度を選択
        else:
            next_velocity = min_dist_dw_coords[0]
    
    return np.array(next_velocity)


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
    
    # 並進方向の速度指令値の算出
    desired_vx = max_vel * ex_body / (distance + 1e-6)  # 0除算を避けるため、小さな値を加える
    desired_vy = max_vel * ey_body / (distance + 1e-6)

    # 位置偏差を補正するのに要する時間
    time_to_goal_position = distance / max_vel

    # 姿勢誤差
    e_theta = normalize_angle(look_ahead_pose[2] - current_pose[2])
    
    # 姿勢偏差を補正するのに設ける時間
    time_to_goal_orientation = K * time_to_goal_position
    
    # 回転速度の算出
    desired_w = e_theta / time_to_goal_orientation

    return np.array([desired_vx, desired_vy, desired_w])


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

    if float(np.linalg.norm(desired_velocity)) == 0.0:
        return np.clip(np.zeros(3), dw_min, dw_max)

    intersects, alpha_min, alpha_max = calc_ray_box_intersection_alpha_range(desired_velocity, dw_min, dw_max)
    if intersects:
        alpha = alpha_max if is_accel else alpha_min
    else:
        alpha = calc_closest_alpha_to_box_from_ray(desired_velocity, dw_min, dw_max, prefer_large_alpha=is_accel)

    return np.clip(alpha * desired_velocity, dw_min, dw_max)


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
