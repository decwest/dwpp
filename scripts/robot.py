import numpy as np
from config import DT, A_MAX, AW_MAX, V_MAX, V_MIN, W_MAX, W_MIN

def forward_simulation_differential(current_pose: np.ndarray, current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    next_velocity = calc_accel_constrained_velocity(current_velocity, next_velocity_ref)
    
    if next_velocity[1] == 0.0:
        # 直進運動
        next_pose = current_pose + np.array([
            next_velocity[0] * np.cos(current_pose[2]),
            next_velocity[0] * np.sin(current_pose[2]),
            next_velocity[1]
        ]) * DT
    else:
        # 旋回運動
        R = next_velocity[0] / next_velocity[1]
        next_pose = current_pose + np.array([
            R * (np.sin(current_pose[2] + next_velocity[1] * DT) - np.sin(current_pose[2])),
            R * (-np.cos(current_pose[2] + next_velocity[1] * DT) + np.cos(current_pose[2])),
            next_velocity[1] * DT
        ])
    
    return next_pose, next_velocity
    

def calc_accel_constrained_velocity(current_velocity: np.ndarray, next_velocity_ref: np.ndarray) -> np.ndarray:
    # 加速度制約によるクリッピング
    next_velocity = next_velocity_ref.copy()
    ## 並進加速度の制約
    if next_velocity_ref[0] > min(current_velocity[0] + A_MAX * DT, V_MAX):
        next_velocity[0] = min(current_velocity[0] + A_MAX * DT, V_MAX)
    elif next_velocity_ref[0] < max(current_velocity[0] - A_MAX * DT, V_MIN):
        next_velocity[0] = max(current_velocity[0] - A_MAX * DT, V_MIN)
    ## 角速度の制約
    if next_velocity_ref[1] > min(current_velocity[1] + AW_MAX * DT, W_MAX):
        next_velocity[1] = min(current_velocity[1] + AW_MAX * DT, W_MAX)
    elif next_velocity_ref[1] < max(current_velocity[1] - AW_MAX * DT, W_MIN):
        next_velocity[1] = max(current_velocity[1] - AW_MAX * DT, W_MIN)
    
    return next_velocity


def forward_simulation_omnidirectional(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    next_velocity_ref: np.ndarray,
    vx_min: float = -V_MAX,
    vx_max: float = V_MAX,
    vy_min: float = -V_MAX,
    vy_max: float = V_MAX,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
    a_xy_max: float = A_MAX,
    aw_max: float = AW_MAX
) -> tuple[np.ndarray, np.ndarray]:
    # next_velocity_ref はロボット座標系 [vx, vy, w] を想定
    next_velocity = calc_accel_constrained_velocity_omnidirectional(
        current_velocity=current_velocity,
        next_velocity_ref=next_velocity_ref,
        vx_min=vx_min,
        vx_max=vx_max,
        vy_min=vy_min,
        vy_max=vy_max,
        w_min=w_min,
        w_max=w_max,
        a_xy_max=a_xy_max,
        aw_max=aw_max
    )

    theta = current_pose[2]
    vx_body, vy_body, w = next_velocity

    # ロボット座標系の並進速度をワールド座標系へ変換
    vx_world = vx_body * np.cos(theta) - vy_body * np.sin(theta)
    vy_world = vx_body * np.sin(theta) + vy_body * np.cos(theta)

    next_pose = current_pose + np.array([vx_world, vy_world, w]) * DT
    return next_pose, next_velocity


def calc_accel_constrained_velocity_omnidirectional(
    current_velocity: np.ndarray,
    next_velocity_ref: np.ndarray,
    vx_min: float = -V_MAX,
    vx_max: float = V_MAX,
    vy_min: float = -V_MAX,
    vy_max: float = V_MAX,
    w_min: float = W_MIN,
    w_max: float = W_MAX,
    a_xy_max: float = A_MAX,
    aw_max: float = AW_MAX
) -> np.ndarray:
    # [vx, vy, w] それぞれに速度制約と加速度制約を適用
    velocity_min = np.array([vx_min, vy_min, w_min], dtype=float)
    velocity_max = np.array([vx_max, vy_max, w_max], dtype=float)
    accel_max = np.array([a_xy_max, a_xy_max, aw_max], dtype=float)

    upper = np.minimum(current_velocity + accel_max * DT, velocity_max)
    lower = np.maximum(current_velocity - accel_max * DT, velocity_min)
    next_velocity = np.clip(next_velocity_ref, lower, upper)
    return next_velocity
