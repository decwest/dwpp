from pure_pursuit import pure_pursuit
from path import step_curves
from robot import forward_simulation_differential, forward_simulation_omnidirectional
from time import perf_counter
import numpy as np
from config import DT, method_name_list, GOAL_TORELANCE_DIST, A_MAX, AW_MAX, V_MIN, V_MAX, W_MIN, W_MAX
from draw import draw_animation, draw_paths, draw_velocity_profile, draw_vw_plane_plot
from stats import calc_break_constraints_rate, calc_heading_rmse, calc_rmse
from collections import defaultdict
import concurrent.futures
import pickle

POST_GOAL_HEADING_KP_DIFF = 2.0


def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def calc_goal_heading(path: np.ndarray) -> float:
    if path.shape[1] >= 3:
        return float(path[-1, 2])
    if len(path) <= 1:
        return 0.0
    delta = path[-1, :2] - path[-2, :2]
    if float(np.linalg.norm(delta)) <= 1e-12:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def calc_goal_alignment_velocity_ref_diffdrive(
    current_pose: np.ndarray,
    current_velocity: np.ndarray,
    goal_heading: float,
) -> np.ndarray:
    heading_error = normalize_angle(goal_heading - current_pose[2])
    w_target = float(np.clip(POST_GOAL_HEADING_KP_DIFF * heading_error, W_MIN, W_MAX))

    v_lower = max(float(current_velocity[0]) - A_MAX * DT, V_MIN)
    v_upper = min(float(current_velocity[0]) + A_MAX * DT, V_MAX)
    w_lower = max(float(current_velocity[1]) - AW_MAX * DT, W_MIN)
    w_upper = min(float(current_velocity[1]) + AW_MAX * DT, W_MAX)

    v_ref = float(np.clip(0.0, v_lower, v_upper))
    w_ref = float(np.clip(w_target, w_lower, w_upper))
    return np.array([v_ref, w_ref], dtype=float)


def simulation(path: np.ndarray, path_name: str, initial_pose: np.ndarray, draw: bool, animation: bool)-> tuple[dict, dict, dict]:
    # 一つの曲線に対するシミュレーションと描画

    robot_poses_dict = defaultdict(list)
    robot_velocities_dict = defaultdict(list)
    robot_ref_velocities_dict = defaultdict(list)
    look_ahead_positions_dict = defaultdict(list)
    break_constraints_flags_dict = defaultdict(list)
    curvatures_flags_dict = defaultdict(list)
    regulated_vs_flags_dict = defaultdict(list)
    time_stamp_dict = defaultdict(list)

    for method_name in method_name_list:
        # print(f"Method: {method_name}")
        is_omni = method_name == "dwpp_omni"
        
        # 初期設定
        ## ロボットの状態
        current_pose = initial_pose
        current_velocity = np.array([0.0, 0.0, 0.0]) if is_omni else np.array([0.0, 0.0])
        # データの保存
        robot_poses = [current_pose]
        robot_velocities = [current_velocity]
        robot_ref_velocities = [current_velocity]
        look_ahead_positions = []
        break_constraints_flags = [[False, False, False]] if is_omni else [[False, False]]
        curvatures = [float("nan")] if is_omni else [0.0]
        regulated_vs = [float("nan")] if is_omni else [0.0]
        time_stamps = [0.0]
        
        sim_start_time = perf_counter()
        goal_xy = path[-1, :2]
        goal_heading = calc_goal_heading(path)
        
        while True:
            # 終了条件
            if np.all(np.isclose(current_velocity, 0.0)) and len(robot_velocities) > 1:
                break

            is_dwpp_goal_align = (
                (method_name == "dwpp")
                and (float(np.linalg.norm(current_pose[:2] - goal_xy)) <= GOAL_TORELANCE_DIST)
                and (abs(normalize_angle(goal_heading - current_pose[2])) > 1e-3)
            )

            if is_dwpp_goal_align:
                next_velocity_ref = calc_goal_alignment_velocity_ref_diffdrive(
                    current_pose=current_pose,
                    current_velocity=current_velocity,
                    goal_heading=goal_heading,
                )
                look_ahead_pos = path[-1, :2]
                break_constraints_flag = [False, False]
                curvature = 0.0
                regulated_v = 0.0
            else:
                next_velocity_ref, look_ahead_pos, break_constraints_flag,\
                    curvature, regulated_v = pure_pursuit(current_pose, current_velocity, path, method_name)

            if is_omni:
                next_pose, next_velocity = forward_simulation_omnidirectional(current_pose, current_velocity, next_velocity_ref)
            else:
                next_pose, next_velocity = forward_simulation_differential(current_pose, current_velocity, next_velocity_ref)
            
            robot_poses.append(next_pose)
            robot_velocities.append(next_velocity)
            robot_ref_velocities.append(next_velocity_ref)
            look_ahead_positions.append(look_ahead_pos)
            break_constraints_flags.append(break_constraints_flag)
            curvatures.append(curvature)
            regulated_vs.append(regulated_v)
            time_stamps.append(time_stamps[-1] + DT)
            
            current_pose = next_pose
            current_velocity = next_velocity
            
        sim_end_time = perf_counter()
        # print(f"Simulation time: {sim_end_time - sim_start_time}s")
        
        robot_poses_dict[method_name] = robot_poses
        robot_velocities_dict[method_name] = robot_velocities
        robot_ref_velocities_dict[method_name] = robot_ref_velocities
        look_ahead_positions_dict[method_name] = look_ahead_positions
        break_constraints_flags_dict[method_name] = break_constraints_flags
        curvatures_flags_dict[method_name] = curvatures
        regulated_vs_flags_dict[method_name] = regulated_vs
        time_stamp_dict[method_name] = time_stamps

    # 統計量の算出
    # RMSEの算出
    cross_track_rmse_dict = calc_rmse(robot_poses_dict, path)
    heading_rmse_dict = calc_heading_rmse(robot_poses_dict, path)
    
    # 制約違反した割合の算出
    break_constraints_rate_dict = calc_break_constraints_rate(break_constraints_flags_dict)
    
    # pickleで保存
    with open(f"../results/{path_name}/robot_poses_dict.pkl", "wb") as f:
        pickle.dump(robot_poses_dict, f)
    with open(f"../results/{path_name}/robot_velocities_dict.pkl", "wb") as f:
        pickle.dump(robot_velocities_dict, f)
    with open(f"../results/{path_name}/robot_ref_velocities_dict.pkl", "wb") as f:
        pickle.dump(robot_ref_velocities_dict, f)
    with open(f"../results/{path_name}/look_ahead_positions_dict.pkl", "wb") as f:
        pickle.dump(look_ahead_positions_dict, f)
    with open(f"../results/{path_name}/break_constraints_flags_dict.pkl", "wb") as f:
        pickle.dump(break_constraints_flags_dict, f)
    with open(f"../results/{path_name}/curvatures_flags_dict.pkl", "wb") as f:
        pickle.dump(curvatures_flags_dict, f)
    with open(f"../results/{path_name}/regulated_vs_flags_dict.pkl", "wb") as f:
        pickle.dump(regulated_vs_flags_dict, f)
    with open(f"../results/{path_name}/time_stamp_dict.pkl", "wb") as f:
        pickle.dump(time_stamp_dict, f)
    with open(f"../results/{path_name}/rmse_dict.pkl", "wb") as f:
        pickle.dump(cross_track_rmse_dict, f)
    with open(f"../results/{path_name}/heading_rmse_dict.pkl", "wb") as f:
        pickle.dump(heading_rmse_dict, f)
    with open(f"../results/{path_name}/break_constraints_rate_dict.pkl", "wb") as f:
        pickle.dump(break_constraints_rate_dict, f)
    
    if not draw:
        return cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict
    
    for method_name in method_name_list:
        robot_poses = np.array(robot_poses_dict[method_name])
        look_ahead_positions = np.array(look_ahead_positions_dict[method_name])
        robot_velocities = np.array(robot_velocities_dict[method_name])
        robot_ref_velocities = np.array(robot_ref_velocities_dict[method_name])
        break_constraints_flags = np.array(break_constraints_flags_dict[method_name])
        curvatures = np.array(curvatures_flags_dict[method_name])
        regulated_vs = np.array(regulated_vs_flags_dict[method_name])
        time_stamps = np.array(time_stamp_dict[method_name])
            
        if method_name != "dwpp_omni":
            # 速度プロファイルの描画（差動二輪版のみ）
            draw_velocity_profile(time_stamps, robot_velocities, robot_ref_velocities, break_constraints_flags, method_name, path_name)
        
        if not animation:
            continue
        
        # 各手法ごとのアニメーションの描画
        draw_animation(path, robot_poses, look_ahead_positions, method_name, path_name)
        
        if method_name in ["rpp", "dwpp", "dwpp_wo_rpp"]:
            draw_vw_plane_plot(robot_velocities, robot_ref_velocities, time_stamps, curvatures, regulated_vs, method_name, path_name)
        
    # 軌跡の描画
    draw_paths(path, robot_poses_dict, path_name)
    
    return cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict


def simulate_path(idx, path, initial_pose, prefix, draw=True, animation=False):
    """
    並列処理させたい実行単位(ワーカー)。
    シミュレーション結果(cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict)を返す。
    """
    path_name = f"{prefix}_{idx}"
    cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict = simulation(path, path_name, initial_pose, draw=draw, animation=animation)
    return cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict

# 事前に sin_curves / step_curves を用意
# paths_sin = sin_curves()
paths_step = step_curves()

# 結果をまとめるための辞書 (質問文中の total_***_dict を想定)
# たとえば、各 method_name をキーとして list を用意するなど
total_cross_track_rmse_dict = defaultdict(list)
total_heading_rmse_dict = defaultdict(list)
total_break_constraints_rate_dict = defaultdict(list)

# プロセスプールで並列実行
with concurrent.futures.ProcessPoolExecutor() as executor:
    # sin_curves 用のタスクをすべて投入
    # future_sin = [
    #     executor.submit(simulate_path, idx, path, np.array([0.0, 0.0, np.pi/2]), 'sin_curve')
    #     for idx, path in enumerate(paths_sin)
    # ]
    # step_curves 用のタスクをすべて投入
    future_step = [
        executor.submit(simulate_path, idx, path, np.array([0.0, 0.0, 0.0]), 'step_curve')
        for idx, path in enumerate(paths_step)
    ]
    
    # sin_curves の結果を回収
    # for future in concurrent.futures.as_completed(future_sin):
    #     rmse_dict, break_constraints_rate_dict = future.result()
        
    #     for method_name, rmse in rmse_dict.items():
    #         total_rmse_dict[method_name].append(rmse)
    #     for method_name, rate in break_constraints_rate_dict.items():
    #         total_break_constraints_rate_dict[method_name].append(rate)
    
    # step_curves の結果を回収
    for future in concurrent.futures.as_completed(future_step):
        cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict = future.result()
        
        for method_name, rmse in cross_track_rmse_dict.items():
            total_cross_track_rmse_dict[method_name].append(rmse)
        for method_name, rmse in heading_rmse_dict.items():
            total_heading_rmse_dict[method_name].append(rmse)
        for method_name, rate in break_constraints_rate_dict.items():
            total_break_constraints_rate_dict[method_name].append(rate)

# これで sin_curves と step_curves に対するシミュレーション結果が
# 並列に計算され、total_rmse_dict / total_break_constraints_rate_dict に格納される

# ファイルに出力
with open("../results/results.txt", "w") as f:
    for method_name, total_rmse in total_cross_track_rmse_dict.items():
        print(f"Method: {method_name}", file=f)
        print(f"Cross Track RMSE mean: {np.mean(total_rmse)}", file=f)
        print(f"Cross Track RMSE std: {np.std(total_rmse)}", file=f)
        print(file=f)  # 改行用

    for method_name, total_rmse in total_heading_rmse_dict.items():
        print(f"Method: {method_name}", file=f)
        print(f"Heading RMSE mean: {np.mean(total_rmse)}", file=f)
        print(f"Heading RMSE std: {np.std(total_rmse)}", file=f)
        print(file=f)  # 改行用

    for method_name, total_rate in total_break_constraints_rate_dict.items():
        print(f"Method: {method_name}", file=f)
        print(f"Break constraints rate mean: {np.mean(total_rate)}", file=f)
        print(f"Break constraints rate std: {np.std(total_rate)}", file=f)
        print(file=f)  # 改行用
