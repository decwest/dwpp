from collections import defaultdict
from time import perf_counter
import concurrent.futures
import pickle
from pathlib import Path

import numpy as np

from config import DT, GOAL_REACH_TOLERANCE_DIST_OMNI, GOAL_REACH_TOLERANCE_HEADING
from path import step_curves, straight_line_heading_step_curve
from pure_pursuit import pure_pursuit
from robot import forward_simulation_omnidirectional
from stats import calc_break_constraints_rate, calc_heading_rmse, calc_rmse


OMNI_METHOD_NAME_LIST = ["dwpp_omni", "dwpp_omni_clip"]
MAX_SIM_STEPS = 20000
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def calc_goal_reach_time(
    robot_poses: np.ndarray,
    time_stamps: np.ndarray,
    goal_pose: np.ndarray,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
) -> float:
    distances_to_goal = np.linalg.norm(robot_poses[:, :2] - goal_pose[:2], axis=1)
    heading_errors = np.abs(normalize_angle(robot_poses[:, 2] - goal_pose[2]))
    reached_indices = np.where(
        (distances_to_goal <= goal_tolerance_dist) & (heading_errors <= goal_tolerance_heading)
    )[0]
    if len(reached_indices) == 0:
        return float("nan")
    return float(time_stamps[reached_indices[0]])


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def is_goal_reached(
    current_pose: np.ndarray,
    goal_pose: np.ndarray,
    goal_tolerance_dist: float,
    goal_tolerance_heading: float,
) -> bool:
    distance_to_goal = float(np.linalg.norm(current_pose[:2] - goal_pose[:2]))
    heading_error = float(abs(normalize_angle(current_pose[2] - goal_pose[2])))
    return (distance_to_goal <= goal_tolerance_dist) and (heading_error <= goal_tolerance_heading)


def write_path_summary_text(
    summary_path: Path,
    cross_track_rmse_dict: dict,
    heading_rmse_dict: dict,
    break_constraints_rate_dict: dict,
    goal_reach_time_dict: dict
) -> None:
    with open(summary_path, "w") as f:
        for method_name, rmse in cross_track_rmse_dict.items():
            print(f"Method: {method_name}", file=f)
            print(f"Cross Track RMSE mean: {rmse}", file=f)
            print(f"Cross Track RMSE std: 0.0", file=f)
            print(file=f)

        for method_name, rmse in heading_rmse_dict.items():
            print(f"Method: {method_name}", file=f)
            print(f"Heading RMSE mean: {rmse}", file=f)
            print(f"Heading RMSE std: 0.0", file=f)
            print(file=f)

        for method_name, rate in break_constraints_rate_dict.items():
            rate = np.array(rate, dtype=float)
            print(f"Method: {method_name}", file=f)
            print(f"Break constraints rate mean: {rate}", file=f)
            print(f"Break constraints rate std: {np.zeros_like(rate)}", file=f)
            print(file=f)

        for method_name, goal_reach_time in goal_reach_time_dict.items():
            reached = np.isfinite(goal_reach_time)
            print(f"Method: {method_name}", file=f)
            print(f"Goal reach time mean [s]: {goal_reach_time}", file=f)
            print(f"Goal reach time std [s]: 0.0", file=f)
            print(f"Goal reached ratio: {1 if reached else 0}/1", file=f)
            print(file=f)


def simulation_omni(
    path: np.ndarray,
    path_name: str,
    initial_pose: np.ndarray,
    draw: bool,
    animation: bool
) -> tuple[dict, dict, dict, dict]:
    goal_pose = path[-1, :3]
    robot_poses_dict = defaultdict(list)
    robot_velocities_dict = defaultdict(list)
    robot_ref_velocities_dict = defaultdict(list)
    look_ahead_positions_dict = defaultdict(list)
    break_constraints_flags_dict = defaultdict(list)
    curvatures_flags_dict = defaultdict(list)
    regulated_vs_flags_dict = defaultdict(list)
    time_stamp_dict = defaultdict(list)

    for method_name in OMNI_METHOD_NAME_LIST:
        current_pose = initial_pose.copy()
        current_velocity = np.array([0.0, 0.0, 0.0])

        robot_poses = [current_pose]
        robot_velocities = [current_velocity]
        robot_ref_velocities = [current_velocity]
        look_ahead_positions = []
        break_constraints_flags = [[False, False, False]]
        curvatures = [float("nan")]
        regulated_vs = [float("nan")]
        time_stamps = [0.0]

        sim_start_time = perf_counter()

        for _ in range(MAX_SIM_STEPS):
            if is_goal_reached(
                current_pose,
                goal_pose,
                GOAL_REACH_TOLERANCE_DIST_OMNI,
                GOAL_REACH_TOLERANCE_HEADING,
            ):
                break

            next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v = pure_pursuit(
                current_pose, current_velocity, path, method_name
            )
            next_pose, next_velocity = forward_simulation_omnidirectional(current_pose, current_velocity, next_velocity_ref)

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
        else:
            print(f"[WARN] Reached MAX_SIM_STEPS for {path_name}, method={method_name}")

        sim_end_time = perf_counter()
        print(f"[INFO] {path_name}, {method_name}: {sim_end_time - sim_start_time:.3f}s")

        robot_poses_dict[method_name] = robot_poses
        robot_velocities_dict[method_name] = robot_velocities
        robot_ref_velocities_dict[method_name] = robot_ref_velocities
        look_ahead_positions_dict[method_name] = look_ahead_positions
        break_constraints_flags_dict[method_name] = break_constraints_flags
        curvatures_flags_dict[method_name] = curvatures
        regulated_vs_flags_dict[method_name] = regulated_vs
        time_stamp_dict[method_name] = time_stamps

    cross_track_rmse_dict = calc_rmse(robot_poses_dict, path)
    heading_rmse_dict = calc_heading_rmse(robot_poses_dict, path)
    break_constraints_rate_dict = calc_break_constraints_rate(break_constraints_flags_dict)
    goal_reach_time_dict = {}
    for method_name in OMNI_METHOD_NAME_LIST:
        goal_reach_time_dict[method_name] = calc_goal_reach_time(
            np.array(robot_poses_dict[method_name]),
            np.array(time_stamp_dict[method_name]),
            path[-1],
            GOAL_REACH_TOLERANCE_DIST_OMNI,
            GOAL_REACH_TOLERANCE_HEADING,
        )

    result_path = RESULTS_DIR / path_name
    result_path.mkdir(parents=True, exist_ok=True)
    with open(result_path / "robot_poses_dict.pkl", "wb") as f:
        pickle.dump(robot_poses_dict, f)
    with open(result_path / "robot_velocities_dict.pkl", "wb") as f:
        pickle.dump(robot_velocities_dict, f)
    with open(result_path / "robot_ref_velocities_dict.pkl", "wb") as f:
        pickle.dump(robot_ref_velocities_dict, f)
    with open(result_path / "look_ahead_positions_dict.pkl", "wb") as f:
        pickle.dump(look_ahead_positions_dict, f)
    with open(result_path / "break_constraints_flags_dict.pkl", "wb") as f:
        pickle.dump(break_constraints_flags_dict, f)
    with open(result_path / "curvatures_flags_dict.pkl", "wb") as f:
        pickle.dump(curvatures_flags_dict, f)
    with open(result_path / "regulated_vs_flags_dict.pkl", "wb") as f:
        pickle.dump(regulated_vs_flags_dict, f)
    with open(result_path / "time_stamp_dict.pkl", "wb") as f:
        pickle.dump(time_stamp_dict, f)
    with open(result_path / "rmse_dict.pkl", "wb") as f:
        pickle.dump(cross_track_rmse_dict, f)
    with open(result_path / "heading_rmse_dict.pkl", "wb") as f:
        pickle.dump(heading_rmse_dict, f)
    with open(result_path / "break_constraints_rate_dict.pkl", "wb") as f:
        pickle.dump(break_constraints_rate_dict, f)
    with open(result_path / "goal_reach_time_dict.pkl", "wb") as f:
        pickle.dump(goal_reach_time_dict, f)
    write_path_summary_text(
        result_path / "results_dwpp_omni.txt",
        cross_track_rmse_dict,
        heading_rmse_dict,
        break_constraints_rate_dict,
        goal_reach_time_dict
    )

    if draw:
        from draw import draw_paths
        draw_paths(path, robot_poses_dict, path_name)

    if animation:
        from draw import draw_animation
        for method_name in OMNI_METHOD_NAME_LIST:
            draw_animation(
                path,
                np.array(robot_poses_dict[method_name]),
                np.array(look_ahead_positions_dict[method_name]),
                method_name,
                path_name
            )

    return cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict, goal_reach_time_dict


def simulate_path(idx, path, initial_pose, prefix, draw=True, animation=False):
    path_name = f"{prefix}_{idx}"
    return simulation_omni(path, path_name, initial_pose, draw=draw, animation=animation)


if __name__ == "__main__":
    paths_step = step_curves()
    path_straight_heading_step = straight_line_heading_step_curve(segment_length=1.0, points_per_segment=100)
    total_cross_track_rmse_dict = defaultdict(list)
    total_heading_rmse_dict = defaultdict(list)
    total_break_constraints_rate_dict = defaultdict(list)
    total_goal_reach_time_dict = defaultdict(list)

    tasks = [
        (idx, path, np.array([0.0, 0.0, 0.0]), "step_curve_omni", True, True)
        for idx, path in enumerate(paths_step)
    ]
    tasks.append((0, path_straight_heading_step, np.array([0.0, 0.0, 0.0]), "straight_heading_step_curve_omni", True, True))

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulate_path, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict, goal_reach_time_dict = future.result()
                for method_name, rmse in cross_track_rmse_dict.items():
                    total_cross_track_rmse_dict[method_name].append(rmse)
                for method_name, rmse in heading_rmse_dict.items():
                    total_heading_rmse_dict[method_name].append(rmse)
                for method_name, rate in break_constraints_rate_dict.items():
                    total_break_constraints_rate_dict[method_name].append(rate)
                for method_name, time_to_goal in goal_reach_time_dict.items():
                    total_goal_reach_time_dict[method_name].append(time_to_goal)
    except (PermissionError, OSError):
        print("[WARN] ProcessPoolExecutor is unavailable. Falling back to sequential execution.")
        for task in tasks:
            cross_track_rmse_dict, heading_rmse_dict, break_constraints_rate_dict, goal_reach_time_dict = simulate_path(*task)
            for method_name, rmse in cross_track_rmse_dict.items():
                total_cross_track_rmse_dict[method_name].append(rmse)
            for method_name, rmse in heading_rmse_dict.items():
                total_heading_rmse_dict[method_name].append(rmse)
            for method_name, rate in break_constraints_rate_dict.items():
                total_break_constraints_rate_dict[method_name].append(rate)
            for method_name, time_to_goal in goal_reach_time_dict.items():
                total_goal_reach_time_dict[method_name].append(time_to_goal)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "results_dwpp_omni.txt", "w") as f:
        for method_name, total_rmse in total_cross_track_rmse_dict.items():
            print(f"Method: {method_name}", file=f)
            print(f"Cross Track RMSE mean: {np.mean(total_rmse)}", file=f)
            print(f"Cross Track RMSE std: {np.std(total_rmse)}", file=f)
            print(file=f)

        for method_name, total_rmse in total_heading_rmse_dict.items():
            print(f"Method: {method_name}", file=f)
            print(f"Heading RMSE mean: {np.mean(total_rmse)}", file=f)
            print(f"Heading RMSE std: {np.std(total_rmse)}", file=f)
            print(file=f)

        for method_name, total_rate in total_break_constraints_rate_dict.items():
            print(f"Method: {method_name}", file=f)
            print(f"Break constraints rate mean: {np.mean(total_rate, axis=0)}", file=f)
            print(f"Break constraints rate std: {np.std(total_rate, axis=0)}", file=f)
            print(file=f)

        for method_name, total_goal_reach_time in total_goal_reach_time_dict.items():
            total_goal_reach_time = np.array(total_goal_reach_time, dtype=float)
            reached_mask = np.isfinite(total_goal_reach_time)

            if np.any(reached_mask):
                goal_reach_time_mean = np.mean(total_goal_reach_time[reached_mask])
                goal_reach_time_std = np.std(total_goal_reach_time[reached_mask])
            else:
                goal_reach_time_mean = float("nan")
                goal_reach_time_std = float("nan")

            print(f"Method: {method_name}", file=f)
            print(f"Goal reach time mean [s]: {goal_reach_time_mean}", file=f)
            print(f"Goal reach time std [s]: {goal_reach_time_std}", file=f)
            print(f"Goal reached ratio: {np.sum(reached_mask)}/{len(total_goal_reach_time)}", file=f)
            print(file=f)
