import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI を使わないバックエンドに設定
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from config import method_name_dict, DT, V_MIN, V_MAX, W_MIN, W_MAX, A_MAX, AW_MAX
from collections import defaultdict
import concurrent.futures
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

plt.rcParams["font.size"] = 9  # 全体のフォントサイズが変更されます。
plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
plt.rcParams["mathtext.fontset"] = "stix"  # math fontの設定
plt.rcParams["font.weight"] = "normal"  # フォントの太さを細字に設定
plt.rcParams["axes.linewidth"] = 1.0  # axis line width
plt.rcParams["axes.grid"] = True  # make grid
plt.rcParams["legend.edgecolor"] = "black"  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節


def calc_path_headings(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.array([])
    if path.shape[1] >= 3:
        return path[:, 2]
    if len(path) == 1:
        return np.array([0.0])

    diffs = np.diff(path[:, :2], axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.concatenate([headings, [headings[-1]]])
    return headings


def draw_path_orientation_arrows(
    ax: plt.Axes,
    path: np.ndarray,
    max_arrows: int = 30,
    color: str = "red",
    alpha: float = 0.7,
    label: str = "Path Orientation"
) -> None:
    if len(path) == 0:
        return

    headings = calc_path_headings(path)
    n = len(path)
    step = max(1, n // max_arrows)
    indices = np.arange(0, n, step, dtype=int)
    if indices[-1] != n - 1:
        indices = np.append(indices, n - 1)

    x = path[indices, 0]
    y = path[indices, 1]
    h = headings[indices]

    x_span = float(np.max(path[:, 0]) - np.min(path[:, 0]))
    y_span = float(np.max(path[:, 1]) - np.min(path[:, 1]))
    diag = float(np.hypot(x_span, y_span))
    arrow_length = 2.0 * max(0.1, 0.03 * diag)

    dx = arrow_length * np.cos(h)
    dy = arrow_length * np.sin(h)

    ax.quiver(
        x,
        y,
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        alpha=alpha,
        width=0.0035,
        label=label
    )


# 軌跡の描画
def draw_paths(path: np.ndarray, robot_poses_dist: defaultdict, path_name: str):
    print("drawing paths...")
    result_dir = RESULTS_DIR / path_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots()
    ax.set_xlim(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1)
    ax.set_ylim(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{path_name} Simulation')
    ax.set_aspect('equal')

    # パス
    ax.plot(path[:, 0], path[:, 1], 'k--', label='Path')
    draw_path_orientation_arrows(ax, path)

    # ロボットの位置
    for method_name, robot_poses in robot_poses_dist.items():
        robot_poses = np.array(robot_poses)
        ax.plot(robot_poses[:, 0], robot_poses[:, 1], label=method_name_dict[method_name])

    # 凡例を表示
    ax.legend()
    
    plt.tight_layout()

    # グラフを表示
    plt.savefig(result_dir / "paths.png")
    plt.close()

# 速度プロファイルの描画
def draw_velocity_profile(
    time_stamps: np.ndarray,
    robot_velocities: np.ndarray,
    robot_ref_velocities: np.ndarray,
    break_constraints_flags: np.ndarray,
    method_name: str,
    path_name: str
):
    print("drawing velocity profiles...")
    result_dir = RESULTS_DIR / path_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    #==================================================================#
    # 1. Translational velocity (並進速度) の描画
    #==================================================================#
    fig1 = plt.figure(figsize=(3, 3))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('$t$ [s]')
    ax1.set_ylabel('$v$ [m/s]')
    # ax1.set_title(f'Translational Velocity Profile, Path: {path_name}, Method: {method_name_dict[method_name]}')
    
    # # コマンド速度(並進)をラインプロット
    # ax1.plot(time_stamps, robot_velocities[:, 0], label='Command Velocity')

    # # 並進速度制約を破っているかどうかのフラグ (N×2のうち1列目)
    # mask_break_trans = break_constraints_flags[:, 0]  # Trueの箇所は並進速度制約違反
    # mask_ok_trans = ~mask_break_trans                # Falseの箇所はOK

    # # Reference velocity(並進)を、OKとNGで色分けして散布図
    # ax1.scatter(
    #     time_stamps[mask_ok_trans],
    #     robot_ref_velocities[mask_ok_trans, 0],
    #     color='blue',
    #     label='Reference Velocity (OK)',
    #     s=5
    # )
    # ax1.scatter(
    #     time_stamps[mask_break_trans],
    #     robot_ref_velocities[mask_break_trans, 0],
    #     color='red',
    #     label='Reference Velocity (Broken Constraints)',
    #     s=5
    # )
    
    # 速度の上限・下限線
    ax1.axhline(y=V_MAX, linestyle="--", c="black")
    ax1.axhline(y=V_MIN, linestyle="--", c="black")
    
    # 並進目標速度
    ax1.plot(time_stamps, robot_ref_velocities[:, 0], c="blue", label='Reference Velocity')
    # 並進実現速度
    ax1.plot(time_stamps, robot_velocities[:, 0], c="red", label='Velocity')
    
    # x軸の値の範囲を指定
    ax1.set_xlim(0, 23.0)
    
    # ax1.legend()
    plt.tight_layout()
    
    plt.savefig(result_dir / f"{method_name}_translational_velocity_profile.png")
    plt.close()
    
    
    #==================================================================#
    # 2. Rotational velocity (回転速度) の描画
    #==================================================================#
    fig2 = plt.figure(figsize=(3, 3))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('$t$ [s]')
    ax2.set_ylabel('$w$ [m/s]')
    # ax2.set_title(f'Rotational Velocity Profile, Path: {path_name}, Method: {method_name_dict[method_name]}')
    
    # # コマンド速度(回転)をラインプロット
    # ax2.plot(time_stamps, robot_velocities[:, 1], label='Command Velocity')

    # # 回転速度制約を破っているかどうかのフラグ (N×2のうち2列目)
    # mask_break_rot = break_constraints_flags[:, 1]  # Trueの箇所は回転速度制約違反
    # mask_ok_rot = ~mask_break_rot                  # Falseの箇所はOK

    # # Reference velocity(回転)を、OKとNGで色分けして散布図
    # ax2.scatter(
    #     time_stamps[mask_ok_rot],
    #     robot_ref_velocities[mask_ok_rot, 1],
    #     color='blue',
    #     label='Reference Velocity (OK)',
    #     s=5
    # )
    # ax2.scatter(
    #     time_stamps[mask_break_rot],
    #     robot_ref_velocities[mask_break_rot, 1],
    #     color='red',
    #     label='Reference Velocity (Broken Constraints)',
    #     s=5
    # )
    
    # 速度の上限・下限線
    ax2.axhline(y=W_MAX, linestyle="--", c="black")
    ax2.axhline(y=W_MIN, linestyle="--", c="black")
    
    # 回転目標速度
    ax2.plot(time_stamps, robot_ref_velocities[:, 1], c="blue", label='Reference Velocity')
    # 回転実現速度
    ax2.plot(time_stamps, robot_velocities[:, 1], c="red", label='Velocity')
    
    # x軸の値の範囲を指定
    ax2.set_xlim(0, 23.0)
    
    # ax2.legend()
    plt.tight_layout()
    plt.savefig(result_dir / f"{method_name}_rotational_velocity_profile.png")
    plt.close()
    
    print("Done.")
    
# アニメーションの描画
def draw_animation(path: np.ndarray, robot_poses: np.ndarray, look_ahead_positions: np.ndarray, method_name:str, path_name: str):
    print("drawing animations...")
    result_dir = RESULTS_DIR / path_name
    result_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.set_xlim(np.min(path[:, 0]) - 1, np.max(path[:, 0]) + 1)
    ax.set_ylim(np.min(path[:, 1]) - 1, np.max(path[:, 1]) + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_title(f'{method_name_dict[method_name]} Simulation')
    ax.set_aspect('equal')

    # プロット要素の初期化
    path_line, = ax.plot(path[:, 0], path[:, 1], 'k--', label='Path')  # パス
    draw_path_orientation_arrows(ax, path, max_arrows=40, alpha=0.45)
    robot_point, = ax.plot([], [], 'bo', label='Robot')  # ロボット位置
    robot_path, = ax.plot([], [], linewidth=1, color="cyan")  # ロボット位置
    look_ahead_point, = ax.plot([], [], 'ro', label='Look Ahead')  # ルックアヘッド位置

    robot_xs = []
    robot_ys = []

    # ロボットの向きを示す矢印
    robot_arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=20, color='blue')
    ax.add_patch(robot_arrow)

    def init():
        robot_point.set_data([], [])
        robot_path.set_data([], [])
        look_ahead_point.set_data([], [])
        robot_arrow.set_visible(False)
        return robot_point, robot_path, look_ahead_point, robot_arrow

    def update(frame):
        # ロボットの位置を更新
        robot_x = robot_poses[frame, 0]
        robot_y = robot_poses[frame, 1]
        robot_theta = robot_poses[frame, 2]
        
        robot_xs.append(robot_x)
        robot_ys.append(robot_y)

        robot_point.set_data([robot_x], [robot_y])
        robot_path.set_data(robot_xs, robot_ys)

        # ロボットの向きを示す矢印を更新
        arrow_length = 0.5  # 矢印の長さ
        dx = arrow_length * np.cos(robot_theta)
        dy = arrow_length * np.sin(robot_theta)

        # 矢印の位置を更新
        robot_arrow.set_positions((robot_x, robot_y), (robot_x + dx, robot_y + dy))
        robot_arrow.set_visible(True)

        # ルックアヘッドの位置を更新
        if frame < len(robot_poses) - 1:
            look_ahead_x = look_ahead_positions[frame, 0]
            look_ahead_y = look_ahead_positions[frame, 1]
            look_ahead_point.set_data([look_ahead_x], [look_ahead_y])
        else:
            look_ahead_point.set_data([], [])

        return path_line, robot_point, look_ahead_point, robot_arrow

    ani = FuncAnimation(
        fig, update, frames=len(robot_poses), init_func=init, blit=False, interval=100, repeat=False
    )

    # 凡例を表示
    ax.legend()

    # アニメーションを表示または保存
    # plt.show()
    fps = max(1, int(round(1.0 / DT)))
    output_path = result_dir / f"{method_name}.mp4"
    try:
        ani.save(output_path, writer='ffmpeg', fps=fps)
    except Exception:
        # ffmpeg が使えない環境では GIF で保存
        gif_path = result_dir / f"{method_name}.gif"
        ani.save(gif_path, writer='pillow', fps=fps)
    plt.close()

def plot_single(args):
    idx, curvature, regulated_v, robot_velocity, robot_ref_velocity, prev_robot_velocity, past_robot_velocities, dir_path = args
    
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(V_MIN - 0.05, V_MAX + 0.05)
    ax.set_ylim(W_MIN - 0.2, W_MAX + 0.2)
    ax.set_xlabel('$v$')
    ax.set_ylabel(r'$\omega$')
    ax.set_title('$v\\omega$ plot')
    
    # 速度の上限・下限線
    ax.axvline(x=V_MAX, linestyle="--", c="black")
    ax.axvline(x=V_MIN, linestyle="--", c="black")
    ax.axhline(y=W_MIN, linestyle="--", c="black")
    ax.axhline(y=W_MAX, linestyle="--", c="black")
    
    # regulatedされた上限値
    ax.axvline(x=regulated_v, linestyle="--", c="red", label="regulated v")
    
    # 参照速度, 現在速度, 一時刻前の速度
    ax.scatter(robot_ref_velocity[0], robot_ref_velocity[1], c="green", label="reference velocity")
    ax.scatter(robot_velocity[0], robot_velocity[1], c="red", label="next velocity")
    ax.scatter(prev_robot_velocity[0], prev_robot_velocity[1], c="blue", label="current velocity")
    
    # これまでの速度の軌跡
    past_robot_velocities = np.array(past_robot_velocities)
    ax.plot(past_robot_velocities.T[0], past_robot_velocities.T[1], c="orange")
    
    # dynamic window の計算
    dw_vmax = min(prev_robot_velocity[0] + A_MAX * DT, V_MAX)
    dw_vmin = max(prev_robot_velocity[0] - A_MAX * DT, V_MIN)
    dw_wmax = min(prev_robot_velocity[1] + AW_MAX * DT, W_MAX)
    dw_wmin = max(prev_robot_velocity[1] - AW_MAX * DT, W_MIN)
    if dw_vmax > regulated_v:
        dw_vmax = max(dw_vmin, regulated_v)
    
    ax.plot([dw_vmin, dw_vmin], [dw_wmin, dw_wmax], c="black", label="dynamic window")
    ax.plot([dw_vmax, dw_vmax], [dw_wmin, dw_wmax], c="black")
    ax.plot([dw_vmin, dw_vmax], [dw_wmin, dw_wmin], c="black")
    ax.plot([dw_vmin, dw_vmax], [dw_wmax, dw_wmax], c="black")
    
    # 曲率に対応する直線
    ax.axline((0, 0), slope=curvature, color='red', lw=2, label=r"$\omega={\phi}v$")
    
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f"{dir_path}/{idx:04d}.png")
    plt.close(fig)

def draw_vw_plane_plot(robot_velocities: np.ndarray, robot_ref_velocities: np.ndarray, time_stamps: np.ndarray, 
                       curvatures: np.ndarray, regulated_vs: np.ndarray, method_name: str, path_name: str):
    if cv2 is None:
        raise ImportError("cv2 is required for draw_vw_plane_plot() but is not installed.")
    
    # 保存先ディレクトリの作成
    dir_path = RESULTS_DIR / path_name / "vw_plot" / method_name
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # 各タスクに渡す引数をリストにまとめる
    args_list = []
    for idx, (curvature, regulated_v, robot_velocity, robot_ref_velocity) in enumerate(
            zip(curvatures, regulated_vs, robot_velocities, robot_ref_velocities)):
        
        past_robot_velocities = robot_velocities[:idx+1]
        
        if idx == 0:
            prev_robot_velocity = robot_velocity
        else:
            prev_robot_velocity = robot_velocities[idx - 1]
        args_list.append((idx, curvature, regulated_v, robot_velocity, robot_ref_velocity, prev_robot_velocity, past_robot_velocities, str(dir_path)))
    
    # ProcessPoolExecutor で並列処理を実行
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(plot_single, args_list)
    
    # 画像を動画化
    # 画像のpathを取得
    image_paths = sorted(dir_path.glob("*.png"))
    if len(image_paths) == 0:
        return
    h, w, _ = cv2.imread(str(image_paths[0])).shape
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video_path = RESULTS_DIR / path_name / "vw_plot" / f"{method_name}.mp4"
    fps = max(1, int(round(1.0 / DT)))
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        video.write(image)
    video.release()
    
