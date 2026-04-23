#!/usr/bin/env python3

import rospy
import numpy as np
import math
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from typing import Tuple, List, Optional
import tf
import time
import pickle
import os
from datetime import datetime
from ytlab_whill_modules.srv import FollowPath, FollowPathResponse
import roslib.packages

class PurePursuitController:
    def __init__(self):
        rospy.init_node('pure_pursuit_controller', anonymous=True)
        
        # map tf name
        self.robot_name = rospy.get_param("~robot_name")
        self.map_tf_name = self.robot_name + rospy.get_param("tf/map")
        # base_link frame name
        self.wheel_base_tf_name = rospy.get_param("tf/wheel_base")
        
        # Parameters
        self.MIN_LOOK_AHEAD_DISTANCE = rospy.get_param('~min_look_ahead_distance', 0.4)
        self.MAX_LOOK_AHEAD_DISTANCE = rospy.get_param('~max_look_ahead_distance', 0.5)
        self.LOOK_AHEAD_TIME = rospy.get_param('~look_ahead_time', 1.0)
        self.V_MAX = rospy.get_param('~max_linear_velocity', 0.5)
        self.V_MIN = rospy.get_param('~min_linear_velocity', 0.0)
        self.W_MAX = rospy.get_param('~max_angular_velocity', 1.0)
        self.W_MIN = rospy.get_param('~min_angular_velocity', -1.0)
        self.A_MAX = rospy.get_param('~max_linear_acceleration', 0.5)
        self.AW_MAX = rospy.get_param('~max_angular_acceleration', 1.0)
        self.GOAL_TOLERANCE_DIST = rospy.get_param('~goal_tolerance_distance', 0.05)
        self.APPROACH_VELOCITY_SCALING_DIST = rospy.get_param('~approach_velocity_scaling_distance', 0.3)
        self.MIN_APPROACH_LINEAR_VELOCITY = rospy.get_param('~min_approach_linear_velocity', 0.05)
        self.REGULATED_LINEAR_SCALING_MIN_RADIUS = rospy.get_param('~regulated_linear_scaling_min_radius', 0.9)
        self.REGULATED_LINEAR_SCALING_MIN_SPEED = rospy.get_param('~regulated_linear_scaling_min_speed', 0.0)
        self.CONTROL_FREQUENCY = rospy.get_param('~control_frequency', 20.0)
        self.METHOD_NAME = rospy.get_param('~method_name', 'dwpp')  # pp, app, rpp, dwpp
        
        # State variables
        self.current_pose: Optional[np.ndarray] = None
        self.current_velocity: np.ndarray = np.array([0.0, 0.0])
        self.current_odom_velocity: np.ndarray = np.array([0.0, 0.0])
        self.path: Optional[np.ndarray] = None
        self.goal_reached: bool = False
        self.following_path: bool = False
        self.start_time: float = 0.0
        
        # Data logging
        self.experiment_data = {
            'robot_poses': [],
            'robot_velocities': [],
            'robot_odom_velocities': [],
            'robot_ref_velocities': [],
            'break_constraints_flags': [],
            'curvatures': [],
            'regulated_vs': [],
            'time_stamps': [],
            'look_ahead_positions': []
        }
        self.current_path_name: str = ""
        self.current_method_name: str = ""
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/whill/controller/req_vel', Twist, queue_size=1)
        self.reference_path_pub = rospy.Publisher('~reference_path', Path, queue_size=1)
        self.robot_trajectory_pub = rospy.Publisher('~robot_trajectory', Path, queue_size=1)
        self.visualization_pub = rospy.Publisher('~visualization_markers', MarkerArray, queue_size=1)
        
        # Subscribers
        self.path_sub = rospy.Subscriber('~path', Path, self.path_callback)
        self.odom_sub = rospy.Subscriber('/whill/odom', Odometry, self.odom_callback)
        
        # Service server
        self.follow_path_service = rospy.Service('/dwpp_experiment', FollowPath, self.follow_path_callback)
        
        # tf
        self.tf_listener = tf.TransformListener()
        
        # Validate method name
        valid_methods = ['pp', 'app', 'rpp', 'dwpp']
        if self.METHOD_NAME not in valid_methods:
            rospy.logerr(f"Invalid method name: {self.METHOD_NAME}. Valid options: {valid_methods}")
            rospy.signal_shutdown("Invalid method name")
            return
        
        # Control timer (starts inactive)
        self.dt = 1.0 / self.CONTROL_FREQUENCY
        self.control_timer = None
        
        rospy.loginfo(f"Pure Pursuit Controller initialized")
        rospy.loginfo("Service 'dwpp_experiment' is ready")

    def get_robot_pose(self) -> Optional[np.ndarray]:
        """Get the current robot pose from the tf listener"""
        # map_frame_id -> base_frame_idの変換の取得（ロボットの経路）
        try:
            (Trans, Rot) = self.tf_listener.lookupTransform(
                self.map_tf_name, self.wheel_base_tf_name, rospy.Time(0)
            )
            # trans
            x, y, z = Trans
            # rot (rpy)
            orientation = tf.transformations.euler_from_quaternion(Rot)
            roll, pitch, yaw = orientation
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            print("lookup transform map_frame_id -> base_frame_id failed")
            return None
        
        return np.array([x, y, yaw])

    def path_callback(self, msg: Path) -> None:
        """Callback for path messages"""
        if len(msg.poses) == 0:
            rospy.logwarn("Received empty path")
            return
            
        path_points = []
        for pose_stamped in msg.poses:
            position = pose_stamped.pose.position
            path_points.append([position.x, position.y])
        
        self.path = np.array(path_points)
        self.goal_reached = False
        self.following_path = False
        rospy.loginfo(f"Received path with {len(path_points)} points")

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry messages"""
        self.current_odom_velocity[0] = msg.twist.twist.linear.x
        self.current_odom_velocity[1] = msg.twist.twist.angular.z

    def follow_path_callback(self, req):
        """Service callback for path following"""
        try:
            # Extract path from request
            if len(req.path.poses) == 0:
                return FollowPathResponse(False, "Empty path received", 0.0)
                
            path_points = []
            for pose_stamped in req.path.poses:
                position = pose_stamped.pose.position
                path_points.append([position.x, position.y])
            
            # Set path and method
            self.path = np.array(path_points)
            self.current_method_name = req.method_name
            self.goal_reached = False
            self.following_path = True
            self.start_time = time.time()
            
            # Reset velocity to zero at start of new path following
            self.current_velocity = np.array([0.0, 0.0])
            
            # Initialize experiment data logging
            self.current_path_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._reset_experiment_data()
            
            # Publish reference path for visualization
            self._publish_reference_path()
            
            rospy.loginfo(f"Starting path following with method: {req.method_name}, {len(path_points)} points")
            
            # Start control timer
            if self.control_timer is not None:
                self.control_timer.shutdown()
            self.control_timer = rospy.Timer(rospy.Duration(self.dt), self.control_callback)
            
            # Wait for path following to complete
            rate = rospy.Rate(5)  # Check status at 5Hz
            while self.following_path and not rospy.is_shutdown():
                rate.sleep()
            
            # Stop control timer
            if self.control_timer is not None:
                self.control_timer.shutdown()
                self.control_timer = None
            
            # Calculate completion time
            completion_time = time.time() - self.start_time
            
            # Save experiment data
            if self.goal_reached:
                self._save_experiment_data()
                return FollowPathResponse(True, f"Path following completed successfully. Data saved as {self.current_path_name}", completion_time)
            else:
                self._save_experiment_data()
                return FollowPathResponse(False, f"Path following interrupted. Data saved as {self.current_path_name}", completion_time)
                
        except Exception as e:
            rospy.logerr(f"Error in follow_path service: {str(e)}")
            return FollowPathResponse(False, f"Error: {str(e)}", 0.0)

    def control_callback(self, timer_event) -> None:
        """Main control loop callback"""
        if not self.following_path:
            return
            
        self.current_pose = self.get_robot_pose()
        
        if self.current_pose is None or self.path is None:
            self.publish_zero_velocity()
            return
            
        if self.goal_reached:
            self.publish_zero_velocity()
            self.following_path = False
            return
            
        # Check if goal is reached
        distance_to_goal = np.linalg.norm(self.path[-1] - self.current_pose[:2])
        if distance_to_goal < self.GOAL_TOLERANCE_DIST:
            self.goal_reached = True
            self.publish_zero_velocity()
            self.following_path = False
            rospy.loginfo("Goal reached!")
            return
            
        # Compute Pure Pursuit control
        try:
            next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v = self.pure_pursuit_control_with_data(
                self.current_pose, self.current_velocity, self.path, self.current_method_name
            )
            
            # Apply acceleration constraints
            next_velocity = self.calc_accel_constrained_velocity(
                self.current_velocity, next_velocity_ref
            )
            
            # Log experiment data
            self._log_data(look_ahead_pos, break_constraints_flag, curvature, regulated_v, next_velocity_ref, next_velocity)
            
            # Publish velocity command
            self.publish_velocity(next_velocity)
            
            # Update current velocity
            self.current_velocity = next_velocity
            
            # Publish robot trajectory for visualization
            self._publish_robot_trajectory()
            self._publish_visualization_markers(look_ahead_pos, curvature)
            
        except Exception as e:
            rospy.logerr(f"Error in control loop: {str(e)}")
            self.publish_zero_velocity()

    def pure_pursuit_control_with_data(self, current_pose: np.ndarray, current_velocity: np.ndarray, 
                           path: np.ndarray, method_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Pure Pursuit control algorithm with multiple method support"""
        # Calculate current path index
        current_idx = self.calc_index(current_pose, path)
        
        # Calculate path distances
        path_distances = self.calc_path_distances(path)

        if method_name == "dwpp":
            is_accel = self.decide_accel_or_decel(current_idx, path_distances)
            next_velocity_ref, look_ahead_pos, curvature, regulated_v = self.calc_dwpp_velocity_with_auto_look_ahead(
                current_pose=current_pose,
                current_velocity=current_velocity,
                current_idx=current_idx,
                path=path,
                path_distances=path_distances,
                is_accel=is_accel,
                use_regulated_velocity=True,
            )
        else:
            # Calculate look ahead distance
            look_ahead_distance = self.calc_look_ahead_distance(current_velocity, method_name)

            # Calculate curvature to look ahead position
            curvature, look_ahead_pos = self.calc_curvature_to_look_ahead_position(
                current_pose, current_idx, path, path_distances, look_ahead_distance
            )

            if method_name in ["rpp", "dwpp"]:
                # Calculate regulated translational velocity (Regulated Pure Pursuit)
                regulated_v = self.calc_regulated_translational_velocity(curvature)
            else:
                regulated_v = self.V_MAX

            if method_name in ["pp", "app", "rpp"]:
                # Calculate translational velocity
                v_ref = self.calc_reference_translational_velocity(current_pose, path[-1])
                
                # Regulate translational velocity for RPP
                if method_name == "rpp":
                    v_ref = min(v_ref, regulated_v)
                
                # Calculate angular velocity
                w_ref = curvature * v_ref
                next_velocity_ref = np.array([v_ref, w_ref])
                
            else:
                # Decide acceleration or deceleration
                is_accel = self.decide_accel_or_decel(current_idx, path_distances)
                
                # Calculate optimal velocity considering dynamic window
                next_velocity_ref = self.calc_optimal_velocity_considering_dynamic_window(
                    current_velocity, regulated_v, curvature, is_accel
                )
        
        # Check constraints breaking
        break_constraints_flag = self._check_constraints_breaking(current_velocity, next_velocity_ref)
        
        return next_velocity_ref, look_ahead_pos, break_constraints_flag, curvature, regulated_v

    def calc_index(self, current_pose: np.ndarray, path: np.ndarray) -> int:
        """Find the closest point index on the path"""
        distances = np.linalg.norm(path - current_pose[:2], axis=1)
        return int(np.argmin(distances))

    def calc_path_distances(self, path: np.ndarray) -> np.ndarray:
        """Calculate cumulative distances along the path"""
        differences = np.diff(path, axis=0)
        distances = np.linalg.norm(differences, axis=1)
        return np.concatenate(([0.0], np.cumsum(distances)))

    def calc_min_look_ahead_distance_from_velocity(self, current_velocity: np.ndarray) -> float:
        return abs(float(current_velocity[0])) * self.dt

    def calc_look_ahead_distance(self, current_velocity: np.ndarray, method_name: str) -> float:
        """Calculate look ahead distance based on method"""
        min_look_ahead_distance = self.calc_min_look_ahead_distance_from_velocity(current_velocity)
        max_look_ahead_distance = max(self.MAX_LOOK_AHEAD_DISTANCE, min_look_ahead_distance)

        if method_name in ["app", "rpp", "dwpp"]:
            # Adaptive look ahead distance
            look_ahead_distance = self.LOOK_AHEAD_TIME * abs(float(current_velocity[0]))
            return float(np.clip(
                look_ahead_distance,
                min_look_ahead_distance,
                max_look_ahead_distance,
            ))
        else:  # pp
            # Fixed look ahead distance
            return max(self.MIN_LOOK_AHEAD_DISTANCE, min_look_ahead_distance)

    def calc_curvature_to_look_ahead_position(self, current_pose: np.ndarray, 
                                            current_idx: int, path: np.ndarray, 
                                            path_distances: np.ndarray, 
                                            look_ahead_distance: float) -> Tuple[float, np.ndarray]:
        """Calculate curvature to the look ahead position"""
        # Find look ahead position
        current_distance = path_distances[current_idx]
        look_ahead_pos_distance = current_distance + look_ahead_distance
        look_ahead_idx = min(np.searchsorted(path_distances, look_ahead_pos_distance), 
                            len(path) - 1)
        look_ahead_pos = path[look_ahead_idx]
        return self.calc_curvature_to_point(current_pose, look_ahead_pos), look_ahead_pos

    def calc_curvature_to_point(self, current_pose: np.ndarray, target_pos: np.ndarray) -> float:
        """Calculate curvature to an arbitrary target point"""
        look_ahead_angle = math.atan2(target_pos[1] - current_pose[1], target_pos[0] - current_pose[0]) - current_pose[2]
        distance = float(np.linalg.norm(target_pos - current_pose[:2]))
        if distance == 0.0:
            return 0.0
        return 2.0 * math.sin(look_ahead_angle) / distance

    def calc_reference_translational_velocity(self, current_pose: np.ndarray, goal_pose: np.ndarray) -> float:
        """Calculate reference translational velocity considering approach to goal"""
        v_ref = self.V_MAX
        
        # Reduce velocity when approaching goal
        distance_to_goal = np.linalg.norm(goal_pose - current_pose[:2])
        if distance_to_goal < self.APPROACH_VELOCITY_SCALING_DIST:
            v_ref = max(v_ref * distance_to_goal / self.APPROACH_VELOCITY_SCALING_DIST, 
                       self.MIN_APPROACH_LINEAR_VELOCITY)
        if distance_to_goal < self.GOAL_TOLERANCE_DIST:
            v_ref = 0.0
        
        return v_ref

    def calc_regulated_translational_velocity(self, curvature: float) -> float:
        """Calculate regulated translational velocity based on curvature"""
        if curvature == 0.0:
            return self.V_MAX
        
        curvature_radius = 1.0 / abs(curvature)
        if curvature_radius <= self.REGULATED_LINEAR_SCALING_MIN_RADIUS:
            regulated_v = self.V_MAX * curvature_radius / self.REGULATED_LINEAR_SCALING_MIN_RADIUS
        else:
            regulated_v = self.V_MAX
        
        return max(regulated_v, self.REGULATED_LINEAR_SCALING_MIN_SPEED)

    def decide_accel_or_decel(self, current_idx: int, path_distances: np.ndarray) -> bool:
        """Decide whether to accelerate or decelerate based on remaining distance"""
        goal_distance = path_distances[-1] - path_distances[current_idx]
        decel_distance = (self.V_MAX ** 2) / (2 * self.A_MAX)
        return goal_distance > decel_distance

    def calc_forward_point_indices(self, current_pose: np.ndarray, current_idx: int, path: np.ndarray) -> np.ndarray:
        candidate_indices = np.arange(current_idx, len(path), dtype=np.intp)
        candidate_points = path[candidate_indices]
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
        self,
        current_idx: int,
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

    def calc_dynamic_window_bounds(self, current_velocity: np.ndarray) -> Tuple[float, float, float, float]:
        dw_vmax = min(current_velocity[0] + self.A_MAX * self.dt, self.V_MAX)
        dw_vmin = max(current_velocity[0] - self.A_MAX * self.dt, self.V_MIN)
        dw_wmax = min(current_velocity[1] + self.AW_MAX * self.dt, self.W_MAX)
        dw_wmin = max(current_velocity[1] - self.AW_MAX * self.dt, self.W_MIN)
        return dw_vmin, dw_vmax, dw_wmin, dw_wmax

    def calc_velocity_command_for_curvature(
        self,
        current_velocity: np.ndarray,
        regulated_v: float,
        curvature: float,
        prefer_high_speed: bool,
    ) -> Tuple[np.ndarray, bool, float]:
        dw_vmin, dw_vmax, dw_wmin, dw_wmax = self.calc_dynamic_window_bounds(current_velocity)

        if dw_vmax > regulated_v:
            dw_vmax = max(dw_vmin, regulated_v)

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

        dw_coords = [
            (dw_vmin, dw_wmin), (dw_vmin, dw_wmax),
            (dw_vmax, dw_wmin), (dw_vmax, dw_wmax)
        ]
        distances = []
        for point in dw_coords:
            dist = abs(curvature * point[0] - point[1]) / math.sqrt(curvature**2 + 1)
            distances.append(dist)

        min_dist = min(distances)
        min_dist_coords = [point for point, dist in zip(dw_coords, distances) if dist == min_dist]
        min_dist_coords.sort(key=lambda point: point[0])
        selected_velocity = min_dist_coords[-1] if prefer_high_speed else min_dist_coords[0]
        return np.array(selected_velocity, dtype=float), False, min_dist

    def calc_dwpp_velocity_with_auto_look_ahead(
        self,
        current_pose: np.ndarray,
        current_velocity: np.ndarray,
        current_idx: int,
        path: np.ndarray,
        path_distances: np.ndarray,
        is_accel: bool,
        use_regulated_velocity: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        forward_indices = self.calc_forward_point_indices(current_pose, current_idx, path)
        min_look_ahead_distance = self.calc_min_look_ahead_distance_from_velocity(current_velocity)
        forward_indices = self.apply_min_look_ahead_distance_to_indices(
            current_idx=current_idx,
            path_distances=path_distances,
            candidate_indices=forward_indices,
            min_look_ahead_distance=min_look_ahead_distance,
        )

        if len(forward_indices) == 0:
            fallback_idx = min(current_idx, len(path) - 1)
            forward_indices = np.array([fallback_idx], dtype=np.intp)

        last_candidate = None

        for candidate_idx in forward_indices:
            look_ahead_pos = path[candidate_idx]
            curvature = self.calc_curvature_to_point(current_pose, look_ahead_pos)
            regulated_v = self.calc_regulated_translational_velocity(curvature) if use_regulated_velocity else self.V_MAX
            next_velocity_ref, has_intersection, distance_to_line = self.calc_velocity_command_for_curvature(
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

    def calc_optimal_velocity_considering_dynamic_window(self, current_velocity: np.ndarray, 
                                                       regulated_v: float, curvature: float, 
                                                       is_accel: bool) -> np.ndarray:
        """Calculate optimal velocity considering dynamic window constraints"""
        next_velocity, _, _ = self.calc_velocity_command_for_curvature(
            current_velocity=current_velocity,
            regulated_v=regulated_v,
            curvature=curvature,
            prefer_high_speed=is_accel,
        )
        return next_velocity

    def calc_accel_constrained_velocity(self, current_velocity: np.ndarray, 
                                      next_velocity_ref: np.ndarray) -> np.ndarray:
        """Apply acceleration constraints to reference velocity"""
        next_velocity = next_velocity_ref.copy()
        
        # Linear acceleration constraint
        max_linear_vel = min(current_velocity[0] + self.A_MAX * self.dt, self.V_MAX)
        min_linear_vel = max(current_velocity[0] - self.A_MAX * self.dt, self.V_MIN)
        next_velocity[0] = np.clip(next_velocity[0], min_linear_vel, max_linear_vel)
        
        # Angular acceleration constraint
        max_angular_vel = min(current_velocity[1] + self.AW_MAX * self.dt, self.W_MAX)
        min_angular_vel = max(current_velocity[1] - self.AW_MAX * self.dt, self.W_MIN)
        next_velocity[1] = np.clip(next_velocity[1], min_angular_vel, max_angular_vel)
        
        return next_velocity

    def publish_velocity(self, velocity: np.ndarray) -> None:
        """Publish velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = float(velocity[0])
        cmd_msg.angular.z = float(velocity[1])
        self.cmd_vel_pub.publish(cmd_msg)

    def publish_zero_velocity(self) -> None:
        """Publish zero velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

    def _check_constraints_breaking(self, current_velocity: np.ndarray, next_velocity: np.ndarray) -> np.ndarray:
        """Check if velocity and acceleration constraints are broken"""
        # Check velocity constraints
        max_linear_vel = min(current_velocity[0] + self.A_MAX * self.dt, self.V_MAX)
        min_linear_vel = max(current_velocity[0] - self.A_MAX * self.dt, self.V_MIN)
        v_break = next_velocity[0] > max_linear_vel or next_velocity[0] < min_linear_vel
        
        max_angular_vel = min(current_velocity[1] + self.AW_MAX * self.dt, self.W_MAX)
        min_angular_vel = max(current_velocity[1] - self.AW_MAX * self.dt, self.W_MIN)
        w_break = next_velocity[1] > max_angular_vel or next_velocity[1] < min_angular_vel
        
        return np.array([v_break, w_break])

    def _reset_experiment_data(self) -> None:
        """Reset experiment data for new experiment"""
        self.experiment_data = {
            'robot_poses': [],
            'robot_velocities': [],
            'robot_odom_velocities': [],
            'robot_ref_velocities': [],
            'break_constraints_flags': [],
            'curvatures': [],
            'regulated_vs': [],
            'time_stamps': [],
            'look_ahead_positions': []
        }

    def _log_data(self, look_ahead_pos: np.ndarray, break_constraints_flag: np.ndarray, 
                  curvature: float, regulated_v: float, ref_velocity: np.ndarray, actual_velocity: np.ndarray) -> None:
        """Log experiment data"""
        current_time = time.time() - self.start_time
        
        self.experiment_data['robot_poses'].append(self.current_pose.copy())
        self.experiment_data['robot_velocities'].append(actual_velocity.copy())
        self.experiment_data['robot_odom_velocities'].append(self.current_odom_velocity.copy())
        self.experiment_data['robot_ref_velocities'].append(ref_velocity.copy())
        self.experiment_data['break_constraints_flags'].append(break_constraints_flag.copy())
        self.experiment_data['curvatures'].append(curvature)
        self.experiment_data['regulated_vs'].append(regulated_v)
        self.experiment_data['time_stamps'].append(current_time)
        self.experiment_data['look_ahead_positions'].append(look_ahead_pos.copy())

    def _save_experiment_data(self) -> None:
        """Save experiment data to pickle files"""
        try:
            # Create experiment directory
            exp_dir = roslib.packages.get_pkg_dir("ytlab_whill_modules") + f"/data/experiments/{self.current_path_name}"
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save path
            with open(f"{exp_dir}/path.pkl", "wb") as f:
                pickle.dump(self.path, f)
            
            # Save experiment data
            for key, data in self.experiment_data.items():
                with open(f"{exp_dir}/{key}.pkl", "wb") as f:
                    pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                'method_name': self.current_method_name,
                'start_time': self.start_time,
                'path_name': self.current_path_name,
                'goal_reached': self.goal_reached,
                'total_time': time.time() - self.start_time
            }
            with open(f"{exp_dir}/metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)
            
            rospy.loginfo(f"Experiment data saved to: {exp_dir}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save experiment data: {str(e)}")

    def _publish_reference_path(self) -> None:
        """Publish reference path for Rviz visualization"""
        try:
            if self.path is None:
                return
                
            path_msg = Path()
            path_msg.header.frame_id = self.map_tf_name
            path_msg.header.stamp = rospy.Time.now()
            
            for point in self.path:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = self.map_tf_name
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.pose.position.x = point[0]
                pose_stamped.pose.position.y = point[1]
                pose_stamped.pose.position.z = 0.0
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)
            
            self.reference_path_pub.publish(path_msg)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish reference path: {str(e)}")

    def _publish_robot_trajectory(self) -> None:
        """Publish robot trajectory for Rviz visualization"""
        try:
            if len(self.experiment_data['robot_poses']) == 0:
                return
                
            trajectory_msg = Path()
            trajectory_msg.header.frame_id = self.map_tf_name
            trajectory_msg.header.stamp = rospy.Time.now()
            
            for pose in self.experiment_data['robot_poses']:
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = self.map_tf_name
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.pose.position.x = pose[0]
                pose_stamped.pose.position.y = pose[1]
                pose_stamped.pose.position.z = 0.0
                
                # Convert yaw to quaternion
                yaw = pose[2]
                from scipy.spatial.transform import Rotation
                q = Rotation.from_euler('z', yaw).as_quat()
                pose_stamped.pose.orientation.x = q[0]
                pose_stamped.pose.orientation.y = q[1]
                pose_stamped.pose.orientation.z = q[2]
                pose_stamped.pose.orientation.w = q[3]
                
                trajectory_msg.poses.append(pose_stamped)
            
            self.robot_trajectory_pub.publish(trajectory_msg)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish robot trajectory: {str(e)}")

    def _publish_visualization_markers(self, look_ahead_pos: np.ndarray, curvature: float) -> None:
        """Publish visualization markers for Rviz"""
        try:
            marker_array = MarkerArray()
            
            # Look ahead point marker
            if look_ahead_pos is not None and len(look_ahead_pos) >= 2:
                look_ahead_marker = Marker()
                look_ahead_marker.header.frame_id = self.map_tf_name
                look_ahead_marker.header.stamp = rospy.Time.now()
                look_ahead_marker.ns = "dwpp_visualization"
                look_ahead_marker.id = 0
                look_ahead_marker.type = Marker.SPHERE
                look_ahead_marker.action = Marker.ADD
                look_ahead_marker.pose.position.x = look_ahead_pos[0]
                look_ahead_marker.pose.position.y = look_ahead_pos[1]
                look_ahead_marker.pose.position.z = 0.1
                look_ahead_marker.pose.orientation.w = 1.0
                look_ahead_marker.scale.x = 0.2
                look_ahead_marker.scale.y = 0.2
                look_ahead_marker.scale.z = 0.2
                look_ahead_marker.color.r = 1.0
                look_ahead_marker.color.g = 0.0
                look_ahead_marker.color.b = 0.0
                look_ahead_marker.color.a = 0.8
                marker_array.markers.append(look_ahead_marker)
            
            # Current goal marker
            if self.path is not None and len(self.path) > 0:
                goal_marker = Marker()
                goal_marker.header.frame_id = self.map_tf_name
                goal_marker.header.stamp = rospy.Time.now()
                goal_marker.ns = "dwpp_visualization"
                goal_marker.id = 1
                goal_marker.type = Marker.CYLINDER
                goal_marker.action = Marker.ADD
                goal_marker.pose.position.x = self.path[-1][0]
                goal_marker.pose.position.y = self.path[-1][1]
                goal_marker.pose.position.z = 0.1
                goal_marker.pose.orientation.w = 1.0
                goal_marker.scale.x = 0.4
                goal_marker.scale.y = 0.4
                goal_marker.scale.z = 0.2
                goal_marker.color.r = 0.0
                goal_marker.color.g = 1.0
                goal_marker.color.b = 0.0
                goal_marker.color.a = 0.6
                marker_array.markers.append(goal_marker)
            
            # Method name text marker
            method_marker = Marker()
            method_marker.header.frame_id = self.map_tf_name
            method_marker.header.stamp = rospy.Time.now()
            method_marker.ns = "dwpp_visualization"
            method_marker.id = 2
            method_marker.type = Marker.TEXT_VIEW_FACING
            method_marker.action = Marker.ADD
            if self.current_pose is not None:
                method_marker.pose.position.x = self.current_pose[0]
                method_marker.pose.position.y = self.current_pose[1] + 1.0
                method_marker.pose.position.z = 1.0
            method_marker.pose.orientation.w = 1.0
            method_marker.scale.z = 0.3
            method_marker.color.r = 1.0
            method_marker.color.g = 1.0
            method_marker.color.b = 1.0
            method_marker.color.a = 1.0
            method_marker.text = f"Method: {self.current_method_name.upper()}" if self.current_method_name else "DWPP"
            marker_array.markers.append(method_marker)
            
            self.visualization_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish visualization markers: {str(e)}")

if __name__ == '__main__':
    try:
        controller = PurePursuitController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
