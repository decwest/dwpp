MIN_LOOK_AHEAD_DISTANCE = 0.1
MAX_LOOK_AHEAD_DISTANCE = 0.5
LOOK_AHEAD_TIME = 1.0
V_MAX = 0.5
V_MIN = 0.0
W_MAX = 1.0
W_MIN = -1.0
A_MAX = 0.5
AW_MAX = 1.0
DT = 0.05
APPROACH_VELOCITY_SCALING_DIST = 0.3
MIN_APPROACH_LINEAR_VELOCITY = 0.05
GOAL_TORELANCE_DIST = 0.05
GOAL_REACH_TOLERANCE_DIST_OMNI = 0.10
REGULATED_LINEAR_SCALING_MIN_RADIUS = 0.9
REGULATED_LINEAR_SCALING_MIN_SPEED = 0.0


method_name_dict = {
    "pp": "Pure Pursuit",
    "app": "Adaptive Pure Pursuit",
    "rpp": "Regulated Pure Pursuit",
    # "dwpp_wo_rpp": "Dynamic Window Pure Pursuit without RPP",
    "dwpp": "Dynamic Window Pure Pursuit",
    "dwpp_omni": "Dynamic Window Pure Pursuit (Omnidirectional)",
    "dwpp_omni_clip": "Omnidirectional PP with Ideal Vector Command"
}
# method_name_list = ["pp", "app", "rpp", "dwpp_wo_rpp", "dwpp"]
method_name_list = ["pp", "app", "rpp", "dwpp"]
