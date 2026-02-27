import math


MIN_LOOK_AHEAD_DISTANCE = 0.11
MAX_LOOK_AHEAD_DISTANCE = 0.33
LOOK_AHEAD_TIME = 1.5
V_MAX = 0.22
V_MIN = 0.0
W_MAX = 0.6
W_MIN = -0.6
A_MAX = 0.22
AW_MAX = 0.6
VX_MAX = V_MAX
VX_MIN = -V_MAX
VY_MAX = V_MAX
VY_MIN = -V_MAX
AX_MAX = A_MAX
AY_MAX = A_MAX
DT = 0.05
K = 1
APPROACH_VELOCITY_SCALING_DIST = 0.3
MIN_APPROACH_LINEAR_VELOCITY = 0.05
GOAL_TORELANCE_DIST = 0.02
GOAL_REACH_TOLERANCE_DIST_OMNI = 0.02
GOAL_REACH_TOLERANCE_HEADING = math.radians(1.0)
REGULATED_LINEAR_SCALING_MIN_RADIUS = 0.9
REGULATED_LINEAR_SCALING_MIN_SPEED = 0.0


method_name_dict = {
    "pp": "Pure Pursuit",
    "app": "Adaptive Pure Pursuit",
    "rpp": "Regulated Pure Pursuit",
    # "dwpp_wo_rpp": "Dynamic Window Pure Pursuit without RPP",
    "dwpp": "Dynamic Window Pure Pursuit",
    "dwpp_omni": "Dynamic Window Pure Pursuit (Omnidirectional)",
    "dwpp_omni_clip": "Omnidirectional PP with desired Vector Command"
}
# method_name_list = ["pp", "app", "rpp", "dwpp_wo_rpp", "dwpp"]
method_name_list = ["pp", "app", "rpp", "dwpp"]
