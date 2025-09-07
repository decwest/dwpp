# DWPP: Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints

# 🟢 **Nav2 plugin implementation is now available!**  

👉 https://github.com/Decwest/nav2_dynamic_window_pure_pursuit_controller

## Folder Structure

- `scripts/`  
  Contains simulation scripts for various pure pursuit methods. You can refer to these to understand how DWPP can be implemented.

- `ros1_node_example/`  
  Includes example ROS 1 implementations of DWPP and conventional pure pursuit methods.  
  The file `dwpp_ros_node.py` was used in real-world robot experiments, but it depends on our private configurations and cannot be used directly. Please use it as a reference only.

- `results/`  
  Contains result figures used in the paper.

## Citation

If you use this code, please cite the following paper:

> Fumiya Ohnishi, Masaki Takahashi, “Dynamic Window Pure Pursuit for Robot Path Tracking Considering Velocity and Acceleration Constraints”, Proceedings of the 19th International Conference on Intelligent Autonomous Systems, Genoa, Italy, 2025.
