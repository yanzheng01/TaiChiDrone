import taichi as ti
import numpy as np
from simDriverTaichi.engine import init_simulation, update_drone_motion_kernel, initialize_fields as init_engine_fields
from simDriverTaichi.utils import find_closest_points_kernel, initialize_fields as init_utils_fields
from simDriverTaichi.render import initialize_guis, render_main_window, render_drone_camera_window, close_guis, \
    get_main_gui_events, is_main_gui_running, is_camera_gui_running, is_key_pressed, set_gui_running_state

# 初始化Taichi，使用CPU架构因为当前环境不支持GPU
ti.init(arch=ti.cpu)

# 定义3D点的数量
num_points = 1000
num_closest_points = 5  # 要查找的最近点数量

# 初始化所有Taichi字段
init_engine_fields(num_points)
init_utils_fields(num_points, num_closest_points)

# 初始化GUI
initialize_guis()

# 主循环
camera_distance = 4.0
rotation_angle_y = 0.0
rotation_angle_x = 0.0

# 用于控制无人机的参数
drone_speed = 0.02

# 初始化模拟
init_simulation()

while is_main_gui_running() and is_camera_gui_running():
    # 处理主窗口事件
    for e in get_main_gui_events():
        if e.key == ti.GUI.ESCAPE:
            # 设置GUI运行状态为False以退出循环
            set_gui_running_state(False, False)
        elif e.key == 'r':
            # 重新生成随机点
            init_simulation()
    
    # 处理键盘输入来控制主视角
    if is_key_pressed('a'):
        rotation_angle_y += 0.05
    if is_key_pressed('d'):
        rotation_angle_y -= 0.05
    if is_key_pressed('w'):
        camera_distance = max(1.0, camera_distance - 0.1)
    if is_key_pressed('s'):
        camera_distance += 0.1
    if is_key_pressed('q'):
        rotation_angle_x += 0.05
    if is_key_pressed('e'):
        rotation_angle_x -= 0.05
    
    # 更新无人机运动（在Taichi作用域中）
    update_drone_motion_kernel()
    
    # 查找最近的点
    find_closest_points_kernel()
    
    # 将Taichi数据转换为numpy数组以便处理
    from simDriverTaichi.engine import points, point_sizes, drone_pos, drone_vel, drone_acceleration, drone_dir
    from simDriverTaichi.utils import closest_indices, closest_distances
    
    points_np = points.to_numpy()
    point_sizes_np = point_sizes.to_numpy()
    drone_pos_np = drone_pos[None].to_numpy()
    closest_indices_np = closest_indices.to_numpy()
    closest_distances_np = closest_distances.to_numpy()
    
    # 渲染主窗口
    render_main_window(points_np, point_sizes_np, drone_pos_np, closest_distances_np,
                      rotation_angle_y, rotation_angle_x, num_points, num_closest_points,
                      camera_distance)
    
    # 渲染无人机摄像机窗口
    render_drone_camera_window(points_np, point_sizes_np, drone_pos_np, drone_dir, drone_vel, drone_acceleration)

# 销毁GUI窗口
close_guis()