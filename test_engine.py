import taichi as ti
import numpy as np
import sys
import os

# 添加 simDriver 到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from simDriver.engine import update_drone_motion, render_windows
from simDriver.utils import project_points, find_closest_points

# 初始化Taichi，使用CPU架构因为当前环境不支持GPU
ti.init(arch=ti.cpu)

# 设置窗口分辨率
main_window_res = (800, 600)
camera_window_res = (400, 300)

# 创建主GUI窗口（显示整个3D空间）
main_gui = ti.GUI("3D Space with Drone", res=main_window_res, background_color=0x112F41)

# 创建无人机摄像机视角窗口
camera_gui = ti.GUI("Drone Camera View", res=camera_window_res, background_color=0x000000)

# 定义3D点的数量
num_points = 1000
num_closest_points = 5  # 要查找的最近点数量

# 创建随机3D点 (范围在 [-1, 1])
points = np.random.rand(num_points, 3) * 2 - 1  # 归一化到[-1, 1]区间

# 无人机参数
drone_pos = np.array([0.0, 0.0, -2.0])  # 初始位置
drone_vel = np.array([0.0, 0.0, 0.02])  # 初始速度（向前飞行）
drone_dir = np.array([0.0, 0.0, 1.0])   # 初始朝向（面向z轴正方向）
drone_acceleration = 0.0005  # 初始加速度大小
acceleration_interval = 10  # 每10帧加速一次
frame_count = 0  # 帧计数器

# 转动惯量参数
moment_of_inertia = 50  # 转动惯量
angular_acceleration = np.array([0.0, 0.0, 0.0])  # 角加速度
angular_velocity = np.array([0.0, 0.0, 0.0])  # 角速度

# 最新加速度方向（用于转向）
latest_acceleration_dir = np.array([0.0, 0.0, 1.0])

# 创建随机点大小数组，范围在[1.0, 5.0]之间
point_sizes = np.random.rand(num_points) * 4.0 + 1.0

# 主循环
camera_distance = 4.0
rotation_angle_y = 0.0
rotation_angle_x = 0.0

while main_gui.running and camera_gui.running:
    # 处理主窗口事件
    for e in main_gui.get_events(main_gui.PRESS):
        if e.key == main_gui.ESCAPE:
            main_gui.running = False
            camera_gui.running = False
        elif e.key == 'r':
            # 重新生成随机点
            points = np.random.rand(num_points, 3) * 2 - 1
            # 重新生成随机点大小
            point_sizes = np.random.rand(num_points) * 4.0 + 1.0
    
    # 处理键盘输入来控制主视角
    if main_gui.is_pressed('a'):
        rotation_angle_y += 0.05
    if main_gui.is_pressed('d'):
        rotation_angle_y -= 0.05
    if main_gui.is_pressed('w'):
        camera_distance = max(1.0, camera_distance - 0.1)
    if main_gui.is_pressed('s'):
        camera_distance += 0.1
    if main_gui.is_pressed('q'):
        rotation_angle_x += 0.05
    if main_gui.is_pressed('e'):
        rotation_angle_x -= 0.05
    
    # 生成新的加速度值
    new_drone_acceleration = np.random.uniform(0.0001*10, 0.001*10)
    
    # 更新无人机运动状态
    (drone_pos, drone_vel, drone_dir, frame_count, angular_acceleration, 
     angular_velocity, latest_acceleration_dir, drone_acceleration) = update_drone_motion(
        drone_pos, drone_vel, drone_dir, frame_count, acceleration_interval,
        moment_of_inertia, angular_acceleration, angular_velocity,
        latest_acceleration_dir, new_drone_acceleration)
    
    # 渲染窗口
    render_windows(main_gui, camera_gui, points, point_sizes, drone_pos, drone_dir,
                  rotation_angle_y, rotation_angle_x, camera_distance, main_window_res, 
                  camera_window_res, num_closest_points, drone_vel, drone_acceleration)

# 销毁GUI窗口
main_gui.close()
camera_gui.close()