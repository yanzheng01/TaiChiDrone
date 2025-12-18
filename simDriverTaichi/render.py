import taichi as ti
import numpy as np
from .utils import project_points

# 全局GUI变量
main_gui = None
camera_gui = None
main_window_res = (800, 600)
camera_window_res = (400, 300)

def initialize_guis():
    """初始化GUI窗口"""
    global main_gui, camera_gui
    # 创建主GUI窗口（显示整个3D空间）
    main_gui = ti.GUI("3D Space with Drone", res=main_window_res, background_color=0x112F41)
    
    # 创建无人机摄像机视角窗口
    camera_gui = ti.GUI("Drone Camera View", res=camera_window_res, background_color=0x000000)

def render_main_window(points_np, point_sizes_np, drone_pos_np, closest_distances_np, 
                       rotation_angle_y, rotation_angle_x, num_points, num_closest_points,
                       camera_distance):
    """渲染主窗口"""
    # 应用主视角旋转
    cos_y, sin_y = np.cos(rotation_angle_y), np.sin(rotation_angle_y)
    cos_x, sin_x = np.cos(rotation_angle_x), np.sin(rotation_angle_x)
    rotated_points = np.zeros_like(points_np)
    for i in range(points_np.shape[0]):
        x, y, z = points_np[i, 0], points_np[i, 1], points_np[i, 2]
        # 绕Y轴旋转
        x1 = x * cos_y - z * sin_y
        z1 = x * sin_y + z * cos_y
        
        # 绕X轴旋转
        y2 = y * cos_x - z1 * sin_x
        z2 = y * sin_x + z1 * cos_x
        
        rotated_points[i, 0] = x1
        rotated_points[i, 1] = y2
        rotated_points[i, 2] = z2
    
    # 投影3D点到主窗口屏幕
    projected_points, depths = project_points(rotated_points, 
                                              np.array([0, 0, -camera_distance]), 
                                              np.array([0, 0, 1]), 
                                              main_window_res)
    
    # 将点限制在主窗口屏幕范围内
    valid_indices = (projected_points[:, 0] >= 0) & (projected_points[:, 0] <= 1) & \
                    (projected_points[:, 1] >= 0) & (projected_points[:, 1] <= 1)
    main_valid_points = projected_points[valid_indices]
    main_valid_depths = depths[valid_indices]
    main_valid_sizes = point_sizes_np[valid_indices]  # 获取有效点的大小
    
    # 根据深度生成点的灰度颜色
    if len(main_valid_depths) > 0:
        min_depth, max_depth = np.min(main_valid_depths), np.max(main_valid_depths)
        # 避免除零错误
        depth_range = max_depth - min_depth
        if depth_range < 1e-6:
            normalized_depths = np.zeros_like(main_valid_depths)
        else:
            normalized_depths = (main_valid_depths - min_depth) / depth_range
        
        # 创建灰度颜色数组，近处亮，远处暗
        gray_values = 1.0 - normalized_depths  # 反转深度值，近处更亮
        
        # 转换为十六进制灰度颜色
        gray_intensities = (gray_values * 255).astype(np.uint32)
        main_hex_colors = (gray_intensities << 16) + (gray_intensities << 8) + gray_intensities
    else:
        main_hex_colors = np.array([0xCCCCCC] * len(main_valid_points))  # 默认灰色
        main_valid_sizes = np.array([2.0] * len(main_valid_points))  # 默认大小
    
    # 对无人机位置应用主视角旋转
    rotated_drone_pos = np.zeros_like(drone_pos_np)
    x, y, z = drone_pos_np[0], drone_pos_np[1], drone_pos_np[2]
    # 绕Y轴旋转
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    
    # 绕X轴旋转
    y2 = y * cos_x - z1 * sin_x
    z2 = y * sin_x + z1 * cos_x
    
    rotated_drone_pos[0] = x1
    rotated_drone_pos[1] = y2
    rotated_drone_pos[2] = z2
    
    # 清空主画布
    main_gui.clear(color=0x112F41)
    
    # 绘制主窗口中的点
    if len(main_valid_points) > 0:
        main_gui.circles(main_valid_points, radius=main_valid_sizes, color=main_hex_colors)
    
    # 绘制无人机位置（在主窗口中）
    # 将无人机3D位置投影到主窗口
    drone_projected_pos, _ = project_points(rotated_drone_pos.reshape(1, 3), 
                                          np.array([0, 0, -camera_distance]), 
                                          np.array([0, 0, 1]), 
                                          main_window_res)
    
    # 检查投影后的无人机位置是否在视野内
    if 0 <= drone_projected_pos[0, 0] <= 1 and 0 <= drone_projected_pos[0, 1] <= 1:
        main_gui.circle(drone_projected_pos[0], radius=8.0, color=0xFF0000)  # 红色表示无人机
    else:
        # 如果无人机不在视野内，显示在边缘并给出提示
        clipped_pos = np.clip(drone_projected_pos[0], 0, 1)
        main_gui.circle(clipped_pos, radius=6.0, color=0xFF6666)  # 浅红色表示无人机在视野外
        main_gui.text(content='DRONE OUT OF VIEW', pos=(0.7, 0.05), color=0xFF6666, font_size=16)
    
    # 显示主窗口说明文字
    main_gui.text(content=f'Points: {num_points}', pos=(0.05, 0.95), color=0xFFFFFF, font_size=20)
    main_gui.text(content='[W/S]: Zoom in/out', pos=(0.05, 0.90), color=0xFFFF00, font_size=16)
    main_gui.text(content='[A/D]: Rotate Y-axis', pos=(0.05, 0.85), color=0xFFFF00, font_size=16)
    main_gui.text(content='[Q/E]: Rotate X-axis', pos=(0.05, 0.80), color=0xFFFF00, font_size=16)
    main_gui.text(content='[R]: Regenerate points', pos=(0.05, 0.75), color=0xFFFF00, font_size=16)
    main_gui.text(content='[ESC]: Exit', pos=(0.05, 0.70), color=0xFFFF00, font_size=16)
    main_gui.text(content='Drone flying through space...', pos=(0.05, 0.10), color=0x00FF00, font_size=16)
    
    # 显示无人机位置坐标
    main_gui.text(content=f'Drone Position: ({drone_pos_np[0]:.2f}, {drone_pos_np[1]:.2f}, {drone_pos_np[2]:.2f})', 
                 pos=(0.05, 0.05), color=0xFF0000, font_size=16)
    
    # 显示最近点的信息
    for i in range(min(num_closest_points, len(closest_distances_np))):
        main_gui.text(content=f'Point {i+1}: {closest_distances_np[i]:.4f}', 
                     pos=(0.8, 0.9 - i*0.05), color=0xFFFF00, font_size=14)
    
    main_gui.show()

def render_drone_camera_window(points_np, point_sizes_np, drone_pos_np, drone_dir, drone_vel, drone_acceleration):
    """渲染无人机摄像机窗口"""
    # 投影3D点到无人机摄像机视角
    drone_projected_points, drone_depths = project_points(points_np, drone_pos_np, drone_dir[None].to_numpy(), camera_window_res)
    
    # 将点限制在无人机摄像机视角范围内
    drone_valid_indices = (drone_projected_points[:, 0] >= 0) & (drone_projected_points[:, 0] <= 1) & \
                          (drone_projected_points[:, 1] >= 0) & (drone_projected_points[:, 1] <= 1)
    drone_valid_points = drone_projected_points[drone_valid_indices]
    drone_valid_depths = drone_depths[drone_valid_indices]
    drone_valid_sizes = point_sizes_np[drone_valid_indices]  # 获取有效点的大小
    
    # 根据深度生成无人机视角点的灰度颜色
    if len(drone_valid_depths) > 0:
        min_depth, max_depth = np.min(drone_valid_depths), np.max(drone_valid_depths)
        # 避免除零错误
        depth_range = max_depth - min_depth
        if depth_range < 1e-6:
            normalized_depths = np.zeros_like(drone_valid_depths)
        else:
            normalized_depths = (drone_valid_depths - min_depth) / depth_range
        
        # 创建灰度颜色数组，近处亮，远处暗
        gray_values = 1.0 - normalized_depths  # 反转深度值，近处更亮
        
        # 转换为十六进制灰度颜色
        gray_intensities = (gray_values * 255).astype(np.uint32)
        drone_hex_colors = (gray_intensities << 16) + (gray_intensities << 8) + gray_intensities
    else:
        drone_hex_colors = np.array([0xCCCCCC] * len(drone_valid_points))  # 默认灰色
        drone_valid_sizes = np.array([2.0] * len(drone_valid_points))  # 默认大小
    
    # 清空无人机摄像机画布
    camera_gui.clear(color=0x000000)
    
    # 绘制无人机摄像机视角中的点
    if len(drone_valid_points) > 0:
        camera_gui.circles(drone_valid_points, radius=drone_valid_sizes, color=drone_hex_colors)
    
    # 显示无人机摄像机视角说明文字
    camera_gui.text(content='Drone Camera View', pos=(0.05, 0.95), color=0xFFFFFF, font_size=16)
    camera_gui.text(content=f'Drone Pos: ({drone_pos_np[0]:.2f}, {drone_pos_np[1]:.2f}, {drone_pos_np[2]:.2f})', 
                   pos=(0.05, 0.90), color=0xFFFF00, font_size=12)
    camera_gui.text(content=f'Drone Speed: {np.linalg.norm(drone_vel[None].to_numpy()):.4f}', 
                   pos=(0.05, 0.85), color=0x00FF00, font_size=12)
    camera_gui.text(content=f'Drone Accel: {drone_acceleration[None]:.5f}', 
                   pos=(0.05, 0.80), color=0x00FFFF, font_size=12)
    
    camera_gui.show()

def close_guis():
    """关闭GUI窗口"""
    if main_gui:
        main_gui.close()
    if camera_gui:
        camera_gui.close()

def get_main_gui_events():
    """获取主GUI事件"""
    return main_gui.get_events(main_gui.PRESS) if main_gui else []

def is_main_gui_running():
    """检查主GUI是否仍在运行"""
    return main_gui.running if main_gui else False

def is_camera_gui_running():
    """检查摄像机GUI是否仍在运行"""
    return camera_gui.running if camera_gui else False

def is_key_pressed(key):
    """检查是否按下了某个键"""
    return main_gui.is_pressed(key) if main_gui else False

def set_gui_running_state(main_running, camera_running):
    """设置GUI运行状态"""
    if main_gui:
        main_gui.running = main_running
    if camera_gui:
        camera_gui.running = camera_running