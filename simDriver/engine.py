import taichi as ti
import numpy as np

from simDriver.utils import project_points, find_closest_points

def update_drone_motion(drone_pos, drone_vel, drone_dir, frame_count, acceleration_interval, 
                       moment_of_inertia, angular_acceleration, angular_velocity, 
                       latest_acceleration_dir, drone_acceleration=None, points=None, num_closest_points=5):
    """
    更新无人机运动状态
    
    参数:
    - drone_pos: 无人机位置
    - drone_vel: 无人机速度
    - drone_dir: 无人机朝向
    - frame_count: 帧计数器
    - acceleration_interval: 加速间隔
    - moment_of_inertia: 转动惯量
    - angular_acceleration: 角加速度
    - angular_velocity: 角速度
    - latest_acceleration_dir: 最新加速度方向
    - drone_acceleration: 无人机加速度大小，如果为None则随机生成
    - points: 3D点云数据，用于计算最近点
    - num_closest_points: 最近点的数量
    
    返回:
    - 更新后的无人机状态元组
    - 最近点的索引和距离
    """
    # 控制无人机移动
    drone_pos += drone_vel
    
    # 增加帧计数器
    frame_count += 1
    
    # 定期为无人机加速，方向随机变化
    if frame_count % acceleration_interval == 0:
        # 生成随机方向的加速度
        random_direction = np.random.rand(3) * 2 - 1  # [-1, 1]范围的随机方向
        # 归一化随机方向
        random_direction = random_direction / np.linalg.norm(random_direction)
        # 如果没有提供加速度大小，则随机生成
        if drone_acceleration is None:
            drone_acceleration = np.random.uniform(0.0001, 0.001)
        # 应用加速度
        drone_vel += random_direction * drone_acceleration
        # 保存最新的加速度方向
        latest_acceleration_dir = random_direction
        
        # 计算角加速度（转动惯量的倒数乘以力矩）
        # 力矩由期望方向和当前方向的叉积决定
        torque = np.cross(drone_dir, latest_acceleration_dir)
        angular_acceleration = torque / moment_of_inertia
    
    # 如果无人机飞出一定范围，则重置位置和速度
    if np.linalg.norm(drone_pos) > 5.0:
        drone_pos = np.array([0.0, 0.0, -2.0])
        drone_vel = np.array([0.0, 0.0, 0.02])
        drone_dir = np.array([0.0, 0.0, 1.0])
        latest_acceleration_dir = np.array([0.0, 0.0, 1.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])
        angular_acceleration = np.array([0.0, 0.0, 0.0])
        drone_acceleration = 0.0005  # 重置加速度大小
        frame_count = 0  # 重置帧计数器
    
    # 更新角速度和朝向
    angular_velocity += angular_acceleration
    # 应用阻尼以稳定系统
    angular_velocity *= 0.95
    
    # 使用角速度更新朝向（简化处理）
    # 在实际应用中，应该使用四元数或旋转矩阵来避免万向锁问题
    rotation_delta = angular_velocity * 0.01
    
    # 应用旋转（绕各个轴旋转）
    cos_x, sin_x = np.cos(rotation_delta[0]), np.sin(rotation_delta[0])
    cos_y, sin_y = np.cos(rotation_delta[1]), np.sin(rotation_delta[1])
    cos_z, sin_z = np.cos(rotation_delta[2]), np.sin(rotation_delta[2])
    
    # 绕X轴旋转
    new_y = drone_dir[1] * cos_x - drone_dir[2] * sin_x
    new_z = drone_dir[1] * sin_x + drone_dir[2] * cos_x
    drone_dir[1] = new_y
    drone_dir[2] = new_z
    
    # 绕Y轴旋转
    new_x = drone_dir[0] * cos_y + drone_dir[2] * sin_y
    new_z = -drone_dir[0] * sin_y + drone_dir[2] * cos_y
    drone_dir[0] = new_x
    drone_dir[2] = new_z
    
    # 绕Z轴旋转
    new_x = drone_dir[0] * cos_z - drone_dir[1] * sin_z
    new_y = drone_dir[0] * sin_z + drone_dir[1] * cos_z
    drone_dir[0] = new_x
    drone_dir[1] = new_y
    
    # 归一化朝向向量
    drone_dir = drone_dir / np.linalg.norm(drone_dir)
    
    # 查找最近的点
    closest_indices = None
    closest_distances = None
    if points is not None:
        closest_indices, closest_distances = find_closest_points(drone_pos, points, num_closest_points)
    
    return (drone_pos, drone_vel, drone_dir, frame_count, angular_acceleration, 
            angular_velocity, latest_acceleration_dir, drone_acceleration, 
            closest_indices, closest_distances)

def render_windows(main_gui, camera_gui, points, point_sizes, drone_pos, drone_dir,
                  rotation_angle_y, rotation_angle_x, camera_distance, main_window_res, 
                  camera_window_res, num_closest_points, drone_vel, drone_acceleration):
    """
    渲染主窗口和无人机摄像机窗口
    
    参数:
    - main_gui: 主GUI窗口
    - camera_gui: 无人机摄像机GUI窗口
    - points: 所有3D点
    - point_sizes: 点大小数组
    - drone_pos: 无人机位置
    - drone_dir: 无人机朝向
    - rotation_angle_y: 主视角Y轴旋转角度
    - rotation_angle_x: 主视角X轴旋转角度
    - camera_distance: 主视角相机距离
    - main_window_res: 主窗口分辨率
    - camera_window_res: 无人机摄像机窗口分辨率
    - num_closest_points: 最近点数量
    - drone_vel: 无人机速度
    - drone_acceleration: 无人机加速度大小
    """
    # 应用主视角旋转
    cos_y, sin_y = np.cos(rotation_angle_y), np.sin(rotation_angle_y)
    cos_x, sin_x = np.cos(rotation_angle_x), np.sin(rotation_angle_x)
    rotated_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
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
    main_valid_sizes = point_sizes[valid_indices]  # 获取有效点的大小
    
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
    rotated_drone_pos = np.zeros_like(drone_pos)
    x, y, z = drone_pos[0], drone_pos[1], drone_pos[2]
    # 绕Y轴旋转
    x1 = x * cos_y - z * sin_y
    z1 = x * sin_y + z * cos_y
    
    # 绕X轴旋转
    y2 = y * cos_x - z1 * sin_x
    z2 = y * sin_x + z1 * cos_x
    
    rotated_drone_pos[0] = x1
    rotated_drone_pos[1] = y2
    rotated_drone_pos[2] = z2
    
    # 投影3D点到无人机摄像机视角
    drone_projected_points, drone_depths = project_points(points, drone_pos, drone_dir, camera_window_res)
    
    # 将点限制在无人机摄像机视角范围内
    drone_valid_indices = (drone_projected_points[:, 0] >= 0) & (drone_projected_points[:, 0] <= 1) & \
                          (drone_projected_points[:, 1] >= 0) & (drone_projected_points[:, 1] <= 1)
    drone_valid_points = drone_projected_points[drone_valid_indices]
    drone_valid_depths = drone_depths[drone_valid_indices]
    drone_valid_sizes = point_sizes[drone_valid_indices]  # 获取有效点的大小
    
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
    main_gui.text(content=f'Points: {len(points)}', pos=(0.05, 0.95), color=0xFFFFFF, font_size=20)
    main_gui.text(content='[W/S]: Zoom in/out', pos=(0.05, 0.90), color=0xFFFF00, font_size=16)
    main_gui.text(content='[A/D]: Rotate Y-axis', pos=(0.05, 0.85), color=0xFFFF00, font_size=16)
    main_gui.text(content='[Q/E]: Rotate X-axis', pos=(0.05, 0.80), color=0xFFFF00, font_size=16)
    main_gui.text(content='[R]: Regenerate points', pos=(0.05, 0.75), color=0xFFFF00, font_size=16)
    main_gui.text(content='[ESC]: Exit', pos=(0.05, 0.70), color=0xFFFF00, font_size=16)
    main_gui.text(content='Drone flying through space...', pos=(0.05, 0.10), color=0x00FF00, font_size=16)
    
    # 显示无人机位置坐标
    main_gui.text(content=f'Drone Position: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})', 
                 pos=(0.05, 0.05), color=0xFF0000, font_size=16)
    
    # 查找最近的点
    closest_indices, closest_distances = find_closest_points(drone_pos, points, num_closest_points)
    
    # 显示最近点的信息
    for i in range(min(num_closest_points, len(closest_distances))):
        main_gui.text(content=f'Point {i+1}: {closest_distances[i]:.4f}', 
                     pos=(0.8, 0.9 - i*0.05), color=0xFFFF00, font_size=14)
    
    # 清空无人机摄像机画布
    camera_gui.clear(color=0x000000)
    
    # 绘制无人机摄像机视角中的点
    if len(drone_valid_points) > 0:
        camera_gui.circles(drone_valid_points, radius=drone_valid_sizes, color=drone_hex_colors)
    
    # 显示无人机摄像机视角说明文字
    camera_gui.text(content='Drone Camera View', pos=(0.05, 0.95), color=0xFFFFFF, font_size=16)
    camera_gui.text(content=f'Drone Pos: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})', 
                   pos=(0.05, 0.90), color=0xFFFF00, font_size=12)
    camera_gui.text(content=f'Drone Speed: {np.linalg.norm(drone_vel):.4f}', 
                   pos=(0.05, 0.85), color=0x00FF00, font_size=12)
    camera_gui.text(content=f'Drone Accel: {drone_acceleration:.5f}', 
                   pos=(0.05, 0.80), color=0x00FFFF, font_size=12)
    
    # 更新显示
    main_gui.show()
    camera_gui.show()