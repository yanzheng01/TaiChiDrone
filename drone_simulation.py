import taichi as ti
import numpy as np

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

# 简单的投影函数 - 将3D点投影到2D屏幕
def project_points(points_3d, camera_pos, camera_dir, window_res, fov=90):
    """
    将3D点投影到相机视角的2D屏幕
    camera_pos: 相机位置
    camera_dir: 相机朝向
    """
    # 简化的相机投影
    projected = np.zeros((points_3d.shape[0], 2))
    depths = np.zeros(points_3d.shape[0])
    
    # 计算相机的右向和上向量
    up = np.array([0, 1, 0])
    right = np.cross(camera_dir, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / np.linalg.norm(right)
    cam_up = np.cross(right, camera_dir)
    cam_up = cam_up / np.linalg.norm(cam_up)
    
    for i in range(points_3d.shape[0]):
        # 计算点相对于相机的位置
        rel_pos = points_3d[i] - camera_pos
        
        # 计算点到相机的距离
        distance = np.linalg.norm(rel_pos)
        depths[i] = distance
        
        # 投影到相机坐标系
        forward_dist = np.dot(rel_pos, camera_dir)
        right_dist = np.dot(rel_pos, right)
        up_dist = np.dot(rel_pos, cam_up)
        
        # 透视投影
        if forward_dist > 0:
            # 视野角转换
            scale = 1.0 / (forward_dist * np.tan(np.radians(fov/2)))
            x = right_dist * scale
            y = up_dist * scale
            
            # 转换到屏幕坐标 [0, 1]
            projected[i, 0] = (x + 1) / 2
            projected[i, 1] = (y + 1) / 2
        else:
            projected[i, 0] = -1  # 无效点
            projected[i, 1] = -1
            
    return projected, depths

# 查找最近的N个点
def find_closest_points(drone_position, points, n):
    """
    查找距离无人机最近的n个点
    返回点索引和对应的距离
    """
    # 计算所有点到无人机的距离
    distances = np.linalg.norm(points - drone_position, axis=1)
    
    # 获取距离最近的n个点的索引
    closest_indices = np.argpartition(distances, n)[:n]
    
    # 按距离排序
    closest_indices = closest_indices[np.argsort(distances[closest_indices])]
    
    # 获取对应的距离
    closest_distances = distances[closest_indices]
    
    return closest_indices, closest_distances

# 主循环
camera_distance = 4.0
rotation_angle_y = 0.0
rotation_angle_x = 0.0

# 用于控制无人机的参数
drone_speed = 0.02

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
        # 生成随机加速度大小
        drone_acceleration = np.random.uniform(0.0001*10, 0.001*10)
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
    closest_indices, closest_distances = find_closest_points(drone_pos, points, num_closest_points)
    
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
    main_gui.text(content=f'Points: {num_points}', pos=(0.05, 0.95), color=0xFFFFFF, font_size=20)
    main_gui.text(content='[W/S]: Zoom in/out', pos=(0.05, 0.90), color=0xFFFF00, font_size=16)
    main_gui.text(content='[A/D]: Rotate Y-axis', pos=(0.05, 0.85), color=0xFFFF00, font_size=16)
    main_gui.text(content='[Q/E]: Rotate X-axis', pos=(0.05, 0.80), color=0xFFFF00, font_size=16)
    main_gui.text(content='[R]: Regenerate points', pos=(0.05, 0.75), color=0xFFFF00, font_size=16)
    main_gui.text(content='[ESC]: Exit', pos=(0.05, 0.70), color=0xFFFF00, font_size=16)
    main_gui.text(content='Drone flying through space...', pos=(0.05, 0.10), color=0x00FF00, font_size=16)
    
    # 显示无人机位置坐标
    main_gui.text(content=f'Drone Position: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})', 
                 pos=(0.05, 0.05), color=0xFF0000, font_size=16)
    
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

# 销毁GUI窗口
main_gui.close()
camera_gui.close()