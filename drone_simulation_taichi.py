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

# 创建Taichi字段来存储3D点
points = ti.Vector.field(3, dtype=ti.f32, shape=(num_points,))
point_sizes = ti.field(dtype=ti.f32, shape=(num_points,))

# 无人机参数 - 使用Taichi字段
drone_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
drone_vel = ti.Vector.field(3, dtype=ti.f32, shape=())
drone_dir = ti.Vector.field(3, dtype=ti.f32, shape=())
latest_acceleration_dir = ti.Vector.field(3, dtype=ti.f32, shape=())

# 物理参数
drone_acceleration = ti.field(dtype=ti.f32, shape=())
moment_of_inertia = ti.field(dtype=ti.f32, shape=())
angular_acceleration = ti.Vector.field(3, dtype=ti.f32, shape=())
angular_velocity = ti.Vector.field(3, dtype=ti.f32, shape=())

# 其他参数
acceleration_interval = 10  # 每10帧加速一次
frame_count = ti.field(dtype=ti.i32, shape=())

# 最近点信息
closest_indices = ti.field(dtype=ti.i32, shape=(num_closest_points,))
closest_distances = ti.field(dtype=ti.f32, shape=(num_closest_points,))

# 距离计算临时字段
point_distances = ti.field(dtype=ti.f32, shape=(num_points,))

# 初始化函数
@ti.kernel
def init_simulation():
    # 初始化点云
    for i in range(num_points):
        for j in ti.static(range(3)):
            points[i][j] = ti.random() * 2.0 - 1.0  # 归一化到[-1, 1]区间
        point_sizes[i] = ti.random() * 4.0 + 1.0  # 点大小范围在[1.0, 5.0]之间
    
    # 初始化无人机参数
    drone_pos[None] = ti.Vector([0.0, 0.0, -2.0])  # 初始位置
    drone_vel[None] = ti.Vector([0.0, 0.0, 0.02])  # 初始速度（向前飞行）
    drone_dir[None] = ti.Vector([0.0, 0.0, 1.0])   # 初始朝向（面向z轴正方向）
    latest_acceleration_dir[None] = ti.Vector([0.0, 0.0, 1.0])
    
    # 初始化物理参数
    drone_acceleration[None] = 0.0005  # 初始加速度大小
    moment_of_inertia[None] = 50.0     # 转动惯量
    angular_acceleration[None] = ti.Vector([0.0, 0.0, 0.0])
    angular_velocity[None] = ti.Vector([0.0, 0.0, 0.0])
    
    # 初始化其他参数
    frame_count[None] = 0

# 查找最近的N个点
@ti.kernel
def find_closest_points_kernel():
    """
    查找距离无人机最近的n个点
    """
    # 计算所有点到无人机的距离
    for i in range(num_points):
        dist = 0.0
        for j in ti.static(range(3)):
            diff = points[i][j] - drone_pos[None][j]
            dist += diff * diff
        point_distances[i] = ti.sqrt(dist)
    
    # 简单的选择排序找出最近的几个点
    for i in range(num_closest_points):
        min_index = -1
        min_dist = 1e10
        for j in range(num_points):
            if point_distances[j] < min_dist:
                min_dist = point_distances[j]
                min_index = j
        closest_indices[i] = min_index
        closest_distances[i] = min_dist
        # 将已找到的最小值设为无穷大，以便找到下一个最小值
        point_distances[min_index] = 1e10

# 更新无人机运动
@ti.kernel
def update_drone_motion_kernel():
    """
    更新无人机运动状态
    """
    # 控制无人机移动
    drone_pos[None] += drone_vel[None]
    
    # 增加帧计数器
    frame_count[None] += 1
    
    # 定期为无人机加速，方向随机变化
    if frame_count[None] % acceleration_interval == 0:
        # 生成随机方向的加速度
        random_direction = ti.Vector([
            ti.random() * 2.0 - 1.0,
            ti.random() * 2.0 - 1.0,
            ti.random() * 2.0 - 1.0
        ])
        
        # 归一化随机方向
        norm = random_direction.norm()
        if norm > 1e-6:
            random_direction = random_direction / norm
        
        # 生成随机加速度大小
        acc_magnitude = ti.random() * (0.001 * 10 - 0.0001 * 10) + 0.0001 * 10
        drone_acceleration[None] = acc_magnitude
        
        # 应用加速度
        drone_vel[None] += random_direction * acc_magnitude
        
        # 保存最新的加速度方向
        latest_acceleration_dir[None] = random_direction
        
        # 计算角加速度（转动惯量的倒数乘以力矩）
        # 力矩由期望方向和当前方向的叉积决定
        torque = drone_dir[None].cross(latest_acceleration_dir[None])
        angular_acceleration[None] = torque / moment_of_inertia[None]
    
    # 如果无人机飞出一定范围，则重置位置和速度
    if drone_pos[None].norm() > 5.0:
        drone_pos[None] = ti.Vector([0.0, 0.0, -2.0])
        drone_vel[None] = ti.Vector([0.0, 0.0, 0.02])
        drone_dir[None] = ti.Vector([0.0, 0.0, 1.0])
        latest_acceleration_dir[None] = ti.Vector([0.0, 0.0, 1.0])
        angular_velocity[None] = ti.Vector([0.0, 0.0, 0.0])
        angular_acceleration[None] = ti.Vector([0.0, 0.0, 0.0])
        drone_acceleration[None] = 0.0005  # 重置加速度大小
        frame_count[None] = 0  # 重置帧计数器
    
    # 更新角速度和朝向
    angular_velocity[None] += angular_acceleration[None]
    # 应用阻尼以稳定系统
    angular_velocity[None] *= 0.95
    
    # 使用角速度更新朝向（简化处理）
    # 在实际应用中，应该使用四元数或旋转矩阵来避免万向锁问题
    rotation_delta = angular_velocity[None] * 0.01
    
    # 绕X轴旋转
    new_y = drone_dir[None][1] * ti.cos(rotation_delta[0]) - drone_dir[None][2] * ti.sin(rotation_delta[0])
    new_z = drone_dir[None][1] * ti.sin(rotation_delta[0]) + drone_dir[None][2] * ti.cos(rotation_delta[0])
    drone_dir[None][1] = new_y
    drone_dir[None][2] = new_z
    
    # 绕Y轴旋转
    new_x = drone_dir[None][0] * ti.cos(rotation_delta[1]) + drone_dir[None][2] * ti.sin(rotation_delta[1])
    new_z = -drone_dir[None][0] * ti.sin(rotation_delta[1]) + drone_dir[None][2] * ti.cos(rotation_delta[1])
    drone_dir[None][0] = new_x
    drone_dir[None][2] = new_z
    
    # 绕Z轴旋转
    new_x = drone_dir[None][0] * ti.cos(rotation_delta[2]) - drone_dir[None][1] * ti.sin(rotation_delta[2])
    new_y = drone_dir[None][0] * ti.sin(rotation_delta[2]) + drone_dir[None][1] * ti.cos(rotation_delta[2])
    drone_dir[None][0] = new_x
    drone_dir[None][1] = new_y
    
    # 归一化朝向向量
    norm = drone_dir[None].norm()
    if norm > 1e-6:
        drone_dir[None] = drone_dir[None] / norm

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

# 主循环
camera_distance = 4.0
rotation_angle_y = 0.0
rotation_angle_x = 0.0

# 用于控制无人机的参数
drone_speed = 0.02

# 初始化模拟
init_simulation()

while main_gui.running and camera_gui.running:
    # 处理主窗口事件
    for e in main_gui.get_events(main_gui.PRESS):
        if e.key == main_gui.ESCAPE:
            main_gui.running = False
            camera_gui.running = False
        elif e.key == 'r':
            # 重新生成随机点
            init_simulation()
    
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
    
    # 更新无人机运动（在Taichi作用域中）
    update_drone_motion_kernel()
    
    # 查找最近的点
    find_closest_points_kernel()
    
    # 将Taichi数据转换为numpy数组以便处理
    points_np = points.to_numpy()
    point_sizes_np = point_sizes.to_numpy()
    drone_pos_np = drone_pos[None].to_numpy()
    closest_indices_np = closest_indices.to_numpy()
    closest_distances_np = closest_distances.to_numpy()
    
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
    
    # 更新显示
    main_gui.show()
    camera_gui.show()

# 销毁GUI窗口
main_gui.close()
camera_gui.close()