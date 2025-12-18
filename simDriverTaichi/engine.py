import taichi as ti
import numpy as np

# 全局变量声明
points = None
point_sizes = None
drone_pos = None
drone_vel = None
drone_dir = None
latest_acceleration_dir = None
drone_acceleration = None
moment_of_inertia = None
angular_acceleration = None
angular_velocity = None
frame_count = None
acceleration_interval = 10

def initialize_fields(num_points=1000):
    """初始化所有Taichi字段"""
    global points, point_sizes, drone_pos, drone_vel, drone_dir
    global latest_acceleration_dir, drone_acceleration, moment_of_inertia
    global angular_acceleration, angular_velocity, frame_count
    
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
    frame_count = ti.field(dtype=ti.i32, shape=())

# 初始化函数
@ti.kernel
def init_simulation():
    # 初始化点云
    for i in range(points.shape[0]):
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