import taichi as ti
import numpy as np

# 全局变量声明
points = None
drone_pos = None
closest_indices = None
closest_distances = None
point_distances = None
num_closest_points = 5

def initialize_fields(num_points=1000, num_closest=5):
    """初始化所有Taichi字段"""
    global points, drone_pos, closest_indices, closest_distances, point_distances, num_closest_points
    
    num_closest_points = num_closest
    
    # 创建Taichi字段来存储3D点
    points = ti.Vector.field(3, dtype=ti.f32, shape=(num_points,))

    # 无人机参数 - 使用Taichi字段
    drone_pos = ti.Vector.field(3, dtype=ti.f32, shape=())

    # 最近点信息
    closest_indices = ti.field(dtype=ti.i32, shape=(num_closest_points,))
    closest_distances = ti.field(dtype=ti.f32, shape=(num_closest_points,))

    # 距离计算临时字段
    point_distances = ti.field(dtype=ti.f32, shape=(num_points,))

# 查找最近的N个点
@ti.kernel
def find_closest_points_kernel():
    """
    查找距离无人机最近的n个点
    """
    # 计算所有点到无人机的距离
    for i in range(points.shape[0]):
        dist = 0.0
        for j in ti.static(range(3)):
            diff = points[i][j] - drone_pos[None][j]
            dist += diff * diff
        point_distances[i] = ti.sqrt(dist)
    
    # 简单的选择排序找出最近的几个点
    for i in range(closest_indices.shape[0]):
        min_index = -1
        min_dist = 1e10
        for j in range(points.shape[0]):
            if point_distances[j] < min_dist:
                min_dist = point_distances[j]
                min_index = j
        closest_indices[i] = min_index
        closest_distances[i] = min_dist
        # 将已找到的最小值设为无穷大，以便找到下一个最小值
        point_distances[min_index] = 1e10


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