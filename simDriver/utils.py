import taichi as ti
import numpy as np

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