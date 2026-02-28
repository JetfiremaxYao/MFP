# Alpha-Shape边界检测算法 - 无头版本
import cv2
import numpy as np
import time

# 尝试导入Alpha-Shape相关库
try:
    from scipy.spatial import Delaunay, ConvexHull
    from scipy.spatial.distance import cdist
    ALPHA_SHAPE_AVAILABLE = True
except ImportError:
    ALPHA_SHAPE_AVAILABLE = False

def detect_boundary_alpha_shape(cam, alpha_value=None, min_contour_area=100, **kwargs):
    """
    使用Alpha-Shape（凹包）进行边界检测 - 无头版本
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    alpha_value : float, optional
        Alpha参数，如果为None则自动计算
    min_contour_area : int
        最小轮廓面积（为了保持接口一致，但实际不使用）
    **kwargs : dict
        其他参数（保持接口一致性）
    
    Returns:
    --------
    result : dict
        包含以下键的字典：
        - 'contours': list - 检测到的轮廓列表
        - 'rgb': np.ndarray - RGB图像
        - 'depth': np.ndarray - 深度图像
        - 'execution_time': float - 执行时间（秒）
        - 'num_contours': int - 轮廓数量
        - 'total_contour_points': int - 总轮廓点数
        - 'method': str - 方法名称
        - 'alpha_used': float - 实际使用的alpha值
    """
    start_time = time.time()
    
    if not ALPHA_SHAPE_AVAILABLE:
        execution_time = time.time() - start_time
        result = {
            'contours': [],
            'rgb': None,
            'depth': None,
            'execution_time': execution_time,
            'num_contours': 0,
            'total_contour_points': 0,
            'method': 'alpha_shape',
            'alpha_used': None,
            'success': False,
            'error': 'Alpha-Shape库未安装'
        }
        return result
    
    try:
        # 获取RGB和深度图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 在像素坐标系进行边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # 提取边缘像素坐标
        edge_pixels = np.column_stack(np.where(edges > 0))
        
        if len(edge_pixels) == 0:
            execution_time = time.time() - start_time
            result = {
                'contours': [],
                'rgb': rgb,
                'depth': depth,
                'execution_time': execution_time,
                'num_contours': 0,
                'total_contour_points': 0,
                'method': 'alpha_shape',
                'alpha_used': None,
                'success': False,
                'error': '未检测到边缘'
            }
            return result
        
        # 转换为(u, v)格式 (注意：np.where返回的是(y, x)，需要转换为(x, y))
        edge_pixels = edge_pixels[:, [1, 0]]  # 交换x, y坐标
        
        if len(edge_pixels) < 3:
            execution_time = time.time() - start_time
            result = {
                'contours': [],
                'rgb': rgb,
                'depth': depth,
                'execution_time': execution_time,
                'num_contours': 0,
                'total_contour_points': 0,
                'method': 'alpha_shape',
                'alpha_used': None,
                'success': False,
                'error': '边缘像素点不足'
            }
            return result
        
        # 在像素坐标系进行Alpha-Shape计算
        alpha_shape_pixels = compute_alpha_shape_pixels(edge_pixels, alpha_value)
        
        if len(alpha_shape_pixels) < 3:
            execution_time = time.time() - start_time
            result = {
                'contours': [],
                'rgb': rgb,
                'depth': depth,
                'execution_time': execution_time,
                'num_contours': 0,
                'total_contour_points': 0,
                'method': 'alpha_shape',
                'alpha_used': alpha_value,
                'success': False,
                'error': 'Alpha-Shape计算失败'
            }
            return result
        
        # 转换为轮廓格式
        contours = [alpha_shape_pixels.reshape(-1, 1, 2).astype(np.int32)]
        
        # 计算轮廓统计信息
        num_contours = len(contours)
        total_contour_points = sum(len(cnt) for cnt in contours)
        
        execution_time = time.time() - start_time
        
        result = {
            'contours': contours,
            'rgb': rgb,
            'depth': depth,
            'execution_time': execution_time,
            'num_contours': num_contours,
            'total_contour_points': total_contour_points,
            'method': 'alpha_shape',
            'alpha_used': alpha_value,
            'success': True,
            'error': None
        }
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        result = {
            'contours': [],
            'rgb': None,
            'depth': None,
            'execution_time': execution_time,
            'num_contours': 0,
            'total_contour_points': 0,
            'method': 'alpha_shape',
            'alpha_used': alpha_value,
            'success': False,
            'error': str(e)
        }
        return result

def compute_alpha_shape_pixels(pixels, alpha=None):
    """在像素坐标系计算Alpha-Shape"""
    if len(pixels) < 3:
        return np.array([])
    
    try:
        # 计算自适应Alpha参数
        if alpha is None:
            alpha = compute_optimal_alpha_pixels(pixels)
        
        # 使用Delaunay三角化
        tri = Delaunay(pixels)
        
        # 过滤三角形
        alpha_edges = set()
        for simplex in tri.simplices:
            pts = pixels[simplex]
            a, b, c = pts
            
            # 计算三角形面积
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
            if area < 1e-10:
                continue
            
            # 计算外接圆半径
            circumradius = (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)) / (4 * area)
            
            # Alpha-Shape条件
            if circumradius < 1.0 / alpha:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                    alpha_edges.add(edge)
        
        if len(alpha_edges) == 0:
            # 没有Alpha-Shape边，使用凸包
            hull = ConvexHull(pixels)
            return pixels[hull.vertices]
        
        # 提取边界点
        boundary_points = set()
        for edge in alpha_edges:
            boundary_points.add(edge[0])
            boundary_points.add(edge[1])
        
        boundary_array = pixels[list(boundary_points)]
        
        # 简单排序
        return order_boundary_points_simple(boundary_array)
        
    except Exception as e:
        # 回退到凸包
        try:
            hull = ConvexHull(pixels)
            return pixels[hull.vertices]
        except:
            return pixels

def compute_optimal_alpha_pixels(pixels):
    """为像素坐标系计算最优Alpha参数"""
    if len(pixels) < 2:
        return 1.0
    
    try:
        # 计算最近邻距离
        distances = []
        for i in range(len(pixels)):
            min_dist = float('inf')
            for j in range(len(pixels)):
                if i != j:
                    dist = np.linalg.norm(pixels[i] - pixels[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        
        if len(distances) == 0:
            return 1.0
        
        median_distance = np.median(distances)
        alpha = 2.0 / median_distance
        alpha = np.clip(alpha, 0.1, 10.0)
        
        return alpha
    except:
        return 1.0

def order_boundary_points_simple(points):
    """简化的边界点排序"""
    if len(points) < 3:
        return points
    
    # 使用极角排序
    start_idx = np.argmin(points[:, 0] + points[:, 1])
    start_point = points[start_idx]
    angles = np.arctan2(points[:, 1] - start_point[1], points[:, 0] - start_point[0])
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]

def get_method_info():
    """
    返回方法信息
    
    Returns:
    --------
    info : dict
        方法信息字典
    """
    return {
        'name': 'alpha_shape',
        'description': 'Alpha-Shape凹包边界检测方法',
        'parameters': {
            'alpha_value': {
                'type': 'float',
                'default': None,
                'description': 'Alpha参数，None表示自动计算'
            },
            'min_contour_area': {
                'type': 'int',
                'default': 100,
                'description': '最小轮廓面积（保持接口一致）'
            }
        },
        'advantages': [
            '能检测复杂凹形状',
            '自动适应点云密度',
            '对噪声有一定抗性'
        ],
        'disadvantages': [
            '计算复杂度高',
            '需要额外的科学计算库',
            '参数调整相对复杂'
        ],
        'dependencies': [
            'scipy.spatial.Delaunay',
            'scipy.spatial.ConvexHull'
        ]
    }
