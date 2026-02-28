# RGB-D边界检测算法 - 无头版本
import cv2
import numpy as np
import time

def detect_boundary_rgbd(cam, color_threshold=0.1, depth_threshold=0.02, min_contour_area=100, **kwargs):
    """
    使用RGB-D方法进行边界检测 - 无头版本
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    color_threshold : float
        颜色阈值
    depth_threshold : float
        深度阈值
    min_contour_area : int
        最小轮廓面积
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
    """
    start_time = time.time()
    
    try:
        # 获取RGB和深度图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # RGB-D边界检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用深度信息进行边缘检测
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 结合灰度图和深度图
        combined = cv2.addWeighted(gray, 0.7, depth_normalized, 0.3, 0)
        
        # 边缘检测
        edges = cv2.Canny(combined, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 过滤小轮廓
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # 计算轮廓统计信息
        num_contours = len(filtered_contours)
        total_contour_points = sum(len(cnt) for cnt in filtered_contours)
        
        execution_time = time.time() - start_time
        
        result = {
            'contours': filtered_contours,
            'rgb': rgb,
            'depth': depth,
            'execution_time': execution_time,
            'num_contours': num_contours,
            'total_contour_points': total_contour_points,
            'method': 'rgbd',
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
            'method': 'rgbd',
            'success': False,
            'error': str(e)
        }
        return result

def get_method_info():
    """
    返回方法信息
    
    Returns:
    --------
    info : dict
        方法信息字典
    """
    return {
        'name': 'rgbd',
        'description': 'RGB-D边界检测方法',
        'parameters': {
            'color_threshold': {
                'type': 'float',
                'default': 0.1,
                'description': '颜色阈值'
            },
            'depth_threshold': {
                'type': 'float',
                'default': 0.02,
                'description': '深度阈值'
            },
            'min_contour_area': {
                'type': 'int',
                'default': 100,
                'description': '最小轮廓面积'
            }
        },
        'advantages': [
            '结合RGB和深度信息',
            '对复杂场景适应性好',
            '能处理遮挡情况'
        ],
        'disadvantages': [
            '计算复杂度较高',
            '需要调整多个参数',
            '对深度传感器质量敏感'
        ]
    }
