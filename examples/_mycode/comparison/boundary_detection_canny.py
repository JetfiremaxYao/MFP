# Canny边界检测算法 - 无头版本
import cv2
import numpy as np
import time

def detect_boundary_canny(cam, min_contour_area=100, **kwargs):
    """
    使用Canny边缘检测进行边界检测 - 无头版本
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
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
        
        # Canny边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)  # 使用原始参数
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
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
            'method': 'canny',
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
            'method': 'canny',
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
        'name': 'canny',
        'description': 'Canny边缘检测方法',
        'parameters': {
            'min_contour_area': {
                'type': 'int',
                'default': 100,
                'description': '最小轮廓面积'
            }
        },
        'advantages': [
            '计算速度快',
            '参数简单',
            '对噪声有一定抗性'
        ],
        'disadvantages': [
            '可能产生断裂的边界',
            '对复杂形状适应性有限',
            '需要手动调整阈值参数'
        ]
    }
