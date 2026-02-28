# 基于Alpha-Shape（凹包）的边界追踪实验
import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import open3d as o3d
from sklearn.decomposition import PCA
import threading
import sys
import select
import os
from math import ceil
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy import ndimage
from scipy.interpolate import griddata
import json

# 尝试导入matplotlib，如果失败则跳过可视化
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，将跳过可视化功能")

# 尝试导入Alpha-Shape相关库
try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import pdist, squareform
    ALPHA_SHAPE_AVAILABLE = True
    print("✅ Alpha-Shape相关库已安装")
except ImportError:
    ALPHA_SHAPE_AVAILABLE = False
    print("警告: Alpha-Shape相关库未安装，将跳过Alpha-Shape方法")

# 全局变量用于ESC键检测
esc_pressed = False
scanning_active = True

def check_esc_key():
    """监听ESC键的线程函数"""
    global esc_pressed
    while scanning_active:
        if sys.platform == "darwin":  # macOS
            # macOS下使用select监听stdin
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()
                if key == 'esc' or key == 'q':
                    esc_pressed = True
                    print("\n检测到ESC键，正在结束扫描...")
                    break
        else:  # Linux/Windows
            # 其他平台使用input
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = input().strip().lower()
                    if key == 'esc' or key == 'q':
                        esc_pressed = True
                        print("\n检测到ESC键，正在结束扫描...")
                        break
            except:
                pass
        time.sleep(0.1)

def reset_arm(scene, ed6, motors_dof_idx):
    """机械臂回到初始零位"""
    motors_dof_idx = list(range(6))  # ED6为6自由度机械臂
    ed6.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000]), dofs_idx_local=motors_dof_idx)
    ed6.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200]), dofs_idx_local=motors_dof_idx)
    ed6.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12]),
        dofs_idx_local=motors_dof_idx,
    )
    # 只直接set零位
    ed6.set_dofs_position(np.zeros(6), motors_dof_idx)
    for _ in range(100):
        scene.step()
    print("机械臂已回到初始零位")

def reset_after_detection(scene, ed6, motors_dof_idx, steps=150):
    """检测后平滑插值回零，模拟现实机械臂reset"""
    qpos_now = ed6.get_dofs_position(motors_dof_idx)
    qpos_now = qpos_now.cpu().numpy()  
    qpos_zero = np.zeros_like(qpos_now)
    path = np.linspace(qpos_now, qpos_zero, steps)
    print("检测后平滑回零...")
    for q in path:
        ed6.control_dofs_position(q, motors_dof_idx)
        scene.step()
    print("机械臂已平滑回到初始零位")

def plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
    """机械臂回零，IK逆解，路径插值并执行"""
    time.sleep(2)
    target_quat = np.array([0, 1, 0, 0])
    target_pos = target_pos.copy()
    target_pos[2] += 0.2
    qpos_ik = ed6.inverse_kinematics(
        link=j6_link,
        pos=target_pos,
        quat=target_quat,
    )
    
    # 处理tensor到numpy的转换
    if hasattr(qpos_ik, 'cpu'):
        qpos_ik = qpos_ik.cpu().numpy()
    
    print("IK逆解目标关节角度:", qpos_ik)
    try:
        path = ed6.plan_path(
            qpos_goal=qpos_ik,
            num_waypoints=150,
        )
        if len(path) == 0:
            raise RuntimeError("plan_path返回空路径，自动切换为线性插值")
    except Exception as e:
        print("plan_path失败，切换为线性插值:", e)
        import torch
        if isinstance(qpos_ik, torch.Tensor):
            qpos_ik = qpos_ik.detach().cpu().numpy()
        path = np.linspace(np.zeros(6), qpos_ik, num=200)
        path = torch.from_numpy(path).float().cpu()
    print("路径插值完成，路径点数:", len(path))
    path_debug = scene.draw_debug_path(path, ed6)
    scene.step()
    print("5秒后开始沿IK路径运动...")
    time.sleep(5)
    for idx, waypoint in enumerate(path):
        ed6.control_dofs_position(waypoint, motors_dof_idx)
        for _ in range(3):
            scene.step()
        if idx % 20 == 0:
            print(f"[路径跟踪] 进度: {idx+1}/{len(path)}  J1角度: {waypoint[0]:.4f}")
    print("路径执行完毕")
    scene.clear_debug_object(path_debug)
    scene.step()
    time.sleep(2)

def detect_boundary_canny(cam, min_contour_area=100):
    """
    使用Canny边缘检测进行边界检测
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    min_contour_area : int
        最小轮廓面积（为了保持接口一致，但实际不使用）
    
    Returns:
    --------
    contours : list
        检测到的轮廓列表
    rgb : np.ndarray
        RGB图像
    depth : np.ndarray
        深度图像
    """
    # 获取RGB和深度图像
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Canny边缘检测 - 使用原始参数，与RGBD版本完全一致
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)  # 原始参数：80, 200
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 不进行面积过滤，与RGBD版本保持一致
    # 直接返回所有检测到的轮廓
    return contours, rgb, depth

def detect_boundary_alpha_shape(cam, alpha_value=None, min_contour_area=100):
    """
    使用Alpha-Shape（凹包）进行边界检测 - 修复版本
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    alpha_value : float, optional
        Alpha参数，如果为None则自动计算
    min_contour_area : int
        最小轮廓面积（为了保持接口一致，但实际不使用）
    
    Returns:
    --------
    contours : list
        检测到的轮廓列表
    rgb : np.ndarray
        RGB图像
    depth : np.ndarray
        深度图像
    """
    if not ALPHA_SHAPE_AVAILABLE:
        print("Alpha-Shape库未安装，回退到Canny方法")
        return detect_boundary_canny(cam, min_contour_area)
    
    try:
        # 获取RGB和深度图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 在像素坐标系进行边缘检测 - 使用与_boundary_phantom1.py相同的参数
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)  # 使用与_boundary_phantom1.py相同的参数
        
        # 提取边缘像素坐标
        edge_pixels = np.column_stack(np.where(edges > 0))
        
        if len(edge_pixels) == 0:
            print("未检测到边缘，回退到Canny方法")
            return detect_boundary_canny(cam, min_contour_area)
        
        # 转换为(u, v)格式 (注意：np.where返回的是(y, x)，需要转换为(x, y))
        edge_pixels = edge_pixels[:, [1, 0]]  # 交换x, y坐标
        
        if len(edge_pixels) < 3:
            print("边缘像素点不足，回退到Canny方法")
            return detect_boundary_canny(cam, min_contour_area)
        
        print(f"提取到{len(edge_pixels)}个边缘像素点")
        
        # 在像素坐标系进行Alpha-Shape计算
        alpha_shape_pixels = compute_alpha_shape_pixels(edge_pixels, alpha_value)
        
        if len(alpha_shape_pixels) < 3:
            print("Alpha-Shape计算失败，回退到Canny方法")
            return detect_boundary_canny(cam, min_contour_area)
        
        # 转换为轮廓格式
        contours = [alpha_shape_pixels.reshape(-1, 1, 2).astype(np.int32)]
        print(f"✅ Alpha-Shape成功，生成{len(contours)}个轮廓，{len(alpha_shape_pixels)}个点")
        return contours, rgb, depth
        
    except Exception as e:
        print(f"❌ Alpha-Shape检测失败: {e}，回退到Canny方法")
        import traceback
        traceback.print_exc()
        return detect_boundary_canny(cam, min_contour_area)

def compute_alpha_shape_pixels(pixels, alpha=None):
    """在像素坐标系计算Alpha-Shape"""
    if len(pixels) < 3:
        return np.array([])
    
    try:
        # 计算自适应Alpha参数
        if alpha is None:
            alpha = compute_optimal_alpha_pixels(pixels)
        
        print(f"使用Alpha参数: {alpha}")
        
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
        
        print(f"Alpha-Shape生成了{len(alpha_edges)}条边")
        
        if len(alpha_edges) == 0:
            print("没有Alpha-Shape边，使用凸包")
            from scipy.spatial import ConvexHull
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
        print(f"Alpha-Shape计算失败: {e}")
        # 回退到凸包
        try:
            from scipy.spatial import ConvexHull
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

def compute_alpha_shape(points, alpha=None):
    """
    计算Alpha-Shape（凹包）
    
    Parameters:
    -----------
    points : np.ndarray
        2D点云，形状为(N, 2)
    alpha : float, optional
        Alpha参数，如果为None则自动计算
    
    Returns:
    --------
    alpha_shape_points : np.ndarray
        Alpha-Shape边界点，形状为(M, 2)
    """
    if len(points) < 3:
        return np.array([])
    
    # 如果未指定alpha值，自动计算
    if alpha is None:
        alpha = compute_optimal_alpha(points)
    
    # 计算Delaunay三角剖分
    try:
        tri = Delaunay(points)
    except:
        # 如果Delaunay失败，返回凸包
        hull = ConvexHull(points)
        return points[hull.vertices]
    
    # 计算每个三角形的外接圆半径
    edges = set()
    edge_points = []
    
    for simplex in tri.simplices:
        # 获取三角形的三个顶点
        pts = points[simplex]
        
        # 计算外接圆半径
        a, b, c = pts
        # 使用海伦公式计算面积
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area < 1e-10:  # 避免除零
            continue
            
        # 计算外接圆半径
        circumradius = (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)) / (4 * area)
        
        # 如果外接圆半径小于alpha，则保留这个三角形
        if circumradius < 1.0 / alpha:
            # 添加三角形的边
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                if edge not in edges:
                    edges.add(edge)
                    edge_points.extend([pts[i], pts[(i+1)%3]])
    
    if len(edge_points) == 0:
        # 如果没有找到合适的边，返回凸包
        hull = ConvexHull(points)
        return points[hull.vertices]
    
    # 将边点转换为numpy数组并去重
    edge_points = np.array(edge_points)
    
    # 按顺序排列边界点
    boundary_points = order_boundary_points(edge_points)
    
    return boundary_points

def compute_optimal_alpha(points):
    """
    自动计算最优的Alpha参数
    
    Parameters:
    -----------
    points : np.ndarray
        2D点云，形状为(N, 2)
    
    Returns:
    --------
    alpha : float
        最优Alpha参数
    """
    # 计算点云的平均最近邻距离
    if len(points) < 2:
        return 1.0
    
    # 计算所有点对之间的距离
    distances = pdist(points)
    
    # 计算平均距离
    mean_distance = np.mean(distances)
    
    # 基于平均距离计算alpha值
    # alpha值越大，形状越接近凸包；alpha值越小，形状越复杂
    alpha = 2.0 / mean_distance
    
    # 限制alpha值的范围
    alpha = np.clip(alpha, 0.1, 10.0)
    
    return alpha

def order_boundary_points(points):
    """
    将边界点按顺序排列
    
    Parameters:
    -----------
    points : np.ndarray
        边界点，形状为(N, 2)
    
    Returns:
    --------
    ordered_points : np.ndarray
        按顺序排列的边界点
    """
    if len(points) < 3:
        return points
    
    # 找到最左边的点作为起始点
    start_idx = np.argmin(points[:, 0])
    start_point = points[start_idx]
    
    # 计算所有点相对于起始点的角度
    angles = np.arctan2(points[:, 1] - start_point[1], points[:, 0] - start_point[0])
    
    # 按角度排序
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]

def ensure_window_created(window_name):
    """
    确保窗口被正确创建和显示
    
    Parameters:
    -----------
    window_name : str
        窗口名称
    """
    # 创建一个小的测试图像来确保窗口被创建
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow(window_name, test_img)
    cv2.waitKey(1)  # 短暂等待确保窗口创建

def compare_detection_methods(cam, show_images=False, step_num=None, method="canny"):
    """
    比较三种边界检测方法的效果：Canny、RGB-D、Alpha-Shape
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    show_images : bool
        是否显示比较图像
    step_num : int
        当前步骤数，用于窗口标题
    method : str
        要使用的方法："canny", "rgbd", "alpha_shape", "all"
    
    Returns:
    --------
    contours : list
        检测到的轮廓列表
    """
    step_info = f" (Step {step_num})" if step_num is not None else ""
    
    if method == "canny":
        # 使用Canny边缘检测
        contours, rgb, depth = detect_boundary_canny(cam)
        print(f"  Canny{step_info}: {len(contours)} 个轮廓")
        
        if show_images:
            _show_canny_visualization(contours, rgb, step_num)
        
        return contours
        
    elif method == "rgbd":
        # 使用RGB-D边界检测
        contours, rgb, depth = detect_boundary_rgbd(cam)
        print(f"  RGB-D{step_info}: {len(contours)} 个轮廓")
        
        if show_images:
            _show_rgbd_visualization(contours, rgb, depth, step_num)
        
        return contours
        
    elif method == "alpha_shape":
        # 使用Alpha-Shape边界检测
        contours, rgb, depth = detect_boundary_alpha_shape(cam)
        print(f"  Alpha-Shape{step_info}: {len(contours)} 个轮廓")
        
        if show_images:
            _show_alpha_shape_visualization(contours, rgb, step_num)
        
        return contours
        
    elif method == "all":
        # 比较所有三种方法
        return _compare_all_methods(cam, show_images, step_num)
    
    else:
        print(f"未知的检测方法: {method}，使用Canny方法")
        return compare_detection_methods(cam, show_images, step_num, "canny")

def _show_canny_visualization(contours, rgb, step_num):
    """显示Canny检测结果"""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis_img = img.copy()
    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
    
    step_title = f"Step {step_num} - " if step_num is not None else ""
    cv2.putText(vis_img, f"{step_title}Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Contours: {len(contours)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Detection Methods Comparison (Canny)", vis_img)
    cv2.waitKey(100)

def _show_rgbd_visualization(contours, rgb, depth, step_num):
    """显示RGB-D检测结果"""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis_img = img.copy()
    cv2.drawContours(vis_img, contours, -1, (255, 0, 0), 2)  # 红色
    
    step_title = f"Step {step_num} - " if step_num is not None else ""
    cv2.putText(vis_img, f"{step_title}RGB-D", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Contours: {len(contours)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Detection Methods Comparison (RGB-D)", vis_img)
    cv2.waitKey(100)

def _show_alpha_shape_visualization(contours, rgb, step_num):
    """显示Alpha-Shape检测结果"""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis_img = img.copy()
    cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)  # 蓝色
    
    step_title = f"Step {step_num} - " if step_num is not None else ""
    cv2.putText(vis_img, f"{step_title}Alpha-Shape", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Contours: {len(contours)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Detection Methods Comparison (Alpha-Shape)", vis_img)
    cv2.waitKey(100)

def _compare_all_methods(cam, show_images, step_num):
    """比较所有三种方法"""
    step_info = f" (Step {step_num})" if step_num is not None else ""
    
    # 获取所有三种方法的结果
    contours_canny, rgb, depth = detect_boundary_canny(cam)
    contours_rgbd, _, _ = detect_boundary_rgbd(cam)
    contours_alpha, _, _ = detect_boundary_alpha_shape(cam)
    
    print(f"  方法比较{step_info}:")
    print(f"    Canny: {len(contours_canny)} 个轮廓")
    print(f"    RGB-D: {len(contours_rgbd)} 个轮廓")
    print(f"    Alpha-Shape: {len(contours_alpha)} 个轮廓")
    
    if show_images:
        # 创建三合一比较图像
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        # 创建三个子图像
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # Canny结果（绿色）
        canny_vis = img.copy()
        cv2.drawContours(canny_vis, contours_canny, -1, (0, 255, 0), 2)
        comparison[:, :w] = canny_vis
        
        # RGB-D结果（红色）
        rgbd_vis = img.copy()
        cv2.drawContours(rgbd_vis, contours_rgbd, -1, (255, 0, 0), 2)
        comparison[:, w:2*w] = rgbd_vis
        
        # Alpha-Shape结果（蓝色）
        alpha_vis = img.copy()
        cv2.drawContours(alpha_vis, contours_alpha, -1, (0, 0, 255), 2)
        comparison[:, 2*w:] = alpha_vis
        
        # 添加标签
        step_title = f"Step {step_num} - " if step_num is not None else ""
        cv2.putText(comparison, f"{step_title}Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, f"RGB-D", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, f"Alpha-Shape", (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(comparison, f"Contours: {len(contours_canny)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, f"Contours: {len(contours_rgbd)}", (w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, f"Contours: {len(contours_alpha)}", (2*w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Detection Methods Comparison (All)", comparison)
        cv2.waitKey(100)
    
    # 返回Canny结果作为默认（保持向后兼容）
    return contours_canny

def detect_boundary_rgbd(cam, color_threshold=0.1, depth_threshold=0.02, min_contour_area=100):
    """
    使用RGB-D方法进行边界检测（从RGB-D版本复制）
    
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
    
    Returns:
    --------
    contours : list
        检测到的轮廓列表
    rgb : np.ndarray
        RGB图像
    depth : np.ndarray
        深度图像
    """
    # 获取RGB和深度图像
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # 简化的RGB-D边界检测
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
    
    return filtered_contours, rgb, depth

def detect_pointcloud_closure(points, cube_pos, cube_size, closure_threshold=0.02, min_points=50):
    """
    检测点云是否形成闭环 - 混合策略（图像法+图论法）
    
    Parameters:
    -----------
    points : np.ndarray
        点云数据，形状为(N, 3)
    cube_pos : np.ndarray
        物体中心位置，形状为(3,)
    cube_size : np.ndarray
        物体尺寸，形状为(3,)
    closure_threshold : float
        闭环检测阈值，默认0.02米
    min_points : int
        最少点数要求，默认80个点（GPT-5建议）
    
    Returns:
    --------
    bool : 是否形成闭环
    float : 闭环质量（0-1之间，1表示完美闭环）
    dict : 详细的检测指标
    """
    # 前置否决条件
    if not check_closure_prerequisites(points):
        return False, 0.0, {}
    
    # 计算目标顶面面积（用于参数设置）
    target_face_area = cube_size[0] * cube_size[1]  # 约0.0189 m²
    
    # ===== 方法A：图像法（投影→栅格化→洪泛填充）=====
    closure_a, metrics_a = detect_closure_flood_fill(points, cube_size, target_face_area)
    
    # ===== 方法B：图论法（kNN图→拓扑环检测）=====
    closure_b, metrics_b = detect_closure_graph_topology(points, cube_size, target_face_area)
    
    # ===== 并联判定：任一方法检测到闭环即可 =====
    is_closed = closure_a or closure_b
    
    # 计算综合质量分数
    closure_quality = max(metrics_a.get('flood_fill_quality', 0.0), metrics_b.get('graph_quality', 0.0))
    
    # 准备详细的检测指标
    detailed_metrics = {
        'method_a_closure': closure_a,
        'method_b_closure': closure_b,
        'method_a_quality': metrics_a.get('flood_fill_quality', 0.0),
        'method_b_quality': metrics_b.get('graph_quality', 0.0),
        'overall_quality': closure_quality,
        'target_face_area': target_face_area,
        'point_count': len(points),
        **metrics_a,  # 包含方法A的详细指标
        **metrics_b   # 包含方法B的详细指标
    }
    
    return is_closed, closure_quality, detailed_metrics

def check_closure_prerequisites(points):
    """
    前置否决条件检查
    """
    # 1. 点数检查：至少80个点
    if len(points) < 80:
        return False
    
    # 2. 计算中位最近邻距离
    points_2d = points[:, :2]
    distances = cdist(points_2d, points_2d)
    np.fill_diagonal(distances, np.inf)  # 排除自身
    min_distances = np.min(distances, axis=1)
    med_nn = np.median(min_distances)
    
    # 3. 检查点云密度：中位最近邻距离不能太大
    if med_nn > 0.025:  # 25mm
        return False
    
    return True

def detect_closure_flood_fill(points, cube_size, target_face_area):
    """
    方法A：图像法（投影→栅格化→洪泛填充）
    """
    points_2d = points[:, :2]
    
    # 1. 计算自适应参数
    px, stroke_px, a_min = calculate_flood_fill_parameters(points_2d, target_face_area)
    
    # 2. 栅格化
    grid, min_coords, resolution, pad_px = create_grid_from_points(points_2d, px)
    
    # 3. 加粗边界
    thickened_grid = thicken_boundary(grid, stroke_px)
    
    # 4. 洪泛填充
    filled_grid = flood_fill_from_boundary(thickened_grid)
    
    # 5. 计算内陆面积
    inland_area = calculate_inland_area(filled_grid, resolution)
    
    # 6. 判断闭环
    is_closed = inland_area >= a_min
    
    # 7. 计算质量分数
    quality = min(1.0, inland_area / a_min) if a_min > 0 else 0.0
    
    # 8. Sanity-check：打印关键参数和比例，帮助调试
    inland_ratio = inland_area / target_face_area if target_face_area > 0 else 0.0
    print(f"    [A法调试] 内陆面积: {inland_area*1e6:.1f}mm²")
    print(f"    [A法调试] 目标面积: {target_face_area*1e6:.1f}mm²")
    print(f"    [A法调试] 内陆/目标比例: {inland_ratio:.3f} (应在0.4-1.2范围)")
    print(f"    [A法调试] 像素分辨率: {px*1e3:.1f}mm/px")
    print(f"    [A法调试] 加粗像素: {stroke_px:.1f} (建议2-8)")
    print(f"    [A法调试] 栅格留边: {pad_px}px")
    
    metrics = {
        'flood_fill_closed': is_closed,
        'flood_fill_quality': quality,
        'inland_area_mm2': inland_area * 1e6,  # 转换为mm²
        'min_area_threshold_mm2': a_min * 1e6,
        'pixel_resolution_mm': px * 1e3,
        'stroke_width_px': stroke_px,
        'grid_padding_px': pad_px,
        'inland_ratio': inland_ratio
    }
    
    return is_closed, metrics

def calculate_flood_fill_parameters(points_2d, target_face_area):
    """
    计算洪泛填充的关键参数
    """
    # 1. 计算中位最近邻距离
    distances = cdist(points_2d, points_2d)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    med_nn = np.median(min_distances)
    
    # 2. 自适应像素尺寸
    px = np.clip(0.5 * med_nn, 0.002, 0.008)  # 2-8 mm/px
    
    # 3. 邻近阈值
    t_near = np.clip(2.5 * med_nn, 0.003, 0.025)  # 3-25 mm
    
    # 4. 加粗半径
    stroke_px = np.clip(t_near / px, 1, 20)
    
    # 5. 最小内陆面积阈值
    a_min = 0.20 * target_face_area  # 20%的目标面积
    
    return px, stroke_px, a_min

def create_grid_from_points(points_2d, resolution):
    """
    将点云转换为栅格图像
    """
    # 计算边界框
    min_coords = np.min(points_2d, axis=0)
    max_coords = np.max(points_2d, axis=0)
    
    # 计算网格尺寸
    grid_width = int((max_coords[0] - min_coords[0]) / resolution) + 1
    grid_height = int((max_coords[1] - min_coords[1]) / resolution) + 1
    
    # 创建栅格
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # 将点云映射到栅格
    for point in points_2d:
        x_idx = int((point[0] - min_coords[0]) / resolution)
        y_idx = int((point[1] - min_coords[1]) / resolution)
        if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
            grid[y_idx, x_idx] = 255
    
    # 栅格留边：确保floodFill能正确填充外部背景
    # 经验公式：pad_px = max(10, int(ceil(0.05 / resolution)))
    # 给外圈留≥5cm的物理边距比较稳
    pad_px = max(10, int(np.ceil(0.05 / resolution)))
    pad_px = min(pad_px, 32)  # 上限夹到32像素
    
    # 在栅格外侧添加边距
    grid = np.pad(grid, pad_px, constant_values=0)
    
    return grid, min_coords, resolution, pad_px

def thicken_boundary(grid, stroke_px):
    """
    加粗边界，使用闭运算确保真正闭合
    """
    # 计算核大小：取max(3, 2*round(stroke_px/2)+1)，保证为≥3且奇数
    kernel_size = max(3, 2 * round(stroke_px / 2) + 1)
    if kernel_size % 2 == 0:  # 确保为奇数
        kernel_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 闭运算：dilate ×2 + erode ×1，比单次膨胀稳
    # 这样能跨过2-3px的小缝，又不至于把窄通道全部"焊死"
    thickened = cv2.dilate(grid, kernel, iterations=2)
    thickened = cv2.erode(thickened, kernel, iterations=1)
    
    return thickened

def flood_fill_from_boundary(grid):
    """
    从边界开始洪泛填充
    """
    filled_grid = grid.copy()
    
    # 创建掩码
    mask = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=np.uint8)
    
    # 使用栅格中心点作为起始填充点，确保能正确填充外部背景
    # 由于已经留了边，中心点一定在外部区域
    start_x = grid.shape[1] // 2
    start_y = grid.shape[0] // 2
    cv2.floodFill(filled_grid, mask, (start_x, start_y), 128)
    
    return filled_grid

def calculate_inland_area(filled_grid, resolution):
    """
    计算内陆面积
    """
    # 计算被边界围住的内部洞的像素数量
    # filled_grid中：255=边界，128=外部背景，0=内部洞
    inland_pixels = np.sum(filled_grid == 0)
    
    # 转换为实际面积
    inland_area = inland_pixels * (resolution ** 2)
    
    return inland_area

def detect_closure_graph_topology(points, cube_size, target_face_area):
    """
    方法B：图论法（kNN图→拓扑环检测）
    """
    points_2d = points[:, :2]
    
    # 1. 计算参数
    k, t_near, t_long = calculate_graph_parameters(points_2d)
    
    # 2. 构建kNN图
    knn_graph = build_knn_graph(points_2d, k, t_long)
    
    # 3. 检测拓扑环 - 传递points_2d以计算真实边长
    cycles, degree_stats = detect_topological_cycles(knn_graph, points_2d)
    
    # 4. 判断闭环
    is_closed = evaluate_cycle_completeness(cycles, degree_stats, t_long)
    
    # 5. 计算质量分数
    quality = calculate_graph_quality(cycles, degree_stats)
    
    # 6. Sanity-check：打印图论法关键参数，帮助调试
    print(f"    [B法调试] k近邻数: {k}")
    print(f"    [B法调试] 环数量: {cycles}")
    print(f"    [B法调试] 度数=2比例: {degree_stats.get('degree_2_ratio', 0.0):.3f} (建议≥0.70)")
    print(f"    [B法调试] 最大边长: {degree_stats.get('max_edge_length', 0.0)*1e3:.1f}mm")
    print(f"    [B法调试] 近邻阈值: {t_near*1e3:.1f}mm")
    print(f"    [B法调试] 最长边阈值: {t_long*1e3:.1f}mm")
    
    metrics = {
        'graph_closed': is_closed,
        'graph_quality': quality,
        'cycle_count': cycles,  # cycles 已经是整数，不需要 len()
        'degree_2_ratio': degree_stats.get('degree_2_ratio', 0.0),
        'max_edge_length': degree_stats.get('max_edge_length', 0.0),
        'k_neighbors': k,
        'near_threshold_mm': t_near * 1e3,
        'long_threshold_mm': t_long * 1e3
    }
    
    return is_closed, metrics

def calculate_graph_parameters(points_2d):
    """
    计算图论检测的参数
    """
    # 1. 计算中位最近邻距离
    distances = cdist(points_2d, points_2d)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    med_nn = np.median(min_distances)
    
    # 2. 参数设置 - 调整为更现实的参数
    k = 3  # 从5降到3，减少跨边连线，更像连贯折线
    t_near = np.clip(2.5 * med_nn, 0.003, 0.025)  # 3-25 mm
    t_long = 1.5 * t_near  # 最长边阈值
    
    return k, t_near, t_long

def build_knn_graph(points_2d, k, t_long):
    """
    构建kNN图
    """
    n_points = len(points_2d)
    distances = cdist(points_2d, points_2d)
    np.fill_diagonal(distances, np.inf)
    
    # 构建邻接矩阵
    adjacency_matrix = np.zeros((n_points, n_points), dtype=bool)
    
    for i in range(n_points):
        # 找到最近的k个邻居
        nearest_indices = np.argsort(distances[i])[:k]
        
        for j in nearest_indices:
            if distances[i, j] <= t_long:  # 最长边限制
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True  # 无向图
    
    return adjacency_matrix

def detect_topological_cycles(adjacency_matrix, points_2d=None):
    """
    检测拓扑环
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        邻接矩阵
    points_2d : np.ndarray, optional
        2D点云坐标，用于计算真实边长
    """
    n_points = len(adjacency_matrix)
    
    # 计算度数
    degrees = np.sum(adjacency_matrix, axis=1)
    
    # 递归去叶（度=1的毛刺）
    while True:
        leaf_nodes = np.where(degrees == 1)[0]
        if len(leaf_nodes) == 0:
            break
        
        # 移除叶子节点
        for leaf in leaf_nodes:
            neighbors = np.where(adjacency_matrix[leaf])[0]
            for neighbor in neighbors:
                adjacency_matrix[leaf, neighbor] = False
                adjacency_matrix[neighbor, leaf] = False
                degrees[neighbor] -= 1
            degrees[leaf] = 0
    
    # 计算环的秩（连通分量数）
    visited = np.zeros(n_points, dtype=bool)
    cycle_count = 0
    
    for i in range(n_points):
        if degrees[i] > 0 and not visited[i]:
            # DFS找连通分量
            stack = [i]
            visited[i] = True
            component_size = 1
            
            while stack:
                current = stack.pop()
                neighbors = np.where(adjacency_matrix[current])[0]
                for neighbor in neighbors:
                    if degrees[neighbor] > 0 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
                        component_size += 1
            
            if component_size >= 3:  # 至少3个点才能形成环
                cycle_count += 1
    
    # 计算度数=2的节点比例
    degree_2_nodes = np.sum(degrees == 2)
    total_nodes_with_edges = np.sum(degrees > 0)
    degree_2_ratio = degree_2_nodes / total_nodes_with_edges if total_nodes_with_edges > 0 else 0.0
    
    # 计算最长边长度 - 使用真实距离而不是硬编码值
    max_edge_length = 0.0
    if points_2d is not None:
        for i in range(n_points):
            for j in range(i+1, n_points):
                if adjacency_matrix[i, j]:
                    # 计算真实的欧氏距离
                    edge_length = np.linalg.norm(points_2d[i] - points_2d[j])
                    max_edge_length = max(max_edge_length, edge_length)
    else:
        # 如果没有点云数据，使用估计值
        max_edge_length = 0.01  # 1cm估计值
    
    degree_stats = {
        'degree_2_ratio': degree_2_ratio,
        'max_edge_length': max_edge_length,
        'total_nodes': total_nodes_with_edges
    }
    
    return cycle_count, degree_stats

def evaluate_cycle_completeness(cycles, degree_stats, t_long):
    """
    评估环的完整性
    """
    # 硬条件检查 - 调整为更现实的阈值
    cycle_rank_ok = cycles >= 1  # 存在至少一个独立环
    degree_2_ok = degree_stats['degree_2_ratio'] >= 0.70  # 度数=2的节点比例≥70%（从85%降到70%）
    max_edge_ok = degree_stats['max_edge_length'] <= t_long  # 最长边≤阈值
    
    return cycle_rank_ok and degree_2_ok and max_edge_ok

def calculate_graph_quality(cycles, degree_stats):
    """
    计算图论检测的质量分数
    """
    # 基于环数量和度数分布计算质量
    cycle_score = min(1.0, cycles / 2.0)  # 最多2个环
    degree_score = degree_stats['degree_2_ratio']
    
    quality = (cycle_score * 0.6 + degree_score * 0.4)
    return quality

def track_and_scan_boundary_jump(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, jump_height=0.1, max_steps=30, method="alpha_shape"):
    """
    跳跃式边界扫描 + 表面扫描 - 使用Alpha-Shape边界检测
    - 检测当前帧边界（Alpha-Shape凹包）
    - 按顺时针方向选择距离当前末端最远的边界点
    - 跳跃到该点上方（保持高度在物体上方）
    - 采集边界点云 + 表面点云
    - 重复直到闭环或回到初始点或按ESC键
    
    Parameters:
    -----------
    method : str
        边界检测方法，默认为"alpha_shape"
    """
    global esc_pressed, scanning_active
    
    # 启动ESC键监听线程
    esc_thread = threading.Thread(target=check_esc_key, daemon=True)
    esc_thread.start()
    
    all_points = []
    all_surface_points = []  # 新增：存储所有表面点云
    start_pos = None
    start_recorded = False
    h, w = cam.res[1], cam.res[0]
    center_img = np.array([w // 2, h // 2])
    
    # 闭环检测相关变量 - 每步都检查
    closure_check_interval = 1  # 每步都检查闭环
    last_closure_check_step = 0
    
    # 记录上一次选择的方向，用于顺时针约束
    last_direction = None
    
    # 自适应高度调整参数
    base_height = cube_pos[2] + 0.15  # 基础高度
    min_height = cube_pos[2] + 0.1  # 最小高度
    max_height = cube_pos[2] + 0.25  # 最大高度
    current_height = base_height
    height_adjustment_step = 0.02  # 高度调整步长
    
    # 检测质量统计
    detection_failures = 0
    max_consecutive_failures = 3
    
    # 回到初始点的检测阈值
    return_threshold = 0.03  # 3cm
    
    print(f"开始跳跃式边界扫描 (Alpha-Shape凹包方法)...")
    print("提示：在终端输入 'esc' 或 'q' 可以立即结束扫描")
    print(f"自适应高度范围: {min_height:.3f} - {max_height:.3f} 米")
    
    for step in range(max_steps):
        # 检查ESC键
        if esc_pressed:
            print("用户中断扫描，正在保存点云...")
            break
            
        print(f"\n==== 第{step+1}步跳跃扫描 ====")
        
        # 1. 边界检测（带重试机制）
        max_retries = 3
        retry_count = 0
        contours = None
        
        while retry_count < max_retries:
            # 检查ESC键
            if esc_pressed:
                break
                
            # 使用Alpha-Shape边界检测方法
            contours, rgb, depth = detect_boundary_alpha_shape(cam, min_contour_area=50)
            
            # 确保img变量可用
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            if contours:
                # 检测成功，重置失败计数
                detection_failures = 0
                
                # 每一步都显示Alpha-Shape检测结果
                print(f"第{step+1}步Alpha-Shape检测:")
                compare_detection_methods(cam, show_images=True, step_num=step+1, method="alpha_shape")  # 每一步都显示图像比较
                
                # 检查是否需要调整摄像机角度
                if optimize_camera_angle(cam, contours, center_img):
                    print(f"    检测到轮廓偏离中心，尝试调整摄像机角度...")
                    # 这里可以添加更复杂的角度调整逻辑
                    # 目前简化处理，只记录日志
                
                break
            
            # 检测失败，每次重试增加0.05米高度
            retry_count += 1
            detection_failures += 1
            
            print(f"  第{retry_count}次检测失败，抬高机械臂0.05米重试...")
            
            # 计算新的高度：基础高度 + 重试次数 * 0.05
            new_height = base_height + retry_count * 0.05
            
            # 确保高度在合理范围内
            if new_height > max_height:
                new_height = max_height
                print(f"    已达到最大高度限制: {new_height:.3f}米")
            
            current_height = new_height
            
            # 调整机械臂高度
            current_pos = j6_link.get_pos().cpu().numpy()
            adjusted_pos = current_pos.copy()
            adjusted_pos[2] = current_height
            
            # 使用路径规划移动到调整后的高度
            target_quat = j6_link.get_quat().cpu().numpy()
            qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=adjusted_pos, quat=target_quat)
            if hasattr(qpos_ik, 'cpu'):
                qpos_ik = qpos_ik.cpu().numpy()
            
            if not np.any(np.isnan(qpos_ik)) and not np.any(np.isinf(qpos_ik)):
                ed6.control_dofs_position(qpos_ik, motors_dof_idx)
                for _ in range(30):
                    scene.step()
                cam.move_to_attach()
                print(f"    已调整到高度: {current_height:.3f}米")
            else:
                print(f"    高度调整失败，IK逆解失败")
        
        if esc_pressed:
            break
            
        if not contours:
            print(f"第{step+1}步，经过{max_retries}次重试仍未检测到边界，停止")
            break
        
        # 2. 记录起始位置
        if not start_recorded:
            start_pos = j6_link.get_pos().cpu().numpy()
            start_recorded = True
            print(f"记录起始位置: {start_pos}")
        
        # 3. 获取当前机械臂末端位置
        current_pos = j6_link.get_pos().cpu().numpy()
        print(f"  当前末端位置: {current_pos}")
        
        # 4. 检查是否回到初始点
        if start_recorded and step > 3:  # 至少5步后才检查
            dist_to_start = np.linalg.norm(current_pos - start_pos)
            print(f"  距离起始位置: {dist_to_start:.4f}米")
            if dist_to_start < return_threshold:
                print(f"第{step+1}步，已回到起始位置附近，扫描完成！")
                break
        
        # 5. 找到最远的边界点（按顺时针方向）
        max_dist = 0
        best_pt = None
        best_cnt = None
        best_world_pos = None
        
        # 可视化图像
        vis_img = img.copy()
        
        # 绘制所有检测到的轮廓（绿色）
        cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
        
        for cnt in contours:
            for pt in cnt:
                pt_xy = pt[0]
                cx, cy = int(pt_xy[0]), int(pt_xy[1])
                
                # 获取深度
                if 0 <= cy < h and 0 <= cx < w:
                    d = depth[cy, cx]
                    if d > 0.05:  # 有效深度
                        # 转换到世界坐标系
                        K = cam.intrinsics
                        x = (cx - K[0,2]) * d / K[0,0]
                        y = (cy - K[1,2]) * d / K[1,1]
                        z = d
                        pos_cam = np.array([x, y, z, 1])
                        T_cam2world = np.linalg.inv(cam.extrinsics)
                        pos_world = T_cam2world @ pos_cam
                        
                        # 计算距离
                        dist = np.linalg.norm(pos_world[:3] - current_pos)
                        
                        # 顺时针方向约束
                        if last_direction is not None:
                            # 计算从当前位置到边界点的方向向量
                            direction_vector = pos_world[:2] - current_pos[:2]
                            direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
                            
                            # 检查是否符合顺时针方向
                            if not is_clockwise_direction(direction_angle, last_direction):
                                continue
                        
                        if dist > max_dist:
                            max_dist = dist
                            best_pt = pt_xy
                            best_cnt = cnt
                            best_world_pos = pos_world[:3]
        
        if best_pt is None:
            print(f"第{step+1}步，未找到合适的边界点，停止")
            break
        
        # 6. 标记最远点（红色圆圈）和当前机械臂位置（蓝色圆圈）
        # 标记目标点（红色）
        cv2.circle(vis_img, tuple(best_pt), 8, (0, 0, 255), -1)  # 红色实心圆
        cv2.circle(vis_img, tuple(best_pt), 10, (0, 0, 255), 2)  # 红色边框
        
        # 标记图像中心（当前机械臂位置，蓝色）
        center_x, center_y = int(center_img[0]), int(center_img[1])
        cv2.circle(vis_img, (center_x, center_y), 5, (255, 0, 0), -1)  # 蓝色实心圆
        cv2.circle(vis_img, (center_x, center_y), 7, (255, 0, 0), 2)   # 蓝色边框
        
        # 绘制从中心到目标点的连线（黄色）
        cv2.line(vis_img, (center_x, center_y), tuple(best_pt), (0, 255, 255), 2)
        
        # 显示图像
        cv2.imshow('Jumping Boundary Scan (Alpha-Shape)', vis_img)
        cv2.waitKey(100)
        
        # 7. 打印最远点信息
        print(f"  最远边界点像素坐标: ({best_pt[0]}, {best_pt[1]})")
        print(f"  最远边界点世界坐标: {best_world_pos}")
        print(f"  距离当前末端: {max_dist:.4f}米")
        
        # 8. 计算目标位置（保持在物体上方）
        target_pos = best_world_pos.copy()
        target_pos[2] = current_height  # 使用当前自适应高度
        
        print(f"  目标位置: {target_pos}")
        
        # 9. 更新方向记录
        direction_vector = best_world_pos[:2] - current_pos[:2]
        last_direction = np.arctan2(direction_vector[1], direction_vector[0])
        
        # 10. 采集点云（边界点 + 表面点）
        pc, _ = cam.render_pointcloud(world_frame=True)
        
        # 10.1 采集边界点 - 修复维度不匹配问题
        boundary_points = []
        for cnt in contours:
            # 只取轮廓的边界点
            for pt in cnt:
                px, py = pt[0]
                if 0 <= px < depth.shape[1] and 0 <= py < depth.shape[0]:
                    # 获取该像素位置的点云数据
                    pixel_index = py * depth.shape[1] + px
                    if pixel_index < len(pc):
                        boundary_points.append(pc[pixel_index])
        
        boundary_points = np.array(boundary_points) if len(boundary_points) > 0 else np.zeros((0, 3))
        all_points.append(boundary_points)
        print(f"第{step+1}步，采集到{len(boundary_points)}个边界点")
        
        # 10.2 采集表面点（当前视野内的所有有效点，但限制数量）
        surface_points = []
        
        # 创建表面点mask：排除边界点，保留物体表面的其他点
        # 使用适中的采样策略平衡数据量和精度
        sample_rate = 0.05  # 采样5%的点（从2%提升到5%）
        for py in range(0, depth.shape[0], 3):  # 每隔3个像素采样（从4个提升到3个）
            for px in range(0, depth.shape[1], 3):  # 每隔3个像素采样（从4个提升到3个）
                if depth[py, px] > 0.05:  # 排除无效深度
                    # 检查是否在物体表面范围内
                    if 0.01 < depth[py, px] < 0.5:  # 合理的深度范围
                        # 随机采样进一步减少数据量
                        if np.random.random() < sample_rate:
                            # 获取该像素位置的点云数据
                            pixel_index = py * depth.shape[1] + px
                            if pixel_index < len(pc):
                                surface_points.append(pc[pixel_index])
        
        surface_points = np.array(surface_points) if len(surface_points) > 0 else np.zeros((0, 3))
        if len(surface_points) > 0:
            all_surface_points.append(surface_points)
            print(f"第{step+1}步，采集到{len(surface_points)}个表面点（采样率5%）")
        
        # 11. 闭环检测 - 修改为5步之后才开始检测
        if step >= 4 and step - last_closure_check_step >= closure_check_interval and len(all_points) > 0:
            current_all_points = np.concatenate(all_points, axis=0)
            
            # 计算cube的尺寸（基于已知信息）
            cube_size = np.array([0.105, 0.18, 0.022])  # 从scene.add_entity中获取
            
            is_closed, closure_quality, detailed_metrics = detect_pointcloud_closure(
                current_all_points, cube_pos, cube_size
            )
            
            print(f"  闭环检测: 质量={closure_quality:.3f}, 形成闭环={is_closed}")
            print(f"  详细指标:")
            print(f"    方法A(图像法): {detailed_metrics['method_a_closure']}, 质量={detailed_metrics['method_a_quality']:.3f}")
            print(f"      内陆面积: {detailed_metrics['inland_area_mm2']:.1f}mm²")
            print(f"      面积阈值: {detailed_metrics['min_area_threshold_mm2']:.1f}mm²")
            print(f"      像素分辨率: {detailed_metrics['pixel_resolution_mm']:.1f}mm/px")
            print(f"    方法B(图论法): {detailed_metrics['method_b_closure']}, 质量={detailed_metrics['method_b_quality']:.3f}")
            print(f"      环数量: {detailed_metrics['cycle_count']}")
            print(f"      度数=2比例: {detailed_metrics['degree_2_ratio']:.3f}")
            print(f"      k近邻数: {detailed_metrics['k_neighbors']}")
            print(f"    目标面积: {detailed_metrics['target_face_area']*1e6:.1f}mm²")
            print(f"    点云总数: {detailed_metrics['point_count']}")
            print(f"    并联判定: 任一方法检测到闭环即可")
            
            if is_closed:
                print(f"第{step+1}步，点云已形成闭环，跳跃扫描完成！")
                break
            
            last_closure_check_step = step
        elif step < 4:
            print(f"  前5步跳过闭环检测，当前为第{step+1}步")
        
        # # 12. 实时可视化点云
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(boundary_points)
        # o3d.visualization.draw_geometries([pcd], window_name=f"Step {step+1} Boundary PointCloud (Alpha-Shape)", width=400, height=300)
        # time.sleep(2)
        # # 11.5. 累积点云可视化
        # if len(all_points) > 0:
        #     accumulated_points = np.concatenate(all_points, axis=0)
        #     accumulated_pcd = o3d.geometry.PointCloud()
        #     accumulated_pcd.points = o3d.utility.Vector3dVector(accumulated_points)
        #     o3d.visualization.draw_geometries([accumulated_pcd], window_name="Accumulated Point Cloud", width=400, height=300)
        #     time.sleep(2)
        
        # 12. IK逆解和路径规划
        target_quat = j6_link.get_quat().cpu().numpy()
        qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
        if hasattr(qpos_ik, 'cpu'):
            qpos_ik = qpos_ik.cpu().numpy()
        if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
            print(f"第{step+1}步，IK逆解失败，目标位置可能超出工作空间")
            break
        
        # 13. 验证IK解的有效性
        joint_limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], 
                                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
        for i, (q, limits) in enumerate(zip(qpos_ik, joint_limits)):
            if q < limits[0] or q > limits[1]:
                print(f"第{step+1}步，关节{i}角度{q:.4f}超出范围[{limits[0]:.4f}, {limits[1]:.4f}]")
                break
        else:
            pass
        
        # 14. 路径规划和执行
        try:
            path = ed6.plan_path(qpos_goal=qpos_ik, num_waypoints=100)
            if len(path) == 0:
                raise RuntimeError("plan_path返回空路径，使用直接控制")
        except Exception as e:
            print(f"路径规划失败，使用直接控制: {e}")
            path = [qpos_ik]
        
        print(f"  路径规划完成，路径点数: {len(path)}")
        
        # 15. 执行路径
        for idx, waypoint in enumerate(path):
            # 检查ESC键
            if esc_pressed:
                break
                
            ed6.control_dofs_position(waypoint, motors_dof_idx)
            for _ in range(3):
                scene.step()
            if idx % 20 == 0:
                print(f"    [路径执行] 进度: {idx+1}/{len(path)}")
        
        if esc_pressed:
            break
            
        cam.move_to_attach()
        
        # 16. 验证实际到达位置
        actual_pos = j6_link.get_pos().cpu().numpy()
        pos_error = np.linalg.norm(actual_pos - target_pos)
        print(f"  实际位置: {actual_pos}")
        print(f"  位置误差: {pos_error:.4f}米")
        
        if pos_error > 0.05:
            print(f"第{step+1}步，位置误差过大({pos_error:.4f}m)，可能IK失败或机械臂卡住")
            print(f"  建议：检查工作空间或调整目标位置")
            break
        
        print(f"第{step+1}步跳跃完成\n")
    
    # 结束扫描
    scanning_active = False
    
    if len(all_points) == 0:
        print("未采集到任何边界点云！")
        return np.zeros((0,3)), np.zeros((0,3))
    
    # 合并边界点云
    all_boundary_points = np.concatenate(all_points, axis=0)
    print(f"跳跃式边界扫描完成 (Alpha-Shape方法)，总共采集到{len(all_boundary_points)}个边界点")
    
    # 合并表面点云并去重
    if len(all_surface_points) > 0:
        all_surface_points_combined = np.concatenate(all_surface_points, axis=0)
        print(f"表面扫描完成，总共采集到{len(all_surface_points_combined)}个表面点")
        
        # 去重处理
        all_surface_points_dedup = remove_duplicate_points(all_surface_points_combined)
        print(f"去重后表面点云: {len(all_surface_points_dedup)}个点")
    else:
        all_surface_points_dedup = np.zeros((0,3))
        print("未采集到表面点云")
    
    return all_boundary_points, all_surface_points_dedup

def track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, step_size=0.08, max_steps=30, method="alpha_shape"):
    """
    使用跳跃式边界扫描 + 表面扫描替代原来的切线跟踪（Alpha-Shape方法）
    
    Parameters:
    -----------
    method : str
        边界检测方法，默认为"alpha_shape"
    """
    return track_and_scan_boundary_jump(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, jump_height=0.1, max_steps=max_steps, method=method)

def visualize_and_save_pointcloud(points, filename="boundary_cloud_alpha_shape.ply"):
    """用Open3D可视化并保存点云"""
    if len(points) == 0:
        print("点云为空，无法可视化和保存！")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存为{filename}")
    # 修正 linter 报错，使用 o3d.visualization.draw_geometries
    o3d.visualization.draw_geometries([pcd])  # type: ignore

def is_clockwise_direction(current_angle, last_angle, tolerance=np.pi/4):
    """
    检查当前方向是否符合顺时针约束
    
    Parameters:
    -----------
    current_angle : float
        当前方向角度
    last_angle : float
        上一次方向角度
    tolerance : float
        允许的角度偏差
    
    Returns:
    --------
    bool : 是否符合顺时针方向
    """
    if last_angle is None:
        return True
    
    # 计算角度差
    angle_diff = current_angle - last_angle
    
    # 处理角度跨越±π的情况
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # 顺时针方向：角度差应该为正（或接近0）
    return angle_diff >= -tolerance

def optimize_camera_angle(cam, current_contours, img_center):
    """
    根据检测效果优化摄像机角度
    
    Parameters:
    -----------
    cam : Camera
        摄像机对象
    current_contours : list
        当前检测到的轮廓
    img_center : np.ndarray
        图像中心点
    
    Returns:
    --------
    bool : 是否需要调整角度
    """
    if not current_contours:
        return True  # 没有检测到轮廓，需要调整
    
    # 计算轮廓在图像中的分布
    all_contour_points = []
    for cnt in current_contours:
        all_contour_points.extend(cnt.reshape(-1, 2))
    
    if not all_contour_points:
        return True
    
    all_contour_points = np.array(all_contour_points)
    
    # 计算轮廓中心
    contour_center = np.mean(all_contour_points, axis=0)
    
    # 计算轮廓中心到图像中心的距离
    center_offset = np.linalg.norm(contour_center - img_center)
    
    # 如果轮廓中心偏离图像中心太远，需要调整角度
    max_offset = 100  # 像素阈值
    
    return center_offset > max_offset

def adjust_camera_orientation(scene, ed6, cam, motors_dof_idx, j6_link, target_angle_offset):
    """
    调整摄像机朝向
    
    Parameters:
    -----------
    target_angle_offset : float
        目标角度偏移（弧度）
    """
    current_pos = j6_link.get_pos().cpu().numpy()
    current_quat = j6_link.get_quat().cpu().numpy()
    
    # 计算新的朝向
    # 这里简化处理，实际应用中可能需要更复杂的角度计算
    target_quat = current_quat.copy()
    
    # 使用IK调整机械臂朝向
    qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=current_pos, quat=target_quat)
    if hasattr(qpos_ik, 'cpu'):
        qpos_ik = qpos_ik.cpu().numpy()
    
    if not np.any(np.isnan(qpos_ik)) and not np.any(np.isinf(qpos_ik)):
        ed6.control_dofs_position(qpos_ik, motors_dof_idx)
        for _ in range(30):
            scene.step()
        cam.move_to_attach()
        return True
    
    return False

# ===== 新增：表面点云采集和触诊建议模块 =====

def remove_duplicate_points(points, tolerance=0.001):
    """
    去除重复的点云点
    
    Parameters:
    -----------
    points : np.ndarray
        点云数据，形状为(N, 3)
    tolerance : float
        去重容差，默认1mm
    
    Returns:
    --------
    unique_points : np.ndarray
        去重后的点云，形状为(M, 3)
    """
    if len(points) == 0:
        return points
    
    print(f"开始去重处理，原始点数: {len(points)}")
    
    # 使用scipy的KDTree进行快速去重
    from scipy.spatial import cKDTree
    
    # 构建KDTree
    tree = cKDTree(points)
    
    # 找到每个点的最近邻
    distances, indices = tree.query(points, k=2)  # k=2因为最近邻是自己
    
    # 标记需要保留的点
    keep_mask = np.ones(len(points), dtype=bool)
    
    for i in range(len(points)):
        if keep_mask[i]:  # 如果这个点还没有被标记为删除
            # 找到所有距离小于容差的点
            nearby_indices = tree.query_ball_point(points[i], tolerance)
            # 保留第一个点，删除其他点
            for j in nearby_indices[1:]:  # 跳过自己
                if j > i:  # 只删除索引更大的点，避免重复删除
                    keep_mask[j] = False
    
    unique_points = points[keep_mask]
    print(f"去重完成，保留点数: {len(unique_points)}")
    
    return unique_points

def scan_surface_points(scene, ed6, cam, motors_dof_idx, j6_link, boundary_points, cube_pos, 
                       scan_density=8, scan_height_offset=0.12):
    """
    在边界内部进行表面点云密集采样
    
    Parameters:
    -----------
    boundary_points : np.ndarray
        边界点云，形状为(N, 3)
    cube_pos : np.ndarray
        物体中心位置
    scan_density : int
        每平方厘米的采样点数，默认8个点
    scan_height_offset : float
        扫描高度偏移，默认0.12米
    
    Returns:
    --------
    surface_points : np.ndarray
        表面点云，形状为(M, 3)
    scan_grid : np.ndarray
        扫描网格信息
    """
    global esc_pressed
    
    if len(boundary_points) == 0:
        print("边界点云为空，无法进行表面扫描")
        return np.zeros((0, 3)), None
    
    print(f"\n开始表面点云密集采样...")
    print(f"扫描密度: {scan_density} 点/平方厘米")
    print(f"扫描高度: {cube_pos[2] + scan_height_offset:.3f} 米")
    
    # 1. 计算扫描区域边界框
    points_2d = boundary_points[:, :2]
    min_coords = np.min(points_2d, axis=0)
    max_coords = np.max(points_2d, axis=0)
    
    # 2. 计算网格分辨率（基于扫描密度）
    area_cm2 = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1]) * 10000  # 转换为cm²
    total_points = int(area_cm2 * scan_density)
    
    # 计算网格尺寸
    aspect_ratio = (max_coords[1] - min_coords[1]) / (max_coords[0] - min_coords[0])
    grid_width = int(np.sqrt(total_points / aspect_ratio))
    grid_height = int(grid_width * aspect_ratio)
    
    # 确保网格尺寸合理
    grid_width = max(5, min(grid_width, 50))
    grid_height = max(5, min(grid_height, 50))
    
    print(f"扫描区域: {area_cm2:.1f} cm²")
    print(f"网格尺寸: {grid_width} x {grid_height}")
    print(f"预计采样点数: {grid_width * grid_height}")
    
    # 3. 生成扫描网格
    x_coords = np.linspace(min_coords[0], max_coords[0], grid_width)
    y_coords = np.linspace(min_coords[1], max_coords[1], grid_height)
    X, Y = np.meshgrid(x_coords, y_coords)
    scan_positions = np.column_stack([X.ravel(), Y.ravel()])
    
    # 4. 过滤扫描点（只保留在边界内部的点）
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points_2d)
        inside_mask = []
        for pos in scan_positions:
            # 使用射线法判断点是否在多边形内部
            is_inside = point_in_polygon(pos, points_2d[hull.vertices])
            inside_mask.append(is_inside)
        inside_mask = np.array(inside_mask)
        valid_positions = scan_positions[inside_mask]
    except:
        # 如果凸包计算失败，使用边界框
        print("凸包计算失败，使用边界框进行扫描")
        valid_positions = scan_positions
    
    print(f"有效扫描点数: {len(valid_positions)}")
    
    # 5. 执行表面扫描
    surface_points = []
    scan_height = cube_pos[2] + scan_height_offset
    
    for i, pos_2d in enumerate(valid_positions):
        if esc_pressed:
            print("用户中断表面扫描")
            break
            
        if i % 10 == 0:
            print(f"表面扫描进度: {i+1}/{len(valid_positions)}")
        
        # 计算目标位置
        target_pos = np.array([pos_2d[0], pos_2d[1], scan_height])
        
        # 移动到扫描位置
        target_quat = j6_link.get_quat().cpu().numpy()
        qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
        
        if hasattr(qpos_ik, 'cpu'):
            qpos_ik = qpos_ik.cpu().numpy()
        
        if not np.any(np.isnan(qpos_ik)) and not np.any(np.isinf(qpos_ik)):
            # 执行移动
            ed6.control_dofs_position(qpos_ik, motors_dof_idx)
            for _ in range(20):  # 减少等待时间
                scene.step()
            cam.move_to_attach()
            
            # 采集点云
            pc, _ = cam.render_pointcloud(world_frame=True)
            if len(pc) > 0:
                # 只保留接近目标高度的点
                height_filter = np.abs(pc[:, 2] - scan_height) < 0.05
                if np.any(height_filter):
                    filtered_pc = pc[height_filter]
                    if len(filtered_pc) > 0:
                        surface_points.append(filtered_pc)
    
    if len(surface_points) == 0:
        print("未采集到表面点云")
        return np.zeros((0, 3)), None
    
    # 合并所有表面点
    all_surface_points = np.concatenate(surface_points, axis=0)
    print(f"表面扫描完成，采集到 {len(all_surface_points)} 个表面点")
    
    # 创建扫描网格信息
    scan_grid = {
        'grid_width': grid_width,
        'grid_height': grid_height,
        'min_coords': min_coords,
        'max_coords': max_coords,
        'scan_positions': valid_positions,
        'scan_height': scan_height
    }
    
    return all_surface_points, scan_grid

def point_in_polygon(point, polygon):
    """
    使用射线法判断点是否在多边形内部
    
    Parameters:
    -----------
    point : np.ndarray
        测试点坐标 [x, y]
    polygon : np.ndarray
        多边形顶点坐标，形状为(N, 2)
    
    Returns:
    --------
    bool : 点是否在多边形内部
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def reconstruct_surface(surface_points, scan_grid, resolution=0.0015):
    """
    表面重建：将点云转换为规则网格（优化内存使用）
    
    Parameters:
    -----------
    surface_points : np.ndarray
        表面点云，形状为(N, 3)
    scan_grid : dict
        扫描网格信息
    resolution : float
        重建分辨率，默认1.5mm（平衡精度和内存）
    
    Returns:
    --------
    surface_grid : np.ndarray
        表面高度网格，形状为(H, W)
    grid_coords : tuple
        网格坐标信息 (x_coords, y_coords)
    """
    if len(surface_points) == 0:
        print("表面点云为空，无法进行表面重建")
        return None, None
    
    print(f"\n开始表面重建...")
    print(f"重建分辨率: {resolution*1000:.1f} mm")
    
    # 1. 限制点云数量，避免内存溢出
    max_points = 5000  # 最多使用5000个点（从3000提升到5000）
    if len(surface_points) > max_points:
        print(f"点云数量过多({len(surface_points)})，随机采样{max_points}个点")
        indices = np.random.choice(len(surface_points), max_points, replace=False)
        surface_points = surface_points[indices]
    
    # 2. 计算重建网格
    min_coords = scan_grid['min_coords']
    max_coords = scan_grid['max_coords']
    
    # 限制网格大小，避免内存溢出
    max_grid_size = 70  # 最大网格尺寸（从50提升到70）
    x_range = max_coords[0] - min_coords[0]
    y_range = max_coords[1] - min_coords[1]
    
    # 自适应调整分辨率
    if x_range / resolution > max_grid_size:
        resolution = x_range / max_grid_size
    if y_range / resolution > max_grid_size:
        resolution = max(y_range / max_grid_size, resolution)
    
    print(f"调整后分辨率: {resolution*1000:.1f} mm")
    
    x_coords = np.arange(min_coords[0], max_coords[0] + resolution, resolution)
    y_coords = np.arange(min_coords[1], max_coords[1] + resolution, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    print(f"网格尺寸: {X.shape}")
    
    # 3. 插值计算表面高度
    from scipy.interpolate import griddata
    
    # 准备插值数据
    points_2d = surface_points[:, :2]
    heights = surface_points[:, 2]
    
    # 使用线性插值
    surface_heights = griddata(
        points_2d, heights, (X, Y), 
        method='linear', fill_value=np.nan
    )
    
    print(f"表面重建完成，网格尺寸: {surface_heights.shape}")
    
    return surface_heights, (x_coords, y_coords)

def detect_surface_anomalies(surface_grid, grid_coords, sigma_scales=[1.0, 2.0], 
                           threshold_percentile=90):
    """
    使用简化的高度差检测表面异常（凸起、结节）- 极简内存使用
    
    Parameters:
    -----------
    surface_grid : np.ndarray
        表面高度网格
    grid_coords : tuple
        网格坐标信息
    sigma_scales : list
        未使用（保持接口一致）
    threshold_percentile : int
        异常检测阈值百分位数
    
    Returns:
    --------
    anomalies : list
        检测到的异常区域列表
    anomaly_map : np.ndarray
        异常强度图
    """
    if surface_grid is None:
        print("表面网格为空，无法进行异常检测")
        return [], None
    
    print(f"\n开始表面异常检测（简化方法）...")
    print(f"阈值百分位: {threshold_percentile}%")
    
    # 1. 处理NaN值
    valid_mask = ~np.isnan(surface_grid)
    if not np.any(valid_mask):
        print("表面网格全为NaN，无法进行异常检测")
        return [], None
    
    # 用周围有效值填充NaN
    surface_filled = surface_grid.copy()
    surface_filled[~valid_mask] = np.nanmedian(surface_grid[valid_mask])
    
    # 2. 简化的异常检测：基于高度差
    x_coords, y_coords = grid_coords
    resolution = x_coords[1] - x_coords[0]  # 假设均匀网格
    
    # 计算每个点与周围点的最大高度差
    height_diff_map = np.zeros_like(surface_filled)
    
    # 使用简单的3x3邻域检测
    for i in range(1, surface_filled.shape[0] - 1):
        for j in range(1, surface_filled.shape[1] - 1):
            if valid_mask[i, j]:
                # 计算3x3邻域的最大高度差
                neighborhood = surface_filled[i-1:i+2, j-1:j+2]
                max_height = np.max(neighborhood)
                min_height = np.min(neighborhood)
                height_diff_map[i, j] = max_height - min_height
    
    # 3. 自适应阈值
    valid_diff = height_diff_map[valid_mask]
    threshold = np.percentile(valid_diff, threshold_percentile)
    
    print(f"检测阈值: {threshold:.3f}")
    
    # 4. 检测异常区域
    anomaly_mask = (height_diff_map > threshold) & valid_mask
    
    # 5. 连通组件分析
    from scipy import ndimage
    labeled_anomalies, num_anomalies = ndimage.label(anomaly_mask)
    
    print(f"检测到 {num_anomalies} 个异常区域")
    
    # 6. 提取异常区域信息
    anomalies = []
    for i in range(1, num_anomalies + 1):
        anomaly_region = (labeled_anomalies == i)
        
        # 计算异常区域中心
        y_indices, x_indices = np.where(anomaly_region)
        center_x = np.mean(x_coords[x_indices])
        center_y = np.mean(y_coords[y_indices])
        
        # 计算异常强度（基于高度差）
        anomaly_strength = np.mean(height_diff_map[anomaly_region])
        
        # 计算异常高度
        anomaly_heights = surface_filled[anomaly_region]
        max_height = np.max(anomaly_heights)
        min_height = np.min(anomaly_heights)
        height_elevation = max_height - min_height
        
        anomaly_info = {
            'id': i,
            'center': np.array([center_x, center_y]),
            'strength': anomaly_strength,
            'height_elevation': height_elevation,
            'area_pixels': np.sum(anomaly_region),
            'area_mm2': np.sum(anomaly_region) * (resolution ** 2) * 1e6,
            'region_mask': anomaly_region
        }
        anomalies.append(anomaly_info)
    
    # 按异常强度排序
    anomalies.sort(key=lambda x: x['strength'], reverse=True)
    
    print(f"异常检测完成，发现 {len(anomalies)} 个异常区域")
    for i, anomaly in enumerate(anomalies):
        print(f"  异常{i+1}: 强度={anomaly['strength']:.3f}, "
              f"高度差={anomaly['height_elevation']*1000:.1f}mm, "
              f"面积={anomaly['area_mm2']:.1f}mm²")
    
    return anomalies, height_diff_map

def create_log_kernel(sigma_pixels, kernel_size=None):
    """
    创建LoG（Laplacian of Gaussian）核
    
    Parameters:
    -----------
    sigma_pixels : float
        高斯标准差（像素单位）
    kernel_size : int, optional
        核大小，如果为None则自动计算
    
    Returns:
    --------
    kernel : np.ndarray
        LoG核
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma_pixels) + 1)
    
    # 创建坐标网格
    x = np.arange(kernel_size) - kernel_size // 2
    y = np.arange(kernel_size) - kernel_size // 2
    X, Y = np.meshgrid(x, y)
    
    # 计算距离
    r_squared = X**2 + Y**2
    
    # LoG公式: (r^2 - 2*sigma^2) / (sigma^4) * exp(-r^2/(2*sigma^2))
    log_kernel = (r_squared - 2 * sigma_pixels**2) / (sigma_pixels**4) * \
                 np.exp(-r_squared / (2 * sigma_pixels**2))
    
    # 归一化
    log_kernel = log_kernel - np.mean(log_kernel)
    
    return log_kernel

def generate_palpation_suggestions(anomalies, cube_pos, max_suggestions=5):
    """
    生成触诊建议位置
    
    Parameters:
    -----------
    anomalies : list
        检测到的异常区域列表
    cube_pos : np.ndarray
        物体中心位置
    max_suggestions : int
        最大建议数量
    
    Returns:
    --------
    suggestions : list
        触诊建议列表
    """
    print(f"\n生成触诊建议...")
    
    suggestions = []
    
    if len(anomalies) == 0:
        # 没有检测到异常，建议触诊中心位置
        center_suggestion = {
            'position': cube_pos.copy(),
            'type': 'center',
            'priority': 1,
            'reason': '表面平滑，建议触诊中心位置',
            'confidence': 0.5
        }
        suggestions.append(center_suggestion)
        print("未检测到异常，建议触诊中心位置")
    else:
        # 为每个异常区域生成触诊建议
        for i, anomaly in enumerate(anomalies[:max_suggestions]):
            # 计算触诊位置（异常中心上方）
            palpation_pos = np.array([
                anomaly['center'][0],
                anomaly['center'][1], 
                cube_pos[2] + 0.15  # 触诊高度
            ])
            
            # 计算优先级（基于异常强度）
            priority = i + 1
            
            # 计算置信度
            confidence = min(0.9, anomaly['strength'] / 10.0)
            
            suggestion = {
                'position': palpation_pos,
                'type': 'anomaly',
                'priority': priority,
                'anomaly_id': anomaly['id'],
                'strength': anomaly['strength'],
                'height_elevation': anomaly['height_elevation'],
                'area_mm2': anomaly['area_mm2'],
                'reason': f'检测到异常区域{i+1}，高度差{anomaly["height_elevation"]*1000:.1f}mm',
                'confidence': confidence
            }
            suggestions.append(suggestion)
            
            print(f"建议{i+1}: 异常区域{anomaly['id']}, "
                  f"位置=({palpation_pos[0]:.3f}, {palpation_pos[1]:.3f}, {palpation_pos[2]:.3f}), "
                  f"置信度={confidence:.2f}")
    
    print(f"触诊建议生成完成，共 {len(suggestions)} 个建议")
    return suggestions

def visualize_surface_analysis(surface_grid, grid_coords, anomalies, suggestions, 
                             scan_grid=None, show_plots=True):
    """
    可视化表面分析结果
    
    Parameters:
    -----------
    surface_grid : np.ndarray
        表面高度网格
    grid_coords : tuple
        网格坐标信息
    anomalies : list
        检测到的异常区域
    suggestions : list
        触诊建议
    scan_grid : dict, optional
        扫描网格信息
    show_plots : bool
        是否显示图像
    """
    if surface_grid is None:
        print("表面网格为空，无法可视化")
        return
    
    print(f"\n可视化表面分析结果...")
    
    # 设置matplotlib字体，解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    x_coords, y_coords = grid_coords
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # 1. 表面高度图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    contour = plt.contourf(X, Y, surface_grid, levels=20, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Height (m)', fontsize=10)
    plt.title('Surface Height Map', fontsize=12, fontweight='bold')
    plt.xlabel('X (m)', fontsize=10)
    plt.ylabel('Y (m)', fontsize=10)
    plt.tick_params(labelsize=8)
    
    # 2. 异常区域热力图
    plt.subplot(2, 3, 2)
    if len(anomalies) > 0:
        # 创建异常强度图
        anomaly_map = np.zeros_like(surface_grid)
        for anomaly in anomalies:
            anomaly_map[anomaly['region_mask']] = anomaly['strength']
        
        contour2 = plt.contourf(X, Y, anomaly_map, levels=20, cmap='hot')
        cbar2 = plt.colorbar(contour2)
        cbar2.set_label('Anomaly Strength', fontsize=10)
        plt.title('Anomaly Heat Map', fontsize=12, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Anomalies Detected', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Anomaly Heat Map', fontsize=12, fontweight='bold')
    plt.xlabel('X (m)', fontsize=10)
    plt.ylabel('Y (m)', fontsize=10)
    plt.tick_params(labelsize=8)
    
    # 3. 触诊建议位置
    plt.subplot(2, 3, 3)
    plt.contourf(X, Y, surface_grid, levels=20, cmap='viridis', alpha=0.7)
    
    # 绘制触诊建议点
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        if suggestion['type'] == 'anomaly':
            plt.scatter(pos[0], pos[1], c='red', s=100, marker='X', 
                       label=f'Anomaly {i+1}' if i < 3 else '')
        else:
            plt.scatter(pos[0], pos[1], c='blue', s=100, marker='o', 
                       label='Center Position')
    
    plt.legend(fontsize=8)
    plt.title('Palpation Suggestions', fontsize=12, fontweight='bold')
    plt.xlabel('X (m)', fontsize=10)
    plt.ylabel('Y (m)', fontsize=10)
    plt.tick_params(labelsize=8)
    
    # 4. 3D表面图
    ax = plt.subplot(2, 3, 4, projection='3d')
    ax.plot_surface(X, Y, surface_grid, cmap='viridis', alpha=0.8)
    
    # 标记触诊建议点
    for suggestion in suggestions:
        pos = suggestion['position']
        if suggestion['type'] == 'anomaly':
            ax.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='X')
        else:
            ax.scatter(pos[0], pos[1], pos[2], c='blue', s=100, marker='o')
    
    ax.set_title('3D Surface Map', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Height (m)', fontsize=10)
    ax.tick_params(labelsize=8)
    
    # 5. 异常统计
    plt.subplot(2, 3, 5)
    if len(anomalies) > 0:
        anomaly_ids = [a['id'] for a in anomalies]
        strengths = [a['strength'] for a in anomalies]
        heights = [a['height_elevation'] * 1000 for a in anomalies]  # 转换为mm
        
        x_pos = np.arange(len(anomaly_ids))
        plt.bar(x_pos, strengths, alpha=0.7, label='Anomaly Strength')
        plt.bar(x_pos, heights, alpha=0.7, label='Height Diff (mm)')
        plt.xlabel('Anomaly ID', fontsize=10)
        plt.ylabel('Strength/Height Diff', fontsize=10)
        plt.title('Anomaly Statistics', fontsize=12, fontweight='bold')
        plt.legend(fontsize=8)
        plt.xticks(x_pos, anomaly_ids, fontsize=8)
        plt.tick_params(labelsize=8)
    else:
        plt.text(0.5, 0.5, 'No Anomalies', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Anomaly Statistics', fontsize=12, fontweight='bold')
    plt.tick_params(labelsize=8)
    
    # 6. 触诊建议详情
    plt.subplot(2, 3, 6)
    if len(suggestions) > 0:
        suggestion_types = [s['type'] for s in suggestions]
        confidences = [s['confidence'] for s in suggestions]
        
        colors = ['red' if t == 'anomaly' else 'blue' for t in suggestion_types]
        plt.bar(range(len(suggestions)), confidences, color=colors, alpha=0.7)
        plt.xlabel('Suggestion Index', fontsize=10)
        plt.ylabel('Confidence', fontsize=10)
        plt.title('Palpation Confidence', fontsize=12, fontweight='bold')
        plt.ylim(0, 1)
        
        # 添加标签
        for i, (s_type, conf) in enumerate(zip(suggestion_types, confidences)):
            plt.text(i, conf + 0.02, f'{s_type}\n{conf:.2f}', ha='center', va='bottom', fontsize=8)
        plt.tick_params(labelsize=8)
    else:
        plt.text(0.5, 0.5, 'No Suggestions', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Palpation Confidence', fontsize=12, fontweight='bold')
    plt.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"surface_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"表面分析结果已保存为: {filename}")
    
    # 7. 单独的3D Surface Map图
    print("生成单独的3D Surface Map...")
    fig_3d = plt.figure(figsize=(12, 9))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # 绘制3D表面
    surface_plot = ax_3d.plot_surface(X, Y, surface_grid, cmap='viridis', 
                                     alpha=0.9, linewidth=0, antialiased=True)
    
    # # 标记触诊建议点
    # for i, suggestion in enumerate(suggestions):
    #     pos = suggestion['position']
    #     if suggestion['type'] == 'anomaly':
    #         ax_3d.scatter(pos[0], pos[1], pos[2], c='red', s=150, marker='X', 
    #                      label=f'Anomaly {i+1}' if i < 5 else '')
    #     else:
    #         ax_3d.scatter(pos[0], pos[1], pos[2], c='blue', s=150, marker='o', 
    #                      label='Center Position')
    
    # 设置3D图的属性
    ax_3d.set_title('3D Surface Map - Detailed View', fontsize=16, fontweight='bold', pad=20)
    ax_3d.set_xlabel('X (m)', fontsize=12, labelpad=10)
    ax_3d.set_ylabel('Y (m)', fontsize=12, labelpad=10)
    ax_3d.set_zlabel('Height (m)', fontsize=12, labelpad=10)
    
    # 添加颜色条
    cbar_3d = fig_3d.colorbar(surface_plot, ax=ax_3d, shrink=0.8, aspect=20, pad=0.1)
    cbar_3d.set_label('Height (m)', fontsize=12, labelpad=15)
    
    # 设置视角
    ax_3d.view_init(elev=25, azim=45)
    
    # 添加图例
    if len(suggestions) > 0:
        ax_3d.legend(fontsize=10, loc='upper left')
    
    # 设置刻度
    ax_3d.tick_params(labelsize=10)
    
    # 保存单独的3D图
    filename_3d = f"3d_surface_map_{timestamp}.png"
    fig_3d.savefig(filename_3d, dpi=300, bbox_inches='tight')
    print(f"单独的3D Surface Map已保存为: {filename_3d}")
    
    if show_plots:
        plt.show()
        fig_3d.show()

def perform_surface_analysis(scene, ed6, cam, motors_dof_idx, j6_link, boundary_points, cube_pos):
    """
    执行完整的表面分析流程
    
    Parameters:
    -----------
    boundary_points : np.ndarray
        边界点云
    cube_pos : np.ndarray
        物体中心位置
    
    Returns:
    --------
    suggestions : list
        触诊建议列表
    """
    global esc_pressed
    
    print(f"\n{'='*60}")
    print("开始表面分析和触诊建议生成")
    print(f"{'='*60}")
    
    # 1. 表面点云采集
    surface_points, scan_grid = scan_surface_points(
        scene, ed6, cam, motors_dof_idx, j6_link, 
        boundary_points, cube_pos, scan_density=8
    )
    
    if esc_pressed or len(surface_points) == 0:
        print("表面扫描中断或失败")
        return []
    
    # 2. 表面重建
    surface_grid, grid_coords = reconstruct_surface(surface_points, scan_grid)
    
    if surface_grid is None:
        print("表面重建失败")
        return []
    
    # 3. 异常检测
    anomalies, anomaly_map = detect_surface_anomalies(surface_grid, grid_coords)
    
    # 4. 生成触诊建议
    suggestions = generate_palpation_suggestions(anomalies, cube_pos)
    
    # 5. 可视化结果
    if MATPLOTLIB_AVAILABLE:
        visualize_surface_analysis(surface_grid, grid_coords, anomalies, suggestions, scan_grid)
    else:
        print("matplotlib未安装，跳过可视化")
    
    # 6. 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存表面点云
    if len(surface_points) > 0:
        surface_filename = f"surface_points_{timestamp}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(surface_points)
        o3d.io.write_point_cloud(surface_filename, pcd)
        print(f"表面点云已保存为: {surface_filename}")
    
    # 保存触诊建议
    suggestions_filename = f"palpation_suggestions_{timestamp}.json"
    import json
    suggestions_data = []
    for suggestion in suggestions:
        suggestion_data = {
            'position': suggestion['position'].tolist(),
            'type': suggestion['type'],
            'priority': suggestion['priority'],
            'confidence': suggestion['confidence'],
            'reason': suggestion['reason']
        }
        if suggestion['type'] == 'anomaly':
            suggestion_data.update({
                'anomaly_id': suggestion['anomaly_id'],
                'strength': suggestion['strength'],
                'height_elevation': suggestion['height_elevation'],
                'area_mm2': suggestion['area_mm2']
            })
        suggestions_data.append(suggestion_data)
    
    with open(suggestions_filename, 'w') as f:
        json.dump(suggestions_data, f, indent=2)
    print(f"触诊建议已保存为: {suggestions_filename}")
    
    print(f"\n表面分析完成！")
    print(f"检测到 {len(anomalies)} 个异常区域")
    print(f"生成 {len(suggestions)} 个触诊建议")
    
    return suggestions

def perform_surface_analysis_with_points(surface_points, boundary_points, cube_pos):
    """
    使用已采集的表面点云进行表面分析
    
    Parameters:
    -----------
    surface_points : np.ndarray
        表面点云
    boundary_points : np.ndarray
        边界点云
    cube_pos : np.ndarray
        物体中心位置
    
    Returns:
    --------
    suggestions : list
        触诊建议列表
    """
    print(f"\n{'='*60}")
    print("开始表面分析和触诊建议生成")
    print(f"{'='*60}")
    
    if len(surface_points) == 0:
        print("表面点云为空，无法进行表面分析")
        return []
    
    # 1. 创建扫描网格信息（基于边界点云）
    if len(boundary_points) > 0:
        points_2d = boundary_points[:, :2]
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
    else:
        # 如果没有边界点云，使用表面点云
        points_2d = surface_points[:, :2]
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
    
    scan_grid = {
        'min_coords': min_coords,
        'max_coords': max_coords,
        'scan_height': np.mean(surface_points[:, 2])
    }
    
    # 2. 表面重建
    surface_grid, grid_coords = reconstruct_surface(surface_points, scan_grid)
    
    if surface_grid is None:
        print("表面重建失败")
        return []
    
    # 3. 异常检测
    anomalies, anomaly_map = detect_surface_anomalies(surface_grid, grid_coords)
    
    # 4. 生成触诊建议
    suggestions = generate_palpation_suggestions(anomalies, cube_pos)
    
    # 5. 可视化结果
    if MATPLOTLIB_AVAILABLE:
        visualize_surface_analysis(surface_grid, grid_coords, anomalies, suggestions, scan_grid)
    else:
        print("matplotlib未安装，跳过可视化")
    
    # 6. 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存触诊建议
    suggestions_filename = f"palpation_suggestions_{timestamp}.json"
    suggestions_data = []
    for suggestion in suggestions:
        suggestion_data = {
            'position': suggestion['position'].tolist(),
            'type': suggestion['type'],
            'priority': suggestion['priority'],
            'confidence': suggestion['confidence'],
            'reason': suggestion['reason']
        }
        if suggestion['type'] == 'anomaly':
            suggestion_data.update({
                'anomaly_id': suggestion['anomaly_id'],
                'strength': suggestion['strength'],
                'height_elevation': suggestion['height_elevation'],
                'area_mm2': suggestion['area_mm2']
            })
        suggestions_data.append(suggestion_data)
    
    with open(suggestions_filename, 'w') as f:
        json.dump(suggestions_data, f, indent=2)
    print(f"触诊建议已保存为: {suggestions_filename}")
    
    print(f"\n表面分析完成！")
    print(f"检测到 {len(anomalies)} 个异常区域")
    print(f"生成 {len(suggestions)} 个触诊建议")
    
    return suggestions

def main():
    gs.init(seed=0, precision="32", logging_level="debug")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=True,
            plane_reflection=True,
            ambient_light=(0.1, 0.1, 0.1),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=False,
            enable_collision=True,
            gravity=(0, 0, -9.81),
        ),
        show_viewer=True,
    )
    plane = scene.add_entity(gs.morphs.Plane(collision=True))

    # cube = scene.add_entity(gs.morphs.Box(size=(0.105, 0.18, 0.022), pos=(0.35, 0, 0.02), collision=True))
    cube = scene.add_entity(gs.morphs.Mesh(
        file="genesis/assets/_myobj/ctry.obj",
        pos=(0.35, 0.0, 0.02),
        collision=True,
        
        fixed=True,
    ))

    ed6 = scene.add_entity(gs.morphs.URDF(
        file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
        scale=1.0,
        requires_jac_and_IK=True,
        fixed=True,
    ))
    cam = scene.add_camera(
        model="thinlens",
        res=(640, 480),
        pos=(0, 0, 0),
        lookat=(0, 0, 1),
        up=(0, 0, 1),
        fov=65,  # 设置为60度
        aperture=2.8,
        focus_dist=0.02,  
        GUI=True,
    )
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    # 增加摄像机偏移距离，减少锥形摄像遮挡
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    
    # 初始化机械臂
    reset_arm(scene, ed6, motors_dof_idx)
    
    # 直接给定目标点（替代环视检测）
    cube_pos = np.array([0.35, -0.05, 0.02])  # 直接设置目标位置
    print(f"直接设置目标位置: {cube_pos}")
    
    # 机械臂移动到目标位置
    plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
    
    # === 开始边界追踪与表面扫描 ===
    print("\n开始边界追踪与表面扫描 (Alpha-Shape方法)...")
    time.sleep(5)
    boundary_points, surface_points = track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, method="alpha_shape")
    
    # 保存边界点云
    visualize_and_save_pointcloud(boundary_points, "boundary_cloud_alpha_shape.ply")
    
    # 保存表面点云
    if len(surface_points) > 0:
        visualize_and_save_pointcloud(surface_points, "surface_cloud_alpha_shape.ply")
    
    # === 表面分析和触诊建议生成 ===
    if len(surface_points) > 0:
        print("\n边界扫描和表面扫描完成，开始表面分析...")
        time.sleep(2)
        
        # 执行表面分析（使用表面点云）
        palpation_suggestions = perform_surface_analysis_with_points(
            surface_points, boundary_points, cube_pos
        )
        
        # 显示触诊建议
        if len(palpation_suggestions) > 0:
            print(f"\n{'='*60}")
            print("触诊建议总结")
            print(f"{'='*60}")
            for i, suggestion in enumerate(palpation_suggestions):
                pos = suggestion['position']
                print(f"建议{i+1} ({suggestion['type']}): "
                      f"位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                      f"置信度={suggestion['confidence']:.2f}")
                print(f"  原因: {suggestion['reason']}")
                if suggestion['type'] == 'anomaly':
                    print(f"  异常强度: {suggestion['strength']:.3f}, "
                          f"高度差: {suggestion['height_elevation']*1000:.1f}mm")
        else:
            print("未生成任何触诊建议")
    else:
        print("表面点云为空，跳过表面分析")

if __name__ == "__main__":
    main()
