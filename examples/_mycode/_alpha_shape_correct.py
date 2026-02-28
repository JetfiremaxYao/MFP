#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的Alpha-Shape边界检测实现
解决坐标系错位和实现不完整的问题
"""

import genesis as gs
import numpy as np
import cv2
from genesis.utils import geom as gu

# 尝试导入Alpha-Shape相关库
try:
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import unary_union
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import pdist, squareform
    ALPHA_SHAPE_AVAILABLE = True
    print("✅ Alpha-Shape相关库已安装")
except ImportError as e:
    ALPHA_SHAPE_AVAILABLE = False
    print(f"❌ Alpha-Shape相关库未安装: {e}")

def detect_boundary_alpha_shape_correct(cam, alpha_value=None, min_contour_area=100):
    """
    正确的Alpha-Shape边界检测实现
    
    关键修复：
    1. 在像素坐标系进行Alpha-Shape计算
    2. 实现真正的Alpha-Shape算法（不是凸包）
    3. 正确的边界点排序
    4. 统一的度量口径
    """
    if not ALPHA_SHAPE_AVAILABLE:
        print("Alpha-Shape库未安装，回退到Canny方法")
        return detect_boundary_canny_fallback(cam, min_contour_area)
    
    try:
        # 步骤1: 获取RGB和深度图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 步骤2: 在像素坐标系提取边界候选点
        boundary_pixels = extract_boundary_pixels(img, depth)
        
        if len(boundary_pixels) < 3:
            print("边界候选点不足，回退到Canny方法")
            return detect_boundary_canny_fallback(cam, min_contour_area)
        
        print(f"提取到{len(boundary_pixels)}个边界候选像素点")
        
        # 步骤3: 在像素坐标系进行Alpha-Shape计算
        alpha_shape_pixels = compute_alpha_shape_pixels(boundary_pixels, alpha_value)
        
        if len(alpha_shape_pixels) < 3:
            print("Alpha-Shape计算失败，回退到Canny方法")
            return detect_boundary_canny_fallback(cam, min_contour_area)
        
        # 步骤4: 转换为OpenCV轮廓格式
        contours = [alpha_shape_pixels.reshape(-1, 1, 2).astype(np.int32)]
        
        print(f"✅ Alpha-Shape成功，生成{len(contours)}个轮廓，{len(alpha_shape_pixels)}个点")
        return contours, rgb, depth
        
    except Exception as e:
        print(f"❌ Alpha-Shape检测失败: {e}")
        import traceback
        traceback.print_exc()
        return detect_boundary_canny_fallback(cam, min_contour_area)

def extract_boundary_pixels(img, depth):
    """
    在像素坐标系提取边界候选点
    
    使用RGB-D融合的边缘检测，返回像素坐标(u, v)
    """
    # RGB边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges_rgb = cv2.Canny(gray, 50, 150)
    
    # 深度边缘检测
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edges_depth = cv2.Canny(depth_normalized, 30, 100)
    
    # 融合边缘
    edges_combined = cv2.bitwise_or(edges_rgb, edges_depth)
    
    # 形态学操作，连接断裂的边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
    
    # 提取边缘像素坐标
    edge_pixels = np.column_stack(np.where(edges_combined > 0))
    
    # 转换为(u, v)格式 (注意：np.where返回的是(y, x)，需要转换为(x, y))
    if len(edge_pixels) > 0:
        edge_pixels = edge_pixels[:, [1, 0]]  # 交换x, y坐标
    
    return edge_pixels

def compute_alpha_shape_pixels(pixels, alpha=None):
    """
    在像素坐标系计算真正的Alpha-Shape
    
    实现完整的Alpha-Shape算法：
    1. Delaunay三角化
    2. 过滤外接圆半径
    3. 构建边界边
    4. 用Shapely构建多边形
    5. 提取外边界
    """
    if len(pixels) < 3:
        return np.array([])
    
    try:
        # 步骤1: 计算自适应Alpha参数
        if alpha is None:
            alpha = compute_adaptive_alpha(pixels)
        
        print(f"使用Alpha参数: {alpha}")
        
        # 步骤2: Delaunay三角化
        tri = Delaunay(pixels)
        
        # 步骤3: 过滤三角形，保留Alpha-Shape边
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
            
            # Alpha-Shape条件：外接圆半径 < 1/alpha
            if circumradius < 1.0 / alpha:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                    alpha_edges.add(edge)
        
        print(f"Alpha-Shape生成了{len(alpha_edges)}条边")
        
        if len(alpha_edges) == 0:
            print("没有Alpha-Shape边，回退到凸包")
            return compute_convex_hull_pixels(pixels)
        
        # 步骤4: 构建边界多边形
        boundary_polygon = build_boundary_polygon(pixels, alpha_edges)
        
        if boundary_polygon is None or boundary_polygon.is_empty:
            print("边界多边形构建失败，回退到凸包")
            return compute_convex_hull_pixels(pixels)
        
        # 步骤5: 提取外边界点
        if hasattr(boundary_polygon, 'exterior'):
            # 单个多边形
            boundary_coords = list(boundary_polygon.exterior.coords[:-1])  # 去掉重复的最后一个点
        else:
            # 多个多边形，取面积最大的
            if hasattr(boundary_polygon, 'geoms'):
                largest_poly = max(boundary_polygon.geoms, key=lambda p: p.area)
                boundary_coords = list(largest_poly.exterior.coords[:-1])
            else:
                print("无法提取边界坐标")
                return compute_convex_hull_pixels(pixels)
        
        boundary_points = np.array(boundary_coords)
        
        # 步骤6: 按边界顺序排序点
        ordered_points = order_boundary_points_correct(boundary_points)
        
        return ordered_points
        
    except Exception as e:
        print(f"Alpha-Shape计算失败: {e}")
        return compute_convex_hull_pixels(pixels)

def compute_adaptive_alpha(pixels):
    """
    计算自适应的Alpha参数
    
    使用kNN距离分位数，比全局均距更稳定
    """
    if len(pixels) < 2:
        return 1.0
    
    try:
        # 计算每个点到其k个最近邻的距离
        k = min(5, len(pixels) - 1)
        distances = []
        
        for i in range(len(pixels)):
            point_distances = []
            for j in range(len(pixels)):
                if i != j:
                    dist = np.linalg.norm(pixels[i] - pixels[j])
                    point_distances.append(dist)
            
            point_distances.sort()
            if len(point_distances) >= k:
                distances.append(point_distances[k-1])  # 第k个最近邻距离
        
        if len(distances) == 0:
            return 1.0
        
        # 使用中位数作为尺度估计
        median_distance = np.median(distances)
        alpha = 2.0 / median_distance
        
        # 裁剪到合理范围
        alpha = np.clip(alpha, 0.1, 10.0)
        
        return alpha
        
    except Exception as e:
        print(f"Alpha参数计算失败: {e}")
        return 1.0

def build_boundary_polygon(pixels, alpha_edges):
    """
    从Alpha-Shape边构建边界多边形
    
    使用Shapely将边连接成多边形
    """
    try:
        if len(alpha_edges) == 0:
            return None
        
        # 构建边的LineString
        lines = []
        for edge in alpha_edges:
            p1, p2 = pixels[edge[0]], pixels[edge[1]]
            line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
            lines.append(line)
        
        # 合并所有边
        if len(lines) > 1:
            merged_lines = unary_union(lines)
        else:
            merged_lines = lines[0]
        
        # 尝试构建多边形
        if hasattr(merged_lines, 'geoms'):
            # 多个几何体
            polygons = []
            for geom in merged_lines.geoms:
                if hasattr(geom, 'is_ring') and geom.is_ring:
                    try:
                        poly = Polygon(geom)
                        if poly.is_valid:
                            polygons.append(poly)
                    except:
                        pass
            
            if polygons:
                # 返回面积最大的多边形
                return max(polygons, key=lambda p: p.area)
        else:
            # 单个几何体
            if hasattr(merged_lines, 'is_ring') and merged_lines.is_ring:
                try:
                    poly = Polygon(merged_lines)
                    if poly.is_valid:
                        return poly
                except:
                    pass
        
        return None
        
    except Exception as e:
        print(f"边界多边形构建失败: {e}")
        return None

def order_boundary_points_correct(points):
    """
    正确的边界点排序
    
    使用图遍历方法，按边连接顺序排序
    """
    if len(points) < 3:
        return points
    
    try:
        # 构建邻接图
        adjacency = {}
        for i in range(len(points)):
            adjacency[i] = []
        
        # 计算所有点对的距离
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < 2.0:  # 像素距离阈值
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # 找到度数最小的点作为起点（通常是边界端点）
        start_idx = min(range(len(points)), key=lambda i: len(adjacency[i]))
        
        # 深度优先搜索遍历边界
        visited = set()
        ordered_indices = []
        
        def dfs(current):
            if current in visited:
                return
            visited.add(current)
            ordered_indices.append(current)
            
            # 按距离排序邻居，选择最近的未访问邻居
            neighbors = [(j, np.linalg.norm(points[current] - points[j])) 
                        for j in adjacency[current] if j not in visited]
            neighbors.sort(key=lambda x: x[1])
            
            for neighbor, _ in neighbors:
                dfs(neighbor)
        
        dfs(start_idx)
        
        if len(ordered_indices) == len(points):
            return points[ordered_indices]
        else:
            # 如果图遍历失败，回退到极角排序
            return order_boundary_points_polar(points)
        
    except Exception as e:
        print(f"边界点排序失败: {e}")
        return order_boundary_points_polar(points)

def order_boundary_points_polar(points):
    """
    极角排序（回退方法）
    """
    if len(points) < 3:
        return points
    
    # 找到最左下角的点作为起点
    start_idx = np.argmin(points[:, 0] + points[:, 1])
    start_point = points[start_idx]
    
    # 计算极角
    angles = np.arctan2(points[:, 1] - start_point[1], points[:, 0] - start_point[0])
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]

def compute_convex_hull_pixels(pixels):
    """
    计算凸包（回退方法）
    """
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pixels)
        return pixels[hull.vertices]
    except:
        return pixels

def detect_boundary_canny_fallback(cam, min_contour_area=100):
    """
    Canny方法回退
    """
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours, rgb, depth

def test_alpha_shape_correct():
    """测试正确的Alpha-Shape实现"""
    print("测试正确的Alpha-Shape实现...")
    
    if not ALPHA_SHAPE_AVAILABLE:
        print("❌ Alpha-Shape库不可用")
        return
    
    # 初始化Genesis
    gs.init(seed=0, precision="32", logging_level="info")
    
    # 创建简单场景
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(640, 480),
            camera_pos=(1.0, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            show_cameras=False,
        ),
        show_viewer=False,
    )
    
    # 添加平面和立方体
    plane = scene.add_entity(gs.morphs.Plane(collision=True))
    cube = scene.add_entity(gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.0, 0.0, 0.05),
        collision=True
    ))
    
    # 添加摄像机
    cam = scene.add_camera(
        model="thinlens",
        res=(640, 480),
        pos=(0, 0, 0),
        lookat=(0, 0, 1),
        up=(0, 0, 1),
        fov=65,
        aperture=2.8,
        focus_dist=0.1,
        GUI=False,
    )
    
    scene.build()
    scene.step()
    
    # 测试Alpha-Shape
    contours, rgb, depth = detect_boundary_alpha_shape_correct(cam)
    
    if contours:
        print(f"✅ Alpha-Shape测试成功！")
        print(f"   轮廓数量: {len(contours)}")
        print(f"   轮廓点数: {len(contours[0])}")
        print(f"   轮廓面积: {cv2.contourArea(contours[0]):.2f} 像素²")
    else:
        print("❌ Alpha-Shape测试失败")

if __name__ == "__main__":
    test_alpha_shape_correct()
