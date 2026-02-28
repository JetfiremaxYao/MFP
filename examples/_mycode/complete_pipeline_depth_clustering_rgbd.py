#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的物体检测和边界扫描Pipeline
合并了深度聚类检测和RGB-D边界扫描功能
"""

import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
import open3d as o3d
from sklearn.decomposition import PCA
import threading
import sys
import select
from math import ceil
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy import ndimage
from scipy.interpolate import griddata

# 全局变量用于ESC键检测
esc_pressed = False
scanning_active = True

def check_esc_key():
    """监听ESC键的线程函数"""
    global esc_pressed
    while scanning_active:
        if sys.platform == "darwin":  # macOS
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()
                if key == 'esc' or key == 'q':
                    esc_pressed = True
                    print("\n检测到ESC键，正在结束扫描...")
                    break
        else:  # Linux/Windows
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
    target_pos[2] += 0.2  # 确保机械臂高度跟物体有一定距离
    
    qpos_ik = ed6.inverse_kinematics(
        link=j6_link,
        pos=target_pos,
        quat=target_quat,
    )
    
    # 处理tensor到numpy的转换
    if hasattr(qpos_ik, 'cpu'):
        qpos_ik = qpos_ik.cpu().numpy()
    
    print(f"IK解算目标位置: {target_pos}")
    print(f"IK解算关节角度: {qpos_ik}")
    
    # 检查IK解的有效性
    if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
        raise RuntimeError("IK解算失败，目标位置不可达")
    
    # 检查关节限制
    joint_limits = np.array([
        [-np.pi, np.pi],      # J1
        [-np.pi/2, np.pi/2],  # J2
        [-np.pi, np.pi],      # J3
        [-np.pi, np.pi],      # J4
        [-np.pi/2, np.pi/2],  # J5
        [-np.pi, np.pi]       # J6
    ])
    
    for i, angle in enumerate(qpos_ik):
        if angle < joint_limits[i][0] or angle > joint_limits[i][1]:
            print(f"警告: 关节{i+1}角度{angle:.3f}超出限制[{joint_limits[i][0]:.3f}, {joint_limits[i][1]:.3f}]")
    
    # 获取当前关节位置
    qpos_now = ed6.get_dofs_position(motors_dof_idx)
    if hasattr(qpos_now, 'cpu'):
        qpos_now = qpos_now.cpu().numpy()
    
    # 生成平滑路径
    num_steps = 100
    path = np.linspace(qpos_now, qpos_ik, num_steps)
    
    print("执行IK路径插值...")
    for i, q in enumerate(path):
        ed6.control_dofs_position(q, motors_dof_idx)
        scene.step()
        
        # 每10步显示进度
        if i % 10 == 0:
            progress = (i + 1) / num_steps * 100
            print(f"路径执行进度: {progress:.1f}%")
    
    print("机械臂已移动到目标位置")
    cam.move_to_attach()
    
    return qpos_ik

# ==================== 深度聚类检测功能 ====================

def estimate_cube_range_from_clusters(all_frame_data):
    """根据多帧聚类结果，估算cube的x/y空间范围（世界坐标系）"""
    all_xy = []
    for frame in all_frame_data:
        cluster_points = frame['cluster_points']
        if len(cluster_points) > 0:
            all_xy.extend(cluster_points[:, :2])  # 只取XY坐标
    
    if not all_xy:
        raise RuntimeError("未采集到聚类点，无法估算范围！")
    
    all_xy = np.array(all_xy)
    min_x, min_y = np.min(all_xy, axis=0)
    max_x, max_y = np.max(all_xy, axis=0)
    return min_x, max_x, min_y, max_y

def depth_based_clustering_detection(depth_image, cam, lighting_condition="normal",
                                   min_depth: float = 0.15, 
                                   max_depth: float = 0.6,
                                   eps: float = 0.03, 
                                   min_samples: int = 5):
    """
    基于深度阈值和DBSCAN聚类的物体检测（针对单一规则物体优化）
    
    Parameters:
    -----------
    depth_image : np.ndarray
        深度图像
    cam : Camera
        相机对象
    lighting_condition : str
        光照条件，用于调整深度阈值
    min_depth : float
        最小深度阈值（米）
    max_depth : float
        最大深度阈值（米）
    eps : float
        DBSCAN的邻域半径
    min_samples : int
        DBSCAN的最小样本数
    
    Returns:
    --------
    cluster_center : np.ndarray or None
        最大聚类的中心点（世界坐标）
    cluster_points : np.ndarray
        最大聚类的所有点（世界坐标）
    """
    # 根据光照条件调整深度阈值参数
    if lighting_condition == "bright":
        # 明亮光照：深度检测更稳定，可以使用更严格的阈值
        min_depth = 0.10
        max_depth = 0.7
        eps = 0.025  # 更小的邻域半径
        min_samples = 4
    elif lighting_condition == "dim":
        # 昏暗光照：深度检测可能不稳定，使用更宽松的阈值
        min_depth = 0.05
        max_depth = 0.9
        eps = 0.04   # 更大的邻域半径
        min_samples = 2
    else:  # normal
        # 正常光照：使用原始参数
        min_depth = 0.05
        max_depth = 0.8
        eps = 0.03
        min_samples = 3
    
    print(f"光照条件: {lighting_condition}, 深度阈值: [{min_depth:.3f}, {max_depth:.3f}]m, eps={eps:.3f}m, min_samples={min_samples}")
    
    # 调试信息：深度值统计
    valid_depths = depth_image[depth_image > 0]
    if len(valid_depths) > 0:
        print(f"深度值统计: 最小={np.min(valid_depths):.3f}m, 最大={np.max(valid_depths):.3f}m, 平均={np.mean(valid_depths):.3f}m")
    else:
        print("警告：深度图像中没有有效深度值！")
        return None, np.array([])
    
    # 1. 深度阈值过滤 - 更精确的范围
    depth_mask = (depth_image >= min_depth) & (depth_image <= max_depth)
    valid_pixels_count = np.sum(depth_mask)
    print(f"深度阈值过滤: 范围[{min_depth:.3f}, {max_depth:.3f}]m, 有效像素数: {valid_pixels_count}")
    
    if not np.any(depth_mask):
        print(f"深度阈值范围内无有效点 (深度范围: {min_depth}-{max_depth}m)")
        # 尝试更宽松的阈值
        print("尝试更宽松的深度阈值...")
        min_depth_relaxed = max(0.05, min_depth - 0.1)
        max_depth_relaxed = max_depth + 0.2
        depth_mask = (depth_image >= min_depth_relaxed) & (depth_image <= max_depth_relaxed)
        valid_pixels_count = np.sum(depth_mask)
        print(f"宽松阈值过滤: 范围[{min_depth_relaxed:.3f}, {max_depth_relaxed:.3f}]m, 有效像素数: {valid_pixels_count}")
        
        if not np.any(depth_mask):
            return None, np.array([])
    
    # 2. 获取有效深度点的像素坐标
    valid_pixels = np.where(depth_mask)
    valid_depths = depth_image[valid_pixels]
    
    print(f"有效深度点数: {len(valid_pixels[0])}, 最小样本数要求: {min_samples}")
    
    if len(valid_pixels[0]) < min_samples:
        print(f"有效点数不足 ({len(valid_pixels[0])} < {min_samples})")
        # 降低最小样本数要求
        min_samples_adjusted = max(2, len(valid_pixels[0]) // 2)
        print(f"调整最小样本数为: {min_samples_adjusted}")
        min_samples = min_samples_adjusted
    
    # 3. 将像素坐标转换为相机坐标系
    K = cam.intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    print(f"相机内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 像素坐标转相机坐标
    x_cam = (valid_pixels[1] - cx) * valid_depths / fx
    y_cam = (valid_pixels[0] - cy) * valid_depths / fy
    z_cam = valid_depths
    
    # 组合成相机坐标系下的3D点
    points_cam = np.column_stack([x_cam, y_cam, z_cam])
    
    print(f"相机坐标系点范围: X[{np.min(x_cam):.3f}, {np.max(x_cam):.3f}], Y[{np.min(y_cam):.3f}, {np.max(y_cam):.3f}], Z[{np.min(z_cam):.3f}, {np.max(z_cam):.3f}]")
    
    # 4. 转换到世界坐标系
    T_cam2world = np.linalg.inv(cam.extrinsics)
    points_world = []
    
    for point_cam in points_cam:
        point_homogeneous = np.append(point_cam, 1)
        point_world = T_cam2world @ point_homogeneous
        points_world.append(point_world[:3])
    
    points_world = np.array(points_world)
    
    print(f"世界坐标系点范围: X[{np.min(points_world[:,0]):.3f}, {np.max(points_world[:,0]):.3f}], Y[{np.min(points_world[:,1]):.3f}, {np.max(points_world[:,1]):.3f}], Z[{np.min(points_world[:,2]):.3f}, {np.max(points_world[:,2]):.3f}]")
    
    # 5. === 关键修正1: 地面剔除 ===
    # 过滤掉地面点（z < 0.002米，即2mm）- 更宽松的地面剔除
    ground_filter_mask = points_world[:, 2] > 0.002
    points_world_no_ground = points_world[ground_filter_mask]
    
    print(f"地面剔除: 原始点数={len(points_world)}, 剔除后点数={len(points_world_no_ground)}")
    if len(points_world_no_ground) > 0:
        print(f"剩余点Z范围: [{np.min(points_world_no_ground[:,2]):.3f}, {np.max(points_world_no_ground[:,2]):.3f}]")
    
    if len(points_world_no_ground) < min_samples:
        print(f"地面剔除后点数不足 ({len(points_world_no_ground)} < {min_samples})")
        return None, np.array([])
    
    # 6. 过滤异常点（距离相机太远或太近的点）
    distances = np.linalg.norm(points_world_no_ground, axis=1)
    valid_distance_mask = (distances >= 0.05) & (distances <= 2.0)  # 放宽距离限制
    points_world_filtered = points_world_no_ground[valid_distance_mask]
    
    print(f"距离过滤: 原始点数={len(points_world_no_ground)}, 过滤后点数={len(points_world_filtered)}")
    
    if len(points_world_filtered) < min_samples:
        print(f"距离过滤后点数不足 ({len(points_world_filtered)} < {min_samples})")
        # 使用地面剔除后的点集
        points_world_filtered = points_world_no_ground
        print("使用地面剔除后的点集进行聚类")
    
    # 7. === 关键修正2: 不使用StandardScaler，直接在米单位聚类 ===
    # 移除StandardScaler，使用原始坐标（米单位）
    print(f"开始DBSCAN聚类: eps={eps}m, min_samples={min_samples}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(points_world_filtered)
    
    # 8. 找到最大的聚类
    unique_labels = np.unique(cluster_labels)
    print(f"聚类结果: 标签={unique_labels}, 标签数量={len(unique_labels)}")
    
    if len(unique_labels) == 1 and unique_labels[0] == -1:
        print("DBSCAN未找到有效聚类，尝试调整参数...")
        # 尝试更宽松的聚类参数
        eps_relaxed = eps * 2  # 从2cm增加到4cm
        min_samples_relaxed = max(2, min_samples // 2)
        print(f"调整聚类参数: eps={eps_relaxed}m, min_samples={min_samples_relaxed}")
        
        dbscan = DBSCAN(eps=eps_relaxed, min_samples=min_samples_relaxed)
        cluster_labels = dbscan.fit_predict(points_world_filtered)
        unique_labels = np.unique(cluster_labels)
        print(f"调整后聚类结果: 标签={unique_labels}")
        
        if len(unique_labels) == 1 and unique_labels[0] == -1:
            print("调整参数后仍未找到有效聚类")
            return None, np.array([])
    
    # 排除噪声点（标签为-1）
    valid_clusters = unique_labels[unique_labels != -1]
    
    if len(valid_clusters) == 0:
        print("所有点都被标记为噪声")
        return None, np.array([])
    
    # 找到最大的聚类
    largest_cluster_label = None
    largest_cluster_size = 0
    
    for label in valid_clusters:
        cluster_size = np.sum(cluster_labels == label)
        print(f"聚类 {label}: {cluster_size} 个点")
        if cluster_size > largest_cluster_size:
            largest_cluster_size = cluster_size
            largest_cluster_label = label
    
    if largest_cluster_label is None:
        print("未找到有效聚类")
        return None, np.array([])
    
    # 9. 提取最大聚类的点
    cluster_mask = cluster_labels == largest_cluster_label
    cluster_points = points_world_filtered[cluster_mask]
    
    # 10. 计算聚类中心（使用几何中心）
    cluster_center = np.mean(cluster_points, axis=0)
    
    # 11. 验证聚类质量（检查聚类是否合理）
    cluster_std = np.std(cluster_points, axis=0)
    print(f"聚类标准差: {cluster_std}")
    
    # 放宽聚类质量检查
    if np.any(cluster_std > 0.5):  # 从0.2放宽到0.5
        print(f"聚类较为分散，标准差: {cluster_std}")
        # 不直接返回None，而是继续处理
    
    print(f"找到最大聚类，包含 {largest_cluster_size} 个点")
    print(f"聚类中心: {cluster_center}")
    
    return cluster_center, cluster_points

def detect_cube_position(scene, ed6, cam, motors_dof_idx, lighting_condition="normal"):
    """机械臂环视一圈，使用深度聚类检测cube，返回cube世界坐标和x/y范围"""
    start_time = time.time()
    
    qpos_scan = np.zeros(6)
    qpos_scan[3] = -np.pi / 3   # J4 -60
    qpos_scan[4] = np.pi / 2   # J5 +90°
    num_views = 24
    start_angle = 0
    angles = np.linspace(start_angle, 2 * np.pi + start_angle, num_views, endpoint=False)
    results = []
    all_frame_data = []
    view_details = []  # 视角级详细信息
    
    qpos_scan[0] = 0
    ed6.control_dofs_position(qpos_scan, motors_dof_idx)
    for step in range(100):
        scene.step()
    cam.move_to_attach()
    settle_steps = 100
    miss_count = 0  # 连续未检测到物体的次数
    detected_once = False  # 是否至少检测到过一次
    frame_count = 0
    views_hit = 0  # 命中视角数
    early_stop_index = num_views  # 提前终止的视角索引
    
    # 根据光照条件调整深度聚类参数
    if lighting_condition == "bright":
        # 明亮光照：深度检测更稳定，可以使用更严格的阈值
        min_depth = 0.10
        max_depth = 0.7
        eps = 0.025  # 更小的邻域半径
        min_samples = 4
    elif lighting_condition == "dim":
        # 昏暗光照：深度检测可能不稳定，使用更宽松的阈值
        min_depth = 0.05
        max_depth = 0.9
        eps = 0.04   # 更大的邻域半径
        min_samples = 2
    else:  # normal
        # 正常光照：使用原始参数
        min_depth = 0.05
        max_depth = 0.8
        eps = 0.03
        min_samples = 3
    
    print(f"光照条件: {lighting_condition}, 深度阈值: [{min_depth:.3f}, {max_depth:.3f}]m, eps={eps:.3f}m, min_samples={min_samples}")
    
    for i, angle in enumerate(angles):
        qpos = qpos_scan.copy()
        qpos[0] = angle
        ed6.control_dofs_position(qpos, motors_dof_idx)
        for step in range(settle_steps):
            scene.step()
        cam.move_to_attach()
        
        frame_start_time = time.time()
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        frame_count += 1
        
        # 使用深度聚类检测，传入光照条件
        cluster_center, cluster_points = depth_based_clustering_detection(
            depth, cam, lighting_condition, min_depth, max_depth, eps, min_samples
        )
        
        # 调试：显示深度图像
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # 在深度图像上标记深度阈值范围
        h, w = depth.shape
        cv2.putText(depth_colored, f"Depth: {np.min(depth):.3f}-{np.max(depth):.3f}m", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"Threshold: {min_depth:.3f}-{max_depth:.3f}m", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Depth Image Debug", depth_colored)
        cv2.waitKey(100)
        
        # 初始化视角信息
        view_hit = 0  # 当前视角是否命中
        view_score = 0.0  # 当前视角得分
        view_cx = 0
        view_cy = 0
        depth_valid_count = 0
        
        if cluster_center is not None:
            # 计算聚类到图像中心的距离（用于评估检测质量）
            h, w = rgb.shape[:2]
            center_img = np.array([w // 2, h // 2])
            
            # 将聚类中心投影回图像平面
            K = cam.intrinsics
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # 世界坐标转相机坐标
            T_world2cam = cam.extrinsics
            center_cam = T_world2cam @ np.append(cluster_center, 1)
            center_cam = center_cam[:3]
            
            # 相机坐标转像素坐标
            if center_cam[2] > 0:
                px = int(center_cam[0] * fx / center_cam[2] + cx)
                py = int(center_cam[1] * fy / center_cam[2] + cy)
                
                # 检查投影点是否在图像范围内
                if 0 <= px < w and 0 <= py < h:
                    dist_to_center = np.linalg.norm([px - center_img[0], py - center_img[1]])
                    
                    # 计算得分（聚类大小优先，距离中心越近越好）
                    score = len(cluster_points) / (1 + dist_to_center / 100)
                    
                    # 可视化检测结果
                    img_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    
                    # 绘制聚类中心（红色圆圈）
                    cv2.circle(img_vis, (px, py), 15, (0, 0, 255), 3)
                    cv2.circle(img_vis, (px, py), 5, (0, 0, 255), -1)
                    
                    # 绘制聚类边界框
                    if len(cluster_points) > 0:
                        # 将聚类点投影到图像平面
                        cluster_pixels = []
                        for point in cluster_points[:100]:  # 限制点数以提高性能
                            point_cam = T_world2cam @ np.append(point, 1)
                            point_cam = point_cam[:3]
                            if point_cam[2] > 0:
                                px_cluster = int(point_cam[0] * fx / point_cam[2] + cx)
                                py_cluster = int(point_cam[1] * fy / point_cam[2] + cy)
                                if 0 <= px_cluster < w and 0 <= py_cluster < h:
                                    cluster_pixels.append([px_cluster, py_cluster])
                        
                        if cluster_pixels:
                            cluster_pixels = np.array(cluster_pixels)
                            # 绘制边界框
                            x_min, y_min = np.min(cluster_pixels, axis=0)
                            x_max, y_max = np.max(cluster_pixels, axis=0)
                            cv2.rectangle(img_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # 添加文本信息
                    cv2.putText(img_vis, f"Center: ({px}, {py})", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(img_vis, f"Points: {len(cluster_points)}", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow("Depth Clustering Detection", img_vis)
                    cv2.waitKey(200)
                    
                    results.append({
                        'cluster_center': cluster_center,
                        'cluster_size': len(cluster_points),
                        'dist_to_center': dist_to_center,
                        'angle': angle,
                        'pixel_pos': (px, py),
                        'score': score
                    })
                    
                    # 记录本帧聚类结果用于范围估算
                    all_frame_data.append({
                        'cluster_points': cluster_points,
                        'cluster_center': cluster_center
                    })
                    
                    # 更新视角信息
                    view_hit = 1
                    view_score = score
                    view_cx = px
                    view_cy = py
                    views_hit += 1
                    
                    # 计算有效深度点数
                    depth_valid_count = np.sum((depth > 0.05) & (depth < 2.0))
                    
                    print(f"第{i+1}帧检测到物体，J1角度={np.rad2deg(angle):.1f}° 世界坐标: {cluster_center}")
                    detected_once = True
                    miss_count = 0  # 检测到物体，miss计数清零
                else:
                    print(f"第{i+1}帧：聚类中心投影到图像外，跳过")
                    miss_count += 1
            else:
                miss_count += 1
                # 显示原始图像
                img_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(img_vis, "No object detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Depth Clustering Detection", img_vis)
                cv2.waitKey(100)
        
        # 记录视角详细信息，包含光照条件信息
        view_details.append({
            'angle_deg': np.rad2deg(angle),
            'hit': view_hit,
            'score': view_score,
            'cx': view_cx,
            'cy': view_cy,
            'depth_valid_count': depth_valid_count,
            'lighting_condition': lighting_condition,  # 添加光照条件信息
            'cluster_detected': cluster_center is not None  # 记录是否检测到聚类
        })
        
        # 调试信息：打印当前状态
        print(f"第{i+1}帧状态: detected_once={detected_once}, miss_count={miss_count}, view_hit={view_hit}")
        
        # 停止逻辑：连续两次未检测到物体且已经检测到过物体
        if detected_once and miss_count >= 2:
            early_stop_index = i + 1
            print(f"连续{miss_count}次未检测到物体，提前结束环视。")
            break
    
    cv2.destroyAllWindows()
    detection_time = time.time() - start_time
    
    if not results:
        raise RuntimeError("环视未检测到物体，请调整深度阈值或聚类参数！")
    
    # 选择最佳检测结果（得分最高）
    best = max(results, key=lambda r: r['score'])
    cube_pos = best['cluster_center']
    best_angle_deg = np.rad2deg(best['angle'])
    best_score = best['score']
    
    # 计算范围
    min_x, max_x, min_y, max_y = estimate_cube_range_from_clusters(all_frame_data)
    detected_range = (min_x, max_x, min_y, max_y)
    
    # 计算考虑光照条件的召回率
    # 基础召回率（所有视角）
    base_view_hit_rate = views_hit / num_views
    
    # 考虑光照条件的召回率（如果检测失败，召回率为0）
    lighting_adjusted_hit_rate = base_view_hit_rate if len(results) > 0 else 0.0
    
    print(f"环视完成，最终选定物体世界坐标: {cube_pos}")
    print(f"物体 x范围: {min_x:.4f} ~ {max_x:.4f}, y范围: {min_y:.4f} ~ {max_y:.4f}")
    print(f"物体 大小: x方向{max_x-min_x:.4f}，y方向{max_y-min_y:.4f}")
    print(f"检测耗时: {detection_time:.3f}秒, 处理帧数: {frame_count}")
    print(f"命中视角数: {views_hit}/{num_views}, 基础视角召回率: {base_view_hit_rate:.2%}")
    print(f"光照调整后召回率: {lighting_adjusted_hit_rate:.2%}")
    print(f"最佳角度: {best_angle_deg:.1f}°, 最佳得分: {best_score:.1f}")
    
    # 返回详细信息
    detection_info = {
        'cube_pos': cube_pos,
        'detected_range': detected_range,
        'detection_time': detection_time,
        'frame_count': frame_count,
        'views_total': num_views,
        'views_hit': views_hit,
        'early_stop_index': early_stop_index,
        'view_hit_rate': lighting_adjusted_hit_rate,  # 使用光照调整后的召回率
        'base_view_hit_rate': base_view_hit_rate,  # 保留基础召回率
        'best_angle_deg': best_angle_deg,
        'best_score': best_score,
        'view_details': view_details,
        'lighting_condition': lighting_condition  # 添加光照条件信息
    }
    
    time.sleep(2)  # 减少等待时间
    return cube_pos, detected_range, detection_info

# ==================== RGB-D 边界检测功能 ====================

def detect_boundary_rgbd(cam, color_threshold=0.1, depth_threshold=0.02, min_contour_area=100):
    """RGB-D边界检测主函数"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    
    if rgb is None or depth is None:
        raise RuntimeError("无法获取RGB或深度图像")
    
    print(f"RGB图像形状: {rgb.shape}")
    print(f"深度图像形状: {depth.shape}")
    print(f"深度值范围: {np.min(depth[depth > 0]):.3f} - {np.max(depth[depth > 0]):.3f}")
    
    # 使用深度信息进行边界检测
    contours = _detect_boundary_rgbd_depth(rgb, depth, cam)
    
    if not contours:
        print("RGB-D深度边界检测未找到轮廓")
        return []
    
    # 过滤小轮廓
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            filtered_contours.append(contour)
    
    print(f"RGB-D深度检测找到 {len(filtered_contours)} 个有效轮廓")
    
    return filtered_contours

def _detect_boundary_rgbd_depth(img, depth, cam):
    """基于深度的边界检测"""
    # 创建深度掩码
    valid_depth_mask = (depth > 0.05) & (depth < 1.5)
    
    if not np.any(valid_depth_mask):
        print("深度掩码无有效点")
        return []
    
    # 使用深度梯度检测边界
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # 深度梯度检测
    grad_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 阈值化梯度
    gradient_threshold = np.percentile(gradient_magnitude[valid_depth_mask], 85)
    edge_mask = gradient_magnitude > gradient_threshold
    
    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, cube_range, step_size=0.08, max_steps=30):
    """基于检测到的物体位置进行边界追踪扫描"""
    print(f"开始边界追踪扫描，物体位置: {cube_pos}")
    print(f"物体范围: {cube_range}")
    
    # 启动ESC键监听线程
    global scanning_active, esc_pressed
    scanning_active = True
    esc_pressed = False
    
    esc_thread = threading.Thread(target=check_esc_key, daemon=True)
    esc_thread.start()
    
    try:
        # 计算扫描路径
        min_x, max_x, min_y, max_y = cube_range
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        
        # 创建扫描网格
        x_steps = int(np.ceil((max_x - min_x) / step_size)) + 1
        y_steps = int(np.ceil((max_y - min_y) / step_size)) + 1
        
        scan_positions = []
        for i in range(x_steps):
            for j in range(y_steps):
                x = min_x + i * step_size
                y = min_y + j * step_size
                if x <= max_x and y <= max_y:
                    scan_positions.append((x, y))
        
        print(f"计划扫描 {len(scan_positions)} 个位置")
        
        all_points = []
        step_count = 0
        
        for x, y in scan_positions:
            if esc_pressed:
                print("检测到ESC键，停止扫描")
                break
                
            if step_count >= max_steps:
                print(f"达到最大步数限制 {max_steps}")
                break
            
            # 计算目标位置（物体上方）
            target_pos = np.array([x, y, cube_pos[2] + 0.15])  # 保持一定高度
            
            try:
                # 使用IK移动到目标位置
                qpos_ik = ed6.inverse_kinematics(
                    link=j6_link,
                    pos=target_pos,
                    quat=np.array([0, 1, 0, 0])
                )
                
                if hasattr(qpos_ik, 'cpu'):
                    qpos_ik = qpos_ik.cpu().numpy()
                
                # 检查IK解的有效性
                if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
                    print(f"位置 ({x:.3f}, {y:.3f}) IK解算失败，跳过")
                    continue
                
                # 执行移动
                ed6.control_dofs_position(qpos_ik, motors_dof_idx)
                for _ in range(50):  # 等待机械臂稳定
                    scene.step()
                
                cam.move_to_attach()
                
                # 进行边界检测
                contours = detect_boundary_rgbd(cam)
                
                if contours:
                    # 获取深度图像用于轮廓转换
                    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
                    # 将轮廓点转换为3D点
                    points_3d = convert_contours_to_3d(contours, depth, cam)
                    if len(points_3d) > 0:
                        all_points.extend(points_3d)
                        print(f"步骤 {step_count+1}: 位置 ({x:.3f}, {y:.3f}) 检测到 {len(points_3d)} 个边界点")
                    else:
                        print(f"步骤 {step_count+1}: 位置 ({x:.3f}, {y:.3f}) 未检测到有效3D点")
                else:
                    print(f"步骤 {step_count+1}: 位置 ({x:.3f}, {y:.3f}) 未检测到边界")
                
                step_count += 1
                
            except Exception as e:
                print(f"位置 ({x:.3f}, {y:.3f}) 处理失败: {e}")
                continue
        
        scanning_active = False
        
        if len(all_points) == 0:
            raise RuntimeError("边界扫描未收集到任何点云数据")
        
        print(f"边界扫描完成，总共收集到 {len(all_points)} 个点")
        
        # 保存点云
        save_pointcloud(all_points, "boundary_cloud_complete_pipeline.ply")
        
        return np.array(all_points)
        
    except Exception as e:
        scanning_active = False
        raise RuntimeError(f"边界扫描过程中发生错误: {e}")

def convert_contours_to_3d(contours, depth, cam):
    """将2D轮廓转换为3D点"""
    points_3d = []
    
    K = cam.intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    T_cam2world = np.linalg.inv(cam.extrinsics)
    
    for contour in contours:
        for point_2d in contour:
            u, v = point_2d[0]
            
            if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                d = depth[v, u]
                
                if d > 0.05 and d < 2.0:  # 有效深度范围
                    # 像素坐标转相机坐标
                    x_cam = (u - cx) * d / fx
                    y_cam = (v - cy) * d / fy
                    z_cam = d
                    
                    # 转换到世界坐标系
                    point_cam = np.array([x_cam, y_cam, z_cam, 1])
                    point_world = T_cam2world @ point_cam
                    
                    points_3d.append(point_world[:3])
    
    return points_3d

def save_pointcloud(points, filename):
    """保存点云到PLY文件"""
    if len(points) == 0:
        print("没有点云数据可保存")
        return
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 添加颜色（蓝色）
    colors = np.tile([0, 0, 1], (len(points), 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存文件
    os.makedirs("boundary_data", exist_ok=True)
    filepath = os.path.join("boundary_data", filename)
    o3d.io.write_point_cloud(filepath, pcd)
    
    print(f"点云已保存到: {filepath}")

def visualize_and_save_pointcloud(points, filename="boundary_cloud.ply"):
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

def perform_surface_analysis_with_points(surface_points, boundary_points, cube_pos):
    """
    使用已采集的表面点云进行表面分析
    简化版本，直接基于表面点云生成触诊建议
    """
    print(f"\n{'='*60}")
    print("开始表面分析和触诊建议生成")
    print(f"{'='*60}")
    
    if len(surface_points) == 0:
        print("表面点云为空，无法进行表面分析")
        return []
    
    print(f"表面点云数量: {len(surface_points)}")
    
    # 找到最高点
    heights = surface_points[:, 2]
    max_height_idx = np.argmax(heights)
    highest_point = surface_points[max_height_idx]
    
    print(f"最高点位置: ({highest_point[0]:.4f}, {highest_point[1]:.4f}, {highest_point[2]:.4f})")
    print(f"最高点高度: {highest_point[2]:.4f}m")
    
    # 计算高度统计信息
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    height_range = np.max(heights) - np.min(heights)
    
    print(f"高度统计: 均值={mean_height:.4f}, 标准差={std_height:.4f}, 范围={height_range:.4f}")
    
    # 判断是否为平面表面
    is_flat_surface = height_range < 0.002  # 2mm阈值
    
    suggestions = []
    
    if is_flat_surface:
        print(f"检测到平面表面（高度差={height_range*1000:.1f}mm），建议触诊中心位置")
        # 平面表面：使用点云质心
        centroid = np.mean(surface_points, axis=0)
        palpation_pos = np.array([centroid[0], centroid[1], centroid[2] + 0.15])
        
        center_suggestion = {
            'position': palpation_pos,
            'type': 'center',
            'priority': 1,
            'reason': f'平面表面，高度差={height_range*1000:.1f}mm',
            'confidence': 0.8
        }
        suggestions.append(center_suggestion)
    else:
        print(f"检测到非平面表面（高度差={height_range*1000:.1f}mm），建议触诊最高点")
        
        # 非平面表面：直接使用最高点
        palpation_pos = np.array([highest_point[0], highest_point[1], highest_point[2] + 0.15])
        
        # 计算置信度：基于高度差和相对高度
        height_above_mean = highest_point[2] - mean_height
        relative_height = height_above_mean / std_height if std_height > 0 else 0
        
        # 简化的置信度计算
        if height_above_mean > 0.005:  # 5mm以上
            confidence = 0.9
        elif height_above_mean > 0.002:  # 2-5mm
            confidence = 0.7
        elif height_above_mean > 0.001:  # 1-2mm
            confidence = 0.5
        else:  # 1mm以下
            confidence = 0.3
        
        highest_suggestion = {
            'position': palpation_pos,
            'type': 'highest_point',
            'priority': 1,
            'height': highest_point[2],
            'height_above_mean': height_above_mean,
            'relative_height': relative_height,
            'reason': f'表面最高点，高度={highest_point[2]:.4f}m，超出均值{height_above_mean*1000:.1f}mm',
            'confidence': confidence
        }
        suggestions.append(highest_suggestion)
    
    print(f"触诊建议生成完成，共 {len(suggestions)} 个建议")
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        print(f"  建议{i+1}: 位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"类型={suggestion['type']}, 置信度={suggestion['confidence']:.2f}")
    
    return suggestions

# ==================== 完整Pipeline ====================

def complete_pipeline(scene, ed6, cam, motors_dof_idx, j6_link, lighting_condition="normal"):
    """
    完整的Pipeline：基于farest结构 + 深度聚类检测 + 完整表面分析
    前半部分：Find the Object（深度聚类检测）
    后半部分：vRGBD（边界扫描 + 表面分析）
    """
    try:
        print("="*80)
        print("完整Pipeline启动：深度聚类物体检测 + RGB-D边界扫描 + 表面分析")
        print("基于farest的Pipeline结构")
        print("="*80)
        
        # === 前半部分：Find the Object ===
        print("\n=== 前半部分：Find the Object（深度聚类检测）===")
        
        # 第一步：机械臂初始化
        print("\n[步骤1] 机械臂初始化...")
        reset_arm(scene, ed6, motors_dof_idx)
        
        # 第二步：深度聚类物体检测
        print("\n[步骤2] 深度聚类物体检测...")
        cube_pos, cube_range, detection_info = detect_cube_position(
            scene, ed6, cam, motors_dof_idx, lighting_condition
        )
        
        print(f"物体检测完成:")
        print(f"  位置: {cube_pos}")
        print(f"  范围: {cube_range}")
        print(f"  检测时间: {detection_info['detection_time']:.3f}秒")
        print(f"  视角命中率: {detection_info['views_hit']}/{detection_info['views_total']}")
        
        # 第三步：检测后平滑回零（farest的关键步骤）
        print("\n[步骤3] 检测后平滑回零...")
        reset_after_detection(scene, ed6, motors_dof_idx)
        
        # 第四步：移动到物体上方
        print("\n[步骤4] 移动到检测到的物体上方...")
        plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
        
        # === 后半部分：vRGBD（边界扫描 + 表面分析）===
        print("\n=== 后半部分：vRGBD（边界扫描 + 表面分析）===")
        
        # 第五步：RGB-D边界扫描和表面点云采集
        print("\n[步骤5] RGB-D边界扫描和表面点云采集...")
        time.sleep(5)
        boundary_points, surface_points = track_and_scan_boundary(
            scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, cube_range
        )
        
        print(f"边界扫描完成:")
        print(f"  边界点数量: {len(boundary_points)}")
        print(f"  表面点数量: {len(surface_points)}")
        
        # 第六步：保存点云数据
        print("\n[步骤6] 保存点云数据...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        boundary_filename = f"boundary_cloud_depth_clustering_{timestamp}.ply"
        surface_filename = f"surface_cloud_depth_clustering_{timestamp}.ply"
        
        visualize_and_save_pointcloud(boundary_points, boundary_filename)
        
        if len(surface_points) > 0:
            visualize_and_save_pointcloud(surface_points, surface_filename)
        
        # 第七步：表面分析和触诊建议生成（farest缺少的功能）
        palpation_suggestions = []
        if len(surface_points) > 0:
            print("\n[步骤7] 表面分析和触诊建议生成...")
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
                    if 'strength' in suggestion:
                        print(f"  异常强度: {suggestion['strength']:.3f}")
                    if 'height_above_mean' in suggestion:
                        print(f"  高度差: {suggestion['height_above_mean']*1000:.1f}mm")
            else:
                print("未生成任何触诊建议")
        else:
            print("\n[步骤7] 表面点云为空，跳过表面分析")
        
        # 第八步：返回机械臂到初始位置（farest的最终步骤）
        print("\n[步骤8] 返回机械臂到初始位置...")
        reset_after_detection(scene, ed6, motors_dof_idx)
        
        # === Pipeline完成总结 ===
        print(f"\n{'='*80}")
        print("Pipeline执行完成！")
        print(f"{'='*80}")
        print(f"检测到的物体位置: {cube_pos}")
        print(f"检测到的物体范围: {cube_range}")
        print(f"边界点云数量: {len(boundary_points)}")
        print(f"表面点云数量: {len(surface_points)}")
        print(f"边界点云文件: {boundary_filename}")
        if len(surface_points) > 0:
            print(f"表面点云文件: {surface_filename}")
        print(f"检测耗时: {detection_info.get('detection_time', 0):.2f}秒")
        print(f"命中视角数: {detection_info.get('views_hit', 0)}/{detection_info.get('views_total', 0)}")
        print(f"视角召回率: {detection_info.get('view_hit_rate', 0):.2%}")
        print(f"触诊建议数量: {len(palpation_suggestions)}")
        
        print("\nPipeline成功完成！所有功能已执行完毕。")
        
        return {
            'cube_pos': cube_pos,
            'cube_range': cube_range,
            'detection_info': detection_info,
            'boundary_points': boundary_points,
            'surface_points': surface_points,
            'palpation_suggestions': palpation_suggestions,
            'boundary_filename': boundary_filename,
            'surface_filename': surface_filename if len(surface_points) > 0 else None,
            'success': True
        }
        
    except Exception as e:
        print(f"\n=== Pipeline执行失败 ===")
        print(f"错误信息: {e}")
        
        # 确保机械臂回到安全位置
        try:
            reset_arm(scene, ed6, motors_dof_idx)
        except:
            pass
        
        raise RuntimeError(f"Pipeline执行失败: {e}")

def main():
    """主函数"""
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
    
    # 添加地面
    plane = scene.add_entity(gs.morphs.Plane(collision=True))
    
    # 添加测试物体
    cube = scene.add_entity(gs.morphs.Box(
        size=(0.105, 0.18, 0.022),
        pos=(0.35, 0, 0.02),
        collision=True,
        fixed=True,
    ))
    
    # 添加机械臂
    ed6 = scene.add_entity(gs.morphs.URDF(
        file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
        scale=1.0,
        requires_jac_and_IK=True,
        fixed=True,
    ))
    
    # 添加相机
    cam = scene.add_camera(
        model="thinlens",
        res=(640, 480),
        pos=(0, 0, 0),
        lookat=(0, 0, 1),
        up=(0, 0, 1),
        fov=60,
        aperture=2.8,
        focus_dist=0.02,
        GUI=True,
    )
    
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    
    # 相机安装到机械臂末端
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    
    print("=== 完整物体检测和边界扫描Pipeline ===")
    
    # 询问光照条件
    lighting_input = input("请输入光照条件 (normal/bright/dim，默认normal): ").strip().lower()
    if lighting_input not in ["normal", "bright", "dim"]:
        lighting_input = "normal"
    
    try:
        # 执行完整Pipeline
        results = complete_pipeline(scene, ed6, cam, motors_dof_idx, j6_link, lighting_input)
        
        print("\n=== Pipeline结果总结 ===")
        print(f"物体位置: {results['cube_pos']}")
        print(f"物体范围: {results['cube_range']}")
        print(f"边界点数量: {len(results['boundary_points'])}")
        print(f"表面点数量: {len(results['surface_points'])}")
        print(f"检测时间: {results['detection_info']['detection_time']:.3f}秒")
        print(f"视角命中率: {results['detection_info']['views_hit']}/{results['detection_info']['views_total']}")
        print(f"边界点云文件: {results['boundary_filename']}")
        if results['surface_filename']:
            print(f"表面点云文件: {results['surface_filename']}")
        print(f"触诊建议数量: {len(results['palpation_suggestions'])}")
        
        if len(results['palpation_suggestions']) > 0:
            print("\n触诊建议详情:")
            for i, suggestion in enumerate(results['palpation_suggestions']):
                pos = suggestion['position']
                print(f"  建议{i+1}: 位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                      f"类型={suggestion['type']}, 置信度={suggestion['confidence']:.2f}")
        
    except Exception as e:
        print(f"Pipeline执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
