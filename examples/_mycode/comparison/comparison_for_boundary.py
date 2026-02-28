#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪方法对比系统 - 完整版本
整合三种方法：Canny、RGB-D、Alpha-Shape
包含完整的边界追踪流程：跳跃式扫描 + 闭环检测
测试多种物体和光照条件
"""

import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import open3d as o3d
import threading
import sys
import select
import os
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy import ndimage
from scipy.interpolate import griddata

# 导入我们提取的算法模块
from boundary_detection_canny import detect_boundary_canny
from boundary_detection_rgbd import detect_boundary_rgbd
from boundary_detection_alpha_shape import detect_boundary_alpha_shape

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
                    print("\n检测到ESC键，正在结束测试...")
                    break
        else:  # Linux/Windows
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = input().strip().lower()
                    if key == 'esc' or 'q':
                        esc_pressed = True
                        print("\n检测到ESC键，正在结束测试...")
                        break
            except:
                pass
        time.sleep(0.1)

def reset_arm(scene, ed6, motors_dof_idx):
    """机械臂回到初始零位"""
    motors_dof_idx = list(range(6))
    ed6.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000]), dofs_idx_local=motors_dof_idx)
    ed6.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200]), dofs_idx_local=motors_dof_idx)
    ed6.set_dofs_force_range(
        lower=np.array([-87, -87, -87, -87, -12, -12]),
        upper=np.array([87, 87, 87, 87, 12, 12]),
        dofs_idx_local=motors_dof_idx,
    )
    ed6.set_dofs_position(np.zeros(6), motors_dof_idx)
    for _ in range(100):
        scene.step()
    print("机械臂已回到初始零位")

def plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
    """机械臂移动到目标位置"""
    time.sleep(0.5)
    target_quat = np.array([0, 1, 0, 0])
    target_pos = target_pos.copy()
    target_pos[2] += 0.2
    qpos_ik = ed6.inverse_kinematics(
        link=j6_link,
        pos=target_pos,
        quat=target_quat,
    )
    
    if hasattr(qpos_ik, 'cpu'):
        qpos_ik = qpos_ik.cpu().numpy()
    
    try:
        path = ed6.plan_path(
            qpos_goal=qpos_ik,
            num_waypoints=50,
        )
        if len(path) == 0:
            raise RuntimeError("plan_path返回空路径，使用直接控制")
    except Exception as e:
        print(f"路径规划失败，使用直接控制: {e}")
        import torch
        if isinstance(qpos_ik, torch.Tensor):
            qpos_ik = qpos_ik.detach().cpu().numpy()
        path = np.linspace(np.zeros(6), qpos_ik, num=50)
        path = torch.from_numpy(path).float().cpu()
    
    # 执行路径
    for idx, waypoint in enumerate(path):
        ed6.control_dofs_position(waypoint, motors_dof_idx)
        for _ in range(1):
            scene.step()
    
    cam.move_to_attach()
    time.sleep(0.5)

class BoundaryTrackingComparison:
    """边界追踪方法对比系统"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
        # 测试物体配置
        self.test_objects = [
            {
                'name': 'Cube',
                'type': 'box',
                'params': {'size': (0.105, 0.18, 0.022), 'pos': (0.35, 0, 0.02)},
                'file': None
            },
            {
                'name': 'ctry.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': '../../genesis/assets/_myobj/ctry.obj'
            },
            {
                'name': 'tuoyuan.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': '../../genesis/assets/_myobj/tuoyuan.obj'
            }
        ]
        
        # 检测方法
        self.detection_methods = {
            'canny': detect_boundary_canny,
            'rgbd': detect_boundary_rgbd,
            'alpha_shape': detect_boundary_alpha_shape
        }
        
        # 光照条件
        self.lighting_conditions = [
            {'name': 'normal', 'ambient': (0.1, 0.1, 0.1), 'directional': (0.5, 0.5, 0.5)},
            {'name': 'bright', 'ambient': (0.3, 0.3, 0.3), 'directional': (0.8, 0.8, 0.8)},
            {'name': 'dim', 'ambient': (0.05, 0.05, 0.05), 'directional': (0.2, 0.2, 0.2)}
        ]
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"evaluation_results/boundary_tracking_comparison_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"边界追踪对比系统初始化完成")
        print(f"测试物体数量: {len(self.test_objects)}")
        print(f"检测方法数量: {len(self.detection_methods)}")
        print(f"光照条件数量: {len(self.lighting_conditions)}")
        print(f"结果将保存到: {self.results_dir}")
    
    def _setup_scene(self, obj_config, lighting_config):
        """设置仿真场景"""
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
                show_cameras=False,  # 关闭摄像机显示
                plane_reflection=True,
                ambient_light=lighting_config['ambient'],
            ),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=False,
                enable_collision=True,
                gravity=(0, 0, -9.81),
            ),
            show_viewer=False,  # 关闭可视化
        )
        
        # 添加平面
        plane = scene.add_entity(gs.morphs.Plane(collision=True))
        
        # 添加测试物体
        if obj_config['type'] == 'box':
            cube = scene.add_entity(gs.morphs.Box(
                size=obj_config['params']['size'],
                pos=obj_config['params']['pos'],
                collision=True
            ))
        else:  # mesh
            cube = scene.add_entity(gs.morphs.Mesh(
                file=obj_config['file'],
                pos=obj_config['params']['pos'],
                collision=obj_config['params']['collision'],
                fixed=obj_config['params']['fixed']
            ))
        
        # 添加机械臂
        ed6 = scene.add_entity(gs.morphs.URDF(
            file="../../genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
            scale=1.0,
            requires_jac_and_IK=True,
            fixed=True,
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
            focus_dist=0.02,
            GUI=False,
        )
        
        scene.build()
        
        # 设置机械臂
        motors_dof_idx = list(range(6))
        j6_link = ed6.get_link("J6")
        offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
        cam.attach(j6_link, offset_T)
        scene.step()
        cam.move_to_attach()
        
        # 初始化机械臂
        reset_arm(scene, ed6, motors_dof_idx)
        
        return scene, ed6, cam, motors_dof_idx, j6_link
    
    def track_boundary_jump(self, scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, method_name, max_steps=15):
        """
        跳跃式边界追踪 - 无头版本
        
        Parameters:
        -----------
        method_name : str
            检测方法名称 ('canny', 'rgbd', 'alpha_shape')
        
        Returns:
        --------
        result : dict
            包含边界追踪结果的字典
        """
        global esc_pressed
        
        all_points = []
        start_pos = None
        start_recorded = False
        h, w = cam.res[1], cam.res[0]
        center_img = np.array([w // 2, h // 2])
        
        # 闭环检测相关变量
        closure_check_interval = 1
        last_closure_check_step = 0
        
        # 记录上一次选择的方向
        last_direction = None
        
        # 自适应高度调整参数
        base_height = cube_pos[2] + 0.15
        min_height = cube_pos[2] + 0.1
        max_height = cube_pos[2] + 0.25
        current_height = base_height
        
        # 检测质量统计
        detection_failures = 0
        max_consecutive_failures = 3
        
        # 回到初始点的检测阈值
        return_threshold = 0.03
        
        print(f"开始跳跃式边界扫描 ({method_name}方法)...")
        
        for step in range(max_steps):
            # 检查ESC键
            if esc_pressed:
                print("用户中断扫描")
                break
                
            print(f"  第{step+1}步边界检测...")
            
            # 1. 边界检测（带重试机制）
            max_retries = 3
            retry_count = 0
            contours = None
            
            while retry_count < max_retries:
                if esc_pressed:
                    break
                    
                # 使用指定的检测方法
                detection_func = self.detection_methods[method_name]
                result = detection_func(cam)
                
                if result['success'] and len(result['contours']) > 0:
                    contours = result['contours']
                    detection_failures = 0
                    break
                
                # 检测失败，重试
                retry_count += 1
                detection_failures += 1
                
                if retry_count < max_retries:
                    print(f"    第{retry_count}次检测失败，重试...")
                    time.sleep(0.5)
            
            if esc_pressed:
                break
                
            if not contours:
                print(f"第{step+1}步，经过{max_retries}次重试仍未检测到边界")
                break
            
            # 2. 记录起始位置
            if not start_recorded:
                start_pos = j6_link.get_pos().cpu().numpy()
                start_recorded = True
            
            # 3. 获取当前机械臂末端位置
            current_pos = j6_link.get_pos().cpu().numpy()
            
            # 4. 检查是否回到初始点
            if start_recorded and step > 3:
                dist_to_start = np.linalg.norm(current_pos - start_pos)
                if dist_to_start < return_threshold:
                    print(f"第{step+1}步，已回到起始位置附近，扫描完成！")
                    break
            
            # 5. 找到最远的边界点
            max_dist = 0
            best_pt = None
            best_world_pos = None
            
            for cnt in contours:
                for pt in cnt:
                    pt_xy = pt[0]
                    cx, cy = int(pt_xy[0]), int(pt_xy[1])
                    
                    # 获取深度
                    if 0 <= cy < h and 0 <= cx < w:
                        d = result['depth'][cy, cx]
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
                                direction_vector = pos_world[:2] - current_pos[:2]
                                direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
                                
                                if not self._is_clockwise_direction(direction_angle, last_direction):
                                    continue
                            
                            if dist > max_dist:
                                max_dist = dist
                                best_pt = pt_xy
                                best_world_pos = pos_world[:3]
            
            if best_pt is None:
                print(f"第{step+1}步，未找到合适的边界点")
                break
            
            # 6. 计算目标位置
            target_pos = best_world_pos.copy()
            target_pos[2] = current_height
            
            # 7. 更新方向记录
            direction_vector = best_world_pos[:2] - current_pos[:2]
            last_direction = np.arctan2(direction_vector[1], direction_vector[0])
            
            # 8. 采集边界点云
            pc, _ = cam.render_pointcloud(world_frame=True)
            
            # 创建边界mask
            boundary_mask = np.zeros_like(result['depth'], dtype=bool)
            for cnt in contours:
                for pt in cnt:
                    px, py = pt[0]
                    if 0 <= px < result['depth'].shape[1] and 0 <= py < result['depth'].shape[0]:
                        boundary_mask[py, px] = True
            
            boundary_points = pc[boundary_mask]
            all_points.append(boundary_points)
            print(f"第{step+1}步，采集到{len(boundary_points)}个边界点")
            
            # 9. 闭环检测
            if step >= 4 and step - last_closure_check_step >= closure_check_interval and len(all_points) > 0:
                current_all_points = np.concatenate(all_points, axis=0)
                cube_size = np.array([0.105, 0.18, 0.022])
                
                is_closed, closure_quality, detailed_metrics = self._detect_pointcloud_closure(
                    current_all_points, cube_pos, cube_size
                )
                
                print(f"  闭环检测: 质量={closure_quality:.3f}, 形成闭环={is_closed}")
                
                if is_closed:
                    print(f"第{step+1}步，点云已形成闭环，跳跃扫描完成！")
                    break
                
                last_closure_check_step = step
            
            # 10. IK逆解和路径规划
            target_quat = j6_link.get_quat().cpu().numpy()
            qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
            if hasattr(qpos_ik, 'cpu'):
                qpos_ik = qpos_ik.cpu().numpy()
            
            if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
                print(f"第{step+1}步，IK逆解失败")
                break
            
            # 11. 执行路径
            try:
                path = ed6.plan_path(qpos_goal=qpos_ik, num_waypoints=50)
                if len(path) == 0:
                    path = [qpos_ik]
            except Exception as e:
                path = [qpos_ik]
            
            for idx, waypoint in enumerate(path):
                if esc_pressed:
                    break
                ed6.control_dofs_position(waypoint, motors_dof_idx)
                for _ in range(1):
                    scene.step()
            
            if esc_pressed:
                break
                
            cam.move_to_attach()
        
        # 合并边界点云
        if len(all_points) == 0:
            print("未采集到任何边界点云！")
            return {
                'boundary_points': np.zeros((0, 3)),
                'total_steps': 0,
                'success': False,
                'error': '未采集到边界点云'
            }
        
        all_boundary_points = np.concatenate(all_points, axis=0)
        print(f"跳跃式边界扫描完成，总共采集到{len(all_boundary_points)}个边界点")
        
        return {
            'boundary_points': all_boundary_points,
            'total_steps': step + 1,
            'success': True,
            'error': None
        }
    
    def _is_clockwise_direction(self, current_angle, last_angle, tolerance=np.pi/4):
        """检查当前方向是否符合顺时针约束"""
        if last_angle is None:
            return True
        
        angle_diff = current_angle - last_angle
        
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return angle_diff >= -tolerance
    
    def _detect_pointcloud_closure(self, points, cube_pos, cube_size, closure_threshold=0.02, min_points=50):
        """检测点云是否形成闭环"""
        if not self._check_closure_prerequisites(points):
            return False, 0.0, {}
        
        target_face_area = cube_size[0] * cube_size[1]
        
        # 方法A：图像法
        closure_a, metrics_a = self._detect_closure_flood_fill(points, cube_size, target_face_area)
        
        # 方法B：图论法
        closure_b, metrics_b = self._detect_closure_graph_topology(points, cube_size, target_face_area)
        
        # 并联判定
        is_closed = closure_a or closure_b
        closure_quality = max(metrics_a.get('flood_fill_quality', 0.0), metrics_b.get('graph_quality', 0.0))
        
        detailed_metrics = {
            'method_a_closure': closure_a,
            'method_b_closure': closure_b,
            'method_a_quality': metrics_a.get('flood_fill_quality', 0.0),
            'method_b_quality': metrics_b.get('graph_quality', 0.0),
            'overall_quality': closure_quality,
            'target_face_area': target_face_area,
            'point_count': len(points),
            **metrics_a,
            **metrics_b
        }
        
        return is_closed, closure_quality, detailed_metrics
    
    def _check_closure_prerequisites(self, points):
        """前置否决条件检查"""
        if len(points) < 80:
            return False
        
        points_2d = points[:, :2]
        distances = cdist(points_2d, points_2d)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        med_nn = np.median(min_distances)
        
        if med_nn > 0.025:
            return False
        
        return True
    
    def _detect_closure_flood_fill(self, points, cube_size, target_face_area):
        """方法A：图像法（投影→栅格化→洪泛填充）"""
        points_2d = points[:, :2]
        
        # 计算自适应参数
        px, stroke_px, a_min = self._calculate_flood_fill_parameters(points_2d, target_face_area)
        
        # 栅格化
        grid, min_coords, resolution, pad_px = self._create_grid_from_points(points_2d, px)
        
        # 加粗边界
        thickened_grid = self._thicken_boundary(grid, stroke_px)
        
        # 洪泛填充
        filled_grid = self._flood_fill_from_boundary(thickened_grid)
        
        # 计算内陆面积
        inland_area = self._calculate_inland_area(filled_grid, resolution)
        
        # 判断闭环
        is_closed = inland_area >= a_min
        quality = min(1.0, inland_area / a_min) if a_min > 0 else 0.0
        
        metrics = {
            'flood_fill_closed': is_closed,
            'flood_fill_quality': quality,
            'inland_area_mm2': inland_area * 1e6,
            'min_area_threshold_mm2': a_min * 1e6,
            'pixel_resolution_mm': px * 1e3,
            'stroke_width_px': stroke_px,
            'grid_padding_px': pad_px,
            'inland_ratio': inland_area / target_face_area if target_face_area > 0 else 0.0
        }
        
        return is_closed, metrics
    
    def _calculate_flood_fill_parameters(self, points_2d, target_face_area):
        """计算洪泛填充的关键参数"""
        distances = cdist(points_2d, points_2d)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        med_nn = np.median(min_distances)
        
        px = np.clip(0.5 * med_nn, 0.002, 0.008)
        t_near = np.clip(2.5 * med_nn, 0.003, 0.025)
        stroke_px = np.clip(t_near / px, 1, 20)
        a_min = 0.20 * target_face_area
        
        return px, stroke_px, a_min
    
    def _create_grid_from_points(self, points_2d, resolution):
        """将点云转换为栅格图像"""
        min_coords = np.min(points_2d, axis=0)
        max_coords = np.max(points_2d, axis=0)
        
        grid_width = int((max_coords[0] - min_coords[0]) / resolution) + 1
        grid_height = int((max_coords[1] - min_coords[1]) / resolution) + 1
        
        grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        for point in points_2d:
            x_idx = int((point[0] - min_coords[0]) / resolution)
            y_idx = int((point[1] - min_coords[1]) / resolution)
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                grid[y_idx, x_idx] = 255
        
        pad_px = max(10, int(np.ceil(0.05 / resolution)))
        pad_px = min(pad_px, 32)
        grid = np.pad(grid, pad_px, constant_values=0)
        
        return grid, min_coords, resolution, pad_px
    
    def _thicken_boundary(self, grid, stroke_px):
        """加粗边界"""
        kernel_size = max(3, 2 * round(stroke_px / 2) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        thickened = cv2.dilate(grid, kernel, iterations=2)
        thickened = cv2.erode(thickened, kernel, iterations=1)
        
        return thickened
    
    def _flood_fill_from_boundary(self, grid):
        """从边界开始洪泛填充"""
        filled_grid = grid.copy()
        mask = np.zeros((grid.shape[0] + 2, grid.shape[1] + 2), dtype=np.uint8)
        
        start_x = grid.shape[1] // 2
        start_y = grid.shape[0] // 2
        cv2.floodFill(filled_grid, mask, (start_x, start_y), 128)
        
        return filled_grid
    
    def _calculate_inland_area(self, filled_grid, resolution):
        """计算内陆面积"""
        inland_pixels = np.sum(filled_grid == 0)
        inland_area = inland_pixels * (resolution ** 2)
        return inland_area
    
    def _detect_closure_graph_topology(self, points, cube_size, target_face_area):
        """方法B：图论法（kNN图→拓扑环检测）"""
        points_2d = points[:, :2]
        
        # 计算参数
        k, t_near, t_long = self._calculate_graph_parameters(points_2d)
        
        # 构建kNN图
        knn_graph = self._build_knn_graph(points_2d, k, t_long)
        
        # 检测拓扑环
        cycles, degree_stats = self._detect_topological_cycles(knn_graph, points_2d)
        
        # 判断闭环
        is_closed = self._evaluate_cycle_completeness(cycles, degree_stats, t_long)
        
        # 计算质量分数
        quality = self._calculate_graph_quality(cycles, degree_stats)
        
        metrics = {
            'graph_closed': is_closed,
            'graph_quality': quality,
            'cycle_count': cycles,
            'degree_2_ratio': degree_stats.get('degree_2_ratio', 0.0),
            'max_edge_length': degree_stats.get('max_edge_length', 0.0),
            'k_neighbors': k,
            'near_threshold_mm': t_near * 1e3,
            'long_threshold_mm': t_long * 1e3
        }
        
        return is_closed, metrics
    
    def _calculate_graph_parameters(self, points_2d):
        """计算图论检测的参数"""
        distances = cdist(points_2d, points_2d)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        med_nn = np.median(min_distances)
        
        k = 3
        t_near = np.clip(2.5 * med_nn, 0.003, 0.025)
        t_long = 1.5 * t_near
        
        return k, t_near, t_long
    
    def _build_knn_graph(self, points_2d, k, t_long):
        """构建kNN图"""
        n_points = len(points_2d)
        distances = cdist(points_2d, points_2d)
        np.fill_diagonal(distances, np.inf)
        
        adjacency_matrix = np.zeros((n_points, n_points), dtype=bool)
        
        for i in range(n_points):
            nearest_indices = np.argsort(distances[i])[:k]
            
            for j in nearest_indices:
                if distances[i, j] <= t_long:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
        
        return adjacency_matrix
    
    def _detect_topological_cycles(self, adjacency_matrix, points_2d=None):
        """检测拓扑环"""
        n_points = len(adjacency_matrix)
        degrees = np.sum(adjacency_matrix, axis=1)
        
        # 递归去叶
        while True:
            leaf_nodes = np.where(degrees == 1)[0]
            if len(leaf_nodes) == 0:
                break
            
            for leaf in leaf_nodes:
                neighbors = np.where(adjacency_matrix[leaf])[0]
                for neighbor in neighbors:
                    adjacency_matrix[leaf, neighbor] = False
                    adjacency_matrix[neighbor, leaf] = False
                    degrees[neighbor] -= 1
                degrees[leaf] = 0
        
        # 计算环的秩
        visited = np.zeros(n_points, dtype=bool)
        cycle_count = 0
        
        for i in range(n_points):
            if degrees[i] > 0 and not visited[i]:
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
                
                if component_size >= 3:
                    cycle_count += 1
        
        # 计算度数=2的节点比例
        degree_2_nodes = np.sum(degrees == 2)
        total_nodes_with_edges = np.sum(degrees > 0)
        degree_2_ratio = degree_2_nodes / total_nodes_with_edges if total_nodes_with_edges > 0 else 0.0
        
        # 计算最长边长度
        max_edge_length = 0.0
        if points_2d is not None:
            for i in range(n_points):
                for j in range(i+1, n_points):
                    if adjacency_matrix[i, j]:
                        edge_length = np.linalg.norm(points_2d[i] - points_2d[j])
                        max_edge_length = max(max_edge_length, edge_length)
        else:
            max_edge_length = 0.01
        
        degree_stats = {
            'degree_2_ratio': degree_2_ratio,
            'max_edge_length': max_edge_length,
            'total_nodes': total_nodes_with_edges
        }
        
        return cycle_count, degree_stats
    
    def _evaluate_cycle_completeness(self, cycles, degree_stats, t_long):
        """评估环的完整性"""
        cycle_rank_ok = cycles >= 1
        degree_2_ok = degree_stats['degree_2_ratio'] >= 0.70
        max_edge_ok = degree_stats['max_edge_length'] <= t_long
        
        return cycle_rank_ok and degree_2_ok and max_edge_ok
    
    def _calculate_graph_quality(self, cycles, degree_stats):
        """计算图论检测的质量分数"""
        cycle_score = min(1.0, cycles / 2.0)
        degree_score = degree_stats['degree_2_ratio']
        
        quality = (cycle_score * 0.6 + degree_score * 0.4)
        return quality
    
    def calculate_accuracy_metrics(self, boundary_points, reference_points=None):
        """
        计算准确性指标
        
        Parameters:
        -----------
        boundary_points : np.ndarray
            检测到的边界点云
        reference_points : np.ndarray, optional
            参考点云（如果可用）
        
        Returns:
        --------
        metrics : dict
            准确性指标字典
        """
        if len(boundary_points) == 0:
            return {
                'chamfer_distance': float('inf'),
                'hausdorff_distance': float('inf'),
                'mean_residual': float('inf'),
                'median_residual': float('inf'),
                'coverage_3mm': 0.0,
                'coverage_adaptive': 0.0
            }
        
        # 如果没有参考点云，使用边界点云本身作为参考
        if reference_points is None:
            reference_points = boundary_points
        
        # 计算Chamfer距离
        chamfer_dist = self._calculate_chamfer_distance(boundary_points, reference_points)
        
        # 计算Hausdorff距离
        hausdorff_dist = self._calculate_hausdorff_distance(boundary_points, reference_points)
        
        # 计算残差
        mean_residual, median_residual = self._calculate_residuals(boundary_points, reference_points)
        
        # 计算覆盖率
        coverage_3mm = self._calculate_coverage(boundary_points, reference_points, threshold=0.003)
        coverage_adaptive = self._calculate_adaptive_coverage(boundary_points, reference_points)
        
        return {
            'chamfer_distance': chamfer_dist,
            'hausdorff_distance': hausdorff_dist,
            'mean_residual': mean_residual,
            'median_residual': median_residual,
            'coverage_3mm': coverage_3mm,
            'coverage_adaptive': coverage_adaptive
        }
    
    def _calculate_chamfer_distance(self, points1, points2):
        """计算Chamfer距离"""
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')
        
        # 计算从points1到points2的最小距离
        distances_1_to_2 = cdist(points1, points2)
        min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
        
        # 计算从points2到points1的最小距离
        distances_2_to_1 = cdist(points2, points1)
        min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
        
        # Chamfer距离
        chamfer_dist = np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)
        
        return chamfer_dist
    
    def _calculate_hausdorff_distance(self, points1, points2):
        """计算Hausdorff距离"""
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')
        
        distances = cdist(points1, points2)
        
        # 从points1到points2的最大最小距离
        max_min_dist_1_to_2 = np.max(np.min(distances, axis=1))
        
        # 从points2到points1的最大最小距离
        max_min_dist_2_to_1 = np.max(np.min(distances, axis=0))
        
        # Hausdorff距离
        hausdorff_dist = max(max_min_dist_1_to_2, max_min_dist_2_to_1)
        
        return hausdorff_dist
    
    def _calculate_residuals(self, points1, points2):
        """计算残差"""
        if len(points1) == 0 or len(points2) == 0:
            return float('inf'), float('inf')
        
        distances = cdist(points1, points2)
        min_distances = np.min(distances, axis=1)
        
        mean_residual = np.mean(min_distances)
        median_residual = np.median(min_distances)
        
        return mean_residual, median_residual
    
    def _calculate_coverage(self, points1, points2, threshold=0.003):
        """计算覆盖率"""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        distances = cdist(points1, points2)
        min_distances = np.min(distances, axis=1)
        
        # 计算在阈值内的点比例
        covered_points = np.sum(min_distances <= threshold)
        coverage = covered_points / len(points1)
        
        return coverage
    
    def _calculate_adaptive_coverage(self, points1, points2):
        """计算自适应覆盖率"""
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # 计算自适应阈值（基于点云密度）
        distances = cdist(points1, points1)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        adaptive_threshold = np.median(min_distances) * 2.0
        
        return self._calculate_coverage(points1, points2, adaptive_threshold)
    
    def calculate_stability_metrics(self, execution_times, residuals_list):
        """计算稳定性指标"""
        if len(execution_times) < 2:
            return {
                'std_execution_time': 0.0,
                'cv_execution_time': 0.0,
                'std_residual': 0.0,
                'cv_residual': 0.0
            }
        
        # 执行时间稳定性
        std_exec_time = np.std(execution_times)
        mean_exec_time = np.mean(execution_times)
        cv_exec_time = std_exec_time / mean_exec_time if mean_exec_time > 0 else 0.0
        
        # 残差稳定性
        if residuals_list and len(residuals_list) > 1:
            std_residual = np.std(residuals_list)
            mean_residual = np.mean(residuals_list)
            cv_residual = std_residual / mean_residual if mean_residual > 0 else 0.0
        else:
            std_residual = 0.0
            cv_residual = 0.0
        
        return {
            'std_execution_time': std_exec_time,
            'cv_execution_time': cv_exec_time,
            'std_residual': std_residual,
            'cv_residual': cv_residual
        }
    
    def run_single_test(self, obj_config, method_name, lighting_config, run_id):
        """运行单次测试"""
        print(f"  运行 {run_id+1}: {method_name} + {lighting_config['name']} 光照")
        
        # 设置场景
        scene, ed6, cam, motors_dof_idx, j6_link = self._setup_scene(obj_config, lighting_config)
        
        try:
            # 移动到目标位置
            target_pos = np.array(obj_config['params']['pos'])
            plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam)
            
            # 执行边界追踪
            start_time = time.time()
            tracking_result = self.track_boundary_jump(
                scene, ed6, cam, motors_dof_idx, j6_link, target_pos, method_name, max_steps=10
            )
            execution_time = time.time() - start_time
            
            if not tracking_result['success']:
                return {
                    'object_name': obj_config['name'],
                    'object_type': obj_config['type'],
                    'method': method_name,
                    'lighting': lighting_config['name'],
                    'run_id': run_id,
                    'execution_time': execution_time,
                    'status': 'Fail',
                    'error': tracking_result['error'],
                    'point_count': 0,
                    'total_steps': 0
                }
            
            # 计算评估指标
            boundary_points = tracking_result['boundary_points']
            accuracy_metrics = self.calculate_accuracy_metrics(boundary_points)
            
            # 记录结果
            result = {
                'object_name': obj_config['name'],
                'object_type': obj_config['type'],
                'method': method_name,
                'lighting': lighting_config['name'],
                'run_id': run_id,
                'execution_time': execution_time,
                'status': 'Success' if len(boundary_points) > 50 else 'Partial',
                'point_count': len(boundary_points),
                'total_steps': tracking_result['total_steps'],
                'accuracy_metrics': accuracy_metrics
            }
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'object_name': obj_config['name'],
                'object_type': obj_config['type'],
                'method': method_name,
                'lighting': lighting_config['name'],
                'run_id': run_id,
                'execution_time': execution_time,
                'status': 'Fail',
                'error': str(e),
                'point_count': 0,
                'total_steps': 0
            }
        finally:
            # 清理场景资源
            try:
                scene.clear()
            except:
                pass
    
    def run_comparison(self, num_runs=3):
        """运行完整对比测试"""
        print(f"\n{'='*80}")
        print("开始边界追踪方法对比测试")
        print(f"{'='*80}")
        
        total_tests = len(self.test_objects) * len(self.detection_methods) * len(self.lighting_conditions) * num_runs
        current_test = 0
        
        # 启动ESC键监听
        esc_thread = threading.Thread(target=check_esc_key, daemon=True)
        esc_thread.start()
        
        for obj_idx, obj_config in enumerate(self.test_objects):
            print(f"\n{'='*60}")
            print(f"测试物体 {obj_idx+1}/{len(self.test_objects)}: {obj_config['name']}")
            print(f"{'='*60}")
            
            for method_name in self.detection_methods.keys():
                print(f"\n方法: {method_name.upper()}")
                print("-" * 40)
                
                for lighting_config in self.lighting_conditions:
                    print(f"  光照条件: {lighting_config['name']}")
                    
                    # 存储该条件下的执行时间和残差用于稳定性分析
                    execution_times = []
                    residuals_list = []
                    
                    for run_id in range(num_runs):
                        current_test += 1
                        progress = (current_test / total_tests) * 100
                        
                        print(f"    进度: {progress:.1f}%", end=" ")
                        
                        # 检查ESC键
                        if esc_pressed:
                            print("\n用户中断测试")
                            return
                        
                        # 运行测试
                        result = self.run_single_test(obj_config, method_name, lighting_config, run_id)
                        
                        if result['status'] == 'Fail':
                            print(f"❌ 失败: {result.get('error', 'Unknown error')}")
                        else:
                            print(f"✅ {result['status']} (时间: {result['execution_time']:.1f}s, 点数: {result['point_count']})")
                        
                        # 记录结果
                        self.results.append(result)
                        execution_times.append(result['execution_time'])
                        
                        # 收集残差数据
                        if 'accuracy_metrics' in result:
                            residuals_list.append(result['accuracy_metrics']['mean_residual'])
                        
                        # 短暂休息
                        time.sleep(0.1)
                    
                    # 计算稳定性指标
                    stability_metrics = self.calculate_stability_metrics(execution_times, residuals_list)
                    
                    # 更新该条件下所有结果的稳定性指标
                    for result in self.results[-num_runs:]:
                        result['stability_metrics'] = stability_metrics
        
        # 结束扫描
        scanning_active = False
        
        print(f"\n{'='*80}")
        print("所有测试完成！")
        print(f"{'='*80}")
        
        # 保存结果
        self._save_results()
        
        # 生成分析报告
        self._generate_analysis()
        
        print(f"\n结果已保存到: {self.results_dir}")
    
    def _save_results(self):
        """保存测试结果"""
        # 保存详细结果JSON
        json_file = self.results_dir / "detailed_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存CSV格式
        df = pd.DataFrame(self.results)
        csv_file = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"详细结果已保存到: {json_file}")
        print(f"CSV结果已保存到: {csv_file}")
    
    def _generate_analysis(self):
        """生成分析报告"""
        print("\n生成分析报告...")
        
        # 创建汇总统计
        summary_stats = []
        
        for obj_name in set(r['object_name'] for r in self.results):
            for method in set(r['method'] for r in self.results):
                for lighting in set(r['lighting'] for r in self.results):
                    subset = [r for r in self.results if 
                             r['object_name'] == obj_name and 
                             r['method'] == method and 
                             r['lighting'] == lighting]
                    
                    if len(subset) == 0:
                        continue
                    
                    # 计算统计量
                    successful_runs = [r for r in subset if r['status'] in ['Success', 'Partial']]
                    
                    summary = {
                        'object_name': obj_name,
                        'method': method,
                        'lighting': lighting,
                        'total_runs': len(subset),
                        'successful_runs': len(successful_runs),
                        'success_rate': len(successful_runs) / len(subset) if subset else 0,
                        'avg_execution_time': np.mean([r['execution_time'] for r in subset]),
                        'std_execution_time': np.std([r['execution_time'] for r in subset]),
                        'avg_point_count': np.mean([r['point_count'] for r in subset]),
                        'avg_total_steps': np.mean([r['total_steps'] for r in subset])
                    }
                    
                    # 添加准确性指标（仅对成功运行）
                    if successful_runs and 'accuracy_metrics' in successful_runs[0]:
                        acc_metrics = [r['accuracy_metrics'] for r in successful_runs]
                        summary.update({
                            'avg_chamfer_distance': np.mean([m['chamfer_distance'] for m in acc_metrics]),
                            'avg_hausdorff_distance': np.mean([m['hausdorff_distance'] for m in acc_metrics]),
                            'avg_coverage_3mm': np.mean([m['coverage_3mm'] for m in acc_metrics]),
                            'avg_mean_residual': np.mean([m['mean_residual'] for m in acc_metrics])
                        })
                    
                    summary_stats.append(summary)
        
        # 保存汇总统计
        summary_df = pd.DataFrame(summary_stats)
        summary_file = self.results_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # 生成方法对比
        method_comparison = []
        for method in set(r['method'] for r in self.results):
            method_data = [r for r in self.results if r['method'] == method]
            successful_data = [r for r in method_data if r['status'] in ['Success', 'Partial']]
            
            comparison = {
                'method': method,
                'total_tests': len(method_data),
                'successful_tests': len(successful_data),
                'overall_success_rate': len(successful_data) / len(method_data) if method_data else 0,
                'avg_execution_time': np.mean([r['execution_time'] for r in method_data]),
                'std_execution_time': np.std([r['execution_time'] for r in method_data]),
                'avg_point_count': np.mean([r['point_count'] for r in method_data]),
                'avg_total_steps': np.mean([r['total_steps'] for r in method_data])
            }
            
            # 添加准确性指标
            if successful_data and 'accuracy_metrics' in successful_data[0]:
                acc_metrics = [r['accuracy_metrics'] for r in successful_data]
                comparison.update({
                    'avg_chamfer_distance': np.mean([m['chamfer_distance'] for m in acc_metrics]),
                    'avg_hausdorff_distance': np.mean([m['hausdorff_distance'] for m in acc_metrics]),
                    'avg_coverage_3mm': np.mean([m['coverage_3mm'] for m in acc_metrics]),
                    'avg_mean_residual': np.mean([m['mean_residual'] for m in acc_metrics])
                })
            
            method_comparison.append(comparison)
        
        # 保存方法对比
        comparison_df = pd.DataFrame(method_comparison)
        comparison_file = self.results_dir / "method_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"汇总统计已保存到: {summary_file}")
        print(f"方法对比已保存到: {comparison_file}")
        
        # 显示总结
        self._display_summary(comparison_df)
    
    def _display_summary(self, comparison_df):
        """在终端显示对比总结"""
        print(f"\n{'='*80}")
        print("边界追踪方法对比总结")
        print(f"{'='*80}")
        
        print(f"\n各方法表现:")
        for _, row in comparison_df.iterrows():
            print(f"  {row['method'].upper()}:")
            print(f"    成功率: {row['overall_success_rate']:.1%}")
            print(f"    执行时间: {row['avg_execution_time']:.1f}±{row['std_execution_time']:.1f}s")
            print(f"    平均点数: {row['avg_point_count']:.0f}")
            print(f"    平均步数: {row['avg_total_steps']:.1f}")
            
            if 'avg_chamfer_distance' in row:
                print(f"    Chamfer距离: {row['avg_chamfer_distance']:.3f}m")
                print(f"    3mm覆盖率: {row['avg_coverage_3mm']:.3f}")
        
        # 找出最佳方法
        if len(comparison_df) > 0:
            best_success = comparison_df.loc[comparison_df['overall_success_rate'].idxmax(), 'method']
            fastest = comparison_df.loc[comparison_df['avg_execution_time'].idxmin(), 'method']
            
            print(f"\n🏆 推荐最佳方法: {best_success.upper()}")
            print(f"   该方法在成功率方面表现最佳")
            print(f"⚡ 最快方法: {fastest.upper()}")
            print(f"   该方法在执行速度方面表现最佳")
        
        print(f"\n详细结果请查看: {self.results_dir}")
        print(f"{'='*80}")

if __name__ == "__main__":
    print("边界追踪方法对比系统")
    print("=" * 60)
    
    # 初始化Genesis
    gs.init(seed=0, precision="32", logging_level="info")
    
    # 创建对比系统
    comparison = BoundaryTrackingComparison()
    
    # 运行对比测试
    comparison.run_comparison(num_runs=3)
    
    print("\n对比测试完成！")
