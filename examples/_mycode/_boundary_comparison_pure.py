#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯数据边界追踪方法对比系统
整合三种方法：Canny、RGB-D、Alpha-Shape
测试五个物体：Cube、tuoyuan.obj、buguize.obj、withtumor.obj、ctry.obj
输出纯CSV数据用于分析
"""

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
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入Alpha-Shape相关库
try:
    from shapely.geometry import Point, Polygon
    from shapely.ops import cascaded_union
    from scipy.spatial import Delaunay
    from scipy.spatial.distance import pdist, squareform
    ALPHA_SHAPE_AVAILABLE = True
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
                    if key == 'esc' or key == 'q':
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
    """机械臂回零，IK逆解，路径插值并执行"""
    time.sleep(1)  # 减少等待时间
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
            num_waypoints=100,  # 减少路径点
        )
        if len(path) == 0:
            raise RuntimeError("plan_path返回空路径，自动切换为线性插值")
    except Exception as e:
        print(f"路径规划失败，使用直接控制: {e}")
        import torch
        if isinstance(qpos_ik, torch.Tensor):
            qpos_ik = qpos_ik.detach().cpu().numpy()
        path = np.linspace(np.zeros(6), qpos_ik, num=100)
        path = torch.from_numpy(path).float().cpu()
    
    # 执行路径
    for idx, waypoint in enumerate(path):
        ed6.control_dofs_position(waypoint, motors_dof_idx)
        for _ in range(2):  # 减少步数
            scene.step()
    
    cam.move_to_attach()
    time.sleep(1)  # 减少等待时间

# ===== 边界检测方法 =====

def detect_boundary_canny(cam, min_contour_area=100):
    """使用Canny边缘检测进行边界检测"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours, rgb, depth

def detect_boundary_rgbd(cam, color_threshold=0.1, depth_threshold=0.02, min_contour_area=100):
    """使用RGB-D方法进行边界检测"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    combined = cv2.addWeighted(gray, 0.7, depth_normalized, 0.3, 0)
    
    edges = cv2.Canny(combined, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    return filtered_contours, rgb, depth

def detect_boundary_alpha_shape(cam, alpha_value=None, min_contour_area=100):
    """使用Alpha-Shape（凹包）进行边界检测"""
    if not ALPHA_SHAPE_AVAILABLE:
        print("Alpha-Shape库未安装，回退到Canny方法")
        return detect_boundary_canny(cam, min_contour_area)
    
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    pc, _ = cam.render_pointcloud(world_frame=True)
    
    if len(pc) == 0:
        print("未获取到点云数据，回退到Canny方法")
        return detect_boundary_canny(cam, min_contour_area)
    
    points_2d = pc[:, :2]
    valid_mask = ~np.isnan(points_2d).any(axis=1) & ~np.isinf(points_2d).any(axis=1)
    points_2d = points_2d[valid_mask]
    
    if len(points_2d) < 3:
        print("有效点云数量不足，回退到Canny方法")
        return detect_boundary_canny(cam, min_contour_area)
    
    try:
        alpha_shape_points = compute_alpha_shape(points_2d, alpha_value)
        
        if len(alpha_shape_points) < 3:
            print("Alpha-Shape计算失败，回退到Canny方法")
            return detect_boundary_canny(cam, min_contour_area)
        
        contours = [alpha_shape_points.reshape(-1, 1, 2).astype(np.int32)]
        return contours, rgb, depth
        
    except Exception as e:
        print(f"Alpha-Shape计算出错: {e}，回退到Canny方法")
        return detect_boundary_canny(cam, min_contour_area)

def compute_alpha_shape(points, alpha=None):
    """计算Alpha-Shape（凹包）"""
    if len(points) < 3:
        return np.array([])
    
    if alpha is None:
        alpha = compute_optimal_alpha(points)
    
    try:
        tri = Delaunay(points)
    except:
        hull = ConvexHull(points)
        return points[hull.vertices]
    
    edges = set()
    edge_points = []
    
    for simplex in tri.simplices:
        pts = points[simplex]
        a, b, c = pts
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
        if area < 1e-10:
            continue
            
        circumradius = (np.linalg.norm(a - b) * np.linalg.norm(b - c) * np.linalg.norm(c - a)) / (4 * area)
        
        if circumradius < 1.0 / alpha:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                if edge not in edges:
                    edges.add(edge)
                    edge_points.extend([pts[i], pts[(i+1)%3]])
    
    if len(edge_points) == 0:
        hull = ConvexHull(points)
        return points[hull.vertices]
    
    edge_points = np.array(edge_points)
    boundary_points = order_boundary_points(edge_points)
    
    return boundary_points

def compute_optimal_alpha(points):
    """自动计算最优的Alpha参数"""
    if len(points) < 2:
        return 1.0
    
    distances = pdist(points)
    mean_distance = np.mean(distances)
    alpha = 2.0 / mean_distance
    alpha = np.clip(alpha, 0.1, 10.0)
    
    return alpha

def order_boundary_points(points):
    """将边界点按顺序排列"""
    if len(points) < 3:
        return points
    
    start_idx = np.argmin(points[:, 0])
    start_point = points[start_idx]
    angles = np.arctan2(points[:, 1] - start_point[1], points[:, 0] - start_point[0])
    sorted_indices = np.argsort(angles)
    
    return points[sorted_indices]

# ===== 评估指标计算 =====

def calculate_accuracy_metrics(contours, rgb, depth, cam):
    """计算准确度指标"""
    if not contours:
        return {
            'num_contours': 0,
            'total_contour_area': 0,
            'avg_contour_area': 0,
            'max_contour_area': 0,
            'contour_perimeter': 0,
            'coverage_ratio': 0
        }
    
    # 基础轮廓统计
    num_contours = len(contours)
    contour_areas = [cv2.contourArea(cnt) for cnt in contours]
    total_contour_area = sum(contour_areas)
    avg_contour_area = total_contour_area / num_contours if num_contours > 0 else 0
    max_contour_area = max(contour_areas) if contour_areas else 0
    
    # 轮廓周长
    contour_perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
    total_perimeter = sum(contour_perimeters)
    
    # 覆盖率（轮廓面积占图像面积的比例）
    image_area = rgb.shape[0] * rgb.shape[1]
    coverage_ratio = total_contour_area / image_area if image_area > 0 else 0
    
    return {
        'num_contours': num_contours,
        'total_contour_area': total_contour_area,
        'avg_contour_area': avg_contour_area,
        'max_contour_area': max_contour_area,
        'contour_perimeter': total_perimeter,
        'coverage_ratio': coverage_ratio
    }

def calculate_stability_metrics(execution_times):
    """计算稳定性指标"""
    if len(execution_times) < 2:
        return {
            'std_execution_time': 0,
            'cv_execution_time': 0,
            'min_execution_time': execution_times[0] if execution_times else 0,
            'max_execution_time': execution_times[0] if execution_times else 0
        }
    
    std_time = np.std(execution_times)
    mean_time = np.mean(execution_times)
    cv_time = std_time / mean_time if mean_time > 0 else 0
    
    return {
        'std_execution_time': std_time,
        'cv_execution_time': cv_time,
        'min_execution_time': min(execution_times),
        'max_execution_time': max(execution_times)
    }

def calculate_performance_metrics(contours, execution_time, num_points):
    """计算性能指标"""
    # 成功率（检测到有效轮廓）
    success = len(contours) > 0
    
    # 效率指标
    contours_per_second = len(contours) / execution_time if execution_time > 0 else 0
    points_per_second = num_points / execution_time if execution_time > 0 else 0
    
    return {
        'success': success,
        'contours_per_second': contours_per_second,
        'points_per_second': points_per_second,
        'efficiency_score': len(contours) / execution_time if execution_time > 0 else 0
    }

class PureBoundaryComparison:
    """纯数据边界追踪方法对比系统"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        self._genesis_initialized = False
        
        # 测试物体配置
        self.test_objects = [
            {
                'name': 'Cube',
                'type': 'box',
                'params': {'size': (0.105, 0.18, 0.022), 'pos': (0.35, 0, 0.02)},
                'file': None
            },
            {
                'name': 'tuoyuan.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': 'genesis/assets/_myobj/tuoyuan.obj'
            },
            {
                'name': 'buguize.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': 'genesis/assets/_myobj/buguize.obj'
            },
            {
                'name': 'withtumor.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': 'genesis/assets/_myobj/withtumor.obj'
            },
            {
                'name': 'ctry.obj',
                'type': 'mesh',
                'params': {'pos': (0.35, 0.0, 0.02), 'collision': True, 'fixed': True},
                'file': 'genesis/assets/_myobj/ctry.obj'
            }
        ]
        
        # 检测方法
        self.detection_methods = {
            'canny': detect_boundary_canny,
            'rgbd': detect_boundary_rgbd,
            'alpha_shape': detect_boundary_alpha_shape
        }
        
        # 如果Alpha-Shape不可用，移除该方法
        if not ALPHA_SHAPE_AVAILABLE:
            del self.detection_methods['alpha_shape']
            print("Alpha-Shape不可用，将只比较Canny和RGB-D方法")
        
        # 光照条件
        self.lighting_conditions = [
            {'name': 'normal', 'ambient': (0.1, 0.1, 0.1), 'directional': (0.5, 0.5, 0.5)},
            {'name': 'bright', 'ambient': (0.3, 0.3, 0.3), 'directional': (0.8, 0.8, 0.8)},
            {'name': 'dim', 'ambient': (0.05, 0.05, 0.05), 'directional': (0.2, 0.2, 0.2)}
        ]
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"evaluation_results/pure_boundary_comparison_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"纯数据边界追踪对比系统初始化完成")
        print(f"测试物体数量: {len(self.test_objects)}")
        print(f"检测方法数量: {len(self.detection_methods)}")
        print(f"光照条件数量: {len(self.lighting_conditions)}")
        print(f"总测试次数: {len(self.test_objects) * len(self.detection_methods) * len(self.lighting_conditions) * 5}")
        print(f"结果将保存到: {self.results_dir}")
    
    def _setup_scene(self, obj_config, lighting_config):
        """设置仿真场景"""
        # 只在第一次初始化Genesis
        if not hasattr(self, '_genesis_initialized'):
            gs.init(seed=0, precision="32", logging_level="info")
            self._genesis_initialized = True
        
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
            file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
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
    
    def _run_single_test(self, obj_config, method_name, lighting_config, run_id):
        """运行单次测试"""
        # 设置场景
        scene, ed6, cam, motors_dof_idx, j6_link = self._setup_scene(obj_config, lighting_config)
        
        # 移动到目标位置
        target_pos = np.array(obj_config['params']['pos'])
        plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam)
        
        # 执行边界检测
        start_time = time.time()
        
        try:
            detection_func = self.detection_methods[method_name]
            contours, rgb, depth = detection_func(cam)
            
            # 获取点云
            pc, _ = cam.render_pointcloud(world_frame=True)
            num_points = len(pc)
            
            execution_time = time.time() - start_time
            
            # 计算评估指标
            accuracy_metrics = calculate_accuracy_metrics(contours, rgb, depth, cam)
            performance_metrics = calculate_performance_metrics(contours, execution_time, num_points)
            
            # 记录结果
            result = {
                'object_name': obj_config['name'],
                'object_type': obj_config['type'],
                'method': method_name,
                'lighting': lighting_config['name'],
                'run_id': run_id,
                'execution_time': execution_time,
                'num_points': num_points,
                'success': performance_metrics['success'],
                **accuracy_metrics,
                **performance_metrics
            }
            
            return result, None
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                'object_name': obj_config['name'],
                'object_type': obj_config['type'],
                'method': method_name,
                'lighting': lighting_config['name'],
                'run_id': run_id,
                'execution_time': execution_time,
                'num_points': 0,
                'success': False,
                'error': str(e)
            }
            return result, str(e)
        finally:
            # 清理场景资源
            try:
                scene.clear()
            except:
                pass
    
    def run_comparison(self, num_runs=5):
        """运行完整对比测试"""
        print(f"\n{'='*80}")
        print("开始纯数据边界追踪方法对比测试")
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
                    
                    # 存储该条件下的执行时间用于稳定性分析
                    execution_times = []
                    
                    for run_id in range(num_runs):
                        current_test += 1
                        progress = (current_test / total_tests) * 100
                        
                        print(f"    运行 {run_id+1}/{num_runs} (总进度: {progress:.1f}%)", end=" ")
                        
                        # 检查ESC键
                        if esc_pressed:
                            print("\n用户中断测试")
                            return
                        
                        # 运行测试
                        result, error = self._run_single_test(obj_config, method_name, lighting_config, run_id)
                        
                        if error:
                            print(f"❌ 失败: {error}")
                        else:
                            print(f"✅ 成功 (时间: {result['execution_time']:.3f}s)")
                        
                        # 记录结果
                        self.results.append(result)
                        execution_times.append(result['execution_time'])
                        
                        # 短 
                        time.sleep(0.5)
                    
                    # 计算稳定性指标
                    stability_metrics = calculate_stability_metrics(execution_times)
                    
                    # 更新该条件下所有结果的平均稳定性指标
                    for result in self.results[-num_runs:]:
                        result.update(stability_metrics)
        
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
        # 保存详细结果CSV
        df = pd.DataFrame(self.results)
        csv_file = self.results_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False)
        
        # 保存JSON格式
        json_file = self.results_dir / "detailed_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"详细结果已保存到: {csv_file}")
        print(f"JSON结果已保存到: {json_file}")
    
    def _generate_analysis(self):
        """生成分析报告"""
        print("\n生成分析报告...")
        
        df = pd.DataFrame(self.results)
        
        # 生成汇总统计
        summary_stats = []
        
        for obj_name in df['object_name'].unique():
            for method in df['method'].unique():
                for lighting in df['lighting'].unique():
                    subset = df[(df['object_name'] == obj_name) & 
                               (df['method'] == method) & 
                               (df['lighting'] == lighting)]
                    
                    if len(subset) == 0:
                        continue
                    
                    summary = {
                        'object_name': obj_name,
                        'method': method,
                        'lighting': lighting,
                        'total_runs': len(subset),
                        'successful_runs': subset['success'].sum(),
                        'success_rate': subset['success'].mean(),
                        'avg_execution_time': subset['execution_time'].mean(),
                        'std_execution_time': subset['execution_time'].std(),
                        'avg_num_contours': subset['num_contours'].mean(),
                        'avg_contour_area': subset['avg_contour_area'].mean(),
                        'avg_coverage_ratio': subset['coverage_ratio'].mean(),
                        'avg_num_points': subset['num_points'].mean(),
                        'avg_efficiency_score': subset['efficiency_score'].mean()
                    }
                    summary_stats.append(summary)
        
        # 保存汇总统计
        summary_df = pd.DataFrame(summary_stats)
        summary_file = self.results_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # 生成方法对比统计
        method_comparison = []
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            comparison = {
                'method': method,
                'total_tests': len(method_data),
                'overall_success_rate': method_data['success'].mean(),
                'avg_execution_time': method_data['execution_time'].mean(),
                'std_execution_time': method_data['execution_time'].std(),
                'avg_num_contours': method_data['num_contours'].mean(),
                'avg_contour_area': method_data['avg_contour_area'].mean(),
                'avg_coverage_ratio': method_data['coverage_ratio'].mean(),
                'avg_efficiency_score': method_data['efficiency_score'].mean(),
                'stability_score': 1 - (method_data['execution_time'].std() / method_data['execution_time'].mean()) if method_data['execution_time'].mean() > 0 else 0
            }
            method_comparison.append(comparison)
        
        # 保存方法对比
        comparison_df = pd.DataFrame(method_comparison)
        comparison_file = self.results_dir / "method_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"汇总统计已保存到: {summary_file}")
        print(f"方法对比已保存到: {comparison_file}")
        
        # 生成文本报告
        self._generate_text_report(comparison_df)
    
    def _generate_text_report(self, comparison_df):
        """生成文本分析报告"""
        report_file = self.results_dir / "comparison_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("纯数据边界追踪方法对比报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试物体: {', '.join([obj['name'] for obj in self.test_objects])}\n")
            f.write(f"检测方法: {', '.join(comparison_df['method'].tolist())}\n")
            f.write(f"光照条件: {', '.join([light['name'] for light in self.lighting_conditions])}\n")
            f.write(f"总测试次数: {len(self.results)}\n\n")
            
            f.write("方法对比结果:\n")
            f.write("-" * 40 + "\n")
            
            for _, row in comparison_df.iterrows():
                f.write(f"\n方法: {row['method'].upper()}\n")
                f.write(f"  总测试次数: {row['total_tests']}\n")
                f.write(f"  整体成功率: {row['overall_success_rate']:.2%}\n")
                f.write(f"  平均执行时间: {row['avg_execution_time']:.3f}±{row['std_execution_time']:.3f}s\n")
                f.write(f"  平均轮廓数量: {row['avg_num_contours']:.1f}\n")
                f.write(f"  平均轮廓面积: {row['avg_contour_area']:.2f}\n")
                f.write(f"  平均覆盖率: {row['avg_coverage_ratio']:.3f}\n")
                f.write(f"  平均效率分数: {row['avg_efficiency_score']:.2f}\n")
                f.write(f"  稳定性分数: {row['stability_score']:.3f}\n")
            
            # 找出最佳方法
            f.write(f"\n\n最佳方法分析:\n")
            f.write("-" * 40 + "\n")
            
            best_success = comparison_df.loc[comparison_df['overall_success_rate'].idxmax(), 'method']
            fastest = comparison_df.loc[comparison_df['avg_execution_time'].idxmin(), 'method']
            most_stable = comparison_df.loc[comparison_df['stability_score'].idxmax(), 'method']
            most_efficient = comparison_df.loc[comparison_df['avg_efficiency_score'].idxmax(), 'method']
            
            f.write(f"最高成功率: {best_success}\n")
            f.write(f"最快执行速度: {fastest}\n")
            f.write(f"最稳定: {most_stable}\n")
            f.write(f"最高效率: {most_efficient}\n")
            
            # 综合评分
            f.write(f"\n综合评分 (成功率40% + 稳定性30% + 效率30%):\n")
            f.write("-" * 40 + "\n")
            
            comparison_df['综合评分'] = (
                comparison_df['overall_success_rate'] * 0.4 +
                comparison_df['stability_score'] * 0.3 +
                (comparison_df['avg_efficiency_score'] / comparison_df['avg_efficiency_score'].max()) * 0.3
            )
            
            # 按综合评分排序
            comparison_df_sorted = comparison_df.sort_values('综合评分', ascending=False)
            
            for i, (_, row) in enumerate(comparison_df_sorted.iterrows()):
                f.write(f"{i+1}. {row['method'].upper()}: {row['综合评分']:.3f}\n")
            
            best_overall = comparison_df_sorted.iloc[0]['method']
            f.write(f"\n推荐最佳方法: {best_overall}\n")
            f.write(f"推荐理由: 综合评分最高，在成功率、稳定性和效率方面表现最佳\n")
        
        print(f"分析报告已保存到: {report_file}")
        
        # 在终端显示总结
        self._display_summary(comparison_df)
    
    def _display_summary(self, comparison_df):
        """在终端显示对比总结"""
        print(f"\n{'='*80}")
        print("边界追踪方法对比总结")
        print(f"{'='*80}")
        
        print(f"\n测试概况:")
        print(f"  测试物体: {len(self.test_objects)} 个")
        print(f"  检测方法: {len(self.detection_methods)} 个")
        print(f"  光照条件: {len(self.lighting_conditions)} 种")
        print(f"  总测试次数: {len(self.results)} 次")
        
        print(f"\n各方法表现:")
        for _, row in comparison_df.iterrows():
            print(f"  {row['method'].upper()}:")
            print(f"    成功率: {row['overall_success_rate']:.1%}")
            print(f"    执行时间: {row['avg_execution_time']:.3f}±{row['std_execution_time']:.3f}s")
            print(f"    稳定性: {row['stability_score']:.3f}")
            print(f"    效率: {row['avg_efficiency_score']:.2f}")
        
        # 综合评分
        comparison_df['综合评分'] = (
            comparison_df['overall_success_rate'] * 0.4 +
            comparison_df['stability_score'] * 0.3 +
            (comparison_df['avg_efficiency_score'] / comparison_df['avg_efficiency_score'].max()) * 0.3
        )
        
        comparison_df_sorted = comparison_df.sort_values('综合评分', ascending=False)
        
        print(f"\n综合排名:")
        for i, (_, row) in enumerate(comparison_df_sorted.iterrows()):
            print(f"  {i+1}. {row['method'].upper()}: {row['综合评分']:.3f}")
        
        best_method = comparison_df_sorted.iloc[0]['method']
        print(f"\n🏆 推荐最佳方法: {best_method.upper()}")
        print(f"   该方法在成功率、稳定性和效率方面综合表现最佳")
        
        print(f"\n详细结果请查看: {self.results_dir}")
        print(f"{'='*80}")

def main():
    """主函数"""
    print("纯数据边界追踪方法对比系统")
    print("=" * 60)
    print("本系统将对比以下边界追踪方法：")
    print("1. Canny边缘检测")
    print("2. RGB-D边界检测")
    if ALPHA_SHAPE_AVAILABLE:
        print("3. Alpha-Shape凹包检测")
    else:
        print("3. Alpha-Shape凹包检测 (不可用)")
    print()
    print("测试配置：")
    print("- 测试物体: Cube, tuoyuan.obj, buguize.obj, withtumor.obj, ctry.obj")
    print("- 光照条件: normal, bright, dim")
    print("- 重复次数: 5次/方法/物体/光照")
    print("- 输出格式: 纯CSV数据")
    print()
    
    # 创建对比系统
    comparison = PureBoundaryComparison()
    
    # 运行对比测试
    comparison.run_comparison(num_runs=5)
    
    print("\n对比测试完成！")

if __name__ == "__main__":
    main()
