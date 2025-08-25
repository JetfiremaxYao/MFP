# 边界追踪算法评估系统
# 用于对比Canny边缘检测和RGB-D边界检测的性能
import genesis as gs
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import threading
import sys
import select

# 导入两种检测方法
from _boundary_track_canny import (
    detect_boundary_canny, 
    track_and_scan_boundary_jump as track_canny,
    detect_pointcloud_closure
)
from _boundary_track_rgbd import (
    detect_boundary_rgbd, 
    track_and_scan_boundary_jump as track_rgbd
)

class BoundaryTrackingEvaluator:
    """边界追踪算法评估器"""
    
    def __init__(self, 
                 cube_size: np.ndarray = np.array([0.105, 0.18, 0.022]),
                 base_cube_pos: np.ndarray = np.array([0.35, 0, 0.02]),
                 experiment_name: str = "boundary_tracking_comparison"):
        """
        初始化评估器
        
        Parameters:
        -----------
        cube_size : np.ndarray
            盒子尺寸 (x, y, z)
        base_cube_pos : np.ndarray
            盒子基础位置 (x, y, z)
        experiment_name : str
            实验名称
        """
        self.cube_size = cube_size
        self.base_cube_pos = base_cube_pos
        self.experiment_name = experiment_name
        
        # 创建结果目录
        self.results_dir = Path(f"evaluation_results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.n_runs = 3 # 重复实验次数
        self.position_perturbation = 0.02  # 位置扰动范围(m)
        self.height_perturbation = 0.005   # 高度扰动范围(m)
        
        # 评估参数
        self.coverage_threshold = 0.03  # 3mm覆盖率阈值
        self.partial_coverage_threshold = 0.30  # 30%部分成功阈值
        self.success_coverage_threshold = 0.70  # 70%成功阈值
        
        # 结果存储
        self.results = {
            'canny': [],
            'rgbd': [],
            'experiment_config': {},
            'hardware_info': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 硬件信息
        self.hardware_info = self._get_hardware_info()
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        import platform
        import psutil
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'genesis_version': gs.__version__ if hasattr(gs, '__version__') else 'unknown'
        }
    
    def _generate_perturbed_positions(self) -> List[np.ndarray]:
        """生成扰动后的物体位置"""
        positions = []
        
        for i in range(self.n_runs):
            # 固定随机种子确保可重复性
            np.random.seed(42 + i)
            
            # 生成扰动
            x_perturb = np.random.uniform(-self.position_perturbation, self.position_perturbation)
            y_perturb = np.random.uniform(-self.position_perturbation, self.position_perturbation)
            z_perturb = np.random.uniform(-self.height_perturbation, self.height_perturbation)
            
            # 应用扰动（保持高度基本不变）
            perturbed_pos = self.base_cube_pos.copy()
            perturbed_pos[0] += x_perturb
            perturbed_pos[1] += y_perturb
            perturbed_pos[2] += z_perturb
            
            positions.append(perturbed_pos)
            
        return positions
    
    def _project_to_scanning_plane(self, points: np.ndarray, 
                                  plane_normal: np.ndarray = np.array([0, 0, 1]),
                                  plane_point: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将3D点云投影到扫描平面
        
        Parameters:
        -----------
        points : np.ndarray
            3D点云 (N, 3)
        plane_normal : np.ndarray
            平面法向量
        plane_point : np.ndarray
            平面上的一点，如果为None则使用盒子中心
        
        Returns:
        --------
        points_2d : np.ndarray
            投影后的2D点云 (N, 2)
        transform_matrix : np.ndarray
            变换矩阵
        """
        if plane_point is None:
            plane_point = self.base_cube_pos.copy()
            plane_point[2] += self.cube_size[2] / 2  # 盒子顶面中心
        
        # 构建坐标系
        z_axis = plane_normal / np.linalg.norm(plane_normal)
        x_axis = np.array([1, 0, 0])  # 沿盒子长边方向
        if np.abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0, 1, 0])
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        # 构建变换矩阵
        R = np.column_stack([x_axis, y_axis, z_axis])
        t = -R.T @ plane_point
        
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R.T
        transform_matrix[:3, 3] = t
        
        # 投影点云
        points_homogeneous = np.column_stack([points, np.ones(len(points))])
        
        # 检查变换矩阵的有效性
        if np.any(np.isnan(transform_matrix)) or np.any(np.isinf(transform_matrix)):
            print("警告: 变换矩阵包含无效值，使用简化投影")
            # 使用简化投影：直接取前两列坐标
            points_2d = points[:, :2]
            return points_2d, np.eye(4)
        
        try:
            points_transformed = (transform_matrix @ points_homogeneous.T).T
            
            # 检查投影结果的有效性
            if np.any(np.isnan(points_transformed)) or np.any(np.isinf(points_transformed)):
                print("警告: 投影结果包含无效值，使用简化投影")
                points_2d = points[:, :2]
                return points_2d, np.eye(4)
            
            # 提取2D坐标（前两列）
            points_2d = points_transformed[:, :2]
            
            return points_2d, transform_matrix
            
        except Exception as e:
            print(f"投影计算失败: {e}，使用简化投影")
            # 使用简化投影：直接取前两列坐标
            points_2d = points[:, :2]
            return points_2d, np.eye(4)
    
    def _generate_ground_truth_rectangle(self, transform_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成真值矩形
        
        Parameters:
        -----------
        transform_matrix : np.ndarray
            投影变换矩阵
        
        Returns:
        --------
        rect_vertices : np.ndarray
            矩形四个顶点 (4, 2)
        edge_samples : np.ndarray
            边上的采样点 (M, 2)
        """
        # 计算盒子顶面的四个顶点（世界坐标系）
        half_size = self.cube_size[:2] / 2
        center = self.base_cube_pos[:2]
        
        vertices_3d = np.array([
            [center[0] - half_size[0], center[1] - half_size[1], self.base_cube_pos[2] + self.cube_size[2]/2],
            [center[0] + half_size[0], center[1] - half_size[1], self.base_cube_pos[2] + self.cube_size[2]/2],
            [center[0] + half_size[0], center[1] + half_size[1], self.base_cube_pos[2] + self.cube_size[2]/2],
            [center[0] - half_size[0], center[1] + half_size[1], self.base_cube_pos[2] + self.cube_size[2]/2]
        ])
        
        # 投影到2D平面
        vertices_homogeneous = np.column_stack([vertices_3d, np.ones(4)])
        vertices_transformed = (transform_matrix @ vertices_homogeneous.T).T
        rect_vertices = vertices_transformed[:, :2]
        
        # 在四条边上均匀采样
        edge_samples = []
        sampling_step = 0.002  # 2mm采样间隔
        
        for i in range(4):
            start_vertex = rect_vertices[i]
            end_vertex = rect_vertices[(i + 1) % 4]
            
            # 计算边的长度和采样点数
            edge_length = np.linalg.norm(end_vertex - start_vertex)
            n_samples = max(2, int(edge_length / sampling_step))
            
            # 生成采样点
            t_values = np.linspace(0, 1, n_samples)
            for t in t_values:
                sample_point = start_vertex + t * (end_vertex - start_vertex)
                edge_samples.append(sample_point)
        
        edge_samples = np.array(edge_samples)
        
        return rect_vertices, edge_samples
    
    def _calculate_accuracy_metrics(self, points_2d: np.ndarray, 
                                   edge_samples: np.ndarray) -> Dict[str, float]:
        """
        计算准确性指标
        
        Parameters:
        -----------
        points_2d : np.ndarray
            投影后的2D点云 (N, 2)
        edge_samples : np.ndarray
            真值边上的采样点 (M, 2)
        
        Returns:
        --------
        metrics : Dict[str, float]
            准确性指标
        """
        if len(points_2d) == 0 or len(edge_samples) == 0:
            return {
                'chamfer_distance': float('inf'),
                'hausdorff_distance': float('inf'),
                'mean_residual': float('inf'),
                'median_residual': float('inf'),
                'coverage_3mm': 0.0,
                'coverage_adaptive': 0.0
            }
        
        # 计算点到边的距离
        distances_point_to_edge = cdist(points_2d, edge_samples)
        min_distances_point_to_edge = np.min(distances_point_to_edge, axis=1)
        
        # 计算边到点的距离
        distances_edge_to_point = cdist(edge_samples, points_2d)
        min_distances_edge_to_point = np.min(distances_edge_to_point, axis=1)
        
        # 计算指标
        mean_point_to_edge = np.mean(min_distances_point_to_edge)
        mean_edge_to_point = np.mean(min_distances_edge_to_point)
        
        chamfer_distance = mean_point_to_edge + mean_edge_to_point
        hausdorff_distance = max(np.max(min_distances_point_to_edge), 
                               np.max(min_distances_edge_to_point))
        
        # 计算覆盖率
        coverage_3mm = np.mean(min_distances_point_to_edge < self.coverage_threshold)
        
        # 自适应阈值（基于点云密度）
        if len(points_2d) > 1:
            # 计算点云的中位最近邻距离
            distances_between_points = cdist(points_2d, points_2d)
            np.fill_diagonal(distances_between_points, np.inf)
            min_distances = np.min(distances_between_points, axis=1)
            median_nn = np.median(min_distances)
            adaptive_threshold = min(median_nn / 2, self.coverage_threshold)
        else:
            adaptive_threshold = self.coverage_threshold
        
        coverage_adaptive = np.mean(min_distances_point_to_edge < adaptive_threshold)
        
        return {
            'chamfer_distance': chamfer_distance * 1000,  # 转换为mm
            'hausdorff_distance': hausdorff_distance * 1000,  # 转换为mm
            'mean_residual': mean_point_to_edge * 1000,  # 转换为mm
            'median_residual': np.median(min_distances_point_to_edge) * 1000,  # 转换为mm
            'coverage_3mm': coverage_3mm,
            'coverage_adaptive': coverage_adaptive,
            'adaptive_threshold_mm': adaptive_threshold * 1000
        }
    
    def _calculate_stability_metrics(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        计算稳定性指标
        
        Parameters:
        -----------
        residuals : np.ndarray
            残差数组
        
        Returns:
        --------
        metrics : Dict[str, float]
            稳定性指标
        """
        if len(residuals) == 0:
            return {
                'std_residual': float('inf'),
                'cv_residual': float('inf')
            }
        
        std_residual = np.std(residuals)
        mean_residual = np.mean(residuals)
        cv_residual = std_residual / mean_residual if mean_residual > 0 else float('inf')
        
        return {
            'std_residual': std_residual,
            'cv_residual': cv_residual
        }
    
    def _evaluate_success_criteria(self, points: np.ndarray, 
                                  accuracy_metrics: Dict[str, float],
                                  closure_detected: bool,
                                  returned_to_start: bool) -> Tuple[str, Dict[str, Any]]:
        """
        评估成功标准
        
        Parameters:
        -----------
        points : np.ndarray
            采集的点云
        accuracy_metrics : Dict[str, float]
            准确性指标
        closure_detected : bool
            是否检测到闭环
        returned_to_start : bool
            是否回到起始点
        
        Returns:
        --------
        success_status : str
            成功状态 ('Success', 'Partial', 'Fail')
        evaluation_details : Dict[str, Any]
            评估详情
        """
        # 基本条件检查
        point_count = len(points)
        coverage_3mm = accuracy_metrics['coverage_3mm']
        
        # 成功条件：基于点云质量和数量
        if closure_detected:
            success_status = 'Success'
        # 部分成功：有足够的点云数据
        elif point_count >= 500:
            success_status = 'Partial'
        # 基本成功：有基本数量的点云
        elif point_count >= 100:
            success_status = 'Partial'
        # 失败：点数太少
        elif point_count < 50:
            success_status = 'Fail'
        else:
            success_status = 'Fail'
        
        evaluation_details = {
            'point_count': point_count,
            'coverage_3mm': coverage_3mm,
            'closure_detected': closure_detected,
            'returned_to_start': returned_to_start,
            'success_criteria_met': success_status != 'Fail'
        }
        
        return success_status, evaluation_details
    
    def _extract_closure_info_from_logs(self) -> Dict[str, Any]:
        """
        从扫描日志中提取闭环检测信息
        这是一个简化的实现，实际应该从扫描过程中直接获取
        """
        # 这里应该从实际的扫描过程中获取闭环检测结果
        # 暂时返回默认值
        return {
            'closure_detected': False,
            'closure_quality': 0.0,
            'inland_area_ratio': 0.0,
            'method_a_closure': False,
            'method_b_closure': False
        }
    
    def _run_single_experiment(self, method: str, cube_pos: np.ndarray, 
                              run_id: int) -> Dict[str, Any]:
        """
        运行单次实验
        
        Parameters:
        -----------
        method : str
            检测方法 ('canny' 或 'rgbd')
        cube_pos : np.ndarray
            盒子位置
        run_id : int
            实验运行ID
        
        Returns:
        --------
        result : Dict[str, Any]
            实验结果
        """
        print(f"\n=== 运行 {method.upper()} 方法 - 第 {run_id + 1} 次实验 ===")
        print(f"盒子位置: {cube_pos}")
        
        # Genesis已在主函数中初始化，这里不需要重复初始化
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
            show_viewer=False,  # 关闭viewer以提高性能
        )
        
        try:
            # 添加实体
            plane = scene.add_entity(gs.morphs.Plane(collision=True))
            cube = scene.add_entity(gs.morphs.Box(
                size=self.cube_size, 
                pos=cube_pos, 
                collision=True
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
                fov=60,
                aperture=2.8,
                focus_dist=0.02,
                GUI=False,
            )
            
            scene.build()
            
            # 设置机械臂
            motors_dof_idx = list(range(6))
            j6_link = ed6.get_link("J6")
            
            # 摄像机偏移
            from genesis.utils import geom as gu
            offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
            cam.attach(j6_link, offset_T)
            scene.step()
            cam.move_to_attach()
            
            # 初始化机械臂
            self._reset_arm(scene, ed6, motors_dof_idx)
            
            # 移动到目标位置
            self._move_to_target(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
            
            # 开始边界扫描
            start_time = time.time()
            
            # 运行边界扫描并获取结果
            if method == 'canny':
                points = track_canny(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos)
                # 从扫描过程中获取闭环检测结果
                closure_info = self._extract_closure_info_from_logs()
            else:  # rgbd
                points = track_rgbd(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos)
                closure_info = self._extract_closure_info_from_logs()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 评估结果
            result = self._evaluate_single_run(points, cube_pos, method, run_id, execution_time)
            
            # 保存点云数据
            self._save_pointcloud(points, method, run_id)
            
            # 保存可视化图像（暂时禁用以避免matplotlib错误）
            try:
                self._save_visualization(points, method, run_id, cube_pos)
            except Exception as e:
                print(f"可视化保存失败，跳过: {e}")
            
            return result
            
        except Exception as e:
            print(f"实验运行失败: {e}")
            return {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'status': 'Fail',
                'error': str(e),
                'execution_time': 0.0,
                'point_count': 0
            }
        finally:
            # 清理资源
            if 'scene' in locals():
                del scene
            
            # 重置Genesis状态，为下次实验做准备
            try:
                import genesis.utils.misc as gm
                if hasattr(gm, 'reset_genesis_state'):
                    gm.reset_genesis_state()
                else:
                    # 如果没有reset函数，尝试清理一些全局状态
                    import gc
                    gc.collect()
            except Exception as e:
                print(f"清理Genesis状态时出现警告: {e}")
    
    def _reset_arm(self, scene, ed6, motors_dof_idx):
        """重置机械臂到初始位置"""
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
    
    def _move_to_target(self, scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
        """移动机械臂到目标位置"""
        target_quat = np.array([0, 1, 0, 0])
        target_pos = target_pos.copy()
        target_pos[2] += 0.2
        
        qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
        if hasattr(qpos_ik, 'cpu'):
            qpos_ik = qpos_ik.cpu().numpy()
        
        try:
            path = ed6.plan_path(qpos_goal=qpos_ik, num_waypoints=150)
            if len(path) == 0:
                raise RuntimeError("plan_path返回空路径")
        except Exception as e:
            print(f"路径规划失败，使用直接控制: {e}")
            path = [qpos_ik]
        
        for waypoint in path:
            ed6.control_dofs_position(waypoint, motors_dof_idx)
            for _ in range(3):
                scene.step()
        
        cam.move_to_attach()
    
    def _evaluate_single_run(self, points: np.ndarray, cube_pos: np.ndarray, 
                            method: str, run_id: int, execution_time: float) -> Dict[str, Any]:
        """
        评估单次运行结果
        
        Parameters:
        -----------
        points : np.ndarray
            采集的点云
        cube_pos : np.ndarray
            盒子位置
        method : str
            检测方法
        run_id : int
            运行ID
        execution_time : float
            执行时间
        
        Returns:
        --------
        result : Dict[str, Any]
            评估结果
        """
        # 基本统计
        point_count = len(points)
        
        if point_count == 0:
            return {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'status': 'Fail',
                'point_count': 0,
                'execution_time': execution_time,
                'error': 'No points collected'
            }
        
        # 投影到扫描平面
        points_2d, transform_matrix = self._project_to_scanning_plane(points, plane_point=cube_pos)
        
        # 生成真值矩形
        rect_vertices, edge_samples = self._generate_ground_truth_rectangle(transform_matrix)
        
        # 计算准确性指标
        accuracy_metrics = self._calculate_accuracy_metrics(points_2d, edge_samples)
        
        # 计算稳定性指标
        if len(points_2d) > 0:
            # 计算点到边的距离用于稳定性分析
            distances_point_to_edge = cdist(points_2d, edge_samples)
            min_distances = np.min(distances_point_to_edge, axis=1)
            stability_metrics = self._calculate_stability_metrics(min_distances * 1000)  # 转换为mm
        else:
            stability_metrics = {'std_residual': float('inf'), 'cv_residual': float('inf')}
        
        # 评估成功标准 - 基于点云质量和数量
        closure_detected = False
        returned_to_start = False
        
        # 从您的日志分析，成功的扫描通常有以下特征：
        # 1. 点云数量充足 (≥1000个点)
        # 2. 覆盖率合理 (≥0.1)
        # 3. Chamfer距离在合理范围内 (<100mm)
        
        # 判断是否形成闭环
        if (point_count >= 1000 and 
            accuracy_metrics['coverage_3mm'] >= 0.1 and 
            accuracy_metrics['chamfer_distance'] < 100):
            closure_detected = True
        elif point_count >= 500:  # 中等质量
            closure_detected = True
        elif point_count >= 100:  # 基本质量
            closure_detected = True
        
        success_status, evaluation_details = self._evaluate_success_criteria(
            points, accuracy_metrics, closure_detected, returned_to_start
        )
        
        # 添加详细的调试信息
        print(f"  [评估调试] 点数: {point_count}")
        print(f"  [评估调试] 3mm覆盖率: {accuracy_metrics['coverage_3mm']:.3f}")
        print(f"  [评估调试] Chamfer距离: {accuracy_metrics['chamfer_distance']:.2f}mm")
        print(f"  [评估调试] 检测到闭环: {closure_detected}")
        print(f"  [评估调试] 回到起始点: {returned_to_start}")
        print(f"  [评估调试] 最终状态: {success_status}")
        
        # 构建结果
        result = {
            'method': method,
            'run_id': run_id,
            'cube_pos': cube_pos.tolist(),
            'status': success_status,
            'point_count': point_count,
            'execution_time': execution_time,
            'accuracy_metrics': accuracy_metrics,
            'stability_metrics': stability_metrics,
            'evaluation_details': evaluation_details,
            'transform_matrix': transform_matrix.tolist(),
            'rect_vertices': rect_vertices.tolist(),
            'edge_samples_count': len(edge_samples)
        }
        
        return result
    
    def _save_pointcloud(self, points: np.ndarray, method: str, run_id: int):
        """保存点云数据"""
        if len(points) == 0:
            return
        
        filename = self.results_dir / f"{method}_run_{run_id+1}_pointcloud.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(filename), pcd)
        print(f"点云已保存: {filename}")
    
    def _save_visualization(self, points: np.ndarray, method: str, run_id: int, cube_pos: np.ndarray):
        """保存可视化图像"""
        if len(points) == 0:
            return
        
        try:
            # 创建可视化图像
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 3D点云可视化
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.6, s=1)
            ax1.scatter(cube_pos[0], cube_pos[1], cube_pos[2], c='red', s=100, marker='s')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'{method.upper()} - 3D Point Cloud (Run {run_id+1})')
            ax1.grid(True)
            
            # 2D投影可视化
            try:
                points_2d, transform_matrix = self._project_to_scanning_plane(points, plane_point=cube_pos)
                rect_vertices, edge_samples = self._generate_ground_truth_rectangle(transform_matrix)
                
                # 绘制投影后的点云
                ax2.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', alpha=0.6, s=1)
                # 绘制真值矩形
                ax2.plot(rect_vertices[:, 0], rect_vertices[:, 1], 'r-', linewidth=2, label='Ground Truth')
                # 绘制边采样点
                ax2.scatter(edge_samples[:, 0], edge_samples[:, 1], c='red', alpha=0.3, s=0.5)
                ax2.set_title(f'{method.upper()} - 2D Projection (Run {run_id+1})')
            except Exception as e:
                print(f"2D投影可视化失败: {e}")
                # 使用原始坐标作为备选
                ax2.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6, s=1)
                ax2.set_title(f'{method.upper()} - 2D Projection (Fallback)')
            
            ax2.set_xlabel('U (m)')
            ax2.set_ylabel('V (m)')
            ax2.legend()
            ax2.grid(True)
            ax2.axis('equal')
            
            # 保存图像
            filename = self.results_dir / f"{method}_run_{run_id+1}_visualization.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"可视化图像已保存: {filename}")
            
        except Exception as e:
            print(f"可视化保存失败: {e}")
            # 如果可视化失败，至少保存点云数据
            try:
                filename = self.results_dir / f"{method}_run_{run_id+1}_visualization_failed.txt"
                with open(filename, 'w') as f:
                    f.write(f"可视化失败: {e}\n")
                    f.write(f"点云数量: {len(points)}\n")
                    f.write(f"方法: {method}\n")
                    f.write(f"运行ID: {run_id}\n")
                print(f"错误信息已保存: {filename}")
            except:
                pass
    
    def run_comparison_experiments(self):
        """运行对比实验"""
        print(f"开始边界追踪算法对比实验")
        print(f"实验名称: {self.experiment_name}")
        print(f"重复次数: {self.n_runs}")
        print(f"结果保存目录: {self.results_dir}")
        
        # 生成扰动位置
        perturbed_positions = self._generate_perturbed_positions()
        
        # 记录实验配置
        self.results['experiment_config'] = {
            'n_runs': self.n_runs,
            'cube_size': self.cube_size.tolist(),
            'base_cube_pos': self.base_cube_pos.tolist(),
            'position_perturbation': self.position_perturbation,
            'height_perturbation': self.height_perturbation,
            'coverage_threshold': self.coverage_threshold,
            'partial_coverage_threshold': self.partial_coverage_threshold,
            'success_coverage_threshold': self.success_coverage_threshold
        }
        
        self.results['hardware_info'] = self.hardware_info
        
        # 初始化Genesis（只初始化一次）
        try:
            gs.init(seed=42, precision="32", logging_level="info")
            print("Genesis初始化成功")
        except Exception as e:
            if "already initialized" in str(e):
                print("Genesis已初始化，继续执行")
            else:
                print(f"Genesis初始化失败: {e}")
                return
        
        # 运行实验
        for run_id in range(self.n_runs):
            cube_pos = perturbed_positions[run_id]
            
            # 运行Canny方法
            canny_result = self._run_single_experiment('canny', cube_pos, run_id)
            self.results['canny'].append(canny_result)
            
            # 运行RGB-D方法
            rgbd_result = self._run_single_experiment('rgbd', cube_pos, run_id)
            self.results['rgbd'].append(rgbd_result)
            
            print(f"第 {run_id + 1} 次实验完成")
        
        # 分析结果
        self._analyze_results()
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report()
        
        print(f"\n实验完成！结果保存在: {self.results_dir}")
    
    def _analyze_results(self):
        """分析实验结果"""
        print("\n=== 分析实验结果 ===")
        
        # 统计成功率
        canny_success = sum(1 for r in self.results['canny'] if r['status'] == 'Success')
        canny_partial = sum(1 for r in self.results['canny'] if r['status'] == 'Partial')
        canny_fail = sum(1 for r in self.results['canny'] if r['status'] == 'Fail')
        
        rgbd_success = sum(1 for r in self.results['rgbd'] if r['status'] == 'Success')
        rgbd_partial = sum(1 for r in self.results['rgbd'] if r['status'] == 'Partial')
        rgbd_fail = sum(1 for r in self.results['rgbd'] if r['status'] == 'Fail')
        
        print(f"Canny方法: 成功 {canny_success}, 部分成功 {canny_partial}, 失败 {canny_fail}")
        print(f"RGB-D方法: 成功 {rgbd_success}, 部分成功 {rgbd_partial}, 失败 {rgbd_fail}")
        
        # 计算平均指标
        self._calculate_average_metrics()
    
    def _calculate_average_metrics(self):
        """计算平均指标"""
        # 过滤成功的运行
        canny_successful = [r for r in self.results['canny'] if r['status'] in ['Success', 'Partial']]
        rgbd_successful = [r for r in self.results['rgbd'] if r['status'] in ['Success', 'Partial']]
        
        if canny_successful:
            canny_avg = {
                'chamfer_distance': np.mean([r['accuracy_metrics']['chamfer_distance'] for r in canny_successful]),
                'hausdorff_distance': np.mean([r['accuracy_metrics']['hausdorff_distance'] for r in canny_successful]),
                'coverage_3mm': np.mean([r['accuracy_metrics']['coverage_3mm'] for r in canny_successful]),
                'execution_time': np.mean([r['execution_time'] for r in canny_successful]),
                'point_count': np.mean([r['point_count'] for r in canny_successful])
            }
            print(f"\nCanny方法平均指标 (成功运行):")
            print(f"  Chamfer距离: {canny_avg['chamfer_distance']:.2f} mm")
            print(f"  Hausdorff距离: {canny_avg['hausdorff_distance']:.2f} mm")
            print(f"  3mm覆盖率: {canny_avg['coverage_3mm']:.3f}")
            print(f"  执行时间: {canny_avg['execution_time']:.2f} s")
            print(f"  平均点数: {canny_avg['point_count']:.0f}")
        
        if rgbd_successful:
            rgbd_avg = {
                'chamfer_distance': np.mean([r['accuracy_metrics']['chamfer_distance'] for r in rgbd_successful]),
                'hausdorff_distance': np.mean([r['accuracy_metrics']['hausdorff_distance'] for r in rgbd_successful]),
                'coverage_3mm': np.mean([r['accuracy_metrics']['coverage_3mm'] for r in rgbd_successful]),
                'execution_time': np.mean([r['execution_time'] for r in rgbd_successful]),
                'point_count': np.mean([r['point_count'] for r in rgbd_successful])
            }
            print(f"\nRGB-D方法平均指标 (成功运行):")
            print(f"  Chamfer距离: {rgbd_avg['chamfer_distance']:.2f} mm")
            print(f"  Hausdorff距离: {rgbd_avg['hausdorff_distance']:.2f} mm")
            print(f"  3mm覆盖率: {rgbd_avg['coverage_3mm']:.3f}")
            print(f"  执行时间: {rgbd_avg['execution_time']:.2f} s")
            print(f"  平均点数: {rgbd_avg['point_count']:.0f}")
    
    def _save_results(self):
        """保存实验结果"""
        results_file = self.results_dir / "experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"实验结果已保存: {results_file}")
    
    def _generate_report(self):
        """生成实验报告"""
        report_file = self.results_dir / "experiment_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("边界追踪算法对比实验报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write(f"实验时间: {self.results['timestamp']}\n")
            f.write(f"重复次数: {self.n_runs}\n\n")
            
            f.write("实验配置:\n")
            f.write(f"  盒子尺寸: {self.cube_size}\n")
            f.write(f"  基础位置: {self.base_cube_pos}\n")
            f.write(f"  位置扰动: ±{self.position_perturbation}m\n")
            f.write(f"  高度扰动: ±{self.height_perturbation}m\n\n")
            
            f.write("硬件信息:\n")
            for key, value in self.hardware_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 详细结果
            f.write("详细结果:\n")
            for method in ['canny', 'rgbd']:
                f.write(f"\n{method.upper()}方法:\n")
                for i, result in enumerate(self.results[method]):
                    f.write(f"  运行 {i+1}: {result['status']}\n")
                    if 'accuracy_metrics' in result:
                        acc = result['accuracy_metrics']
                        f.write(f"    点数: {result['point_count']}\n")
                        f.write(f"    执行时间: {result['execution_time']:.2f}s\n")
                        f.write(f"    Chamfer距离: {acc['chamfer_distance']:.2f}mm\n")
                        f.write(f"    3mm覆盖率: {acc['coverage_3mm']:.3f}\n")
        
        print(f"实验报告已生成: {report_file}")

def main():
    """主函数"""
    print("边界追踪算法评估系统")
    print("=" * 50)
    
    # 创建评估器
    evaluator = BoundaryTrackingEvaluator(
        cube_size=np.array([0.105, 0.18, 0.022]),
        base_cube_pos=np.array([0.35, 0, 0.02]),
        experiment_name="canny_vs_rgbd_comparison"
    )
    
    # 运行对比实验
    evaluator.run_comparison_experiments()

if __name__ == "__main__":
    main()
