# 物体检测方法对比评估系统
# 专门用于对比HSV+Depth和Depth+Clustering两种方法
import genesis as gs
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import threading
import sys
import select
from sklearn.cluster import DBSCAN

# 导入两种检测方法
from _find_the_object_hsv+depth import (
    ObjectDetectionEvaluator as HSVDepthEvaluator,
    detect_cube_position as detect_hsv_depth,
    reset_arm, reset_after_detection
)
from _find_the_object_depth_clustering import (
    ObjectDetectionEvaluator as DepthClusteringEvaluator,
    detect_cube_position as detect_depth_clustering
)

class ObjectDetectionComparison:
    """物体检测方法对比评估器"""
    
    def __init__(self, 
                 cube_size: np.ndarray = np.array([0.105, 0.18, 0.022]),
                 base_cube_pos: np.ndarray = np.array([0.35, 0, 0.02]),
                 experiment_name: str = "hsv_depth_vs_clustering"):
        """
        初始化对比评估器
        
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
        self.n_runs = 10  # 重复实验次数
        self.position_perturbation = 0.02  # 位置扰动范围(m)
        self.height_perturbation = 0.005   # 高度扰动范围(m)
        
        # 评估参数
        self.success_threshold = 0.05  # 5cm成功阈值
        self.partial_success_threshold = 0.10  # 10cm部分成功阈值
        
        # 结果存储
        self.results = {
            'hsv_depth': [],
            'depth_clustering': [],
            'experiment_config': {},
            'hardware_info': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 硬件信息
        self.hardware_info = self._get_hardware_info()
        
        # 初始化评估器
        self.hsv_evaluator = HSVDepthEvaluator("HSV+Depth")
        self.clustering_evaluator = DepthClusteringEvaluator("Depth+Clustering")
        
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
            
            # 应用扰动
            perturbed_pos = self.base_cube_pos.copy()
            perturbed_pos[0] += x_perturb
            perturbed_pos[1] += y_perturb
            perturbed_pos[2] += z_perturb
            
            positions.append(perturbed_pos)
            
        return positions
    
    def _setup_scene(self, cube_pos: np.ndarray):
        """设置实验场景"""
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
        
        return scene, ed6, cam, motors_dof_idx, j6_link
    
    def _calculate_true_range(self, cube_pos: np.ndarray) -> Tuple[float, float, float, float]:
        """计算真实范围"""
        min_x = cube_pos[0] - self.cube_size[0]/2
        max_x = cube_pos[0] + self.cube_size[0]/2
        min_y = cube_pos[1] - self.cube_size[1]/2
        max_y = cube_pos[1] + self.cube_size[1]/2
        return (min_x, max_x, min_y, max_y)
    
    def _run_single_experiment(self, method: str, cube_pos: np.ndarray, 
                              run_id: int) -> Dict[str, Any]:
        """
        运行单次实验
        
        Parameters:
        -----------
        method : str
            检测方法 ('hsv_depth' 或 'depth_clustering')
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
        
        try:
            # 设置场景
            scene, ed6, cam, motors_dof_idx, j6_link = self._setup_scene(cube_pos)
            
            # 计算真实范围
            true_range = self._calculate_true_range(cube_pos)
            
            # 初始化机械臂
            reset_arm(scene, ed6, motors_dof_idx)
            
            # 开始检测
            start_time = time.time()
            
            if method == 'hsv_depth':
                detected_pos, detected_range = detect_hsv_depth(
                    scene, ed6, cam, motors_dof_idx,
                    true_position=cube_pos,
                    true_range=true_range
                )
            else:  # depth_clustering
                detected_pos, detected_range = detect_depth_clustering(
                    scene, ed6, cam, motors_dof_idx,
                    true_position=cube_pos,
                    true_range=true_range
                )
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            # 计算误差
            position_error = np.linalg.norm(detected_pos - cube_pos)
            range_error = self._calculate_range_error(true_range, detected_range)
            
            # 判断成功状态
            if position_error < self.success_threshold:
                status = 'Success'
            elif position_error < self.partial_success_threshold:
                status = 'Partial'
            else:
                status = 'Fail'
            
            # 构建结果
            result = {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'detected_pos': detected_pos.tolist(),
                'true_range': true_range,
                'detected_range': detected_range,
                'position_error': position_error,
                'range_error': range_error,
                'detection_time': detection_time,
                'status': status
            }
            
            print(f"检测结果: 位置误差={position_error:.4f}m, 状态={status}")
            
            # 平滑回零
            reset_after_detection(scene, ed6, motors_dof_idx)
            
            return result
            
        except Exception as e:
            print(f"实验运行失败: {e}")
            return {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'status': 'Fail',
                'error': str(e),
                'detection_time': 0.0,
                'position_error': float('inf'),
                'range_error': float('inf')
            }
        finally:
            # 清理资源
            if 'scene' in locals():
                del scene
    
    def _calculate_range_error(self, true_range: Tuple, detected_range: Tuple) -> float:
        """计算范围检测误差"""
        true_min_x, true_max_x, true_min_y, true_max_y = true_range
        detected_min_x, detected_max_x, detected_min_y, detected_max_y = detected_range
        
        # 计算X和Y方向的误差
        x_error = abs((true_max_x - true_min_x) - (detected_max_x - detected_min_x))
        y_error = abs((true_max_y - true_min_y) - (detected_max_y - detected_min_y))
        
        return np.sqrt(x_error**2 + y_error**2)
    
    def run_comparison_experiments(self):
        """运行对比实验"""
        print(f"开始物体检测方法对比实验")
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
            'success_threshold': self.success_threshold,
            'partial_success_threshold': self.partial_success_threshold
        }
        
        self.results['hardware_info'] = self.hardware_info
        
        # 初始化Genesis
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
            
            # 运行HSV+Depth方法
            hsv_result = self._run_single_experiment('hsv_depth', cube_pos, run_id)
            self.results['hsv_depth'].append(hsv_result)
            
            # 运行Depth+Clustering方法
            clustering_result = self._run_single_experiment('depth_clustering', cube_pos, run_id)
            self.results['depth_clustering'].append(clustering_result)
            
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
        hsv_success = sum(1 for r in self.results['hsv_depth'] if r['status'] == 'Success')
        hsv_partial = sum(1 for r in self.results['hsv_depth'] if r['status'] == 'Partial')
        hsv_fail = sum(1 for r in self.results['hsv_depth'] if r['status'] == 'Fail')
        
        clustering_success = sum(1 for r in self.results['depth_clustering'] if r['status'] == 'Success')
        clustering_partial = sum(1 for r in self.results['depth_clustering'] if r['status'] == 'Partial')
        clustering_fail = sum(1 for r in self.results['depth_clustering'] if r['status'] == 'Fail')
        
        print(f"HSV+Depth方法: 成功 {hsv_success}, 部分成功 {hsv_partial}, 失败 {hsv_fail}")
        print(f"Depth+Clustering方法: 成功 {clustering_success}, 部分成功 {clustering_partial}, 失败 {clustering_fail}")
        
        # 计算平均指标
        self._calculate_average_metrics()
    
    def _calculate_average_metrics(self):
        """计算平均指标"""
        # 过滤成功的运行
        hsv_successful = [r for r in self.results['hsv_depth'] if r['status'] in ['Success', 'Partial']]
        clustering_successful = [r for r in self.results['depth_clustering'] if r['status'] in ['Success', 'Partial']]
        
        if hsv_successful:
            hsv_avg = {
                'position_error': np.mean([r['position_error'] for r in hsv_successful]),
                'range_error': np.mean([r['range_error'] for r in hsv_successful]),
                'detection_time': np.mean([r['detection_time'] for r in hsv_successful]),
                'success_rate': len(hsv_successful) / len(self.results['hsv_depth'])
            }
            print(f"\nHSV+Depth方法平均指标 (成功运行):")
            print(f"  位置误差: {hsv_avg['position_error']*1000:.2f} mm")
            print(f"  范围误差: {hsv_avg['range_error']*1000:.2f} mm")
            print(f"  检测时间: {hsv_avg['detection_time']:.2f} s")
            print(f"  成功率: {hsv_avg['success_rate']:.2%}")
        
        if clustering_successful:
            clustering_avg = {
                'position_error': np.mean([r['position_error'] for r in clustering_successful]),
                'range_error': np.mean([r['range_error'] for r in clustering_successful]),
                'detection_time': np.mean([r['detection_time'] for r in clustering_successful]),
                'success_rate': len(clustering_successful) / len(self.results['depth_clustering'])
            }
            print(f"\nDepth+Clustering方法平均指标 (成功运行):")
            print(f"  位置误差: {clustering_avg['position_error']*1000:.2f} mm")
            print(f"  范围误差: {clustering_avg['range_error']*1000:.2f} mm")
            print(f"  检测时间: {clustering_avg['detection_time']:.2f} s")
            print(f"  成功率: {clustering_avg['success_rate']:.2%}")
    
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
            f.write("物体检测方法对比实验报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write(f"实验时间: {self.results['timestamp']}\n")
            f.write(f"重复次数: {self.n_runs}\n\n")
            
            f.write("实验配置:\n")
            f.write(f"  盒子尺寸: {self.cube_size}\n")
            f.write(f"  基础位置: {self.base_cube_pos}\n")
            f.write(f"  位置扰动: ±{self.position_perturbation}m\n")
            f.write(f"  高度扰动: ±{self.height_perturbation}m\n")
            f.write(f"  成功阈值: {self.success_threshold}m\n")
            f.write(f"  部分成功阈值: {self.partial_success_threshold}m\n\n")
            
            f.write("硬件信息:\n")
            for key, value in self.hardware_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 详细结果
            f.write("详细结果:\n")
            for method in ['hsv_depth', 'depth_clustering']:
                f.write(f"\n{method.upper()}方法:\n")
                for i, result in enumerate(self.results[method]):
                    f.write(f"  运行 {i+1}: {result['status']}\n")
                    if 'position_error' in result:
                        f.write(f"    位置误差: {result['position_error']*1000:.2f}mm\n")
                        f.write(f"    范围误差: {result['range_error']*1000:.2f}mm\n")
                        f.write(f"    检测时间: {result['detection_time']:.2f}s\n")
        
        print(f"实验报告已生成: {report_file}")
        
        # 生成可视化图表
        self._generate_visualization()
    
    def _generate_visualization(self):
        """生成可视化图表"""
        try:
            # 创建图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 准备数据
            hsv_data = self.results['hsv_depth']
            clustering_data = self.results['depth_clustering']
            
            # 1. 成功率对比
            hsv_success = sum(1 for r in hsv_data if r['status'] == 'Success')
            hsv_partial = sum(1 for r in hsv_data if r['status'] == 'Partial')
            hsv_fail = sum(1 for r in hsv_data if r['status'] == 'Fail')
            
            clustering_success = sum(1 for r in clustering_data if r['status'] == 'Success')
            clustering_partial = sum(1 for r in clustering_data if r['status'] == 'Partial')
            clustering_fail = sum(1 for r in clustering_data if r['status'] == 'Fail')
            
            x = np.arange(2)
            width = 0.25
            
            ax1.bar(x - width, [hsv_success, clustering_success], width, label='Success', color='green')
            ax1.bar(x, [hsv_partial, clustering_partial], width, label='Partial', color='orange')
            ax1.bar(x + width, [hsv_fail, clustering_fail], width, label='Fail', color='red')
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Count')
            ax1.set_title('Success Rate Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(['HSV+Depth', 'Depth+Clustering'])
            ax1.legend()
            
            # 2. 位置误差对比
            hsv_errors = [r['position_error']*1000 for r in hsv_data if r['status'] != 'Fail']
            clustering_errors = [r['position_error']*1000 for r in clustering_data if r['status'] != 'Fail']
            
            ax2.boxplot([hsv_errors, clustering_errors], labels=['HSV+Depth', 'Depth+Clustering'])
            ax2.set_ylabel('Position Error (mm)')
            ax2.set_title('Position Error Comparison')
            
            # 3. 检测时间对比
            hsv_times = [r['detection_time'] for r in hsv_data if r['status'] != 'Fail']
            clustering_times = [r['detection_time'] for r in clustering_data if r['status'] != 'Fail']
            
            ax3.boxplot([hsv_times, clustering_times], labels=['HSV+Depth', 'Depth+Clustering'])
            ax3.set_ylabel('Detection Time (s)')
            ax3.set_title('Detection Time Comparison')
            
            # 4. 范围误差对比
            hsv_range_errors = [r['range_error']*1000 for r in hsv_data if r['status'] != 'Fail']
            clustering_range_errors = [r['range_error']*1000 for r in clustering_data if r['status'] != 'Fail']
            
            ax4.boxplot([hsv_range_errors, clustering_range_errors], labels=['HSV+Depth', 'Depth+Clustering'])
            ax4.set_ylabel('Range Error (mm)')
            ax4.set_title('Range Error Comparison')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.results_dir / "comparison_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存: {chart_file}")
            
        except Exception as e:
            print(f"生成可视化图表失败: {e}")

def main():
    """主函数"""
    print("物体检测方法对比评估系统")
    print("=" * 50)
    
    # 创建评估器
    evaluator = ObjectDetectionComparison(
        cube_size=np.array([0.105, 0.18, 0.022]),
        base_cube_pos=np.array([0.35, 0, 0.02]),
        experiment_name="hsv_depth_vs_clustering"
    )
    
    # 运行对比实验
    evaluator.run_comparison_experiments()

if __name__ == "__main__":
    main()
