#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版物体检测评估系统
- 移除召回率指标
- 增加多种物体类型
- 随机化物体位置
- 专注于时间性能对比
"""

import genesis as gs
import numpy as np
import time
import json
import os
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# 导入两种检测方法
from _find_the_object_hsv_depth_fixed import (
    ObjectDetectionEvaluator as HSVDepthEvaluator,
    detect_cube_position as detect_hsv_depth,
    reset_arm, reset_after_detection
)
from _find_the_object_depth_clustering_fixed import (
    ObjectDetectionEvaluator as DepthClusteringEvaluator,
    detect_cube_position as detect_depth_clustering
)

@dataclass
class EnhancedTrialResult:
    """增强版trial结果记录"""
    # 基本信息
    method: str
    lighting: str
    trial_id: int
    seed: int
    
    # 物体信息
    target_objects: List[Dict]  # 目标物体列表
    detected_objects: List[Dict]  # 检测到的物体列表
    
    # 精度指标
    pos_errors: List[float]  # 各物体的位置误差
    range_errors: List[float]  # 各物体的范围误差
    
    # 时间指标
    total_detection_time_s: float  # 总检测时间
    first_object_time_s: float  # 找到第一个物体的时间
    all_objects_time_s: float  # 找到所有物体的时间
    avg_detection_time_per_object: float  # 平均每个物体的检测时间
    
    # 成功指标
    objects_found: int  # 找到的物体数量
    objects_total: int  # 总物体数量
    detection_success_rate: float  # 检测成功率
    
    # 详细时间记录
    time_breakdown: Dict[str, float]  # 时间分解

class EnhancedObjectDetectionEvaluator:
    """增强版物体检测评估器"""
    
    def __init__(self, 
                 base_radius: float = 0.3,
                 experiment_name: str = "enhanced_object_detection"):
        """
        初始化增强版评估器
        
        Parameters:
        -----------
        base_radius : float
            随机放置物体的半径范围
        experiment_name : str
            实验名称
        """
        self.base_radius = base_radius
        self.experiment_name = experiment_name
        
        # 创建结果目录
        self.results_dir = Path(f"evaluation_results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.n_trials_per_condition = 5  # 增加重复次数
        self.lighting_conditions = ["normal", "bright", "dim"]
        
        # 物体类型配置
        self.object_types = [
            {
                'type': 'box',
                'size': np.array([0.105, 0.18, 0.022]),
                'color': (0.8, 0.2, 0.2),  # 红色
                'name': 'red_box'
            },
            {
                'type': 'box', 
                'size': np.array([0.08, 0.08, 0.08]),
                'color': (0.2, 0.8, 0.2),  # 绿色
                'name': 'green_cube'
            },
            {
                'type': 'sphere',
                'size': np.array([0.06, 0.06, 0.06]),
                'color': (0.2, 0.2, 0.8),  # 蓝色
                'name': 'blue_sphere'
            },
            {
                'type': 'cylinder',
                'size': np.array([0.05, 0.05, 0.15]),
                'color': (0.8, 0.8, 0.2),  # 黄色
                'name': 'yellow_cylinder'
            }
        ]
        
        # 结果存储
        self.trial_results = []
        self.experiment_config = {}
        
        # 初始化评估器
        self.hsv_evaluator = HSVDepthEvaluator("HSV+Depth")
        self.clustering_evaluator = DepthClusteringEvaluator("Depth+Clustering")
    
    def _generate_random_positions(self, n_objects: int, seed: int) -> List[np.ndarray]:
        """生成随机物体位置"""
        np.random.seed(seed)
        positions = []
        
        for i in range(n_objects):
            # 在半径范围内随机生成位置
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.1, self.base_radius)
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = np.random.uniform(0.01, 0.05)  # 高度随机
            
            positions.append(np.array([x, y, z]))
        
        return positions
    
    def _setup_enhanced_scene(self, objects_config: List[Dict], lighting: str = "normal"):
        """设置增强版场景"""
        # 根据光照条件调整环境光
        if lighting == "normal":
            ambient_light = (0.1, 0.1, 0.1)
        elif lighting == "bright":
            ambient_light = (0.3, 0.3, 0.3)
        elif lighting == "dim":
            ambient_light = (0.03, 0.03, 0.03)
        else:
            ambient_light = (0.1, 0.1, 0.1)
        
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
                ambient_light=ambient_light,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=False,
                enable_collision=True,
                gravity=(0, 0, -9.81),
            ),
            show_viewer=False,
        )
        
        # 添加地面
        plane = scene.add_entity(gs.morphs.Plane(collision=True))
        
        # 添加多个物体
        added_objects = []
        for obj_config in objects_config:
            if obj_config['type'] == 'box':
                obj = scene.add_entity(gs.morphs.Box(
                    size=obj_config['size'],
                    pos=obj_config['position'],
                    collision=True,
                    color=obj_config['color']
                ))
            elif obj_config['type'] == 'sphere':
                obj = scene.add_entity(gs.morphs.Sphere(
                    radius=obj_config['size'][0]/2,
                    pos=obj_config['position'],
                    collision=True,
                    color=obj_config['color']
                ))
            elif obj_config['type'] == 'cylinder':
                obj = scene.add_entity(gs.morphs.Cylinder(
                    radius=obj_config['size'][0]/2,
                    height=obj_config['size'][2],
                    pos=obj_config['position'],
                    collision=True,
                    color=obj_config['color']
                ))
            
            added_objects.append({
                'entity': obj,
                'name': obj_config['name'],
                'position': obj_config['position'],
                'size': obj_config['size'],
                'type': obj_config['type']
            })
        
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
        
        return scene, ed6, cam, motors_dof_idx, j6_link, added_objects
    
    def _run_enhanced_trial(self, method: str, objects_config: List[Dict], 
                           lighting: str, trial_id: int) -> EnhancedTrialResult:
        """运行增强版单次trial"""
        print(f"\n=== 运行 {method.upper()} 方法 - {lighting} - 第 {trial_id} 次trial ===")
        print(f"物体数量: {len(objects_config)}")
        
        # 设置随机种子确保可重复性
        seed = hash(f"{method}_{lighting}_{trial_id}") % (2**32)
        np.random.seed(seed)
        
        try:
            # 设置场景
            scene, ed6, cam, motors_dof_idx, j6_link, added_objects = self._setup_enhanced_scene(
                objects_config, lighting)
            
            # 初始化机械臂
            reset_arm(scene, ed6, motors_dof_idx)
            
            # 开始检测
            start_time = time.time()
            first_object_time = None
            all_objects_time = None
            
            detected_objects = []
            pos_errors = []
            range_errors = []
            time_breakdown = {}
            
            # 这里需要根据实际的检测方法来修改
            # 暂时使用简化的检测逻辑
            if method == 'hsv_depth':
                # 使用HSV+Depth方法检测所有物体
                for i, obj in enumerate(added_objects):
                    obj_start_time = time.time()
                    
                    # 调用检测方法（需要修改以支持多物体检测）
                    detected_pos, detected_range, detection_info = detect_hsv_depth(
                        scene, ed6, cam, motors_dof_idx,
                        true_position=obj['position'],
                        true_range=self._calculate_true_range(obj['position'], obj['size']),
                        lighting_condition=lighting,
                        background_complexity="simple"
                    )
                    
                    obj_end_time = time.time()
                    obj_detection_time = obj_end_time - obj_start_time
                    
                    # 计算误差
                    pos_err = np.linalg.norm(detected_pos - obj['position'])
                    dx, dy, range_err = self._calculate_range_error(
                        self._calculate_true_range(obj['position'], obj['size']), 
                        detected_range
                    )
                    
                    detected_objects.append({
                        'name': obj['name'],
                        'detected_position': detected_pos,
                        'true_position': obj['position'],
                        'position_error': pos_err,
                        'range_error': range_err,
                        'detection_time': obj_detection_time
                    })
                    
                    pos_errors.append(pos_err)
                    range_errors.append(range_err)
                    
                    if first_object_time is None:
                        first_object_time = obj_detection_time
                    
                    time_breakdown[f'object_{i}_time'] = obj_detection_time
                
                all_objects_time = time.time() - start_time
                
            else:  # depth_clustering
                # 使用Depth+Clustering方法检测所有物体
                for i, obj in enumerate(added_objects):
                    obj_start_time = time.time()
                    
                    # 调用检测方法
                    detected_pos, detected_range, detection_info = detect_depth_clustering(
                        scene, ed6, cam, motors_dof_idx,
                        true_position=obj['position'],
                        true_range=self._calculate_true_range(obj['position'], obj['size']),
                        lighting_condition=lighting,
                        background_complexity="simple"
                    )
                    
                    obj_end_time = time.time()
                    obj_detection_time = obj_end_time - obj_start_time
                    
                    # 计算误差
                    pos_err = np.linalg.norm(detected_pos - obj['position'])
                    dx, dy, range_err = self._calculate_range_error(
                        self._calculate_true_range(obj['position'], obj['size']), 
                        detected_range
                    )
                    
                    detected_objects.append({
                        'name': obj['name'],
                        'detected_position': detected_pos,
                        'true_position': obj['position'],
                        'position_error': pos_err,
                        'range_error': range_err,
                        'detection_time': obj_detection_time
                    })
                    
                    pos_errors.append(pos_err)
                    range_errors.append(range_err)
                    
                    if first_object_time is None:
                        first_object_time = obj_detection_time
                    
                    time_breakdown[f'object_{i}_time'] = obj_detection_time
                
                all_objects_time = time.time() - start_time
            
            total_detection_time = time.time() - start_time
            
            # 构建结果
            result = EnhancedTrialResult(
                method=method,
                lighting=lighting,
                trial_id=trial_id,
                seed=seed,
                target_objects=[{
                    'name': obj['name'],
                    'position': obj['position'].tolist(),
                    'size': obj['size'].tolist(),
                    'type': obj['type']
                } for obj in added_objects],
                detected_objects=detected_objects,
                pos_errors=pos_errors,
                range_errors=range_errors,
                total_detection_time_s=total_detection_time,
                first_object_time_s=first_object_time if first_object_time else 0,
                all_objects_time_s=all_objects_time,
                avg_detection_time_per_object=total_detection_time / len(added_objects),
                objects_found=len(detected_objects),
                objects_total=len(added_objects),
                detection_success_rate=len(detected_objects) / len(added_objects),
                time_breakdown=time_breakdown
            )
            
            print(f"检测完成: 找到 {len(detected_objects)}/{len(added_objects)} 个物体")
            print(f"总检测时间: {total_detection_time:.2f}秒")
            
            # 平滑回零
            reset_after_detection(scene, ed6, motors_dof_idx)
            
            return result
            
        except Exception as e:
            print(f"Trial运行失败: {e}")
            # 返回失败结果
            return EnhancedTrialResult(
                method=method,
                lighting=lighting,
                trial_id=trial_id,
                seed=seed,
                target_objects=[],
                detected_objects=[],
                pos_errors=[],
                range_errors=[],
                total_detection_time_s=0.0,
                first_object_time_s=0.0,
                all_objects_time_s=0.0,
                avg_detection_time_per_object=0.0,
                objects_found=0,
                objects_total=len(objects_config),
                detection_success_rate=0.0,
                time_breakdown={}
            )
        finally:
            # 清理资源
            if 'scene' in locals():
                del scene
    
    def _calculate_true_range(self, position: np.ndarray, size: np.ndarray) -> Tuple[float, float, float, float]:
        """计算真实范围"""
        min_x = position[0] - size[0]/2
        max_x = position[0] + size[0]/2
        min_y = position[1] - size[1]/2
        max_y = position[1] + size[1]/2
        return (min_x, max_x, min_y, max_y)
    
    def _calculate_range_error(self, true_range: Tuple, detected_range: Tuple) -> Tuple[float, float, float]:
        """计算范围误差"""
        true_min_x, true_max_x, true_min_y, true_max_y = true_range
        detected_min_x, detected_max_x, detected_min_y, detected_max_y = detected_range
        
        dx = abs((true_max_x - true_min_x) - (detected_max_x - detected_min_x))
        dy = abs((true_max_y - true_min_y) - (detected_max_y - detected_min_y))
        range_err = np.sqrt(dx**2 + dy**2)
        
        return dx, dy, range_err
    
    def run_enhanced_evaluation(self):
        """运行增强版评估"""
        print(f"开始增强版物体检测方法评估")
        print(f"实验名称: {self.experiment_name}")
        print(f"结果保存目录: {self.results_dir}")
        
        # 记录实验配置
        self.experiment_config = {
            'n_trials_per_condition': self.n_trials_per_condition,
            'lighting_conditions': self.lighting_conditions,
            'object_types': self.object_types,
            'base_radius': self.base_radius,
            'n_objects_per_trial': len(self.object_types)
        }
        
        # 初始化Genesis
        try:
            gs.init(seed=0, precision="32", logging_level="info")
            print("Genesis初始化成功")
        except Exception as e:
            if "already initialized" in str(e):
                print("Genesis已初始化，继续执行")
            else:
                print(f"Genesis初始化失败: {e}")
                return
        
        # 运行实验
        total_trials = len(self.lighting_conditions) * self.n_trials_per_condition * 2
        current_trial = 0
        
        for lighting in self.lighting_conditions:
            for trial_id in range(self.n_trials_per_condition):
                # 生成随机物体配置
                positions = self._generate_random_positions(len(self.object_types), 42 + trial_id)
                objects_config = []
                for i, obj_type in enumerate(self.object_types):
                    obj_config = obj_type.copy()
                    obj_config['position'] = positions[i]
                    objects_config.append(obj_config)
                
                # 运行HSV+Depth方法
                current_trial += 1
                print(f"\n进度: {current_trial}/{total_trials}")
                hsv_result = self._run_enhanced_trial('hsv_depth', objects_config, lighting, trial_id)
                self.trial_results.append(hsv_result)
                
                # 运行Depth+Clustering方法
                current_trial += 1
                print(f"\n进度: {current_trial}/{total_trials}")
                clustering_result = self._run_enhanced_trial('depth_clustering', objects_config, lighting, trial_id)
                self.trial_results.append(clustering_result)
        
        # 分析结果
        self._analyze_enhanced_results()
        
        # 保存结果
        self._save_enhanced_results()
        
        print(f"\n增强版实验完成！结果保存在: {self.results_dir}")
    
    def _analyze_enhanced_results(self):
        """分析增强版实验结果"""
        print("\n=== 分析增强版实验结果 ===")
        
        # 按方法分组统计
        for method in ['hsv_depth', 'depth_clustering']:
            method_results = [r for r in self.trial_results if r.method == method]
            print(f"\n{method.upper()}方法统计:")
            
            if method_results:
                avg_total_time = np.mean([r.total_detection_time_s for r in method_results])
                avg_first_object_time = np.mean([r.first_object_time_s for r in method_results])
                avg_objects_found = np.mean([r.objects_found for r in method_results])
                avg_success_rate = np.mean([r.detection_success_rate for r in method_results])
                
                print(f"  平均总检测时间: {avg_total_time:.2f}秒")
                print(f"  平均找到第一个物体时间: {avg_first_object_time:.2f}秒")
                print(f"  平均找到物体数量: {avg_objects_found:.1f}/{len(self.object_types)}")
                print(f"  平均检测成功率: {avg_success_rate:.1%}")
    
    def _save_enhanced_results(self):
        """保存增强版实验结果"""
        # 保存JSON格式
        json_file = self.results_dir / "enhanced_experiment_results.json"
        results_dict = {
            'experiment_config': self.experiment_config,
            'timestamp': datetime.now().isoformat(),
            'trial_results': [asdict(result) for result in self.trial_results]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"增强版结果已保存: {json_file}")

def main():
    """主函数"""
    print("增强版物体检测方法评估系统")
    print("="*50)
    
    # 创建增强版评估器
    evaluator = EnhancedObjectDetectionEvaluator(
        base_radius=0.3,  # 30cm半径范围
        experiment_name="enhanced_object_detection"
    )
    
    # 运行增强版评估
    evaluator.run_enhanced_evaluation()

if __name__ == "__main__":
    main()

