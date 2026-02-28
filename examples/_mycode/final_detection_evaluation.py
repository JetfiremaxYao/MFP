#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版物体检测评估系统
- 使用4种物体：ttt.obj, ctry.obj, tuoyuan.obj, 盒子
- 每次只检测一个物体
- 位置范围：0.3-0.8m半径
- 专注于检测时间对比
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
class FinalTrialResult:
    """最终版trial结果记录"""
    # 基本信息
    method: str
    object_type: str
    lighting: str
    trial_id: int
    seed: int
    
    # 物体信息
    object_position: List[float]  # 物体位置 [x, y, z]
    object_size: List[float]  # 物体尺寸 [x, y, z]
    
    # 检测结果
    detected_position: List[float]  # 检测到的位置
    detected_range: Tuple[float, float, float, float]  # 检测到的范围
    
    # 精度指标
    pos_error_mm: float  # 位置误差（毫米）
    range_error_mm: float  # 范围误差（毫米）
    
    # 时间指标
    detection_time_s: float  # 检测时间（秒）
    
    # 成功判定
    success_pos_2cm: bool  # 位置误差 < 2cm
    success_range_2cm: bool  # 范围误差 < 2cm
    success_both: bool  # 两个条件都满足

class FinalObjectDetectionEvaluator:
    """最终版物体检测评估器"""
    
    def __init__(self, 
                 min_radius: float = 0.3,
                 max_radius: float = 0.45,
                 experiment_name: str = "final_object_detection"):
        """
        初始化最终版评估器
        
        Parameters:
        -----------
        min_radius : float
            最小半径（米）
        max_radius : float
            最大半径（米）
        experiment_name : str
            实验名称
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.experiment_name = experiment_name
        
        # 创建结果目录
        self.results_dir = Path(f"evaluation_results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验配置
        self.n_trials_per_condition = 6  # 每组条件重复5次
        self.lighting_conditions = ["normal", "bright", "dim"]  # 三种光照条件
        
        # 物体类型配置
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        # 定义两个基础位置
        self.base_positions = [
            np.array([0.35, 0, 0.02]),   # 右侧位置
            np.array([-0.35, 0, 0.02])   # 左侧位置
        ]
        
        self.object_types = [
            {
                'name': 'cube',
                'size': np.array([0.105, 0.18, 0.022]),  # 与unified evaluation一致
                'type': 'box'
            }
        ]
        
        # 结果存储
        self.trial_results = []
        self.experiment_config = {}
        
        # 初始化评估器
        self.hsv_evaluator = HSVDepthEvaluator("HSV+Depth")
        self.clustering_evaluator = DepthClusteringEvaluator("Depth+Clustering")
    
    def _generate_random_position(self, seed: int, position_index: int) -> np.ndarray:
        """生成随机物体位置（基于基础位置+扰动）"""
        np.random.seed(seed)
        
        # 选择基础位置
        base_pos = self.base_positions[position_index % len(self.base_positions)]
        
        # 生成扰动：xy只能变大（正值），z不变
        x_perturb = np.random.uniform(0, 0.05)  # 0到5cm的正扰动
        y_perturb = np.random.uniform(0, 0.05)  # 0到5cm的正扰动
        z_perturb = 0.0  # z不变
        
        # 应用扰动
        position = base_pos.copy()
        position[0] += x_perturb
        position[1] += y_perturb
        position[2] += z_perturb
        
        return position
    
    def _setup_single_object_scene(self, object_config: Dict, position: np.ndarray, lighting: str = "normal"):
        """设置单物体场景"""
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
        
        # 添加单个物体
        if object_config['type'] == 'mesh':
            obj = scene.add_entity(gs.morphs.Mesh(
                file=object_config['file'],
                pos=position,
                collision=True,
                fixed=True
            ))
        elif object_config['type'] == 'box':
            obj = scene.add_entity(gs.morphs.Box(
                size=object_config['size'],
                pos=position,
                collision=True
            ))
        
        # 添加机械臂
        project_root = Path(__file__).parent.parent.parent
        urdf_path = project_root / 'genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf'
        
        ed6 = scene.add_entity(gs.morphs.URDF(
            file=str(urdf_path),
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
        
        return scene, ed6, cam, motors_dof_idx, j6_link, obj
    
    def _run_single_object_trial(self, method: str, object_config: Dict, position: np.ndarray,
                                lighting: str, trial_id: int) -> FinalTrialResult:
        """运行单物体检测trial"""
        print(f"\n=== 运行 {method.upper()} 方法 - {object_config['name']} - {lighting} - 第 {trial_id} 次trial ===")
        print(f"物体位置: {position}")
        
        # 设置随机种子确保可重复性
        seed = hash(f"{method}_{object_config['name']}_{lighting}_{trial_id}") % (2**32)
        np.random.seed(seed)
        
        # 计算真实范围（在try块外定义size变量）
        if object_config['type'] == 'mesh':
            # 对于mesh对象使用默认尺寸
            size = np.array([0.1, 0.1, 0.1])
        else:
            size = object_config['size']
        
        try:
            # 设置场景
            scene, ed6, cam, motors_dof_idx, j6_link, obj = self._setup_single_object_scene(
                object_config, position, lighting)
            
            true_range = self._calculate_true_range(position, size)
            
            # 初始化机械臂
            reset_arm(scene, ed6, motors_dof_idx)
            
            # 开始检测
            start_time = time.time()
            
            try:
                if method == 'hsv_depth':
                    detection_result = detect_hsv_depth(
                        scene, ed6, cam, motors_dof_idx,
                        true_position=position,
                        true_range=true_range,
                        lighting_condition=lighting,
                        background_complexity="simple"
                    )
                else:  # depth_clustering
                    detection_result = detect_depth_clustering(
                        scene, ed6, cam, motors_dof_idx,
                        true_position=position,
                        true_range=true_range,
                        lighting_condition=lighting,
                        background_complexity="simple"
                    )
                
                # 检查检测结果是否有效
                if detection_result is None or len(detection_result) != 3:
                    print(f"检测失败：返回结果无效")
                    raise RuntimeError("检测函数返回无效结果")
                
                detected_pos, detected_range, detection_info = detection_result
                
                # 检查检测到的位置是否有效
                if detected_pos is None or len(detected_pos) != 3:
                    print(f"检测失败：位置信息无效")
                    raise RuntimeError("检测位置信息无效")
                
                # 检查检测到的范围是否有效
                if detected_range is None or len(detected_range) != 4:
                    print(f"检测失败：范围信息无效")
                    raise RuntimeError("检测范围信息无效")
                    
            except Exception as detection_error:
                print(f"检测过程失败: {detection_error}")
                raise detection_error
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            # 计算误差
            pos_err = np.linalg.norm(detected_pos - position)
            dx, dy, range_err = self._calculate_range_error(true_range, detected_range)
            
            # 判断成功状态
            success_pos_2cm = pos_err < 0.02  # 2cm
            success_range_2cm = range_err < 0.02  # 2cm
            success_both = success_pos_2cm and success_range_2cm
            
            # 构建结果
            result = FinalTrialResult(
                method=method,
                object_type=object_config['name'],
                lighting=lighting,
                trial_id=trial_id,
                seed=seed,
                object_position=position.tolist(),
                object_size=size.tolist(),
                detected_position=detected_pos.tolist(),
                detected_range=detected_range,
                pos_error_mm=pos_err * 1000,  # 转换为毫米
                range_error_mm=range_err * 1000,  # 转换为毫米
                detection_time_s=detection_time,
                success_pos_2cm=success_pos_2cm,
                success_range_2cm=success_range_2cm,
                success_both=success_both
            )
            
            print(f"检测结果: 位置误差={pos_err*1000:.1f}mm, 范围误差={range_err*1000:.1f}mm, 时间={detection_time:.2f}s")
            print(f"成功状态: 位置2cm={success_pos_2cm}, 范围2cm={success_range_2cm}")
            
            # 平滑回零
            reset_after_detection(scene, ed6, motors_dof_idx)
            
            return result
            
        except Exception as e:
            print(f"Trial运行失败: {e}")
            # 返回失败结果
            return FinalTrialResult(
                method=method,
                object_type=object_config['name'],
                lighting=lighting,
                trial_id=trial_id,
                seed=seed,
                object_position=position.tolist(),
                object_size=size.tolist(),
                detected_position=[0, 0, 0],
                detected_range=(0, 0, 0, 0),
                pos_error_mm=float('inf'),
                range_error_mm=float('inf'),
                detection_time_s=0.0,
                success_pos_2cm=False,
                success_range_2cm=False,
                success_both=False
            )
        finally:
            # 清理资源
            if 'scene' in locals():
                del scene
            import gc
            gc.collect()  # 强制垃圾回收
    
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
    
    def run_final_evaluation(self):
        """运行最终版评估"""
        print(f"开始最终版物体检测方法评估")
        print(f"实验名称: {self.experiment_name}")
        print(f"结果保存目录: {self.results_dir}")
        print(f"物体类型: {[obj['name'] for obj in self.object_types]}")
        print(f"基础位置: {[pos.tolist() for pos in self.base_positions]}")
        print(f"扰动规则: xy只能变大(0-5cm), z不变")
        
        # 记录实验配置
        self.experiment_config = {
            'n_trials_per_condition': self.n_trials_per_condition,
            'lighting_conditions': self.lighting_conditions,
            'object_types': self.object_types,
            'base_positions': [pos.tolist() for pos in self.base_positions],
            'position_perturbation': 'xy_positive_only_0_to_5cm_z_fixed',
            'n_objects': len(self.object_types),
            'n_positions': len(self.base_positions)
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
        total_trials = len(self.object_types) * len(self.base_positions) * len(self.lighting_conditions) * self.n_trials_per_condition * 2
        current_trial = 0
        
        for obj_config in self.object_types:
            for pos_idx, base_pos in enumerate(self.base_positions):
                for lighting in self.lighting_conditions:
                    for trial_id in range(self.n_trials_per_condition):
                        # 生成随机位置（确保相同物体在相同条件下位置一致）
                        position_seed = hash(f"{obj_config['name']}_{pos_idx}_{lighting}_{trial_id}") % (2**32)
                        position = self._generate_random_position(position_seed, pos_idx)
                        
                        # 运行HSV+Depth方法
                        current_trial += 1
                        print(f"\n进度: {current_trial}/{total_trials}")
                        print(f"基础位置: {base_pos}, 实际位置: {position}")
                        hsv_result = self._run_single_object_trial('hsv_depth', obj_config, position, lighting, trial_id)
                        self.trial_results.append(hsv_result)
                        
                        # 运行Depth+Clustering方法
                        current_trial += 1
                        print(f"\n进度: {current_trial}/{total_trials}")
                        clustering_result = self._run_single_object_trial('depth_clustering', obj_config, position, lighting, trial_id)
                        self.trial_results.append(clustering_result)
        
        # 分析结果
        self._analyze_final_results()
        
        # 保存结果
        self._save_final_results()
        
        print(f"\n最终版实验完成！结果保存在: {self.results_dir}")
    
    def _analyze_final_results(self):
        """分析最终版实验结果"""
        print("\n=== 分析最终版实验结果 ===")
        
        # 按方法和物体类型分组统计
        for method in ['hsv_depth', 'depth_clustering']:
            method_results = [r for r in self.trial_results if r.method == method]
            print(f"\n{method.upper()}方法统计:")
            
            for obj_type in self.object_types:
                obj_name = obj_type['name']
                obj_results = [r for r in method_results if r.object_type == obj_name]
                
                if obj_results:
                    avg_time = np.mean([r.detection_time_s for r in obj_results])
                    avg_pos_err = np.mean([r.pos_error_mm for r in obj_results])
                    avg_range_err = np.mean([r.range_error_mm for r in obj_results])
                    success_rate = np.mean([r.success_both for r in obj_results])
                    
                    print(f"  {obj_name}:")
                    print(f"    平均检测时间: {avg_time:.2f}秒")
                    print(f"    平均位置误差: {avg_pos_err:.1f}mm")
                    print(f"    平均范围误差: {avg_range_err:.1f}mm")
                    print(f"    成功率: {success_rate:.1%}")
    
    def _save_final_results(self):
        """保存最终版实验结果"""
        import pandas as pd
        
        # 主要保存CSV格式，包含所有实验数据
        csv_file = self.results_dir / "final_trial_results.csv"
        
        # 转换结果为字典列表，确保所有数据类型正确
        results_data = []
        for result in self.trial_results:
            result_dict = asdict(result)
            
            # 处理特殊数据类型
            if 'object_position' in result_dict and isinstance(result_dict['object_position'], np.ndarray):
                result_dict['object_position'] = result_dict['object_position'].tolist()
            if 'object_size' in result_dict and isinstance(result_dict['object_size'], np.ndarray):
                result_dict['object_size'] = result_dict['object_size'].tolist()
            if 'detected_position' in result_dict and isinstance(result_dict['detected_position'], np.ndarray):
                result_dict['detected_position'] = result_dict['detected_position'].tolist()
            if 'detected_range' in result_dict and isinstance(result_dict['detected_range'], tuple):
                result_dict['detected_range'] = list(result_dict['detected_range'])
            
            # 确保所有numpy类型转换为Python原生类型
            for key, value in result_dict.items():
                if isinstance(value, np.integer):
                    result_dict[key] = int(value)
                elif isinstance(value, np.floating):
                    result_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result_dict[key] = value.tolist()
                elif isinstance(value, bool):
                    result_dict[key] = bool(value)
            
            results_data.append(result_dict)
        
        # 保存CSV
        df = pd.DataFrame(results_data)
        df.to_csv(csv_file, index=False)
        print(f"CSV结果已保存: {csv_file}")
        print(f"保存了 {len(df)} 条记录")
        
        # 保存简化的JSON配置信息
        config_file = self.results_dir / "experiment_config.json"
        config_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'n_trials_per_condition': self.n_trials_per_condition,
            'lighting_conditions': self.lighting_conditions,
            'object_types': [obj['name'] for obj in self.object_types],
            'total_trials': len(self.trial_results)
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"实验配置已保存: {config_file}")

def main():
    """主函数"""
    print("最终版物体检测方法评估系统")
    print("="*50)
    
    # 创建最终版评估器
    evaluator = FinalObjectDetectionEvaluator(
        min_radius=0.3,  # 30cm
        max_radius=0.45,  # 45cm
        experiment_name="final_object_detection"
    )
    
    # 运行最终版评估
    evaluator.run_final_evaluation()

if __name__ == "__main__":
    main()
