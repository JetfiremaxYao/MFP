# 统一物体检测方法评估系统
# 按照标准化框架对比HSV+Depth和Depth+Clustering两种方法
import genesis as gs
import numpy as np
import time
import json
import os
import csv
from datetime import datetime
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示问题
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.cluster import DBSCAN

# 导入两种检测方法（修复版）
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
class TrialResult:
    """单次trial的结果记录"""
    # 基本信息
    method: str
    lighting: str
    background: str
    trial_id: int
    seed: int
    
    # 准确度指标
    pos_err_m: float  # 位置误差（米）
    dx_m: float      # X方向范围误差（米）
    dy_m: float      # Y方向范围误差（米）
    range_err_m: float  # 合成范围误差（米）
    
    # 成功判定
    success_pos_2cm: bool   # 位置误差 < 2cm
    success_range_2cm: bool # 范围误差 < 2cm
    success_pos_5cm: bool   # 位置误差 < 5cm（松标准）
    success_range_5cm: bool # 范围误差 < 5cm（松标准）
    
    # 速度/成本指标
    detection_time_s: float  # 检测时间（秒）
    frame_count: int         # 处理帧数
    fps: float              # 平均帧率
    
    # 召回/鲁棒性指标
    views_total: int        # 总视角数（24）
    views_hit: int          # 命中视角数
    early_stop_index: int   # 提前终止的视角索引
    view_hit_rate: float    # 视角召回率
    
    # 诊断信息
    best_angle_deg: float   # 最佳角度（度）
    best_score: float       # 最佳得分
    
    # 视角级详细信息（可选）
    view_details: Optional[List[Dict]] = None
    
    def to_dict(self):
        """转换为字典，处理不可序列化的类型"""
        result_dict = asdict(self)
        # 处理view_details中的不可序列化类型
        if result_dict['view_details'] is not None:
            # 确保view_details中的所有值都是可序列化的
            for detail in result_dict['view_details']:
                for key, value in detail.items():
                    if isinstance(value, np.integer):
                        detail[key] = int(value)
                    elif isinstance(value, np.floating):
                        detail[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        detail[key] = value.tolist()
        return result_dict

class UnifiedObjectDetectionEvaluator:
    """统一物体检测评估器"""
    
    def __init__(self, 
                 cube_size: np.ndarray = np.array([0.105, 0.18, 0.022]),
                 base_cube_pos: np.ndarray = np.array([0.35, 0, 0.02]),
                 experiment_name: str = "unified_object_detection"):
        """
        初始化统一评估器
        
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
        self.n_trials_per_condition = 3 # 每组条件重复次数
        self.lighting_conditions = ["normal", "bright", "dim"]
        self.background_conditions = ["simple"]  # 暂时只使用简单背景
        
        # 评估参数
        self.success_threshold_strict = 0.02  # 2cm严格阈值
        self.success_threshold_relaxed = 0.05  # 5cm宽松阈值
        
        # 结果存储
        self.trial_results = []
        self.experiment_config = {}
        self.hardware_info = {}
        
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
    
    def _setup_scene(self, cube_pos: np.ndarray, lighting: str = "normal", background: str = "simple"):
        """设置实验场景"""
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
            show_viewer=False,  # 关闭viewer避免计时污染
        )
        
        # 添加地面
        plane = scene.add_entity(gs.morphs.Plane(collision=True))
        
        # 添加目标物体
        cube = scene.add_entity(gs.morphs.Box(
            size=self.cube_size, 
            pos=cube_pos, 
            collision=True
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
    
    def _calculate_range_error(self, true_range: Tuple, detected_range: Tuple) -> Tuple[float, float, float]:
        """计算范围误差"""
        true_min_x, true_max_x, true_min_y, true_max_y = true_range
        detected_min_x, detected_max_x, detected_min_y, detected_max_y = detected_range
        
        # 计算X和Y方向的误差
        dx = abs((true_max_x - true_min_x) - (detected_max_x - detected_min_x))
        dy = abs((true_max_y - true_min_y) - (detected_max_y - detected_min_y))
        
        # 合成误差
        range_err = np.sqrt(dx**2 + dy**2)
        
        return dx, dy, range_err
    
    def _run_single_trial(self, method: str, cube_pos: np.ndarray, 
                         lighting: str, background: str, trial_id: int) -> TrialResult:
        """
        运行单次trial
        
        Parameters:
        -----------
        method : str
            检测方法 ('hsv_depth' 或 'depth_clustering')
        cube_pos : np.ndarray
            盒子位置
        lighting : str
            光照条件
        background : str
            背景条件
        trial_id : int
            trial ID
        
        Returns:
        --------
        TrialResult
            trial结果
        """
        print(f"\n=== 运行 {method.upper()} 方法 - {lighting}/{background} - 第 {trial_id} 次trial ===")
        print(f"盒子位置: {cube_pos}")
        
        # 设置随机种子确保可重复性
        seed = hash(f"{method}_{lighting}_{background}_{trial_id}") % (2**32)
        np.random.seed(seed)
        
        try:
            # 设置场景
            scene, ed6, cam, motors_dof_idx, j6_link = self._setup_scene(cube_pos, lighting, background)
            
            # 计算真实范围
            true_range = self._calculate_true_range(cube_pos)
            
            # 初始化机械臂
            reset_arm(scene, ed6, motors_dof_idx)
            
            # 开始检测
            start_time = time.time()
            
            if method == 'hsv_depth':
                detected_pos, detected_range, detection_info = detect_hsv_depth(
                    scene, ed6, cam, motors_dof_idx,
                    true_position=cube_pos,
                    true_range=true_range,
                    lighting_condition=lighting,
                    background_complexity=background
                )
            else:  # depth_clustering
                detected_pos, detected_range, detection_info = detect_depth_clustering(
                    scene, ed6, cam, motors_dof_idx,
                    true_position=cube_pos,
                    true_range=true_range,
                    lighting_condition=lighting,
                    background_complexity=background
                )
            
            end_time = time.time()
            detection_time = end_time - start_time
            
            # 计算误差
            pos_err = np.linalg.norm(detected_pos - cube_pos)
            dx, dy, range_err = self._calculate_range_error(true_range, detected_range)
            
            # 判断成功状态
            success_pos_2cm = pos_err < self.success_threshold_strict
            success_range_2cm = range_err < self.success_threshold_strict
            success_pos_5cm = pos_err < self.success_threshold_relaxed
            success_range_5cm = range_err < self.success_threshold_relaxed
            
            # 从检测信息中获取详细数据
            frame_count = detection_info['frame_count']
            views_total = detection_info['views_total']
            views_hit = detection_info['views_hit']
            early_stop_index = detection_info['early_stop_index']
            view_hit_rate = detection_info['view_hit_rate']
            best_angle_deg = detection_info['best_angle_deg']
            best_score = detection_info['best_score']
            view_details = detection_info['view_details']
            
            # 计算FPS
            fps = frame_count / detection_time if detection_time > 0 else 0
            
            # 构建结果
            result = TrialResult(
                method=method,
                lighting=lighting,
                background=background,
                trial_id=trial_id,
                seed=seed,
                pos_err_m=pos_err,
                dx_m=dx,
                dy_m=dy,
                range_err_m=range_err,
                success_pos_2cm=success_pos_2cm,
                success_range_2cm=success_range_2cm,
                success_pos_5cm=success_pos_5cm,
                success_range_5cm=success_range_5cm,
                detection_time_s=detection_time,
                frame_count=frame_count,
                fps=fps,
                views_total=views_total,
                views_hit=views_hit,
                early_stop_index=early_stop_index,
                view_hit_rate=view_hit_rate,
                best_angle_deg=best_angle_deg,
                best_score=best_score,
                view_details=view_details
            )
            
            print(f"检测结果: 位置误差={pos_err*1000:.1f}mm, 范围误差={range_err*1000:.1f}mm")
            print(f"成功状态: 位置2cm={success_pos_2cm}, 范围2cm={success_range_2cm}")
            
            # 平滑回零
            reset_after_detection(scene, ed6, motors_dof_idx)
            
            return result
            
        except Exception as e:
            print(f"Trial运行失败: {e}")
            # 返回失败结果
            return TrialResult(
                method=method,
                lighting=lighting,
                background=background,
                trial_id=trial_id,
                seed=seed,
                pos_err_m=float('inf'),
                dx_m=float('inf'),
                dy_m=float('inf'),
                range_err_m=float('inf'),
                success_pos_2cm=False,
                success_range_2cm=False,
                success_pos_5cm=False,
                success_range_5cm=False,
                detection_time_s=0.0,
                frame_count=0,
                fps=0.0,
                views_total=24,
                views_hit=0,
                early_stop_index=0,
                view_hit_rate=0.0,
                best_angle_deg=0.0,
                best_score=0.0,
                view_details=None
            )
        finally:
            # 清理资源
            if 'scene' in locals():
                del scene
    
    def run_comprehensive_evaluation(self):
        """运行综合评估"""
        print(f"开始统一物体检测方法评估")
        print(f"实验名称: {self.experiment_name}")
        print(f"结果保存目录: {self.results_dir}")
        
        # 记录实验配置
        self.experiment_config = {
            'n_trials_per_condition': self.n_trials_per_condition,
            'lighting_conditions': self.lighting_conditions,
            'background_conditions': self.background_conditions,
            'cube_size': self.cube_size.tolist(),
            'base_cube_pos': self.base_cube_pos.tolist(),
            'success_threshold_strict': self.success_threshold_strict,
            'success_threshold_relaxed': self.success_threshold_relaxed
        }
        
        self.hardware_info = self._get_hardware_info()
        
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
        total_trials = len(self.lighting_conditions) * len(self.background_conditions) * self.n_trials_per_condition * 2
        current_trial = 0
        
        for lighting in self.lighting_conditions:
            for background in self.background_conditions:
                for trial_id in range(self.n_trials_per_condition):
                    # 生成扰动位置
                    np.random.seed(42 + trial_id)
                    x_perturb = np.random.uniform(-0.02, 0.02)
                    y_perturb = np.random.uniform(-0.02, 0.02)
                    z_perturb = np.random.uniform(-0.005, 0.005)
                    
                    cube_pos = self.base_cube_pos.copy()
                    cube_pos[0] += x_perturb
                    cube_pos[1] += y_perturb
                    cube_pos[2] += z_perturb
                    
                    # 运行HSV+Depth方法
                    current_trial += 1
                    print(f"\n进度: {current_trial}/{total_trials}")
                    hsv_result = self._run_single_trial('hsv_depth', cube_pos, lighting, background, trial_id)
                    self.trial_results.append(hsv_result)
                    
                    # 运行Depth+Clustering方法
                    current_trial += 1
                    print(f"\n进度: {current_trial}/{total_trials}")
                    clustering_result = self._run_single_trial('depth_clustering', cube_pos, lighting, background, trial_id)
                    self.trial_results.append(clustering_result)
        
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
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame([result.to_dict() for result in self.trial_results])
        
        # 按方法和条件分组统计
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            print(f"\n{method.upper()}方法统计:")
            
            for lighting in self.lighting_conditions:
                lighting_df = method_df[method_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    print(f"  {lighting}光照条件:")
                    print(f"    位置误差: {lighting_df['pos_err_m'].mean()*1000:.2f}±{lighting_df['pos_err_m'].std()*1000:.2f}mm")
                    print(f"    范围误差: {lighting_df['range_err_m'].mean()*1000:.2f}±{lighting_df['range_err_m'].std()*1000:.2f}mm")
                    print(f"    检测时间: {lighting_df['detection_time_s'].mean():.2f}±{lighting_df['detection_time_s'].std():.2f}s")
                    print(f"    成功率(2cm): {lighting_df['success_pos_2cm'].mean():.1%}")
                    print(f"    视角召回率: {lighting_df['view_hit_rate'].mean():.1%}")
    
    def _save_results(self):
        """保存实验结果"""
        # 保存CSV格式
        csv_file = self.results_dir / "trial_results.csv"
        df = pd.DataFrame([result.to_dict() for result in self.trial_results])
        df.to_csv(csv_file, index=False)
        print(f"CSV结果已保存: {csv_file}")
        
        # 保存JSON格式
        json_file = self.results_dir / "experiment_results.json"
        results_dict = {
            'experiment_config': self.experiment_config,
            'hardware_info': self.hardware_info,
            'timestamp': datetime.now().isoformat(),
            'trial_results': [result.to_dict() for result in self.trial_results]
        }
        
        # 确保所有值都是JSON可序列化的
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):
                return obj
            else:
                return obj
        
        serializable_results = make_json_serializable(results_dict)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"JSON结果已保存: {json_file}")
    
    def _generate_report(self):
        """生成实验报告"""
        report_file = self.results_dir / "experiment_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("统一物体检测方法评估报告\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"实验名称: {self.experiment_name}\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总trial数: {len(self.trial_results)}\n\n")
            
            f.write("实验配置:\n")
            for key, value in self.experiment_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("硬件信息:\n")
            for key, value in self.hardware_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 详细结果统计
            df = pd.DataFrame([result.to_dict() for result in self.trial_results])
            
            f.write("详细结果统计:\n")
            for method in ['hsv_depth', 'depth_clustering']:
                f.write(f"\n{method.upper()}方法:\n")
                method_df = df[df['method'] == method]
                
                for lighting in self.lighting_conditions:
                    lighting_df = method_df[method_df['lighting'] == lighting]
                    if len(lighting_df) > 0:
                        f.write(f"  {lighting}光照条件:\n")
                        f.write(f"    位置误差: {lighting_df['pos_err_m'].mean()*1000:.2f}±{lighting_df['pos_err_m'].std()*1000:.2f}mm\n")
                        f.write(f"    范围误差: {lighting_df['range_err_m'].mean()*1000:.2f}±{lighting_df['range_err_m'].std()*1000:.2f}mm\n")
                        f.write(f"    检测时间: {lighting_df['detection_time_s'].mean():.2f}±{lighting_df['detection_time_s'].std():.2f}s\n")
                        f.write(f"    成功率(2cm): {lighting_df['success_pos_2cm'].mean():.1%}\n")
                        f.write(f"    视角召回率: {lighting_df['view_hit_rate'].mean():.1%}\n")
        
        print(f"实验报告已生成: {report_file}")
        
        # 生成可视化图表
        self._generate_visualization()
    
    def _generate_visualization(self):
        """生成可视化图表"""
        try:
            print("开始生成可视化图表...")
            
            # 检查数据
            if not self.trial_results:
                print("警告：没有trial结果数据，跳过可视化")
                return
            
            df = pd.DataFrame([result.to_dict() for result in self.trial_results])
            print(f"数据框形状: {df.shape}")
            print(f"数据列: {list(df.columns)}")
            print(f"方法类型: {df['method'].unique()}")
            print(f"光照条件: {df['lighting'].unique()}")
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('统一物体检测方法评估结果', fontsize=16, fontweight='bold')
            
            # 1. 位置误差箱线图
            print("绘制位置误差箱线图...")
            sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', ax=axes[0, 0])
            axes[0, 0].set_title('位置误差对比', fontsize=12)
            axes[0, 0].set_ylabel('位置误差 (m)', fontsize=10)
            axes[0, 0].set_xlabel('光照条件', fontsize=10)
            axes[0, 0].tick_params(axis='both', which='major', labelsize=9)
            
            # 2. 范围误差箱线图
            print("绘制范围误差箱线图...")
            sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', ax=axes[0, 1])
            axes[0, 1].set_title('范围误差对比', fontsize=12)
            axes[0, 1].set_ylabel('范围误差 (m)', fontsize=10)
            axes[0, 1].set_xlabel('光照条件', fontsize=10)
            axes[0, 1].tick_params(axis='both', which='major', labelsize=9)
            
            # 3. 检测时间箱线图
            print("绘制检测时间箱线图...")
            sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=axes[0, 2])
            axes[0, 2].set_title('检测时间对比', fontsize=12)
            axes[0, 2].set_ylabel('检测时间 (s)', fontsize=10)
            axes[0, 2].set_xlabel('光照条件', fontsize=10)
            axes[0, 2].tick_params(axis='both', which='major', labelsize=9)
            
            # 4. 成功率柱状图
            print("绘制成功率柱状图...")
            success_rates = []
            methods = []
            lightings = []
            
            for method in ['hsv_depth', 'depth_clustering']:
                for lighting in self.lighting_conditions:
                    subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
                    if len(subset) > 0:
                        success_rate = subset['success_pos_2cm'].mean()
                        success_rates.append(success_rate)
                        methods.append(method)
                        lightings.append(lighting)
            
            if success_rates:
                success_df = pd.DataFrame({
                    'method': methods,
                    'lighting': lightings,
                    'success_rate': success_rates
                })
                
                sns.barplot(data=success_df, x='lighting', y='success_rate', hue='method', ax=axes[1, 0])
                axes[1, 0].set_title('成功率对比 (2cm阈值)', fontsize=12)
                axes[1, 0].set_ylabel('成功率', fontsize=10)
                axes[1, 0].set_xlabel('光照条件', fontsize=10)
                axes[1, 0].tick_params(axis='both', which='major', labelsize=9)
            
            # 5. 视角召回率柱状图
            print("绘制视角召回率柱状图...")
            recall_rates = []
            methods = []
            lightings = []
            
            for method in ['hsv_depth', 'depth_clustering']:
                for lighting in self.lighting_conditions:
                    subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
                    if len(subset) > 0:
                        recall_rate = subset['view_hit_rate'].mean()
                        recall_rates.append(recall_rate)
                        methods.append(method)
                        lightings.append(lighting)
            
            if recall_rates:
                recall_df = pd.DataFrame({
                    'method': methods,
                    'lighting': lightings,
                    'recall_rate': recall_rates
                })
                
                sns.barplot(data=recall_df, x='lighting', y='recall_rate', hue='method', ax=axes[1, 1])
                axes[1, 1].set_title('视角召回率对比', fontsize=12)
                axes[1, 1].set_ylabel('召回率', fontsize=10)
                axes[1, 1].set_xlabel('光照条件', fontsize=10)
                axes[1, 1].tick_params(axis='both', which='major', labelsize=9)
            
            # 6. FPS对比
            print("绘制FPS对比图...")
            sns.boxplot(data=df, x='lighting', y='fps', hue='method', ax=axes[1, 2])
            axes[1, 2].set_title('FPS对比', fontsize=12)
            axes[1, 2].set_ylabel('FPS', fontsize=10)
            axes[1, 2].set_xlabel('光照条件', fontsize=10)
            axes[1, 2].tick_params(axis='both', which='major', labelsize=9)
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.results_dir / "evaluation_charts.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"可视化图表已保存: {chart_file}")
            
            # 生成额外的详细图表
            self._generate_detailed_charts(df)
            
        except Exception as e:
            print(f"生成可视化图表失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_detailed_charts(self, df):
        """生成详细的补充图表"""
        try:
            print("生成详细补充图表...")
            
            # 创建新的图表
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('详细性能分析', fontsize=16, fontweight='bold')
            
            # 1. 误差分布直方图
            for i, method in enumerate(['hsv_depth', 'depth_clustering']):
                method_df = df[df['method'] == method]
                if len(method_df) > 0:
                    axes[0, 0].hist(method_df['pos_err_m'] * 1000, alpha=0.7, label=method, bins=10)
            
            axes[0, 0].set_title('位置误差分布', fontsize=12)
            axes[0, 0].set_xlabel('位置误差 (mm)', fontsize=10)
            axes[0, 0].set_ylabel('频次', fontsize=10)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 检测时间vs误差散点图
            for method in ['hsv_depth', 'depth_clustering']:
                method_df = df[df['method'] == method]
                if len(method_df) > 0:
                    axes[0, 1].scatter(method_df['detection_time_s'], method_df['pos_err_m'] * 1000, 
                                     alpha=0.7, label=method)
            
            axes[0, 1].set_title('检测时间 vs 位置误差', fontsize=12)
            axes[0, 1].set_xlabel('检测时间 (s)', fontsize=10)
            axes[0, 1].set_ylabel('位置误差 (mm)', fontsize=10)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 成功率热力图
            success_matrix = []
            for method in ['hsv_depth', 'depth_clustering']:
                row = []
                for lighting in self.lighting_conditions:
                    subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
                    if len(subset) > 0:
                        success_rate = subset['success_pos_2cm'].mean()
                        row.append(success_rate)
                    else:
                        row.append(0.0)
                success_matrix.append(row)
            
            if success_matrix:
                sns.heatmap(success_matrix, 
                           xticklabels=self.lighting_conditions,
                           yticklabels=['HSV+Depth', 'Depth+Clustering'],
                           annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 0])
                axes[1, 0].set_title('成功率热力图 (2cm阈值)', fontsize=12)
            
            # 4. 综合性能雷达图
            # 计算各方法的综合性能指标
            performance_data = []
            method_names = []
            
            for method in ['hsv_depth', 'depth_clustering']:
                method_df = df[df['method'] == method]
                if len(method_df) > 0:
                    # 归一化各项指标 (0-1之间，1为最好)
                    accuracy = 1.0 / (1.0 + method_df['pos_err_m'].mean() * 1000 / 50)  # 50mm为基准
                    speed = 1.0 / (1.0 + method_df['detection_time_s'].mean() / 30)  # 30s为基准
                    success_rate = method_df['success_pos_2cm'].mean()
                    recall_rate = method_df['view_hit_rate'].mean()
                    
                    performance_data.append([accuracy, speed, success_rate, recall_rate])
                    method_names.append(method)
            
            if performance_data:
                # 雷达图
                categories = ['精度', '速度', '成功率', '召回率']
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # 闭合
                
                ax = axes[1, 1]
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                for i, (data, name) in enumerate(zip(performance_data, method_names)):
                    data += data[:1]  # 闭合
                    ax.plot(angles, data, 'o-', linewidth=2, label=name, alpha=0.7)
                    ax.fill(angles, data, alpha=0.1)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_ylim(0, 1)
                ax.set_title('综合性能雷达图', fontsize=12)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
            
            plt.tight_layout()
            
            # 保存详细图表
            detailed_chart_file = self.results_dir / "detailed_analysis_charts.png"
            plt.savefig(detailed_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"详细分析图表已保存: {detailed_chart_file}")
            
        except Exception as e:
            print(f"生成详细图表失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("统一物体检测方法评估系统")
    print("="*50)
    
    # 创建评估器
    evaluator = UnifiedObjectDetectionEvaluator(
        cube_size=np.array([0.105, 0.18, 0.022]),
        base_cube_pos=np.array([0.35, 0, 0.02]),
        experiment_name="unified_object_detection"
    )
    
    # 运行综合评估
    evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()
