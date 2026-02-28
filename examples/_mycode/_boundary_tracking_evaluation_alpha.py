# 边界追踪方法评估脚本 - 包含Alpha-Shape方法
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
                    print("\n检测到ESC键，正在结束扫描...")
                    break
        else:  # Linux/Windows
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = input().strip().lower()
                    if key == 'esc' or 'q':
                        esc_pressed = True
                        print("\n检测到ESC键，正在结束扫描...")
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
    time.sleep(2)
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

# 导入边界检测方法
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

class BoundaryTrackingEvaluator:
    """边界追踪方法评估器"""
    
    def __init__(self, 
                 cube_size: np.ndarray = np.array([0.105, 0.18, 0.022]),
                 base_cube_pos: np.ndarray = np.array([0.35, 0.08, 0.02]),
                 experiment_name: str = "boundary_tracking_comparison_alpha"):
        
        self.cube_size = cube_size
        self.base_cube_pos = base_cube_pos
        self.experiment_name = experiment_name
        
        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"evaluation_results/{experiment_name}_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化结果存储
        self.results = {
            'canny': [],
            'rgbd': [],
            'alpha_shape': []
        }
        
        # 检测方法映射
        self.detection_methods = {
            'canny': detect_boundary_canny,
            'rgbd': detect_boundary_rgbd,
            'alpha_shape': detect_boundary_alpha_shape
        }
        
        # 如果Alpha-Shape不可用，移除该方法
        if not ALPHA_SHAPE_AVAILABLE:
            del self.detection_methods['alpha_shape']
            del self.results['alpha_shape']
            print("Alpha-Shape不可用，将只比较Canny和RGB-D方法")
        
        print(f"评估器初始化完成，结果将保存到: {self.results_dir}")
        print(f"将比较的方法: {list(self.detection_methods.keys())}")
    
    def _setup_scene(self, cube_pos: np.ndarray):
        """设置仿真场景"""
        gs.init(seed=0, precision="32", logging_level="info")
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
            show_viewer=False,  # 关闭可视化以提高速度
        )
        
        # 添加平面
        plane = scene.add_entity(gs.morphs.Plane(collision=True))
        
        # 添加物体
        cube = scene.add_entity(gs.morphs.Mesh(
            file="genesis/assets/_myobj/ctry.obj",
            pos=cube_pos,
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
    
    def _run_single_evaluation(self, method: str, cube_pos: np.ndarray, run_id: int):
        """运行单次评估"""
        print(f"\n开始评估 {method} 方法 (运行 {run_id+1})")
        
        # 设置场景
        scene, ed6, cam, motors_dof_idx, j6_link = self._setup_scene(cube_pos)
        
        # 移动到目标位置
        target_pos = cube_pos.copy()
        target_pos[2] += 0.2
        plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam)
        
        # 执行边界检测
        start_time = time.time()
        
        try:
            detection_func = self.detection_methods[method]
            contours, rgb, depth = detection_func(cam)
            
            # 计算检测指标
            num_contours = len(contours)
            total_contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
            avg_contour_area = total_contour_area / num_contours if num_contours > 0 else 0
            
            # 计算点云质量
            pc, _ = cam.render_pointcloud(world_frame=True)
            num_points = len(pc)
            point_density = num_points / (640 * 480)  # 点密度
            
            execution_time = time.time() - start_time
            
            # 评估结果
            result = {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'num_contours': num_contours,
                'total_contour_area': total_contour_area,
                'avg_contour_area': avg_contour_area,
                'num_points': num_points,
                'point_density': point_density,
                'execution_time': execution_time,
                'success': True,
                'error': None
            }
            
            print(f"  {method} 评估完成:")
            print(f"    轮廓数量: {num_contours}")
            print(f"    总轮廓面积: {total_contour_area:.2f}")
            print(f"    平均轮廓面积: {avg_contour_area:.2f}")
            print(f"    点云数量: {num_points}")
            print(f"    执行时间: {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            result = {
                'method': method,
                'run_id': run_id,
                'cube_pos': cube_pos.tolist(),
                'num_contours': 0,
                'total_contour_area': 0,
                'avg_contour_area': 0,
                'num_points': 0,
                'point_density': 0,
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
            print(f"  {method} 评估失败: {e}")
        
        return result
    
    def run_evaluation(self, num_runs: int = 5, num_positions: int = 3):
        """运行完整评估"""
        print(f"\n开始边界追踪方法评估")
        print(f"评估方法: {list(self.detection_methods.keys())}")
        print(f"每个方法运行次数: {num_runs}")
        print(f"测试位置数量: {num_positions}")
        
        # 生成测试位置
        positions = []
        for i in range(num_positions):
            pos = self.base_cube_pos.copy()
            pos[0] += (i - num_positions//2) * 0.05  # X方向偏移
            pos[1] += (i % 2) * 0.03  # Y方向偏移
            positions.append(pos)
        
        print(f"测试位置: {[pos.tolist() for pos in positions]}")
        
        # 运行评估
        for method in self.detection_methods.keys():
            print(f"\n{'='*50}")
            print(f"评估方法: {method.upper()}")
            print(f"{'='*50}")
            
            for pos_idx, cube_pos in enumerate(positions):
                print(f"\n位置 {pos_idx+1}/{num_positions}: {cube_pos}")
                
                for run_id in range(num_runs):
                    result = self._run_single_evaluation(method, cube_pos, run_id)
                    result['position_id'] = pos_idx
                    self.results[method].append(result)
                    
                    # 短暂休息
                    time.sleep(1)
        
        # 保存结果
        self._save_results()
        
        # 生成分析报告
        self._generate_analysis()
        
        print(f"\n评估完成！结果保存在: {self.results_dir}")
    
    def _save_results(self):
        """保存评估结果"""
        # 保存JSON格式的详细结果
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存CSV格式的汇总结果
        all_results = []
        for method, method_results in self.results.items():
            all_results.extend(method_results)
        
        df = pd.DataFrame(all_results)
        csv_file = self.results_dir / "results_summary.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"结果已保存到: {results_file} 和 {csv_file}")
    
    def _generate_analysis(self):
        """生成分析报告"""
        print("\n生成分析报告...")
        
        # 创建汇总数据
        summary_data = []
        for method, method_results in self.results.items():
            if not method_results:
                continue
                
            successful_runs = [r for r in method_results if r['success']]
            if not successful_runs:
                continue
            
            summary = {
                'method': method,
                'total_runs': len(method_results),
                'successful_runs': len(successful_runs),
                'success_rate': len(successful_runs) / len(method_results),
                'avg_contours': np.mean([r['num_contours'] for r in successful_runs]),
                'avg_contour_area': np.mean([r['avg_contour_area'] for r in successful_runs]),
                'avg_points': np.mean([r['num_points'] for r in successful_runs]),
                'avg_execution_time': np.mean([r['execution_time'] for r in successful_runs]),
                'std_execution_time': np.std([r['execution_time'] for r in successful_runs])
            }
            summary_data.append(summary)
        
        # 保存汇总数据
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # 生成可视化图表
        self._create_visualizations(summary_df)
        
        # 生成文本报告
        self._generate_text_report(summary_df)
        
        print(f"分析报告已生成")
    
    def _create_visualizations(self, summary_df):
        """创建可视化图表"""
        if len(summary_df) == 0:
            print("没有数据可以可视化")
            return
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('边界追踪方法比较分析', fontsize=16, fontweight='bold')
        
        # 1. 成功率比较
        axes[0, 0].bar(summary_df['method'], summary_df['success_rate'], 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('成功率比较')
        axes[0, 0].set_ylabel('成功率')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(summary_df['success_rate']):
            axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
        
        # 2. 平均轮廓数量比较
        axes[0, 1].bar(summary_df['method'], summary_df['avg_contours'],
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('平均轮廓数量比较')
        axes[0, 1].set_ylabel('轮廓数量')
        for i, v in enumerate(summary_df['avg_contours']):
            axes[0, 1].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
        
        # 3. 平均执行时间比较
        axes[1, 0].bar(summary_df['method'], summary_df['avg_execution_time'],
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 0].set_title('平均执行时间比较')
        axes[1, 0].set_ylabel('执行时间 (秒)')
        for i, v in enumerate(summary_df['avg_execution_time']):
            axes[1, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # 4. 平均点云数量比较
        axes[1, 1].bar(summary_df['method'], summary_df['avg_points'],
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('平均点云数量比较')
        axes[1, 1].set_ylabel('点云数量')
        for i, v in enumerate(summary_df['avg_points']):
            axes[1, 1].text(i, v + 100, f'{v:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        chart_file = self.results_dir / "comparison_charts.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到: {chart_file}")
    
    def _generate_text_report(self, summary_df):
        """生成文本报告"""
        report_file = self.results_dir / "evaluation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("边界追踪方法评估报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估方法: {', '.join(summary_df['method'].tolist())}\n")
            f.write(f"总运行次数: {summary_df['total_runs'].sum()}\n\n")
            
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"\n方法: {row['method'].upper()}\n")
                f.write(f"  成功率: {row['success_rate']:.2%}\n")
                f.write(f"  平均轮廓数量: {row['avg_contours']:.1f}\n")
                f.write(f"  平均轮廓面积: {row['avg_contour_area']:.2f}\n")
                f.write(f"  平均点云数量: {row['avg_points']:.0f}\n")
                f.write(f"  平均执行时间: {row['avg_execution_time']:.3f}s\n")
                f.write(f"  执行时间标准差: {row['std_execution_time']:.3f}s\n")
            
            # 找出最佳方法
            f.write(f"\n\n最佳方法分析:\n")
            f.write("-" * 30 + "\n")
            
            best_success = summary_df.loc[summary_df['success_rate'].idxmax(), 'method']
            fastest = summary_df.loc[summary_df['avg_execution_time'].idxmin(), 'method']
            most_contours = summary_df.loc[summary_df['avg_contours'].idxmax(), 'method']
            
            f.write(f"最高成功率: {best_success}\n")
            f.write(f"最快执行速度: {fastest}\n")
            f.write(f"最多轮廓检测: {most_contours}\n")
        
        print(f"文本报告已保存到: {report_file}")

def main():
    """主函数"""
    print("边界追踪方法评估系统")
    print("=" * 50)
    
    # 创建评估器
    evaluator = BoundaryTrackingEvaluator()
    
    # 运行评估
    evaluator.run_evaluation(num_runs=3, num_positions=2)
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()
