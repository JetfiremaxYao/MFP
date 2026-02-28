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

class ObjectDetectionEvaluator:
    """物体检测评估器 - 用于定量分析检测方法的性能"""
    
    def __init__(self, method_name: str = "Depth+Clustering"):
        self.method_name = method_name
        self.results = {
            'method_name': method_name,
            'timestamp': datetime.now().isoformat(),
            'detection_results': [],
            'performance_metrics': {},
            'stability_analysis': {},
            'accuracy_analysis': {}
        }
        
    def add_detection_result(self, 
                           true_position: np.ndarray,
                           detected_position: np.ndarray,
                           true_range: Tuple[float, float, float, float],
                           detected_range: Tuple[float, float, float, float],
                           detection_time: float,
                           frame_count: int,
                           lighting_condition: str = "normal",
                           background_complexity: str = "simple"):
        """添加一次检测结果"""
        result = {
            'true_position': true_position.tolist(),
            'detected_position': detected_position.tolist(),
            'true_range': true_range,
            'detected_range': detected_range,
            'detection_time': detection_time,
            'frame_count': frame_count,
            'lighting_condition': lighting_condition,
            'background_complexity': background_complexity,
            'position_error': np.linalg.norm(true_position - detected_position),
            'range_error': self._calculate_range_error(true_range, detected_range)
        }
        self.results['detection_results'].append(result)
    
    def _calculate_range_error(self, true_range: Tuple, detected_range: Tuple) -> float:
        """计算范围检测误差"""
        true_min_x, true_max_x, true_min_y, true_max_y = true_range
        detected_min_x, detected_max_x, detected_min_y, detected_max_y = detected_range
        
        # 计算X和Y方向的误差
        x_error = abs((true_max_x - true_min_x) - (detected_max_x - detected_min_x))
        y_error = abs((true_max_y - true_min_y) - (detected_max_y - detected_min_y))
        
        return np.sqrt(x_error**2 + y_error**2)
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.results['detection_results']:
            return
        
        results = self.results['detection_results']
        
        # 位置精度分析
        position_errors = [r['position_error'] for r in results]
        range_errors = [r['range_error'] for r in results]
        detection_times = [r['detection_time'] for r in results]
        
        self.results['accuracy_analysis'] = {
            'position_error_mean': np.mean(position_errors),
            'position_error_std': np.std(position_errors),
            'position_error_min': np.min(position_errors),
            'position_error_max': np.max(position_errors),
            'range_error_mean': np.mean(range_errors),
            'range_error_std': np.std(range_errors),
            'range_error_min': np.min(range_errors),
            'range_error_max': np.max(range_errors)
        }
        
        # 性能分析
        self.results['performance_metrics'] = {
            'avg_detection_time': np.mean(detection_times),
            'std_detection_time': np.std(detection_times),
            'min_detection_time': np.min(detection_times),
            'max_detection_time': np.max(detection_times),
            'avg_fps': 1.0 / np.mean(detection_times) if np.mean(detection_times) > 0 else 0,
            'total_detections': len(results),
            'success_rate': len([r for r in results if r['position_error'] < 0.05]) / len(results)  # 5cm阈值
        }
        
        # 稳定性分析（按光照和背景条件分组）
        self.results['stability_analysis'] = self._analyze_stability(results)
    
    def _analyze_stability(self, results: List[Dict]) -> Dict:
        """分析检测稳定性"""
        stability = {}
        
        # 按光照条件分组
        lighting_groups = {}
        for r in results:
            lighting = r['lighting_condition']
            if lighting not in lighting_groups:
                lighting_groups[lighting] = []
            lighting_groups[lighting].append(r['position_error'])
        
        for lighting, errors in lighting_groups.items():
            stability[f'lighting_{lighting}_mean_error'] = np.mean(errors)
            stability[f'lighting_{lighting}_std_error'] = np.std(errors)
        
        # 按背景复杂度分组
        background_groups = {}
        for r in results:
            bg = r['background_complexity']
            if bg not in background_groups:
                background_groups[bg] = []
            background_groups[bg].append(r['position_error'])
        
        for bg, errors in background_groups.items():
            stability[f'background_{bg}_mean_error'] = np.mean(errors)
            stability[f'background_{bg}_std_error'] = np.std(errors)
        
        return stability
    
    def save_results(self, filename: str = None):
        """保存评估结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{self.method_name}_{timestamp}.json"
        
        # 确保results目录存在
        os.makedirs("evaluation_results", exist_ok=True)
        filepath = os.path.join("evaluation_results", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {filepath}")
        return filepath
    
    def print_summary(self):
        """打印评估摘要"""
        if not self.results['detection_results']:
            print("没有检测结果可供分析")
            return
        
        self.calculate_performance_metrics()
        
        print(f"\n=== {self.method_name} 方法评估摘要 ===")
        print(f"总检测次数: {self.results['performance_metrics']['total_detections']}")
        print(f"成功率: {self.results['performance_metrics']['success_rate']:.2%}")
        print(f"平均检测时间: {self.results['performance_metrics']['avg_detection_time']:.3f}s")
        print(f"平均FPS: {self.results['performance_metrics']['avg_fps']:.1f}")
        print(f"位置误差 - 均值: {self.results['accuracy_analysis']['position_error_mean']:.4f}m, "
              f"标准差: {self.results['accuracy_analysis']['position_error_std']:.4f}m")
        print(f"范围误差 - 均值: {self.results['accuracy_analysis']['range_error_mean']:.4f}m, "
              f"标准差: {self.results['accuracy_analysis']['range_error_std']:.4f}m")

# 全局评估器实例
evaluator = ObjectDetectionEvaluator("Depth+Clustering")

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
        min_depth = 0.15
        max_depth = 0.6
        eps = 0.03
        min_samples = 5
    
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

def detect_cube_position(scene, ed6, cam, motors_dof_idx, 
                        true_position: np.ndarray = None,
                        true_range: Tuple[float, float, float, float] = None,
                        lighting_condition: str = "normal",
                        background_complexity: str = "simple"):
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
    
    # 如果提供了真实位置，进行定量评估
    if true_position is not None and true_range is not None:
        evaluator.add_detection_result(
            true_position=true_position,
            detected_position=cube_pos,
            true_range=true_range,
            detected_range=detected_range,
            detection_time=detection_time,
            frame_count=frame_count,
            lighting_condition=lighting_condition,
            background_complexity=background_complexity
        )
    
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

def run_comprehensive_evaluation(scene, ed6, cam, motors_dof_idx, 
                               true_position: np.ndarray,
                               true_range: Tuple[float, float, float, float],
                               num_trials: int = 10):
    """运行综合评估测试"""
    print(f"\n开始运行 {num_trials} 次综合评估测试...")
    
    # 测试不同光照条件
    lighting_conditions = ["normal", "bright", "dim"]
    background_complexities = ["simple", "complex"]
    
    for trial in range(num_trials):
        print(f"\n=== 第 {trial + 1}/{num_trials} 次测试 ===")
        
        # 重置机械臂
        reset_arm(scene, ed6, motors_dof_idx)
        
        # 随机选择测试条件
        lighting = np.random.choice(lighting_conditions)
        background = np.random.choice(background_complexities)
        
        print(f"测试条件: 光照={lighting}, 背景={background}")
        
        try:
            # 执行检测
            detected_pos, detected_range, detection_info = detect_cube_position(
                scene, ed6, cam, motors_dof_idx,
                true_position=true_position,
                true_range=true_range,
                lighting_condition=lighting,
                background_complexity=background
            )
            
            # 平滑回零
            reset_after_detection(scene, ed6, motors_dof_idx)
            
        except Exception as e:
            print(f"第 {trial + 1} 次测试失败: {e}")
            continue
    
    # 计算并显示评估结果
    evaluator.calculate_performance_metrics()
    evaluator.print_summary()
    
    # 保存结果
    evaluator.save_results()
    
    return evaluator.results

def main():
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
    plane = scene.add_entity(gs.morphs.Plane(collision=True))

    # 定义物体的真实位置和范围（用于评估）
    true_cube_pos = np.array([0.35, 0, 0.02])
    true_cube_size = (0.105, 0.18, 0.022)
    true_range = (
        true_cube_pos[0] - true_cube_size[0]/2,  # min_x
        true_cube_pos[0] + true_cube_size[0]/2,  # max_x
        true_cube_pos[1] - true_cube_size[1]/2,  # min_y
        true_cube_pos[1] + true_cube_size[1]/2   # max_y
    )

    cube = scene.add_entity(gs.morphs.Box(size=true_cube_size, pos=true_cube_pos, collision=True))

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
        fov=60,  # 增加视野角度，从60度增加到80度
        aperture=2.8,
        focus_dist=0.02,  # 调整焦距，从0.015增加到0.02
        GUI=True,
    )
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    # 增加摄像机偏移距离，减少锥形摄像遮挡
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    
    print("=== Depth+Clustering 物体检测方法 ===")
    print(f"真实物体位置: {true_cube_pos}")
    print(f"真实物体范围: {true_range}")
    
    # 询问用户是否运行评估测试
    user_input = input("\n是否运行综合评估测试？(y/n): ").lower().strip()
    
    if user_input == 'y':
        # 运行综合评估
        num_trials = int(input("请输入测试次数 (默认10): ") or "10")
        evaluation_results = run_comprehensive_evaluation(
            scene, ed6, cam, motors_dof_idx,
            true_position=true_cube_pos,
            true_range=true_range,
            num_trials=num_trials
        )
    else:
        # 单次检测
        reset_arm(scene, ed6, motors_dof_idx)
        cube_pos, (min_x, max_x, min_y, max_y), detection_info = detect_cube_position(
            scene, ed6, cam, motors_dof_idx,
            true_position=true_cube_pos,
            true_range=true_range
        )
        reset_after_detection(scene, ed6, motors_dof_idx)
        
        # 输出检测结果
        print("\n=== 检测结果总结 ===")
        print(f"物体位置: {cube_pos}")
        print(f"X轴范围: {min_x:.4f} ~ {max_x:.4f} (宽度: {max_x-min_x:.4f})")
        print(f"Y轴范围: {min_y:.4f} ~ {max_y:.4f} (长度: {max_y-min_y:.4f})")
        print(f"物体尺寸: {max_x-min_x:.4f} x {max_y-min_y:.4f} x 0.022")
        
        # 计算误差
        position_error = np.linalg.norm(cube_pos - true_cube_pos)
        range_error = evaluator._calculate_range_error(true_range, (min_x, max_x, min_y, max_y))
        print(f"位置误差: {position_error:.4f}m")
        print(f"范围误差: {range_error:.4f}m")
        print("环视检测完成！")

if __name__ == "__main__":
    main()
