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

class ObjectDetectionEvaluator:
    """物体检测评估器 - 用于定量分析检测方法的性能"""
    
    def __init__(self, method_name: str = "HSV+Depth"):
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
evaluator = ObjectDetectionEvaluator("HSV+Depth")

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

def estimate_cube_range(all_frame_data):
    """根据多帧分割结果，估算cube的x/y空间范围（世界坐标系）"""
    all_xy = []
    for frame in all_frame_data:
        contour = frame['contour']
        depth = frame['depth']
        K = frame['K']
        T = frame['T']
        for pt in contour:
            px, py = pt[0]
            d = depth[py, px]
            if d > 0.05:
                x = (px - K[0,2]) * d / K[0,0]
                y = (py - K[1,2]) * d / K[1,1]
                z = d
                pos_cam = np.array([x, y, z, 1])
                pos_world = T @ pos_cam
                all_xy.append(pos_world[:2])
    if not all_xy:
        raise RuntimeError("未采集到cube边界点，无法估算范围！")
    all_xy = np.array(all_xy)
    min_x, min_y = np.min(all_xy, axis=0)
    max_x, max_y = np.max(all_xy, axis=0)
    return min_x, max_x, min_y, max_y


def detect_cube_position(scene, ed6, cam, motors_dof_idx, 
                        true_position: np.ndarray = None,
                        true_range: Tuple[float, float, float, float] = None,
                        lighting_condition: str = "normal",
                        background_complexity: str = "simple"):
    """机械臂环视一圈，检测cube，返回cube世界坐标和x/y范围（遇到连续两次未检测到物体即停止）"""
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
    
    # 根据光照条件调整HSV阈值参数
    if lighting_condition == "bright":
        # 明亮光照：提高亮度阈值，降低饱和度阈值
        lower = np.array([0, 0, 140])  # 提高亮度下限
        upper = np.array([180, 40, 255])  # 降低饱和度上限
    elif lighting_condition == "dim":
        # 昏暗光照：降低亮度阈值，提高饱和度阈值
        lower = np.array([0, 0, 80])   # 降低亮度下限
        upper = np.array([180, 60, 255])  # 提高饱和度上限
    else:  # normal
        # 正常光照：使用原始阈值
        lower = np.array([0, 0, 120])
        upper = np.array([180, 50, 255])
    
    print(f"光照条件: {lighting_condition}, HSV阈值: lower={lower}, upper={upper}")
    
    for i, angle in enumerate(angles):
        qpos = qpos_scan.copy()
        qpos[0] = angle
        ed6.control_dofs_position(qpos, motors_dof_idx)
        for step in range(settle_steps):
            scene.step()
        cam.move_to_attach()
        
        frame_start_time = time.time()
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 使用根据光照条件调整的阈值
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_count += 1
        
        h, w = img.shape[:2]
        view_hit = 0  # 当前视角是否命中
        view_score = 0.0  # 当前视角得分
        view_cx = 0
        view_cy = 0
        depth_valid_count = 0
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_to_center = np.linalg.norm([cx - w/2, cy - h/2])
                d = depth[cy, cx]
                if d > 0.05:
                    K = cam.intrinsics
                    x = (cx - K[0,2]) * d / K[0,0]
                    y = (cy - K[1,2]) * d / K[1,1]
                    z = d
                    pos_cam = np.array([x, y, z, 1])
                    T_cam2world = np.linalg.inv(cam.extrinsics)
                    pos_world = T_cam2world @ pos_cam
                    
                    # 计算得分（面积优先，距离中心越近越好）
                    score = area / (1 + dist_to_center / 100)
                    
                    results.append({
                        'cx': cx, 'cy': cy, 'area': area, 'd': d,
                        'pos_world': pos_world[:3],
                        'dist_to_center': dist_to_center,
                        'angle': angle,
                        'score': score
                    })
                    
                    # 记录本帧分割结果用于范围估算
                    all_frame_data.append({
                        'contour': c,
                        'depth': depth,
                        'K': K,
                        'T': T_cam2world
                    })
                    
                    # 更新视角信息
                    view_hit = 1
                    view_score = score
                    view_cx = cx
                    view_cy = cy
                    views_hit += 1
                    
                    # 计算有效深度点数
                    depth_valid_count = np.sum((depth > 0.05) & (depth < 2.0))
                    
                    print(f"第{i+1}帧检测到cube，J1角度={np.rad2deg(angle):.1f}° 世界坐标: {pos_world[:3]}")
                miss_count = 0  # 检测到物体，miss计数清零
                detected_once = True
            else:
                miss_count += 1
        else:
            miss_count += 1
        
        # 记录视角详细信息，包含光照条件信息
        view_details.append({
            'angle_deg': np.rad2deg(angle),
            'hit': view_hit,
            'score': view_score,
            'cx': view_cx,
            'cy': view_cy,
            'depth_valid_count': depth_valid_count,
            'lighting_condition': lighting_condition,  # 添加光照条件信息
            'hsv_lower': lower.tolist(),  # 记录使用的HSV阈值
            'hsv_upper': upper.tolist()
        })
        
        if detected_once and miss_count >= 2:  # 连续两次未检测到物体
            early_stop_index = i + 1
            print(f"连续{miss_count}次未检测到物体，提前结束环视。")
            break
    
    detection_time = time.time() - start_time
    
    if not results:
        raise RuntimeError("环视未检测到cube，请调整颜色阈值或cube位置！")
    
    best = max(results, key=lambda r: (r['score']))
    cube_pos = best['pos_world']
    best_angle_deg = np.rad2deg(best['angle'])
    best_score = best['score']
    
    # 计算范围
    min_x, max_x, min_y, max_y = estimate_cube_range(all_frame_data)
    detected_range = (min_x, max_x, min_y, max_y)
    
    # 计算考虑光照条件的召回率
    # 基础召回率（所有视角）
    base_view_hit_rate = views_hit / num_views
    
    # 考虑光照条件的召回率（如果检测失败，召回率为0）
    lighting_adjusted_hit_rate = base_view_hit_rate if len(results) > 0 else 0.0
    
    print(f"环视完成，最终选定cube世界坐标: {cube_pos}")
    print(f"cube x范围: {min_x:.4f} ~ {max_x:.4f}, y范围: {min_y:.4f} ~ {max_y:.4f}")
    print(f"cube 大小: x方向{max_x-min_x:.4f}，y方向{max_y-min_y:.4f}")
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
            detected_pos, detected_range = detect_cube_position(
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

    # cube = scene.add_entity(gs.morphs.Box(size=true_cube_size, pos=true_cube_pos, collision=True))
    cube = scene.add_entity(gs.morphs.Mesh(
        file="genesis/assets/_myobj/ctry.obj",
        pos=(0.30, 0.05, 0.02),
        collision=True,
        fixed=True,
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
        fov=80,  # 增加视野角度，从60度增加到80度
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
    
    print("=== HSV+Depth 物体检测方法 ===")
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
        cube_pos, (min_x, max_x, min_y, max_y) = detect_cube_position(
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