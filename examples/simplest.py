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

# 全局变量用于ESC键检测
esc_pressed = False
scanning_active = True

def check_esc_key():
    """监听ESC键的线程函数"""
    global esc_pressed
    while scanning_active:
        if sys.platform == "darwin":  # macOS
            # macOS下使用select监听stdin
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()
                if key == 'esc' or key == 'q':
                    esc_pressed = True
                    print("\n检测到ESC键，正在结束扫描...")
                    break
        else:  # Linux/Windows
            # 其他平台使用input
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = input().strip().lower()
                    if key == 'esc' or key == 'q':
                        esc_pressed = True
                        print("\n检测到ESC键，正在结束扫描...")
                        break
            except:
                pass
        time.sleep(0.1)

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


def detect_cube_position(scene, ed6, cam, motors_dof_idx):
    """机械臂环视一圈，检测cube，返回cube世界坐标和x/y范围（遇到连续两次未检测到物体即停止）"""
    # print("开始环视检测cube位置... (按Enter继续)")
    # input()
    qpos_scan = np.zeros(6)
    qpos_scan[3] = -np.pi / 3   # J4 -60
    qpos_scan[4] = np.pi / 2   # J5 +90°
    num_views = 24
    start_angle = 0
    angles = np.linspace(start_angle, 2 * np.pi + start_angle, num_views, endpoint=False)
    results = []
    all_frame_data = []
    qpos_scan[0] = 0
    ed6.control_dofs_position(qpos_scan, motors_dof_idx)
    for step in range(100):
        scene.step()
    cam.move_to_attach()
    settle_steps = 100
    miss_count = 0  # 连续未检测到物体的次数
    detected_once = False  # 是否至少检测到过一次
    for i, angle in enumerate(angles):
        qpos = qpos_scan.copy()
        qpos[0] = angle
        ed6.control_dofs_position(qpos, motors_dof_idx)
        for step in range(settle_steps):
            scene.step()
        cam.move_to_attach()
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 120])
        upper = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_to_center = np.linalg.norm([cx - w/2, cy - h/2])
                img_vis = img.copy()
                cv2.circle(img_vis, (cx, cy), 10, (0,0,255), 2)
                cv2.imshow("Detection", img_vis)
                cv2.waitKey(200)
                d = depth[cy, cx]
                if d > 0.05:
                    K = cam.intrinsics
                    x = (cx - K[0,2]) * d / K[0,0]
                    y = (cy - K[1,2]) * d / K[1,1]
                    z = d
                    pos_cam = np.array([x, y, z, 1])
                    T_cam2world = np.linalg.inv(cam.extrinsics)
                    pos_world = T_cam2world @ pos_cam
                    results.append({
                        'cx': cx, 'cy': cy, 'area': area, 'd': d,
                        'pos_world': pos_world[:3],
                        'dist_to_center': dist_to_center,
                        'angle': angle
                    })
                    # 记录本帧分割结果用于范围估算
                    all_frame_data.append({
                        'contour': c,
                        'depth': depth,
                        'K': K,
                        'T': T_cam2world
                    })
                    print(f"第{i+1}帧检测到cube，J1角度={np.rad2deg(angle):.1f}° 世界坐标: {pos_world[:3]}")
                miss_count = 0  # 检测到物体，miss计数清零
                detected_once = True
            else:
                miss_count += 1
        else:
            miss_count += 1
            cv2.imshow("Detection", img)
            cv2.waitKey(100)
        if detected_once and miss_count >= 1:
            print(f"连续{miss_count}次未检测到物体，提前结束环视。")
            break
    cv2.destroyAllWindows()
    if not results:
        raise RuntimeError("环视未检测到cube，请调整颜色阈值或cube位置！")
    best = max(results, key=lambda r: (r['area'], -r['dist_to_center']))
    cube_pos = best['pos_world']
    # 计算范围
    min_x, max_x, min_y, max_y = estimate_cube_range(all_frame_data)
    print(f"环视完成，最终选定cube世界坐标: {cube_pos}")
    print(f"cube x范围: {min_x:.4f} ~ {max_x:.4f}, y范围: {min_y:.4f} ~ {max_y:.4f}")
    print(f"cube 大小: x方向{max_x-min_x:.4f}，y方向{max_y-min_y:.4f}")
    time.sleep(10)
    return cube_pos, (min_x, max_x, min_y, max_y)

def plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
    """机械臂回零，IK逆解，路径插值并执行"""
    # reset_arm(scene, ed6, motors_dof_idx)  # 由reset_after_detection替代
    # print("机械臂已回到初始零位，等待2秒... (按Enter继续)")
    # input()
    time.sleep(2)
    target_quat = np.array([0, 1, 0, 0])
    target_pos = target_pos.copy()
    target_pos[2] += 0.2
    qpos_ik = ed6.inverse_kinematics(
        link=j6_link,
        pos=target_pos,
        quat=target_quat,
    )
    
    # 处理tensor到numpy的转换
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


def detect_object_boundary(cam):
    """用Canny检测当前帧物体边界，返回边界像素mask"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    return edges > 0  # 返回bool型mask

# def get_boundary_pointcloud(cam, boundary_mask):
#     """用mask筛选点云，只保留边界点"""
#     pc, _ = cam.render_pointcloud(world_frame=True)
#     points = pc[boundary_mask]
#     return points

def detect_pointcloud_closure(points, closure_threshold=0.02, min_points=50):
    """
    检测点云是否形成闭环
    
    Parameters:
    -----------
    points : np.ndarray
        点云数据，形状为(N, 3)
    closure_threshold : float
        闭环检测阈值，默认0.02米
    min_points : int
        最少点数要求，默认50个点
    
    Returns:
    --------
    bool : 是否形成闭环
    float : 闭环质量（0-1之间，1表示完美闭环）
    """
    if len(points) < min_points:
        return False, 0.0
    
    # 计算点云的边界框
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    bbox_size = max_coords - min_coords
    
    # 如果边界框太小，可能不是有效的闭环
    if np.any(bbox_size < 0.01):  # 小于1cm的边界框
        return False, 0.0
    
    # 方法1: 检查首尾点距离
    if len(points) > 0:
        start_end_dist = np.linalg.norm(points[0] - points[-1])
        closure_quality_1 = max(0, 1 - start_end_dist / closure_threshold)
    else:
        closure_quality_1 = 0.0
    
    # 方法2: 检查点云密度分布
    # 计算每个点到其他点的平均距离
    if len(points) > 10:
        from scipy.spatial.distance import cdist
        distances = cdist(points, points)
        np.fill_diagonal(distances, np.inf)  # 排除自身
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)
        
        # 如果平均最小距离很小，说明点云密集，可能是闭环
        density_quality = max(0, 1 - avg_min_distance / 0.01)  # 1cm作为密集阈值
    else:
        density_quality = 0.0
    
    # 方法3: 检查点云的空间分布
    # 计算点云的重心
    centroid = np.mean(points, axis=0)
    distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
    std_distance = np.std(distances_to_centroid)
    mean_distance = np.mean(distances_to_centroid)
    
    # 如果距离标准差相对于平均距离较小，说明点云分布较均匀
    if mean_distance > 0:
        uniformity_quality = max(0, 1 - std_distance / mean_distance)
    else:
        uniformity_quality = 0.0
    
    # 综合评分
    closure_quality = (closure_quality_1 * 0.4 + density_quality * 0.3 + uniformity_quality * 0.3)
    
    # 判断是否形成闭环
    is_closed = closure_quality > 0.6  # 60%以上质量认为形成闭环
    
    return is_closed, closure_quality

def track_and_scan_boundary_jump(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, jump_height=0.1, max_steps=30):
    """
    跳跃式边界扫描
    - 检测当前帧边界
    - 按顺时针方向选择距离当前末端最远的边界点
    - 跳跃到该点上方（保持高度在物体上方）
    - 采集点云
    - 重复直到闭环或回到初始点或按ESC键
    """
    global esc_pressed, scanning_active
    
    # 启动ESC键监听线程
    esc_thread = threading.Thread(target=check_esc_key, daemon=True)
    esc_thread.start()
    
    all_points = []
    start_pos = None
    start_recorded = False
    h, w = cam.res[1], cam.res[0]
    center_img = np.array([w // 2, h // 2])
    
    # 闭环检测相关变量
    closure_check_interval = 3  # 每3步检查一次闭环
    last_closure_check_step = 0
    
    # 记录上一次选择的方向，用于顺时针约束
    last_direction = None
    
    # 计算目标高度（物体上方0.1米）
    target_height = cube_pos[2] + 0.1
    
    # 回到初始点的检测阈值
    return_threshold = 0.03  # 3cm
    
    print("开始跳跃式边界扫描...")
    print("提示：在终端输入 'esc' 或 'q' 可以立即结束扫描")
    
    for step in range(max_steps):
        # 检查ESC键
        if esc_pressed:
            print("用户中断扫描，正在保存点云...")
            break
            
        print(f"\n==== 第{step+1}步跳跃扫描 ====")
        
        # 1. 边界检测（带重试机制）
        max_retries = 3
        retry_count = 0
        contours = None
        
        while retry_count < max_retries:
            # 检查ESC键
            if esc_pressed:
                break
                
            rgb, depth, _, _ = cam.render(rgb=True, depth=True)
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if contours:
                break
            
            # 检测失败，抬高机械臂重试
            retry_count += 1
            print(f"  第{retry_count}次检测失败，抬高机械臂0.05米重试...")
            
            # 抬高机械臂
            current_pos = j6_link.get_pos().cpu().numpy()
            elevated_pos = current_pos.copy()
            elevated_pos[2] += 0.05
            
            # 使用路径规划移动到抬高位置
            target_quat = j6_link.get_quat().cpu().numpy()
            qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=elevated_pos, quat=target_quat)
            if hasattr(qpos_ik, 'cpu'):
                qpos_ik = qpos_ik.cpu().numpy()
            
            if not np.any(np.isnan(qpos_ik)) and not np.any(np.isinf(qpos_ik)):
                ed6.control_dofs_position(qpos_ik, motors_dof_idx)
                for _ in range(30):
                    scene.step()
                cam.move_to_attach()
        
        if esc_pressed:
            break
            
        if not contours:
            print(f"第{step+1}步，经过{max_retries}次重试仍未检测到边界，停止")
            break
        
        # 2. 记录起始位置
        if not start_recorded:
            start_pos = j6_link.get_pos().cpu().numpy()
            start_recorded = True
            print(f"记录起始位置: {start_pos}")
        
        # 3. 获取当前机械臂末端位置
        current_pos = j6_link.get_pos().cpu().numpy()
        print(f"  当前末端位置: {current_pos}")
        
        # 4. 检查是否回到初始点
        if start_recorded and step > 3:  # 至少5步后才检查
            dist_to_start = np.linalg.norm(current_pos - start_pos)
            print(f"  距离起始位置: {dist_to_start:.4f}米")
            if dist_to_start < return_threshold:
                print(f"第{step+1}步，已回到起始位置附近，扫描完成！")
                break
        
        # 5. 找到最远的边界点（按顺时针方向）
        max_dist = 0
        best_pt = None
        best_cnt = None
        best_world_pos = None
        
        # 可视化图像
        vis_img = img.copy()
        
        for cnt in contours:
            # 绘制边界轮廓（黄色）
            cv2.drawContours(vis_img, [cnt], -1, (0, 255, 255), 2)
            
            for pt in cnt:
                pt_xy = pt[0]
                cx, cy = int(pt_xy[0]), int(pt_xy[1])
                
                # 获取深度
                if 0 <= cy < h and 0 <= cx < w:
                    d = depth[cy, cx]
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
                            # 计算从当前位置到边界点的方向向量
                            direction_vector = pos_world[:2] - current_pos[:2]
                            direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
                            
                            # 检查是否符合顺时针方向
                            if not is_clockwise_direction(direction_angle, last_direction):
                                continue
                        
                        if dist > max_dist:
                            max_dist = dist
                            best_pt = pt_xy
                            best_cnt = cnt
                            best_world_pos = pos_world[:3]
        
        if best_pt is None:
            print(f"第{step+1}步，未找到合适的边界点，停止")
            break
        
        # 6. 标记最远点（红色圆圈）
        cv2.circle(vis_img, tuple(best_pt), 8, (0, 0, 255), -1)  # 红色实心圆
        cv2.circle(vis_img, tuple(best_pt), 10, (0, 0, 255), 2)  # 红色边框
        
        # 显示图像
        cv2.imshow('Jumping Boundary Scan', vis_img)
        cv2.waitKey(100)
        
        # 7. 打印最远点信息
        print(f"  最远边界点像素坐标: ({best_pt[0]}, {best_pt[1]})")
        print(f"  最远边界点世界坐标: {best_world_pos}")
        print(f"  距离当前末端: {max_dist:.4f}米")
        
        # 8. 计算目标位置（保持在物体上方）
        target_pos = best_world_pos.copy()
        target_pos[2] = target_height  # 使用预计算的目标高度
        
        print(f"  目标位置: {target_pos}")
        
        # 9. 更新方向记录
        direction_vector = best_world_pos[:2] - current_pos[:2]
        last_direction = np.arctan2(direction_vector[1], direction_vector[0])
        
        # 10. 采集点云（当前帧所有边界点）
        mask = np.zeros_like(edges, dtype=bool)
        for cnt in contours:
            for pt in cnt:
                mask[pt[0][1], pt[0][0]] = True
        pc, _ = cam.render_pointcloud(world_frame=True)
        points = pc[mask]
        all_points.append(points)
        print(f"第{step+1}步，采集到{len(points)}个边界点")
        
        # 11. 闭环检测
        if step - last_closure_check_step >= closure_check_interval and len(all_points) > 0:
            current_all_points = np.concatenate(all_points, axis=0)
            is_closed, closure_quality = detect_pointcloud_closure(current_all_points)
            
            print(f"  闭环检测: 质量={closure_quality:.3f}, 形成闭环={is_closed}")
            
            if is_closed:
                print(f"第{step+1}步，点云已形成闭环，跳跃扫描完成！")
                break
            
            last_closure_check_step = step
        
        # 12. 实时可视化点云
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd], window_name=f"Step {step+1} PointCloud", width=400, height=300)
        
        # 13. IK逆解和路径规划
        target_quat = j6_link.get_quat().cpu().numpy()
        qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
        if hasattr(qpos_ik, 'cpu'):
            qpos_ik = qpos_ik.cpu().numpy()
        if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
            print(f"第{step+1}步，IK逆解失败，目标位置可能超出工作空间")
            break
        
        # 14. 验证IK解的有效性
        joint_limits = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi], 
                                [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
        for i, (q, limits) in enumerate(zip(qpos_ik, joint_limits)):
            if q < limits[0] or q > limits[1]:
                print(f"第{step+1}步，关节{i}角度{q:.4f}超出范围[{limits[0]:.4f}, {limits[1]:.4f}]")
                break
        else:
            pass
        
        # 15. 路径规划和执行
        try:
            path = ed6.plan_path(qpos_goal=qpos_ik, num_waypoints=100)
            if len(path) == 0:
                raise RuntimeError("plan_path返回空路径，使用直接控制")
        except Exception as e:
            print(f"路径规划失败，使用直接控制: {e}")
            path = [qpos_ik]
        
        print(f"  路径规划完成，路径点数: {len(path)}")
        
        # 16. 执行路径
        for idx, waypoint in enumerate(path):
            # 检查ESC键
            if esc_pressed:
                break
                
            ed6.control_dofs_position(waypoint, motors_dof_idx)
            for _ in range(3):
                scene.step()
            if idx % 20 == 0:
                print(f"    [路径执行] 进度: {idx+1}/{len(path)}")
        
        if esc_pressed:
            break
            
        cam.move_to_attach()
        
        # 17. 验证实际到达位置
        actual_pos = j6_link.get_pos().cpu().numpy()
        pos_error = np.linalg.norm(actual_pos - target_pos)
        print(f"  实际位置: {actual_pos}")
        print(f"  位置误差: {pos_error:.4f}米")
        
        if pos_error > 0.05:
            print(f"第{step+1}步，位置误差过大({pos_error:.4f}m)，可能IK失败或机械臂卡住")
            print(f"  建议：检查工作空间或调整目标位置")
            break
        
        print(f"第{step+1}步跳跃完成\n")
    
    # 结束扫描
    scanning_active = False
    
    if len(all_points) == 0:
        print("未采集到任何边界点云！")
        return np.zeros((0,3))
    all_points = np.concatenate(all_points, axis=0)
    print(f"跳跃式边界扫描完成，总共采集到{len(all_points)}个边界点")
    return all_points

def is_clockwise_direction(current_angle, last_angle, tolerance=np.pi/4):
    """
    检查当前方向是否符合顺时针约束
    
    Parameters:
    -----------
    current_angle : float
        当前方向角度
    last_angle : float
        上一次方向角度
    tolerance : float
        允许的角度偏差
    
    Returns:
    --------
    bool : 是否符合顺时针方向
    """
    if last_angle is None:
        return True
    
    # 计算角度差
    angle_diff = current_angle - last_angle
    
    # 处理角度跨越±π的情况
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi
    
    # 顺时针方向：角度差应该为正（或接近0）
    return angle_diff >= -tolerance

def track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, step_size=0.08, max_steps=30):
    """
    使用跳跃式边界扫描替代原来的切线跟踪
    """
    return track_and_scan_boundary_jump(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, jump_height=0.1, max_steps=max_steps)

def visualize_and_save_pointcloud(points, filename="boundary_cloud.ply"):
    """用Open3D可视化并保存点云"""
    if len(points) == 0:
        print("点云为空，无法可视化和保存！")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存为{filename}")
    # 修正 linter 报错，使用 o3d.visualization.draw_geometries
    o3d.visualization.draw_geometries([pcd])  # type: ignore

# def xy_grid_scan(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, x_range, y_range):
#     all_points = []
#     z_height = cube_pos[2] + 0.1  # z高度始终为cube_pos[2]+0.1
#     x_step = 0.08
#     y_step = 0.08
#     x_vals = np.arange(x_range[0], x_range[1] + x_step, x_step)
#     y_vals = np.arange(y_range[0], y_range[1] + y_step, y_step)
#     scan_path = [(x, y) for i, x in enumerate(x_vals)
#                  for y in (y_vals if i % 2 == 0 else reversed(y_vals))]

#     for idx, (x, y) in enumerate(scan_path):
#         target_pos = np.array([x, y, z_height])
#         target_quat = np.array([0, 1, 0, 0])
#         qpos_ik = ed6.inverse_kinematics(j6_link, target_pos, target_quat).cpu().numpy()
#         if np.any(np.isnan(qpos_ik)):
#             print(f"第{idx+1}步，IK失败，跳过")
#             continue
#         ed6.control_dofs_position(qpos_ik, motors_dof_idx)
#         for _ in range(30):
#             scene.step()
#         cam.move_to_attach()
#         pc, _ = cam.render_pointcloud(world_frame=True)
#         all_points.append(pc)
#         print(f"[{idx+1}/{len(scan_path)}] 点({x:.3f}, {y:.3f})，点云数: {len(pc)}")
#         # 实时可视化
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(pc)
#         o3d.visualization.draw_geometries([pcd], window_name=f"Scan {idx+1}", width=400, height=300)

#     return np.concatenate(all_points, axis=0) if all_points else np.zeros((0, 3))



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
            show_cameras=False,
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

    cube = scene.add_entity(gs.morphs.Box(size=(0.105, 0.18, 0.022), pos=(0.35, 0, 0.02), collision=True))


    # cube = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.35, 0.0, 0.02), collision=True))
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
        focus_dist=0.015,
        GUI=True,
    )
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.05]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    reset_arm(scene, ed6, motors_dof_idx)
    cube_pos, (min_x, max_x, min_y, max_y) = detect_cube_position(scene, ed6, cam, motors_dof_idx)
    reset_after_detection(scene, ed6, motors_dof_idx)  # 检测后平滑回零
    plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
    # === 新增：自动边界跟踪与点云采集 ===
    print("\n开始自动边界跟踪与点云采集...")
    time.sleep(5)
    points = track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos)
    visualize_and_save_pointcloud(points)

if __name__ == "__main__":
    main()