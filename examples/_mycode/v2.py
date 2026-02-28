import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import open3d as o3d
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

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
    qpos_now = qpos_now.cpu().numpy()  # 关键修正
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
    print("开始环视检测cube位置...")
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
    
    # 记录检测方法统计
    detection_methods = []
    
    for i, angle in enumerate(angles):
        qpos = qpos_scan.copy()
        qpos[0] = angle
        ed6.control_dofs_position(qpos, motors_dof_idx)
        for step in range(settle_steps):
            scene.step()
        cam.move_to_attach()
        
        # 获取RGB-D图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 使用多模态检测
        best_contour, detection_mask, detection_method = multi_modal_cube_detection(
            rgb, depth, cam.intrinsics
        )
        
        detection_methods.append(detection_method)
        h, w = img.shape[:2]
        
        if best_contour is not None:
            area = cv2.contourArea(best_contour)
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_to_center = np.linalg.norm([cx - w/2, cy - h/2])
                
                # 可视化检测结果
                img_vis = img.copy()
                cv2.circle(img_vis, (cx, cy), 10, (0,0,255), 2)
                
                # 根据检测方法显示不同颜色
                if detection_method == "depth+color_fusion":
                    color = (0, 255, 0)  # 绿色：融合检测
                elif detection_method == "depth_only":
                    color = (255, 0, 0)  # 蓝色：深度检测
                elif detection_method == "color_only":
                    color = (0, 255, 255)  # 黄色：颜色检测
                else:
                    color = (128, 128, 128)  # 灰色：其他
                
                cv2.putText(img_vis, detection_method, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("Multi-Modal Detection", img_vis)
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
                        'angle': angle,
                        'detection_method': detection_method
                    })
                    # 记录本帧分割结果用于范围估算
                    all_frame_data.append({
                        'contour': best_contour,
                        'depth': depth,
                        'K': K,
                        'T': T_cam2world
                    })
                    print(f"第{i+1}帧检测到cube，J1角度={np.rad2deg(angle):.1f}° 世界坐标: {pos_world[:3]} 方法: {detection_method}")
                miss_count = 0  # 检测到物体，miss计数清零
                detected_once = True
            else:
                miss_count += 1
        else:
            miss_count += 1
            cv2.imshow("Multi-Modal Detection", img)
            cv2.waitKey(100)
            print(f"第{i+1}帧未检测到物体，方法: {detection_method}")
        
        if detected_once and miss_count >= 2:
            print(f"连续{miss_count}次未检测到物体，提前结束环视。")
            break
    
    cv2.destroyAllWindows()
    
    if not results:
        raise RuntimeError("环视未检测到cube，请调整检测参数或cube位置！")
    
    # 统计检测方法
    method_counts = {}
    for method in detection_methods:
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print(f"检测方法统计: {method_counts}")
    
    best = max(results, key=lambda r: (r['area'], -r['dist_to_center']))
    cube_pos = best['pos_world']
    
    # 计算范围
    min_x, max_x, min_y, max_y = estimate_cube_range(all_frame_data)
    print(f"环视完成，最终选定cube世界坐标: {cube_pos}")
    print(f"cube x范围: {min_x:.4f} ~ {max_x:.4f}, y范围: {min_y:.4f} ~ {max_y:.4f}")
    print(f"cube 大小: x方向{max_x-min_x:.4f}，y方向{max_y-min_y:.4f}")
    print(f"最佳检测方法: {best['detection_method']}")
    time.sleep(10)
    return cube_pos, (min_x, max_x, min_y, max_y)

def detect_cube_by_depth_geometry(depth_image, min_depth=0.05, max_depth=2.0):
    """
    基于深度几何特征检测地面凸起物体
    """
    # 1. 深度图预处理
    valid_depth = (depth_image > min_depth) & (depth_image < max_depth)
    
    # 2. 计算深度梯度（检测边缘）
    grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 3. 寻找深度不连续区域（物体边缘）
    edge_threshold = np.percentile(gradient_magnitude[valid_depth], 90)
    edge_mask = gradient_magnitude > edge_threshold
    
    # 4. 形态学操作连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # 5. 寻找轮廓
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, edge_mask

def adaptive_hsv_detection(rgb_image, target_color_hint=None):
    """
    自适应HSV检测，适应不同光照条件
    
    Parameters:
    -----------
    rgb_image : np.ndarray
        RGB图像
    target_color_hint : tuple, optional
        目标颜色提示 (h, s, v)，用于调整检测范围
    
    Returns:
    --------
    contours : list
        检测到的轮廓列表
    mask : np.ndarray
        检测掩码
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # 1. 计算图像统计信息
    h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
    s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
    v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
    
    # 2. 自适应阈值（基于统计信息）
    if target_color_hint is None:
        # 默认检测半透明浅色物体
        lower = np.array([
            max(0, h_mean - 2*h_std),      # 色调范围
            max(0, s_mean - 1.5*s_std),    # 饱和度下限（低饱和度）
            max(0, v_mean - 1.0*v_std)     # 亮度下限（高亮度）
        ])
        upper = np.array([
            min(180, h_mean + 2*h_std),
            min(255, s_mean + 1.5*s_std),
            min(255, v_mean + 1.0*v_std)
        ])
    else:
        # 使用目标颜色提示
        h, s, v = target_color_hint
        lower = np.array([max(0, h-20), max(0, s-30), max(0, v-40)])
        upper = np.array([min(180, h+20), min(255, s+30), min(255, v+40)])
    
    # 3. 应用HSV阈值
    mask = cv2.inRange(hsv, lower, upper)
    
    # 4. 形态学后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 5. 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, mask

def multi_modal_cube_detection(rgb_image, depth_image, camera_intrinsics, target_color_hint=None):
    """
    多模态融合检测cube：结合深度几何特征和自适应颜色检测
    
    Parameters:
    -----------
    rgb_image : np.ndarray
        RGB图像
    depth_image : np.ndarray
        深度图像
    camera_intrinsics : np.ndarray
        相机内参矩阵
    target_color_hint : tuple, optional
        目标颜色提示
    
    Returns:
    --------
    best_contour : np.ndarray or None
        最佳检测轮廓
    detection_mask : np.ndarray
        融合检测掩码
    detection_method : str
        使用的检测方法
    """
    # 1. 深度几何检测
    depth_contours, depth_mask = detect_cube_by_depth_geometry(depth_image)
    
    # 2. 自适应HSV检测
    color_contours, color_mask = adaptive_hsv_detection(rgb_image, target_color_hint)
    
    # 3. 融合策略
    if depth_contours and color_contours:
        # 两种方法都检测到物体，选择重叠度最高的
        best_contour = None
        best_overlap = 0
        
        for d_contour in depth_contours:
            for c_contour in color_contours:
                # 计算轮廓重叠度
                overlap_mask = np.zeros_like(depth_mask)
                cv2.drawContours(overlap_mask, [d_contour], -1, 1, -1)
                cv2.drawContours(overlap_mask, [c_contour], -1, 1, -1)
                overlap_ratio = np.sum(overlap_mask == 2) / np.sum(overlap_mask > 0)
                
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_contour = d_contour if cv2.contourArea(d_contour) > cv2.contourArea(c_contour) else c_contour
        
        if best_overlap > 0.3:  # 30%以上重叠
            detection_mask = depth_mask | color_mask
            detection_method = "depth+color_fusion"
        else:
            # 重叠度低，选择面积更大的
            best_contour = max(depth_contours + color_contours, key=cv2.contourArea)
            detection_mask = depth_mask if cv2.contourArea(max(depth_contours, key=cv2.contourArea)) > cv2.contourArea(max(color_contours, key=cv2.contourArea)) else color_mask
            detection_method = "largest_area"
    
    elif depth_contours:
        # 只有深度检测成功
        best_contour = max(depth_contours, key=cv2.contourArea)
        detection_mask = depth_mask
        detection_method = "depth_only"
    
    elif color_contours:
        # 只有颜色检测成功
        best_contour = max(color_contours, key=cv2.contourArea)
        detection_mask = color_mask
        detection_method = "color_only"
    
    else:
        # 两种方法都失败
        best_contour = None
        detection_mask = np.zeros_like(depth_mask)
        detection_method = "failed"
    
    return best_contour, detection_mask, detection_method

# def plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
#     """机械臂回零，IK逆解，路径插值并执行"""
#     # reset_arm(scene, ed6, motors_dof_idx)  # 由reset_after_detection替代
#     print("机械臂已回到初始零位，等待2秒...")
#     time.sleep(2)
#     target_quat = np.array([0, 1, 0, 0])
#     target_pos = target_pos.copy()
#     target_pos[2] += 0.2
#     qpos_ik = ed6.inverse_kinematics(
#         link=j6_link,
#         pos=target_pos,
#         quat=target_quat,
#     )
    
#     # 处理tensor到numpy的转换（MPS设备兼容）
#     if hasattr(qpos_ik, 'cpu'):
#         qpos_ik = qpos_ik.cpu().numpy()
    
#     print("IK逆解目标关节角度:", qpos_ik)
#     try:
#         path = ed6.plan_path(
#             qpos_goal=qpos_ik,
#             num_waypoints=150,
#         )
#         if len(path) == 0:
#             raise RuntimeError("plan_path返回空路径，自动切换为线性插值")
#     except Exception as e:
#         print("plan_path失败，切换为线性插值:", e)
#         import torch
#         if isinstance(qpos_ik, torch.Tensor):
#             qpos_ik = qpos_ik.detach().cpu().numpy()
#         path = np.linspace(np.zeros(6), qpos_ik, num=200)
#         path = torch.from_numpy(path).float().cpu()
#     print("路径插值完成，路径点数:", len(path))
#     path_debug = scene.draw_debug_path(path, ed6)
#     scene.step()
#     print("5秒后开始沿IK路径运动...")
#     time.sleep(5)
#     for idx, waypoint in enumerate(path):
#         ed6.control_dofs_position(waypoint, motors_dof_idx)
#         for _ in range(3):
#             scene.step()
#         if idx % 20 == 0:
#             print(f"[路径跟踪] 进度: {idx+1}/{len(path)}  J1角度: {waypoint[0]:.4f}")
#     print("路径执行完毕")
#     time.sleep(2)
#     # print("IK运动完成！相机窗口会保持打开，你可以观察J6末端的视角。按Ctrl+C退出程序。")
#     # try:
#     #     while True:
#     #         scene.step()
#     #         rgb, depth, segmentation, normal = cam.render(
#     #             rgb=True, depth=True, segmentation=True, normal=True
#     #         )
#     # except KeyboardInterrupt:
#     print("程序已退出。")

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

# def track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, x_range=None, y_range=None, step_size=0.08, max_steps=200):
#     """
#     沿Canny边界顺时针等距采样点云，采样步长0.08m，形成闭环。
#     """
#     all_points = []
#     vis_pcd = o3d.geometry.PointCloud()
#     vis_step = 0
#     # 1. 边界检测，提取最大轮廓
#     rgb, depth, _, _ = cam.render(rgb=True, depth=True)
#     img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 30, 100)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if not contours:
#         print("未检测到边界，无法采集点云！")
#         return np.zeros((0,3))
#     cnt = max(contours, key=cv2.contourArea)
#     cnt = cnt[:,0,:]  # N×2
#     # 2. 顺时针排序（cv2.findContours默认顺时针）
#     # 3. 计算每个点的物理位置
#     K = cam.intrinsics
#     T_cam2world = np.linalg.inv(cam.extrinsics)
#     points_2d = []
#     points_3d = []
#     for pt in cnt:
#         cx, cy = int(pt[0]), int(pt[1])
#         d = depth[cy, cx]
#         if d <= 0.05:
#             continue
#         x = (cx - K[0,2]) * d / K[0,0]
#         y = (cy - K[1,2]) * d / K[1,1]
#         z = d
#         pos_cam = np.array([x, y, z, 1])
#         pos_world = (T_cam2world @ pos_cam)[:3]
#         points_2d.append([cx, cy])
#         points_3d.append(pos_world)
#     points_2d = np.array(points_2d)
#     points_3d = np.array(points_3d)
#     if len(points_3d) < 2:
#         print("有效边界点太少，无法采集点云！")
#         return np.zeros((0,3))
#     # 4. 沿边界等距采样（物理距离0.08m）
#     sampled_idx = [0]
#     acc_dist = 0.0
#     for i in range(1, len(points_3d)):
#         dist = np.linalg.norm(points_3d[i] - points_3d[sampled_idx[-1]])
#         if dist >= step_size:
#             sampled_idx.append(i)
#     # 闭环
#     if np.linalg.norm(points_3d[sampled_idx[0]] - points_3d[sampled_idx[-1]]) > step_size/2:
#         sampled_idx.append(0)
#     print(f"采样点数: {len(sampled_idx)} (闭环)")
#     # 5. 顺序遍历采集
#     for idx in range(len(sampled_idx)):
#         pt2d = points_2d[sampled_idx[idx]]
#         pt3d = points_3d[sampled_idx[idx]]
#         cx, cy = int(pt2d[0]), int(pt2d[1])
#         target_pos = pt3d
#         target_quat = np.array([0, 1, 0, 0])
#         # IK逆解和路径规划
#         target_quat = np.array([0, 1, 0, 0])
#         qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
#         if hasattr(qpos_ik, 'cpu'):
#             qpos_ik = qpos_ik.cpu().numpy()
#         if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
#             print(f"第{idx+1}步，IK逆解失败，目标位置可能超出工作空间")
#             continue
        
#         # 路径规划和执行
#         try:
#             path = ed6.plan_path(qpos_goal=qpos_ik, num_waypoints=100)
#             if len(path) == 0:
#                 raise RuntimeError("plan_path返回空路径，使用直接控制")
#         except Exception as e:
#             print(f"路径规划失败，使用直接控制: {e}")
#             path = [qpos_ik]
        
#         # 执行路径
#         for waypoint_idx, waypoint in enumerate(path):
#             ed6.control_dofs_position(waypoint, motors_dof_idx)
#             for _ in range(3):
#                 scene.step()
        
#         cam.move_to_attach()
#         # 采集点云（只保留当前帧所有边界点）
#         mask = np.zeros_like(edges, dtype=bool)
#         for c in contours:
#             for p in c:
#                 mask[p[0][1], p[0][0]] = True
#         pc, _ = cam.render_pointcloud(world_frame=True)
#         points = pc[mask]
#         all_points.append(points)
#         # === 实时debug画面 ===
#         debug_img = img.copy()
#         cv2.drawContours(debug_img, [cnt], -1, (0, 255, 255), 2)  # 黄色线
#         cv2.circle(debug_img, (cx, cy), 7, (0,0,255), 2)  # 当前目标点
#         if idx < len(sampled_idx)-1:
#             next_pt2d = points_2d[sampled_idx[idx+1]]
#             cv2.circle(debug_img, (int(next_pt2d[0]), int(next_pt2d[1])), 7, (255,0,255), 2)  # 下一个目标点
#             move_dir = next_pt2d - pt2d
#             move_dir = move_dir / (np.linalg.norm(move_dir)+1e-8)
#             arrow_end = (int(cx + 30*move_dir[0]), int(cy + 30*move_dir[1]))
#             cv2.arrowedLine(debug_img, (cx, cy), arrow_end, (255,0,0), 2)
#         cv2.imshow('Boundary Tracking Debug', debug_img)
#         cv2.waitKey(50)
#         print(f"第{idx+1}步，采集到{len(points)}个边界点")
#     cv2.destroyAllWindows()
#     if len(all_points) == 0:
#         print("未采集到任何边界点云！")
#         return np.zeros((0,3))
#     all_points = np.concatenate(all_points, axis=0)
#     print(f"边界跟踪完成，总共采集到{len(all_points)}个边界点")
#     # === 采集完后整体可视化 ===
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(all_points)
#     o3d.io.write_point_cloud("boundary_cloud.ply", pcd)
#     print("点云已保存为boundary_cloud.ply")
#     o3d.visualization.draw_geometries([pcd])
#     return all_points

# def visualize_and_save_pointcloud(points, filename="boundary_cloud.ply"):
#     """用Open3D可视化并保存点云"""
#     if len(points) == 0:
#         print("点云为空，无法可视化和保存！")
#         return
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.io.write_point_cloud(filename, pcd)
#     print(f"点云已保存为{filename}")
#     # 修正 linter 报错，使用 o3d.visualization.draw_geometries
#     o3d.visualization.draw_geometries([pcd])  # type: ignore

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
    plane = scene.add_entity(
        gs.morphs.Plane(collision=True),
        surface=gs.surfaces.Metal(
            color=(0.1, 0.1, 0.1),  # 更深的灰黑色，接近黑色
            roughness=0.8,  # 高粗糙度，非镜面
            metallic=0.8,  # 高金属度
        )
    )

    cube = scene.add_entity(gs.morphs.Box(size=(0.18, 0.105, 0.022), pos=(0.5, 0.1, 0.02), collision=True))
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
        fov=40,
        aperture=2.8,
        focus_dist=0.015,
        GUI=True,
    )
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.08]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    reset_arm(scene, ed6, motors_dof_idx)
    cube_pos, (min_x, max_x, min_y, max_y) = detect_cube_position(scene, ed6, cam, motors_dof_idx)
    reset_after_detection(scene, ed6, motors_dof_idx)  # 检测后平滑回零
    print(cube_pos) 
    print(min_x, max_x, min_y, max_y)   
    # plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
    # # === 新增：自动边界跟踪与点云采集 ===
    # print("\n开始自动边界跟踪与点云采集...")
    # time.sleep(5)
    # points = track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos)
    # visualize_and_save_pointcloud(points)

if __name__ == "__main__":
    main()