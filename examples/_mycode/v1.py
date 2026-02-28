import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import open3d as o3d
from sklearn.decomposition import PCA

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
    # return cube_pos, (min_x, max_x, min_y, max_y)
    return cube_pos
    
def plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, target_pos, cam):
    """机械臂回零，IK逆解，路径插值并执行"""
    # reset_arm(scene, ed6, motors_dof_idx)  # 由reset_after_detection替代
    print("机械臂已回到初始零位，等待2秒...")
    time.sleep(2)
    target_quat = np.array([0, 1, 0, 0])
    target_pos = target_pos.copy()
    target_pos[2] += 0.2
    qpos_ik = ed6.inverse_kinematics(
        link=j6_link,
        pos=target_pos,
        quat=target_quat,
    )
    
    # 处理tensor到numpy的转换（MPS设备兼容）
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
    print("2秒后开始沿IK路径运动...")
    time.sleep(2)
    for idx, waypoint in enumerate(path):
        ed6.control_dofs_position(waypoint, motors_dof_idx)
        for _ in range(3):
            scene.step()
        if idx % 20 == 0:
            print(f"[路径跟踪] 进度: {idx+1}/{len(path)}  J1角度: {waypoint[0]:.4f}")
    print("路径执行完毕")
    scene.clear_debug_object(path_debug)
    time.sleep(2)
    print("程序已退出。")

def detect_object_boundary(cam):
    """用Canny检测当前帧物体边界，返回边界像素mask"""
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray, 80, 200)
    _, edges2 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.bitwise_or(edges1, edges2)
    return edges > 0  # 返回bool型mask

def get_boundary_pointcloud(cam, boundary_mask):
    """用mask筛选点云，只保留边界点"""
    pc, _ = cam.render_pointcloud(world_frame=True)
    points = pc[boundary_mask]
    return points

def track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos, step_size=0.05, max_steps=200):
    """
    沿物体边界顺序推进采集点云，目标点推进法，避免横跳。
    - 目标点只有在到达（10像素内）才更新。
    - 目标点始终选当前点最远角点。
    - IK失败时可交互跳过或中断。
    """
    all_points = []
    start_pos = None
    start_recorded = False
    h, w = cam.res[1], cam.res[0]
    center_img = np.array([w // 2, h // 2])
    vis_pcd = o3d.geometry.PointCloud()
    vis_step = 0
    prev_target_pt = None
    prev_move_direction = None
    for step in range(max_steps):
        # 1. 边界检测
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            print(f"第{step+1}步，未检测到边界，停止")
            break
        # 2. 找到最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        # === 最大轮廓debug显示 ===
        debug_img = img.copy()
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 255), 2)  # 黄色线
        cv2.imshow('Max Contour Debug', debug_img)
        cv2.waitKey(1)
        # 3. 多边形逼近，提取角点
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 4. 找到距离中心最近的边界点
        min_dist = 1e9
        best_pt = None
        best_idx = None
        for i, pt in enumerate(cnt):
            pt_xy = pt[0]
            dist = np.linalg.norm(pt_xy - center_img)
            if dist < min_dist:
                min_dist = dist
                best_pt = pt_xy
                best_idx = i
        if best_pt is None:
            print(f"第{step+1}步，未找到最近边界点，停止")
            break
        # === 目标点推进逻辑 ===
        update_target = False
        if step == 0 or prev_target_pt is None:
            update_target = True
        else:
            # 判断是否到达目标点
            if np.linalg.norm(best_pt - prev_target_pt) < 10:
                update_target = True
        if update_target:
            # 选当前点最远角点为新目标点
            target_pt = None
            if len(approx) > 0:
                max_corner_dist = -1
                for k, corner in enumerate(approx):
                    corner_xy = corner[0]
                    dist = np.linalg.norm(corner_xy - best_pt)
                    if dist > max_corner_dist:
                        max_corner_dist = dist
                        target_pt = corner_xy
                print(f"第{step+1}步，检测到角点，选取距离best_pt最远的角点: {target_pt}")
            else:
                # 没有角点，按原逻辑在切线方向上选最远点
                if best_idx is None or cnt is None or len(cnt) == 0:
                    break
                prev_pt = cnt[(best_idx-1)%len(cnt)][0]
                next_pt = cnt[(best_idx+1)%len(cnt)][0]
                tangent = next_pt - prev_pt
                tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
                max_proj = 0
                target_pt = best_pt
                for pt in cnt:
                    pt_xy = pt[0]
                    v = pt_xy - best_pt
                    s = np.dot(v, tangent)
                    if s > max_proj:
                        max_proj = s
                        target_pt = pt_xy
            move_direction = target_pt - best_pt
            move_direction = move_direction / (np.linalg.norm(move_direction) + 1e-8)
            prev_target_pt = np.array(target_pt).copy() if target_pt is not None else None
            prev_move_direction = np.array(move_direction).copy() if move_direction is not None else None
        else:
            # 没到目标点，继续沿上一次方向前进
            move_direction = np.array(prev_move_direction).copy() if prev_move_direction is not None else np.array([1.0, 0.0])
            target_pt = np.array(prev_target_pt).copy() if prev_target_pt is not None else None
        # 可视化：画出所有角点、当前点、目标点
        vis_img = img.copy()
        for k, corner in enumerate(approx):
            cv2.circle(vis_img, tuple(corner[0]), 5, (0,255,0), -1)
        cv2.circle(vis_img, tuple(best_pt), 5, (0,0,255), -1)
        if target_pt is not None:
            cv2.circle(vis_img, tuple(target_pt), 7, (0,255,255), 2)  # 黄色圈目标点
        arrow_end = (int(best_pt[0] + 30*move_direction[0]), int(best_pt[1] + 30*move_direction[1]))
        cv2.arrowedLine(vis_img, tuple(best_pt), arrow_end, (255,0,0), 2)
        cv2.imshow('Boundary Tracking Debug', vis_img)
        cv2.waitKey(50)
        # 6. 记录起始位置
        if not start_recorded:
            start_pos = j6_link.get_pos().cpu().numpy()
            start_recorded = True
            print(f"记录起始位置: {start_pos}")
        # 7. 获取该点深度（用target_pt）
        if target_pt is None:
            print(f"第{step+1}步，未能确定目标点，停止")
            break
        cx, cy = int(target_pt[0]), int(target_pt[1])
        d = depth[cy, cx]
        if d <= 0.05:
            print(f"第{step+1}步，边界点深度无效: {d}")
            break
        # 8. 用相机内参将像素坐标和切线方向转换到相机坐标系
        K = cam.intrinsics
        x = (cx - K[0,2]) * d / K[0,0]
        y = (cy - K[1,2]) * d / K[1,1]
        z = d
        dx = move_direction[0] * d / K[0,0]
        dy = move_direction[1] * d / K[1,1]
        dz = 0
        tangent_cam = np.array([dx, dy, dz, 0])
        # 9. 转到世界坐标系
        T_cam2world = np.linalg.inv(cam.extrinsics)
        tangent_world = (T_cam2world @ tangent_cam)[:3]
        tangent_world = tangent_world / (np.linalg.norm(tangent_world) + 1e-8)
        # 10. 机械臂末端移动
        pos = j6_link.get_pos().cpu().numpy()
        target_pos = pos + tangent_world * step_size
        target_quat = j6_link.get_quat().cpu().numpy()
        # 11. 采集点云（只保留当前帧所有边界点）
        mask = np.zeros_like(edges, dtype=bool)
        for c in contours:
            for pt in c:
                mask[pt[0][1], pt[0][0]] = True
        pc, _ = cam.render_pointcloud(world_frame=True)
        points = pc[mask]
        all_points.append(points)
        # === 实时点云显示 ===
        vis_step += 1
        if vis_step % 2 == 0:
            vis_pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
            o3d.visualization.draw_geometries([vis_pcd], window_name="PointCloud Live", width=640, height=480)
        print(f"第{step+1}步，采集到{len(points)}个边界点")
        # 12. 检查是否回到起始位置
        if start_recorded:
            dist_to_start = np.linalg.norm(pos - start_pos)
            print(f"  距离起始位置: {dist_to_start:.4f}米")
            if dist_to_start < 0.02 and step >20:
                print(f"第{step+1}步，已回到起始位置附近，边界跟踪完成")
                break
        # 13. IK逆解
        try:
            qpos_ik = ed6.inverse_kinematics(link=j6_link, pos=target_pos, quat=target_quat)
            if hasattr(qpos_ik, 'cpu'):
                qpos_ik = qpos_ik.cpu().numpy()
            if np.any(np.isnan(qpos_ik)) or np.any(np.isinf(qpos_ik)):
                raise RuntimeError("IK解无效")
        except Exception as e:
            print(f"第{step+1}步，IK逆解失败: {e}")
            print("按Enter跳过，按Esc退出...")
            key = cv2.waitKey(0)
            if key == 13:  # Enter
                continue
            elif key == 27:  # Esc
                print("用户中断边界跟踪。")
                break
        # 14. 控制机械臂移动
        ed6.control_dofs_position(qpos_ik, motors_dof_idx)
        for _ in range(30):
            scene.step()
        cam.move_to_attach()
        print(f"第{step+1}步移动完成\n")
    cv2.destroyAllWindows()
    # 不再用Visualizer窗口，直接整体可视化
    if len(all_points) == 0:
        print("未采集到任何边界点云！")
        return np.zeros((0,3))
    all_points = np.concatenate(all_points, axis=0)
    print(f"边界跟踪完成，总共采集到{len(all_points)}个边界点")
    # === 采集完后整体可视化 ===
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud("boundary_cloud.ply", pcd)
    print("点云已保存为boundary_cloud.ply")
    o3d.visualization.draw_geometries([pcd])
    return all_points

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

def main():
    gs.init(seed=0, precision="32", logging_level="debug")
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 960),
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=90,
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

    cube = scene.add_entity(gs.morphs.Box(size=(0.105, 0.18, 0.022), pos=(0.35, 0.2, 0.02), collision=True))
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
    cube_pos = detect_cube_position(scene, ed6, cam, motors_dof_idx)
    reset_after_detection(scene, ed6, motors_dof_idx)  # 检测后平滑回零
    plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
    # === 新增：自动边界跟踪与点云采集 ===
    print("\n开始自动边界跟踪与点云采集...")
    time.sleep(5)
    points = track_and_scan_boundary(scene, ed6, cam, motors_dof_idx, j6_link, cube_pos)
    visualize_and_save_pointcloud(points)

if __name__ == "__main__":
    main()
