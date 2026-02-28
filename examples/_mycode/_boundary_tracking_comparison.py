# 三种边界追踪方法比较脚本
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
                    if key == 'esc' or key == 'q':
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

# 边界检测方法
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

def compare_all_methods(cam, show_images=True, step_num=None):
    """比较所有三种方法"""
    step_info = f" (Step {step_num})" if step_num is not None else ""
    
    # 获取所有三种方法的结果
    contours_canny, rgb, depth = detect_boundary_canny(cam)
    contours_rgbd, _, _ = detect_boundary_rgbd(cam)
    
    if ALPHA_SHAPE_AVAILABLE:
        contours_alpha, _, _ = detect_boundary_alpha_shape(cam)
    else:
        contours_alpha = []
    
    print(f"  方法比较{step_info}:")
    print(f"    Canny: {len(contours_canny)} 个轮廓")
    print(f"    RGB-D: {len(contours_rgbd)} 个轮廓")
    if ALPHA_SHAPE_AVAILABLE:
        print(f"    Alpha-Shape: {len(contours_alpha)} 个轮廓")
    else:
        print(f"    Alpha-Shape: 不可用")
    
    if show_images:
        # 创建比较图像
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        
        if ALPHA_SHAPE_AVAILABLE:
            # 创建三个子图像
            comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
            
            # Canny结果（绿色）
            canny_vis = img.copy()
            cv2.drawContours(canny_vis, contours_canny, -1, (0, 255, 0), 2)
            comparison[:, :w] = canny_vis
            
            # RGB-D结果（红色）
            rgbd_vis = img.copy()
            cv2.drawContours(rgbd_vis, contours_rgbd, -1, (255, 0, 0), 2)
            comparison[:, w:2*w] = rgbd_vis
            
            # Alpha-Shape结果（蓝色）
            alpha_vis = img.copy()
            cv2.drawContours(alpha_vis, contours_alpha, -1, (0, 0, 255), 2)
            comparison[:, 2*w:] = alpha_vis
            
            # 添加标签
            step_title = f"Step {step_num} - " if step_num is not None else ""
            cv2.putText(comparison, f"{step_title}Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, f"RGB-D", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, f"Alpha-Shape", (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(comparison, f"Contours: {len(contours_canny)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, f"Contours: {len(contours_rgbd)}", (w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, f"Contours: {len(contours_alpha)}", (2*w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Detection Methods Comparison (All)", comparison)
        else:
            # 只有两种方法
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            
            # Canny结果（绿色）
            canny_vis = img.copy()
            cv2.drawContours(canny_vis, contours_canny, -1, (0, 255, 0), 2)
            comparison[:, :w] = canny_vis
            
            # RGB-D结果（红色）
            rgbd_vis = img.copy()
            cv2.drawContours(rgbd_vis, contours_rgbd, -1, (255, 0, 0), 2)
            comparison[:, w:] = rgbd_vis
            
            # 添加标签
            step_title = f"Step {step_num} - " if step_num is not None else ""
            cv2.putText(comparison, f"{step_title}Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, f"RGB-D", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(comparison, f"Contours: {len(contours_canny)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(comparison, f"Contours: {len(contours_rgbd)}", (w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Detection Methods Comparison (Canny vs RGB-D)", comparison)
        
        cv2.waitKey(100)
    
    # 返回Canny结果作为默认（保持向后兼容）
    return contours_canny

def main():
    """主函数 - 三种方法比较"""
    print("三种边界追踪方法比较系统")
    print("=" * 50)
    
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

    cube = scene.add_entity(gs.morphs.Mesh(
        file="genesis/assets/_myobj/ctry.obj",
        pos=(0.35, 0.0, 0.02),
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
        fov=65,
        aperture=2.8,
        focus_dist=0.02,  
        GUI=True,
    )
    scene.build()
    motors_dof_idx = list(range(6))
    j6_link = ed6.get_link("J6")
    offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
    cam.attach(j6_link, offset_T)
    scene.step()
    cam.move_to_attach()
    
    # 初始化机械臂
    reset_arm(scene, ed6, motors_dof_idx)
    
    # 直接给定目标点
    cube_pos = np.array([0.35, -0.05, 0.02])
    print(f"直接设置目标位置: {cube_pos}")
    
    # 机械臂移动到目标位置
    plan_and_execute_path(scene, ed6, motors_dof_idx, j6_link, cube_pos, cam)
    
    # 开始三种方法比较
    print("\n开始三种边界检测方法比较...")
    time.sleep(3)
    
    # 运行比较
    for step in range(5):  # 比较5步
        print(f"\n==== 第{step+1}步方法比较 ====")
        compare_all_methods(cam, show_images=True, step_num=step+1)
        time.sleep(2)
    
    print("\n三种方法比较完成！")
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
