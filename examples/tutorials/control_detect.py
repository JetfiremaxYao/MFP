from re import T
import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from genesis.utils import geom as gu

########################## init ##########################
gs.init(backend=gs.gpu, seed=0, precision="32", logging_level="debug")

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        # 不出现相机实体
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

########################## entities ##########################
# add plane
plane = scene.add_entity(
    gs.morphs.Plane(
        collision=True,
    )
)

# add cube
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.35, 0.0, 0.02),
        collision=True,
        
    )
)

# add ED6 robot
ed6 = scene.add_entity(
    gs.morphs.URDF(
        file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
        scale=1.0,
        requires_jac_and_IK=True,
        fixed=True,  # 固定机械臂在原点，防止摔倒
    )
)

# add camera attached to J6 end effector
cam = scene.add_camera(
    model="thinlens",
    res=(640, 480),
    pos=(0, 0, 0),  # 初始位置，后续会更新
    lookat=(0, 0, 1),  # 初始朝向，后续会更新
    up=(0, 0, 1),
    fov=30,
    aperture=2.8,      # 景深光圈
    focus_dist=0.5,    # 对焦距离
    GUI=True,  # 弹出相机窗口
)

########################## build ##########################
scene.build()

# 获取ED6机械臂的关节信息
# joints_name = ("J1", "J2", "J3", "J4", "J5", "J6")
# motors_dof_idx = []
# for name in joints_name:
#     joint = ed6.get_joint(name)
#     motors_dof_idx.append(joint.dofs_idx_local[0])

motors_dof_idx = list(range(6))  # ED6为6自由度机械臂

print(f"ED6机械臂关节数量: {len(motors_dof_idx)}")
print(f"关节DOF索引: {motors_dof_idx}")

############ 设置控制参数 ############
# 设置位置增益 (PD控制器的P参数)
# ed6.set_dofs_kp(
#     kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]),
#     dofs_idx_local=motors_dof_idx,
# )
ed6.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000]), dofs_idx_local=motors_dof_idx)
# 设置速度增益 (PD控制器的D参数)
ed6.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200]), dofs_idx_local=motors_dof_idx)
# 设置力范围限制（安全保护）
ed6.set_dofs_force_range(
    lower=np.array([-87, -87, -87, -87, -12, -12]),
    upper=np.array([87, 87, 87, 87, 12, 12]),
    dofs_idx_local=motors_dof_idx,
)

############ 硬重置 - 设置初始姿态 ############
print("执行硬重置，设置初始姿态...")
for i in range(150):
    if i < 50:
        # 第一个姿态
        ed6.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), motors_dof_idx)
    elif i < 100:
        # 第二个姿态
        ed6.set_dofs_position(np.array([1, 0.5, 0.5, -1, 0.5, 0]), motors_dof_idx)
    else:
        # 回到零位
        ed6.set_dofs_position(np.array([0, 0, 0, 0, 0, 0]), motors_dof_idx)
    scene.step()

############ 相机挂载到机械臂末端 ############
print("挂载相机到J6末端...")
j6_link = ed6.get_link("J6")
# 相机刚性挂载在J6末端法兰平面上（z轴负方向0.08m，与地面夹角135度）
offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.08]), np.array([0, 1, 0, 0]))
cam.attach(j6_link, offset_T)
scene.step()           # 让 J6 的全局位姿先更新一帧
cam.move_to_attach() 
print("相机挂载完成")
time.sleep(3)

############## 固定末端姿态与位置，只用J1旋转环视 ##############
# 只用J1旋转，J4/J5等关节角度已规定好，环视采集无需IK
qpos_scan = np.zeros(6)

qpos_scan[3] = -np.pi / 3   # J4 -60
qpos_scan[4] = np.pi / 2   # J5 +90°
print(f"环视初始关节角度: {qpos_scan}")

num_views = 24 # 一圈，采集24个点
start_angle = 0
angles = np.linspace(start_angle, 2 * np.pi + start_angle, num_views, endpoint=False)
print("开始J1扫视一圈采集...")
results = []
traj_positions = []  # 记录所有采样点末端位置
# 环视前，先将机械臂设置为环视初始姿态并打印
qpos_scan[0] = 0  # 强制J1初始为0
ed6.control_dofs_position(qpos_scan, motors_dof_idx)
for step in range(100):
    scene.step()
print(f"[环视初始] 设定J1角度: {qpos_scan[0]:.4f}")
cam.move_to_attach()
j6_pos = ed6.get_link("J6").get_pos().cpu().numpy()
j6_quat = ed6.get_link("J6").get_quat().cpu().numpy()
print("\n机械臂环视姿态已生成：")
print(f"J6末端位置: {j6_pos}")
print(f"J6末端姿态四元数: {j6_quat}")
print(f"当前关节角度: {qpos_scan}")
print(f"当前J1角度: {qpos_scan[0]:.4f}")
print("2秒后开始环视检测物体...\n")
time.sleep(2)

settle_steps = 100
for i, angle in enumerate(angles):
    print(f"\n=== J1角度 {np.rad2deg(angle):.1f}° (第{i+1}/{num_views}点) ===")
    qpos = qpos_scan.copy()
    qpos[0] = angle  # 只动J1
    print(f"[采集前] 设定J1角度: {qpos[0]:.4f}")
    ed6.control_dofs_position(qpos, motors_dof_idx)
    for step in range(settle_steps):
        scene.step()
    cam.move_to_attach()
    # 打印实际J1角度（如有接口）
    print(f"[采集后] 设定J1角度: {qpos[0]:.4f}")
    # 打印机械臂末端位置、姿态、关节角度
    j6_pos = ed6.get_link("J6").get_pos().cpu().numpy()
    j6_quat = ed6.get_link("J6").get_quat().cpu().numpy()
    print(f"J6末端位置: {j6_pos}")
    print(f"J6末端姿态四元数: {j6_quat}")
    print(f"当前关节角度: {qpos}")
    print(f"当前J1角度: {qpos[0]:.4f}")
    traj_positions.append(j6_pos)
    print("相机位置已更新")
    print("暂停2秒，观察当前位姿...")
    time.sleep(0.5)
    print("正在采集图像...")
    try:
        import cv2
    except ImportError:
        import os
        os.system('pip install opencv-python')
        import cv2
    rgb, depth, _, _ = cam.render(rgb=True, depth=True)
    # OpenCV颜色分割（假设cube为灰色/白色，HSV阈值可调整）
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
                print(f"[DEBUG] 检测像素: ({cx}, {cy}), 深度: {d:.4f}")
                print("[DEBUG] 相机内参K:\n", K)
                x = (cx - K[0,2]) * d / K[0,0]
                y = (cy - K[1,2]) * d / K[1,1]
                z = d
                print(f"[DEBUG] 相机坐标系下: x={x:.4f}, y={y:.4f}, z={z:.4f}")
                # 不取反y/z
                pos_cam = np.array([x, y, z, 1])
                print("[DEBUG] pos_cam: ", pos_cam)
                # 用外参逆矩阵
                T_cam2world = np.linalg.inv(cam.extrinsics)
                print("[DEBUG] 用逆矩阵 T_cam2world:\n", T_cam2world)
                pos_world = T_cam2world @ pos_cam
                print("[DEBUG] 转换后世界坐标: ", pos_world[:3])
                results.append({
                    'cx': cx, 'cy': cy, 'area': area, 'd': d,
                    'pos_world': pos_world[:3],
                    'dist_to_center': dist_to_center,
                    'angle': angle
                })
                print(f"第{i+1}帧检测到cube，J1角度={np.rad2deg(angle):.1f}°")
                input("已检测到cube，按回车继续...")
                
    else:
        cv2.imshow("Detection", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

if not results:
    raise RuntimeError("环视未检测到cube，请调整颜色阈值或cube位置！")

# 选面积最大且最靠近图像中心的帧
best = max(results, key=lambda r: (r['area'], -r['dist_to_center']))
cube_pos = best['pos_world']
print(f"环视完成，最终选定cube世界坐标: {cube_pos}")

# 打印当前末端姿态四元数（便于调试）
print("当前J6末端姿态四元数：", j6_link.get_quat().cpu().numpy())




# ========== 回到初始位置（零位）并等待 ==========
print("机械臂回到初始零位...")
init_qpos = np.zeros(6)  # 所有关节都为0，包括J4/J5
ed6.set_dofs_position(init_qpos, motors_dof_idx)
for _ in range(100):
    scene.step()
print("机械臂已回到初始零位，等待2秒...")
time.sleep(2)

# ========== 路径规划与IK ==========
qpos_start = init_qpos.copy()
target_quat = np.array([0, 1, 0, 0])
qpos_ik = ed6.inverse_kinematics(
    link=j6_link,
    pos=cube_pos,
    quat=target_quat,
)
print("IK逆解目标关节角度:", qpos_ik)

try:
    path = ed6.plan_path(
        qpos_goal=qpos_ik,
        num_waypoints=200,
    )
    if len(path) == 0:
        raise RuntimeError("plan_path返回空路径，自动切换为线性插值")
except Exception as e:
    print("plan_path失败，切换为线性插值:", e)
    import torch
    if isinstance(qpos_ik, torch.Tensor):
        qpos_ik = qpos_ik.detach().cpu().numpy()
    if isinstance(qpos_start, torch.Tensor):
        qpos_start = qpos_start.detach().cpu().numpy()
    path = np.linspace(qpos_start, qpos_ik, num=200)
    # 转为torch tensor，兼容draw_debug_path
    path = torch.from_numpy(path).float().cpu()

print("路径插值完成，路径点数:", len(path))
# 可视化路径
path_debug = scene.draw_debug_path(path, ed6)
scene.step()  # 确保可视化生效

print("5秒后开始沿IK路径运动...")
time.sleep(5)
for idx, waypoint in enumerate(path):
    ed6.control_dofs_position(waypoint, motors_dof_idx)
    for _ in range(3):
        scene.step()
    if idx % 20 == 0:
        print(f"[路径跟踪] 进度: {idx+1}/{len(path)}  J1角度: {waypoint[0]:.4f}")

print("路径执行完毕")
time.sleep(2)
# scene.clear_debug_object(path_debug)

print("IK运动完成！相机窗口会保持打开，你可以观察J6末端的视角。按Ctrl+C退出程序。")

# 保持程序运行，显示相机画面
try:
    while True:
        scene.step()
        rgb, depth, segmentation, normal = cam.render(
            rgb=True, depth=True, segmentation=True, normal=True
        )
except KeyboardInterrupt:
    print("程序已退出。") 

# 采集结束后打印所有采样点末端位置轨迹
print("\n所有采样点末端位置轨迹（xyz）：")
for idx, pos in enumerate(traj_positions):
    print(f"点{idx+1}: {pos}") 

    