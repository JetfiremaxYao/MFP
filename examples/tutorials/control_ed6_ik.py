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
    res=(640, 480),
    pos=(0, 0, 0),  # 初始位置，后续会更新
    lookat=(0, 0, -1),  # 朝向J6的z轴负方向（向下）
    fov=30,
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

############ IK运动规划到cube正上方 ############
# 获取cube中心和高度
cube_pos = np.array(cube.get_pos().cpu())
cube_size = np.array(cube.size) if hasattr(cube, 'size') else np.array([0.1, 0.1, 0.1])
cube_height = cube_size[2]

# 目标点：cube中心正上方，z轴+cube高度/2+0.15
target_pos = cube_pos.copy()
target_pos[2] += cube_height / 2 + 0.15

# 末端link对象
j6_link = ed6.get_link("J6")

# 打印当前末端姿态四元数（便于调试）
print("当前J6末端姿态四元数：", j6_link.get_quat().cpu().numpy())

# 指定目标姿态（z轴朝下，默认[0,1,0,0]，如需可用上面打印的四元数）
target_quat = np.array([0, 1, 0, 0])
qpos_ik = ed6.inverse_kinematics(
    link=j6_link,
    pos=target_pos,
    quat=target_quat,
)

# 规划路径（插值100个点）
path = ed6.plan_path(
    qpos_goal=qpos_ik,
    num_waypoints=100,
)

# 可视化路径
path_debug = scene.draw_debug_path(path, ed6)

# 相机刚性挂载在J6末端法兰平面上（z轴负方向0.08m）
offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.08]), np.array([0, 1, 0, 0]))
cam.attach(j6_link, offset_T)

print("5秒后开始沿IK路径运动...")
time.sleep(5)
for waypoint in path:
    ed6.control_dofs_position(waypoint, motors_dof_idx)
    rgb, depth, segmentation, normal = cam.render(
        rgb=True, depth=True, segmentation=True, normal=True
    )
    scene.step()
scene.clear_debug_object(path_debug)

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