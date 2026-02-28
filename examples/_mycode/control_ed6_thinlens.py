import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        # 不展示相机只实现功能
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
plane = scene.add_entity(gs.morphs.Plane())

# add cube
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
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

# add thinlens camera attached to J6 end effector
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
joints_name = ("J1", "J2", "J3", "J4", "J5", "J6")
motors_dof_idx = []
for name in joints_name:
    joint = ed6.get_joint(name)
    motors_dof_idx.append(joint.dofs_idx_local[0])

print(f"ED6机械臂关节数量: {len(motors_dof_idx)}")
print(f"关节名称: {joints_name}")
print(f"关节DOF索引: {motors_dof_idx}")

############ 设置控制参数 ############
# 设置位置增益 (PD控制器的P参数)
ed6.set_dofs_kp(
    kp=np.array([4500, 4500, 3500, 3500, 2000, 2000]),
    dofs_idx_local=motors_dof_idx,
)
# 设置速度增益 (PD控制器的D参数)
ed6.set_dofs_kv(
    kv=np.array([450, 450, 350, 350, 200, 200]),
    dofs_idx_local=motors_dof_idx,
)
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

############ PD控制演示 ############
print("开始PD控制演示...")
for i in range(1250):
    # 更新相机位置到J6末端，朝向z轴正方向，up为x轴方向
    j6_link = ed6.get_link("J6")
    j6_pos = j6_link.get_pos().cpu().numpy()
    j6_quat = j6_link.get_quat().cpu().numpy()
    r = R.from_quat([j6_quat[1], j6_quat[2], j6_quat[3], j6_quat[0]])
    z_axis_world = r.apply([0, 0, 1])
    x_axis_world = r.apply([1, 0, 0])
    cam.set_pose(
        pos=j6_pos,
        lookat=j6_pos + z_axis_world,
        up=x_axis_world,
    )
    
    # 不同时间点的控制命令
    if i == 0:
        # 第一个目标位置
        target_pos = np.array([0.5, 0.2, 0.3, -0.5, 0.8, 0.2])
        ed6.control_dofs_position(target_pos, motors_dof_idx)
        print(f"步骤 {i}: 移动到位置 {target_pos}")
        
    elif i == 250:
        # 第二个目标位置
        target_pos = np.array([-0.3, 0.8, 0.5, -1.2, 0.3, -0.5])
        ed6.control_dofs_position(target_pos, motors_dof_idx)
        print(f"步骤 {i}: 移动到位置 {target_pos}")
        
    elif i == 500:
        # 第三个目标位置
        target_pos = np.array([0, 0, 0, 0, 0, 0])
        ed6.control_dofs_position(target_pos, motors_dof_idx)
        print(f"步骤 {i}: 回到零位 {target_pos}")
        
    elif i == 750:
        # 混合控制：前3个关节用位置控制，后3个关节用速度控制
        ed6.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0])[:3],
            motors_dof_idx[:3],
        )
        ed6.control_dofs_velocity(
            np.array([0.5, 0.3, 0.2, 0, 0, 0])[3:],
            motors_dof_idx[3:],
        )
        print(f"步骤 {i}: 混合控制模式")
        
    elif i == 1000:
        # 力控制模式
        ed6.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0]),
            motors_dof_idx,
        )
        print(f"步骤 {i}: 力控制模式")
    
    # 渲染相机图像
    if i % 10 == 0:  # 每10步渲染一次，提高性能
        rgb, depth, segmentation, normal = cam.render(
            rgb=True, depth=True, segmentation=True, normal=True
        )
    
    # 打印控制力和实际力（每100步打印一次，避免输出过多）
    if i % 100 == 0:
        control_force = ed6.get_dofs_control_force(motors_dof_idx)
        internal_force = ed6.get_dofs_force(motors_dof_idx)
        print(f"步骤 {i}: 控制力={control_force}, 实际力={internal_force}")
    
    scene.step()

print("控制演示完成！")
print("相机窗口会保持打开，你可以观察J6末端的视角。")
print("按Ctrl+C退出程序。")

# 保持程序运行，显示相机画面
try:
    while True:
        scene.step()
        # 继续更新相机位置
        j6_link = ed6.get_link("J6")
        j6_pos = j6_link.get_pos().cpu().numpy()
        j6_quat = j6_link.get_quat().cpu().numpy()
        r = R.from_quat([j6_quat[1], j6_quat[2], j6_quat[3], j6_quat[0]])
        z_axis_world = r.apply([0, 0, 1])
        x_axis_world = r.apply([1, 0, 0])
        cam.set_pose(
            pos=j6_pos,
            lookat=j6_pos + z_axis_world,
            up=x_axis_world,
        )
        rgb, depth, segmentation, normal = cam.render(
            rgb=True, depth=True, segmentation=True, normal=True
        )
except KeyboardInterrupt:
    print("程序已退出。") 