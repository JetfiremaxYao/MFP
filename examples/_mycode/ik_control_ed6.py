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
plane = scene.add_entity(gs.morphs.Plane(collision=True))
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
        collision=True,
    )
)
ed6 = scene.add_entity(
    gs.morphs.URDF(
        file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
        scale=1.0,
        requires_jac_and_IK=True,
        fixed=True,
    )
)
cam = scene.add_camera(
    res=(640, 480),
    pos=(0, 0, 0),
    lookat=(0, 0, -1),
    fov=30,
    GUI=True,
)

########################## build ##########################
scene.build()

# 获取关节索引
joints_name = ("J1", "J2", "J3", "J4", "J5", "J6")
motors_dof_idx = [ed6.get_joint(name).dofs_idx_local[0] for name in joints_name]

# 1. 机械臂初始化为“直立零位”
zero_pos = np.zeros(6)
ed6.control_dofs_position(zero_pos, motors_dof_idx)
for _ in range(100):
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
    cam.render(rgb=True)  # 强制刷新相机窗口
    scene.step()

# 2. IK+路径规划到cube上方
cube_pos = np.array([0.65, 0.0, 0.02])
target_pos = cube_pos + np.array([0, 0, 0.04 + 0.10])
target_quat = np.array([0, 1, 0, 0])
j6_link = ed6.get_link("J6")
qpos_goal = ed6.inverse_kinematics(
    link=j6_link,
    pos=target_pos,
    quat=target_quat,
)
path = ed6.plan_path(
    qpos_goal=qpos_goal,
    num_waypoints=200,
)
for waypoint in path:
    ed6.control_dofs_position(waypoint, motors_dof_idx)
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
    cam.render(rgb=True)
    scene.step()

print("机械臂已对准cube上方，J6法兰面朝向cube。相机实时跟随J6末端。")

# 保持程序运行，显示相机画面
try:
    while True:
        scene.step()
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
        cam.render(rgb=True)
except KeyboardInterrupt:
    print("程序已退出。") 