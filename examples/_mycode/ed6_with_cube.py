import genesis as gs
import numpy as np

########################## init ##########################
gs.init(seed=0, precision="32", logging_level="debug")

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
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
scene.add_entity(gs.morphs.Plane())

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

########################## build ##########################
scene.build()

# get end effector link (J6)
j6_link = ed6.get_link("J6")

# 目标位置：cube正上方（cube中心上方0.04m）
cube_pos = np.array([0.65, 0.0, 0.02])
target_pos = cube_pos + np.array([0, 0, 0.04 + 0.10])  # 末端在cube正上方10cm
# 末端姿态：保持默认
quat = np.array([0, 1, 0, 0])

# 逆解求解
qpos = ed6.inverse_kinematics(
    link=j6_link,
    pos=target_pos,
    quat=quat,
)
ed6.set_qpos(qpos)

print("J6末端目标世界坐标：", j6_link.get_pos().cpu().numpy())

# 保持仿真窗口
while True:
    scene.step()