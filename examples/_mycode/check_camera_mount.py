import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu

# 初始化
# 注意：如你的环境不是cuda可用，可改为cpu
gs.init(backend=gs.gpu, seed=0, precision="32", logging_level="debug")

# 创建场景
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
        show_world_frame=True,  # 便于观察
        world_frame_size=1.0,
        show_link_frame=True,   # 显示机械臂各link坐标系
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

# 加载机械臂
ed6 = scene.add_entity(
    gs.morphs.URDF(
        file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
        scale=1.0,
        requires_jac_and_IK=True,
        fixed=True,
    )
)

# 添加相机
cam = scene.add_camera(
    model="thinlens",
    res=(640, 480),
    pos=(0, 0, 0),  # 初始位置
    lookat=(0, 0, 1),
    up=(0, 0, 1),
    fov=30,
    aperture=2.8,
    focus_dist=0.5,
    GUI=True,
)

scene.build()

# 获取J6末端
j6_link = ed6.get_link("J6")
# 挂载相机，z轴负方向0.08m
offset_T = gu.trans_quat_to_T(np.array([0, 0, 0.08]), np.array([0, 1, 0, 0]))
cam.attach(j6_link, offset_T)

print("相机已挂载到J6末端，窗口显示为相机视角。按Ctrl+C退出。")

# 保持仿真运行，持续显示相机画面
try:
    while True:
        scene.step()
        cam.move_to_attach()
        rgb, depth, seg, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
        time.sleep(0.01)
except KeyboardInterrupt:
    print("退出。") 