import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu

gs.init(backend=gs.gpu, seed=0, precision="32", logging_level="debug")

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

plane = scene.add_entity(gs.morphs.Plane(collision=True))

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.1, 0.1, 0.1),
        pos=(0.34586686, -0.0005347, 0.10007912),
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
    model="thinlens",
    res=(640, 480),
    pos=(0, 0, 0),
    lookat=(0, 0, 1),
    up=(0, 0, 1),
    fov=30,
    aperture=2.8,
    focus_dist=0.5,
    GUI=True,
)

scene.build()

print("场景已加载，cube已放置在指定位置：", cube.get_pos().cpu().numpy())

# 可视化保持
try:
    while True:
        scene.step()
except KeyboardInterrupt:
    print("程序已退出。") 