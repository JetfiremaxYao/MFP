#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试.obj文件加载的简单脚本
"""

import genesis as gs
import numpy as np

def test_obj_loading():
    """测试.obj文件加载"""
    print("初始化Genesis...")
    gs.init(seed=0, precision="32", logging_level="info")
    
    print("创建场景...")
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
    
    print("添加地面...")
    plane = scene.add_entity(gs.morphs.Plane(collision=True))
    
    print("尝试加载.obj文件...")
    try:
        phantom = scene.add_entity(gs.morphs.Mesh(
            file="genesis/assets/_myobj/circlephantom.obj",
            pos=(0.35, 0, 0.02),
            quat=[0, 0, 0.7071068, 0.7071068],  # 绕Z轴旋转90度
            collision=True,
            scale=0.1,  # 与主程序保持一致
            fixed=True  # 固定物体，防止移动
            # Genesis会自动加载同名的MTL文件
        ))
        print("✅ 成功加载.obj文件！")
        print(f"物体类型: {type(phantom)}")
        print(f"物体位置: {phantom.get_pos()}")
        
        # 尝试获取物体的边界框信息
        try:
            # 获取物体的边界框
            bounds = phantom.get_bounds()
            print(f"物体边界框: {bounds}")
        except Exception as e:
            print(f"无法获取边界框: {e}")
            
    except Exception as e:
        print(f"❌ 加载.obj文件失败: {e}")
        print("请检查文件路径和格式是否正确")
        return False
    
    print("构建场景...")
    scene.build()
    
    print("运行场景...")
    for i in range(100):
        scene.step()
        if i % 20 == 0:
            print(f"步骤 {i}")
    
    print("✅ 测试完成！")
    return True

if __name__ == "__main__":
    test_obj_loading()
