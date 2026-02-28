#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的边界检测测试脚本
"""

import genesis as gs
import numpy as np
import cv2
import time

def simple_boundary_test():
    """简单的边界检测测试"""
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
    
    print("加载phantom物体...")
    phantom = scene.add_entity(gs.morphs.Mesh(
        file="genesis/assets/_myobj/phantom.stl",
        pos=(0.35, 0, 0.02),
        quat=[0, 0, 0.7071068, 0.7071068],  # 绕Z轴旋转90度
        collision=True,
        scale=0.1,
        fixed=True
    ))
    
    print("添加摄像机...")
    cam = scene.add_camera(
        model="thinlens",
        res=(640, 480),
        pos=(0.35, 0, 0.15),  # 放在物体上方
        lookat=(0.35, 0, 0.02),  # 看向物体中心
        up=(0, 0, 1),
        fov=60,
        aperture=2.8,
        focus_dist=0.02,
        GUI=True,
    )
    
    print("构建场景...")
    scene.build()
    
    print("开始边界检测测试...")
    print("按ESC退出...")
    
    step_count = 0
    while True:
        step_count += 1
        print(f"\n=== 第{step_count}步测试 ===")
        
        try:
            # 获取图像
            rgb, depth, _, _ = cam.render(rgb=True, depth=True)
            print(f"图像尺寸: RGB={rgb.shape}, Depth={depth.shape}")
            
            # 转换为OpenCV格式
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # 显示原始图像
            cv2.imshow('Original Image', img)
            
            # 显示深度图像
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow('Depth Map', depth_colored)
            
            # 简单的Canny边缘检测
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 尝试不同的阈值
            edges_low = cv2.Canny(gray, 30, 100)
            edges_high = cv2.Canny(gray, 80, 200)
            
            cv2.imshow('Canny Low (30,100)', edges_low)
            cv2.imshow('Canny High (80,200)', edges_high)
            
            # 检测轮廓
            contours_low, _ = cv2.findContours(edges_low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_high, _ = cv2.findContours(edges_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            print(f"检测到轮廓数量 - 低阈值: {len(contours_low)}, 高阈值: {len(contours_high)}")
            
            # 显示轮廓
            img_with_contours_low = img.copy()
            img_with_contours_high = img.copy()
            
            # 绘制低阈值轮廓（绿色）
            for cnt in contours_low:
                area = cv2.contourArea(cnt)
                if area > 5:  # 还原到简单的面积限制
                    cv2.drawContours(img_with_contours_low, [cnt], -1, (0, 255, 0), 2)
                    # 显示面积
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(img_with_contours_low, f"{area:.0f}", (cx, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 绘制高阈值轮廓（红色）
            for cnt in contours_high:
                area = cv2.contourArea(cnt)
                if area > 5:  # 还原到简单的面积限制
                    cv2.drawContours(img_with_contours_high, [cnt], -1, (0, 0, 255), 2)
                    # 显示面积
                    M = cv2.moments(cnt)
                    if M["m01"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(img_with_contours_high, f"{area:.0f}", (cx, cy), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.imshow('Contours Low Threshold', img_with_contours_low)
            cv2.imshow('Contours High Threshold', img_with_contours_high)
            
            # 打印深度信息
            valid_depth = depth[depth > 0.05]
            if len(valid_depth) > 0:
                print(f"深度范围: {np.min(valid_depth):.3f}m - {np.max(valid_depth):.3f}m")
                print(f"有效深度点数: {len(valid_depth)}")
            else:
                print("警告: 没有有效的深度数据！")
            
        except Exception as e:
            print(f"第{step_count}步出现异常: {e}")
            import traceback
            traceback.print_exc()
        
        # 检查按键
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            break
        
        # 场景步进
        scene.step()
        
        # 每10步暂停一下
        if step_count % 10 == 0:
            print(f"已完成{step_count}步，按任意键继续...")
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\n测试完成！")

if __name__ == "__main__":
    simple_boundary_test()
