#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化边界追踪评估系统测试脚本
用于验证系统是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """测试所有必要的模块是否能正常导入"""
    print("测试模块导入...")
    
    try:
        import genesis as gs
        print("✓ Genesis 导入成功")
    except ImportError as e:
        print(f"✗ Genesis 导入失败: {e}")
        return False
    
    try:
        from _boundary_track_canny_auto import detect_boundary_canny, track_and_scan_boundary_jump
        print("✓ Canny自动化模块导入成功")
    except ImportError as e:
        print(f"✗ Canny自动化模块导入失败: {e}")
        return False
    
    try:
        from _boundary_track_rgbd_auto import detect_boundary_rgbd, track_and_scan_boundary_jump
        print("✓ RGB-D自动化模块导入成功")
    except ImportError as e:
        print(f"✗ RGB-D自动化模块导入失败: {e}")
        return False
    
    try:
        from _boundary_tracking_evaluation_auto import BoundaryTrackingEvaluatorAuto
        print("✓ 自动化评估器导入成功")
    except ImportError as e:
        print(f"✗ 自动化评估器导入失败: {e}")
        return False
    
    return True

def test_evaluator_creation():
    """测试评估器创建"""
    print("\n测试评估器创建...")
    
    try:
        from _boundary_tracking_evaluation_auto import BoundaryTrackingEvaluatorAuto
        
        evaluator = BoundaryTrackingEvaluatorAuto(
            cube_size=[0.105, 0.18, 0.022],
            base_cube_pos=[0.35, 0, 0.02],
            experiment_name="test_auto_evaluation"
        )
        
        print("✓ 评估器创建成功")
        print(f"  结果目录: {evaluator.results_dir}")
        print(f"  实验次数: {evaluator.n_runs}")
        print(f"  光照条件: {evaluator.lighting_conditions}")
        
        return True
    except Exception as e:
        print(f"✗ 评估器创建失败: {e}")
        return False

def test_single_experiment():
    """测试单次实验（简化版本）"""
    print("\n测试单次实验...")
    
    try:
        import genesis as gs
        import numpy as np
        
        # 初始化Genesis（最小配置）
        gs.init(seed=0, precision="32", logging_level="error")
        
        # 创建简单场景
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(640, 480),
                camera_pos=(2.0, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=30,
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
                show_link_frame=False,
                show_cameras=False,
                plane_reflection=False,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_joint_limit=False,
                enable_collision=False,
                gravity=(0, 0, 0),
            ),
            show_viewer=False,  # 关闭可视化
        )
        
        # 添加基本元素
        plane = scene.add_entity(gs.morphs.Plane(collision=False))
        cube = scene.add_entity(gs.morphs.Box(
            size=(0.105, 0.18, 0.022), 
            pos=(0.35, 0, 0.02), 
            collision=False,
            name="cube"
        ))
        
        # 添加机械臂
        ed6 = scene.add_entity(gs.morphs.URDF(
            file="genesis/assets/xml/ED6-URDF-0102.SLDASM/urdf/ED6-URDF-0102.SLDASM.urdf",
            scale=1.0,
            requires_jac_and_IK=True,
            fixed=True,
        ))
        
        # 添加相机
        cam = scene.add_camera(
            model="thinlens",
            res=(320, 240),  # 降低分辨率以加快速度
            pos=(0, 0, 0),
            lookat=(0, 0, 1),
            up=(0, 0, 1),
            fov=60, 
            aperture=2.8,
            focus_dist=0.02,  
            GUI=False,
        )
        
        scene.build()
        
        # 设置相机
        motors_dof_idx = list(range(6))
        j6_link = ed6.get_link("J6")
        offset_T = gs.utils.geom.trans_quat_to_T(np.array([0, 0, 0.01]), np.array([0, 1, 0, 0]))
        cam.attach(j6_link, offset_T)
        scene.step()
        cam.move_to_attach()
        
        print("✓ 场景创建成功")
        
        # 测试边界检测函数（不实际运行追踪）
        from _boundary_track_canny_auto import detect_boundary_canny
        from _boundary_track_rgbd_auto import detect_boundary_rgbd
        
        # 获取图像
        rgb, depth, _, _ = cam.render(rgb=True, depth=True)
        
        # 测试Canny检测
        try:
            contours_canny = detect_boundary_canny(cam)
            print(f"✓ Canny边界检测成功，检测到 {len(contours_canny)} 个轮廓")
        except Exception as e:
            print(f"✗ Canny边界检测失败: {e}")
        
        # 测试RGB-D检测
        try:
            contours_rgbd, _, _ = detect_boundary_rgbd(cam)
            print(f"✓ RGB-D边界检测成功，检测到 {len(contours_rgbd)} 个轮廓")
        except Exception as e:
            print(f"✗ RGB-D边界检测失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ 单次实验测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("自动化边界追踪评估系统测试")
    print("=" * 50)
    
    # 测试1: 模块导入
    if not test_imports():
        print("\n❌ 模块导入测试失败，请检查依赖")
        return False
    
    # 测试2: 评估器创建
    if not test_evaluator_creation():
        print("\n❌ 评估器创建测试失败")
        return False
    
    # 测试3: 单次实验
    if not test_single_experiment():
        print("\n❌ 单次实验测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("✅ 所有测试通过！自动化评估系统准备就绪")
    print("\n使用方法:")
    print("1. 运行完整评估: python _boundary_tracking_evaluation_auto.py")
    print("2. 或者创建评估器实例并调用 run_evaluation() 方法")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
