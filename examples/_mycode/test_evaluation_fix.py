#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的边界追踪评估系统
"""
import genesis as gs
import numpy as np
import time
from genesis.utils import geom as gu
import cv2
import open3d as o3d
from sklearn.decomposition import PCA
import threading
import sys
import select
import os
from math import ceil
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy import ndimage
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_single_run():
    """测试单次实验运行"""
    try:
        from boundary_tracking_evaluation import BoundaryTrackingEvaluator
        
        print("=== 测试单次实验运行 ===")
        
        # 创建评估器
        evaluator = BoundaryTrackingEvaluator(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0, 0.02]),
            experiment_name="test_single_run"
        )
        
        # 只运行一次Canny实验
        print("运行Canny方法测试...")
        cube_pos = np.array([0.35, 0, 0.02])
        result = evaluator._run_single_experiment('canny', cube_pos, 0)
        
        print(f"测试结果: {result['status']}")
        print(f"采集点数: {result['point_count']}")
        
        if result['status'] != 'Fail':
            print("✅ 单次实验测试成功！")
            return True
        else:
            print(f"❌ 单次实验测试失败: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_projection():
    """测试投影功能"""
    try:
        from boundary_tracking_evaluation import BoundaryTrackingEvaluator
        
        print("\n=== 测试投影功能 ===")
        
        evaluator = BoundaryTrackingEvaluator()
        
        # 创建测试点云
        test_points = np.array([
            [0.35, 0, 0.031],
            [0.35, 0.09, 0.031],
            [0.455, 0, 0.031],
            [0.455, 0.09, 0.031]
        ])
        
        # 测试投影
        points_2d, transform_matrix = evaluator._project_to_scanning_plane(test_points)
        
        print(f"原始点云形状: {test_points.shape}")
        print(f"投影后形状: {points_2d.shape}")
        print(f"变换矩阵形状: {transform_matrix.shape}")
        
        if not np.any(np.isnan(points_2d)) and not np.any(np.isinf(points_2d)):
            print("✅ 投影功能测试成功！")
            return True
        else:
            print("❌ 投影结果包含无效值")
            return False
            
    except Exception as e:
        print(f"❌ 投影测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("边界追踪评估系统修复测试")
    print("=" * 50)
    
    # 检查依赖
    try:
        import numpy as np
        import genesis as gs
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 依赖检查失败: {e}")
        return
    
    # 测试投影功能
    projection_success = test_projection()
    
    # 测试单次实验（可选，因为需要较长时间）
    print("\n是否要测试单次实验运行？(y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes', '是']:
        experiment_success = test_single_run()
        if projection_success and experiment_success:
            print("\n🎉 所有测试通过！系统修复成功。")
        else:
            print("\n⚠️ 部分测试失败，需要进一步修复。")
    else:
        if projection_success:
            print("\n🎉 投影功能测试通过！")
        else:
            print("\n⚠️ 投影功能测试失败，需要进一步修复。")

if __name__ == "__main__":
    main()
