#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的自动化边界追踪评估系统
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_fixed_system():
    """测试修复后的系统"""
    print("测试修复后的自动化边界追踪评估系统")
    print("=" * 50)
    
    try:
        # 导入必要的模块
        import genesis as gs
        print("✓ Genesis 导入成功")
        
        from _boundary_tracking_evaluation_auto import BoundaryTrackingEvaluatorAuto
        print("✓ 自动化评估器导入成功")
        
        # 创建评估器（最小配置）
        evaluator = BoundaryTrackingEvaluatorAuto(
            cube_size=[0.105, 0.18, 0.022],
            base_cube_pos=[0.35, 0, 0.02],
            experiment_name="test_fixed"
        )
        
        # 修改配置以减少测试时间
        evaluator.n_runs = 1  # 只运行1次
        evaluator.lighting_conditions = ["normal"]  # 只测试正常光照
        
        print("✓ 评估器创建成功")
        print(f"  结果目录: {evaluator.results_dir}")
        print(f"  实验次数: {evaluator.n_runs}")
        print(f"  光照条件: {evaluator.lighting_conditions}")
        
        # 测试场景创建方法
        print("\n测试场景创建方法...")
        scene, ed6, cam, motors_dof_idx, j6_link = evaluator._create_scene_with_lighting("normal")
        print("✓ 场景创建成功")
        
        # 清理场景
        del scene
        print("✓ 场景清理成功")
        
        print("\n✅ 修复验证完成！系统应该可以正常运行了。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_system()
    sys.exit(0 if success else 1)
