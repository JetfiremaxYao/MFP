#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试自动化边界追踪评估系统
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def quick_test():
    """快速测试自动化评估系统"""
    print("快速测试自动化边界追踪评估系统")
    print("=" * 50)
    
    try:
        # 导入必要的模块
        import genesis as gs
        print("✓ Genesis 导入成功")
        
        from _boundary_tracking_evaluation_auto import BoundaryTrackingEvaluatorAuto
        print("✓ 自动化评估器导入成功")
        
        # 创建评估器（简化配置）
        evaluator = BoundaryTrackingEvaluatorAuto(
            cube_size=[0.105, 0.18, 0.022],
            base_cube_pos=[0.35, 0, 0.02],
            experiment_name="quick_test"
        )
        
        # 修改配置以减少测试时间
        evaluator.n_runs = 1  # 只运行1次
        evaluator.lighting_conditions = ["normal"]  # 只测试正常光照
        
        print("✓ 评估器创建成功")
        print(f"  结果目录: {evaluator.results_dir}")
        print(f"  实验次数: {evaluator.n_runs}")
        print(f"  光照条件: {evaluator.lighting_conditions}")
        
        # 运行评估
        print("\n开始运行评估...")
        evaluator.run_evaluation()
        
        print("\n✅ 快速测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
