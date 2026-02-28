#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪算法评估系统运行脚本
简化版本，用于快速执行评估实验
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """主函数"""
    print("=" * 60)
    print("边界追踪算法评估系统")
    print("=" * 60)
    print()
    
    # 检查依赖
    try:
        import genesis as gs
        print(f"✓ Genesis已安装: {gs.__version__ if hasattr(gs, '__version__') else 'unknown'}")
    except ImportError:
        print("✗ 错误: 未安装Genesis")
        print("请先安装Genesis: pip install genesis")
        return
    
    try:
        import numpy as np
        import cv2
        import open3d as o3d
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        from scipy import stats
        print("✓ 所有依赖包已安装")
    except ImportError as e:
        print(f"✗ 错误: 缺少依赖包: {e}")
        print("请安装缺失的包: pip install numpy opencv-python open3d matplotlib seaborn pandas scipy")
        return
    
    # 检查原始脚本是否存在
    required_files = [
        '_boundary_track_canny.py',
        '_boundary_track_rgbd.py',
        'boundary_tracking_evaluation.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not (current_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ 错误: 缺少必要文件: {', '.join(missing_files)}")
        print("请确保所有必要的脚本文件都在当前目录中")
        return
    
    print("✓ 所有必要文件已就绪")
    print()
    
    # 显示配置选项
    print("实验配置:")
    print("- 重复次数: 5次")
    print("- 位置扰动: ±2cm")
    print("- 高度扰动: ±0.5cm")
    print("- 检测方法: Canny vs RGB-D")
    print()
    
    # 询问用户是否继续
    while True:
        response = input("是否开始运行评估实验? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            break
        elif response in ['n', 'no', '否']:
            print("实验已取消")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    print("开始运行评估实验...")
    print("注意: 实验可能需要1-2小时完成")
    print()
    
    try:
        # 导入并运行评估器
        from boundary_tracking_evaluation import BoundaryTrackingEvaluator
        
        # 创建评估器
        evaluator = BoundaryTrackingEvaluator(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0, 0.02]),
            experiment_name="canny_vs_rgbd_comparison"
        )
        
        # 运行实验
        evaluator.run_comparison_experiments()
        
        print()
        print("=" * 60)
        print("评估实验完成！")
        print("=" * 60)
        print()
        
        # 询问是否进行结果分析
        while True:
            response = input("是否立即进行结果分析? (y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                break
            elif response in ['n', 'no', '否']:
                print("您可以稍后手动运行分析:")
                print(f"python boundary_tracking_analysis.py {evaluator.results_dir}/experiment_results.json")
                return
            else:
                print("请输入 y 或 n")
        
        print()
        print("开始结果分析...")
        
        # 导入并运行分析器
        from boundary_tracking_analysis import BoundaryTrackingAnalyzer
        
        # 创建分析器
        results_file = evaluator.results_dir / "experiment_results.json"
        analyzer = BoundaryTrackingAnalyzer(str(results_file))
        
        # 生成综合分析
        analyzer.generate_comprehensive_analysis()
        
        # 生成汇总表格
        analyzer.generate_summary_table()
        
        print()
        print("=" * 60)
        print("结果分析完成！")
        print("=" * 60)
        print()
        print(f"实验结果保存在: {evaluator.results_dir}")
        print(f"分析结果保存在: {analyzer.analysis_dir}")
        print()
        print("生成的文件包括:")
        print("- 实验数据和点云文件")
        print("- 可视化图表和对比图")
        print("- 统计分析报告")
        print("- 汇总统计表格")
        print()
        print("您可以查看这些文件来了解两种方法的性能对比结果")
        
    except Exception as e:
        print(f"✗ 实验运行失败: {e}")
        print("请检查错误信息并修复问题")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
