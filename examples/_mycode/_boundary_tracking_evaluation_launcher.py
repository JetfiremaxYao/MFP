#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪算法评估系统快速启动脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """主函数"""
    print("="*60)
    print("边界追踪算法评估系统")
    print("="*60)
    print()
    print("本系统将对比以下两种边界追踪方法：")
    print("1. Canny边缘检测: 基于图像边缘检测的边界追踪方法")
    print("2. RGB-D边界检测: 基于深度信息的边界追踪方法")
    print()
    
    # 显示实验配置
    print("实验配置：")
    print("- 重复次数: 3次/条件")
    print("- 光照条件: normal/bright/dim (3组)")
    print("- 背景条件: simple (1组)")
    print("- 总trial数: 18次 (3×1×3×2)")
    print("- 成功阈值: 3mm覆盖率、Chamfer距离<100mm")
    print("- 物体尺寸: 0.105 × 0.18 × 0.022 m")
    print("- 基础位置: (0.35, 0.08, 0.02) m")
    print("- 位置扰动: ±2cm")
    print("- 高度扰动: ±5mm")
    print()
    
    print("评估指标：")
    print("1. 准确度: Chamfer距离、Hausdorff距离、覆盖率")
    print("2. 稳定性: 残差标准差、变异系数")
    print("3. 成功率: 成功/部分成功/失败率")
    print("4. 性能: 执行时间、点云数量")
    print("5. 可视化: 误差热力图、覆盖率曲线、雷达图等")
    print()
    
    print("输出文件：")
    print("- CSV数据: boundary_tracking_results.csv")
    print("- 摘要统计: boundary_tracking_summary.csv")
    print("- 分组统计: boundary_tracking_grouped_summary.csv")
    print("- JSON数据: experiment_results.json")
    print("- 实验报告: experiment_report.txt")
    print("- 可视化图表: 多种PNG格式图表")
    print("- 点云数据: PLY格式文件")
    print()
    
    # 询问用户是否继续
    while True:
        response = input("是否开始运行边界追踪评估实验? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            break
        elif response in ['n', 'no', '否']:
            print("实验已取消")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    print("开始运行边界追踪评估实验...")
    print("注意: 实验可能需要30-60分钟完成")
    print()
    
    try:
        # 导入并运行边界追踪评估器
        from _boundary_tracking_evaluation import BoundaryTrackingEvaluator
        
        # 创建评估器
        evaluator = BoundaryTrackingEvaluator(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0.08, 0.02]),
            experiment_name="canny_vs_rgbd_comparison"
        )
        
        # 运行实验
        evaluator.run_comparison_experiments()
        
        print()
        print("="*60)
        print("边界追踪评估实验完成！")
        print("="*60)
        print()
        
        # 显示结果文件位置
        print("结果文件位置：")
        print(f"CSV数据: {evaluator.results_dir}/boundary_tracking_results.csv")
        print(f"摘要统计: {evaluator.results_dir}/boundary_tracking_summary.csv")
        print(f"分组统计: {evaluator.results_dir}/boundary_tracking_grouped_summary.csv")
        print(f"JSON数据: {evaluator.results_dir}/experiment_results.json")
        print(f"实验报告: {evaluator.results_dir}/experiment_report.txt")
        print(f"可视化图表: {evaluator.results_dir}/*.png")
        print(f"点云数据: {evaluator.results_dir}/*.ply")
        print()
        
        # 显示结果摘要
        print("结果摘要：")
        try:
            # 读取实验结果
            import pandas as pd
            df = pd.read_csv(evaluator.results_dir / "boundary_tracking_summary.csv")
            
            # 按方法分组统计
            for method in ['canny', 'rgbd']:
                method_df = df[df['method'] == method]
                if len(method_df) == 0:
                    continue
                    
                print(f"\n{method.upper()}方法:")
                
                # 成功率统计
                success_rate = method_df['success_rate'].iloc[0]
                partial_success_rate = method_df['partial_success_rate'].iloc[0]
                overall_success_rate = method_df['overall_success_rate'].iloc[0]
                print(f"  成功率: {success_rate:.1%}")
                print(f"  部分成功率: {partial_success_rate:.1%}")
                print(f"  总体成功率: {overall_success_rate:.1%}")
                
                # 准确性指标统计
                if 'chamfer_distance_mm_mean' in method_df.columns and not pd.isna(method_df['chamfer_distance_mm_mean'].iloc[0]):
                    chamfer_mean = method_df['chamfer_distance_mm_mean'].iloc[0]
                    chamfer_std = method_df['chamfer_distance_mm_std'].iloc[0]
                    print(f"  Chamfer距离: {chamfer_mean:.1f}±{chamfer_std:.1f}mm")
                
                if 'coverage_3mm_mean' in method_df.columns and not pd.isna(method_df['coverage_3mm_mean'].iloc[0]):
                    coverage_mean = method_df['coverage_3mm_mean'].iloc[0]
                    coverage_std = method_df['coverage_3mm_std'].iloc[0]
                    print(f"  3mm覆盖率: {coverage_mean:.3f}±{coverage_std:.3f}")
                
                # 性能指标统计
                if 'execution_time_s_mean' in method_df.columns and not pd.isna(method_df['execution_time_s_mean'].iloc[0]):
                    time_mean = method_df['execution_time_s_mean'].iloc[0]
                    time_std = method_df['execution_time_s_std'].iloc[0]
                    print(f"  执行时间: {time_mean:.2f}±{time_std:.2f}s")
                
                if 'point_count_mean' in method_df.columns and not pd.isna(method_df['point_count_mean'].iloc[0]):
                    point_mean = method_df['point_count_mean'].iloc[0]
                    point_std = method_df['point_count_std'].iloc[0]
                    print(f"  平均点数: {point_mean:.0f}±{point_std:.0f}")
                
                # 稳定性指标统计
                if 'std_residual_mm_mean' in method_df.columns and not pd.isna(method_df['std_residual_mm_mean'].iloc[0]):
                    std_residual_mean = method_df['std_residual_mm_mean'].iloc[0]
                    std_residual_std = method_df['std_residual_mm_std'].iloc[0]
                    print(f"  残差标准差: {std_residual_mean:.2f}±{std_residual_std:.2f}mm")
                
                if 'cv_residual_mean' in method_df.columns and not pd.isna(method_df['cv_residual_mean'].iloc[0]):
                    cv_residual_mean = method_df['cv_residual_mean'].iloc[0]
                    cv_residual_std = method_df['cv_residual_std'].iloc[0]
                    print(f"  变异系数: {cv_residual_mean:.3f}±{cv_residual_std:.3f}")
            
            # 按光照条件分组统计
            print(f"\n按光照条件分组统计:")
            grouped_df = pd.read_csv(evaluator.results_dir / "boundary_tracking_grouped_summary.csv")
            
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = grouped_df[grouped_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    print(f"\n  {lighting}光照条件:")
                    
                    for method in ['canny', 'rgbd']:
                        method_lighting_df = lighting_df[lighting_df['method'] == method]
                        if len(method_lighting_df) > 0:
                            success_rate = method_lighting_df['success_rate'].iloc[0]
                            chamfer_mean = method_lighting_df['chamfer_distance_mm_mean'].iloc[0] if not pd.isna(method_lighting_df['chamfer_distance_mm_mean'].iloc[0]) else float('inf')
                            coverage_mean = method_lighting_df['coverage_3mm_mean'].iloc[0] if not pd.isna(method_lighting_df['coverage_3mm_mean'].iloc[0]) else 0
                            std_residual_mean = method_lighting_df['std_residual_mm_mean'].iloc[0] if not pd.isna(method_lighting_df['std_residual_mm_mean'].iloc[0]) else float('inf')
                            
                            print(f"    {method}: 成功率={success_rate:.1%}, Chamfer={chamfer_mean:.1f}mm, 覆盖率={coverage_mean:.3f}, 残差标准差={std_residual_mean:.1f}mm")
        
        except Exception as e:
            print(f"读取结果摘要时出错: {e}")
            print("请直接查看CSV文件了解详细结果")
        
        print()
        print("实验完成！请查看生成的文件了解详细结果。")
        print()
        print("后续分析建议：")
        print("1. 使用pandas读取CSV文件进行自定义分析")
        print("2. 查看可视化图表了解性能对比")
        print("3. 使用statistical_analysis.py进行统计显著性检验")
        print("4. 查看JSON文件了解完整的实验配置和数据")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已安装：")
        print("pip install numpy matplotlib seaborn pandas scipy opencv-python open3d")
        return
    except Exception as e:
        print(f"实验运行失败: {e}")
        print("请检查错误信息并重试")
        return

if __name__ == "__main__":
    # 导入必要的包
    try:
        import numpy as np
    except ImportError:
        print("错误: 缺少numpy包")
        print("请运行: pip install numpy")
        sys.exit(1)
    
    main()
