#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三种边界追踪方法统一评估系统启动脚本
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
    print("三种边界追踪方法统一评估系统")
    print("="*60)
    print()
    print("本系统将对比以下三种边界追踪方法：")
    print("1. Canny边缘检测: 基于图像边缘检测的边界追踪方法")
    print("2. RGB-D边界检测: 基于深度信息的边界追踪方法")
    print("3. Alpha-Shape凹包: 基于点云几何的凹包检测方法")
    print()
    
    # 显示实验配置
    print("统一实验配置：")
    print("- 测试物体: ctry.obj (所有方法使用相同物体)")
    print("- 物体尺寸: 0.105 × 0.18 × 0.022 m")
    print("- 基础位置: (0.35, 0.08, 0.02) m")
    print("- 位置扰动: ±2cm")
    print("- 重复次数: 3次/方法/位置")
    print("- 测试位置: 2个不同位置")
    print("- 总trial数: 18次 (3×2×3)")
    print()
    
    print("评估指标：")
    print("1. 成功率: 成功检测到边界的比例")
    print("2. 轮廓质量: 检测到的轮廓数量和面积")
    print("3. 点云质量: 点云数量和密度")
    print("4. 执行时间: 方法执行耗时")
    print("5. 稳定性: 多次运行的标准差")
    print()
    
    print("输出文件：")
    print("- CSV数据: results_summary.csv")
    print("- 详细结果: detailed_results.json")
    print("- 统计摘要: summary_statistics.csv")
    print("- 评估报告: three_methods_evaluation_report.txt")
    print("- 可视化图表: three_methods_comparison_charts.png")
    print()
    
    # 询问用户是否继续
    while True:
        response = input("是否开始运行三种方法统一评估实验? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            break
        elif response in ['n', 'no', '否']:
            print("实验已取消")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    print("开始运行三种方法统一评估实验...")
    print("注意: 实验可能需要20-40分钟完成")
    print()
    
    try:
        # 导入并运行三种方法评估器
        from _boundary_tracking_three_methods_evaluation import ThreeMethodsBoundaryTrackingEvaluator
        
        # 创建评估器
        evaluator = ThreeMethodsBoundaryTrackingEvaluator(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0.08, 0.02]),
            experiment_name="three_methods_comparison"
        )
        
        # 运行实验
        evaluator.run_evaluation(num_runs=3, num_positions=2)
        
        print()
        print("="*60)
        print("三种方法统一评估实验完成！")
        print("="*60)
        print()
        
        # 显示结果文件位置
        print("结果文件位置：")
        print(f"CSV数据: {evaluator.results_dir}/results_summary.csv")
        print(f"详细结果: {evaluator.results_dir}/detailed_results.json")
        print(f"统计摘要: {evaluator.results_dir}/summary_statistics.csv")
        print(f"评估报告: {evaluator.results_dir}/three_methods_evaluation_report.txt")
        print(f"可视化图表: {evaluator.results_dir}/three_methods_comparison_charts.png")
        print()
        
        # 显示结果摘要
        print("结果摘要：")
        try:
            # 读取实验结果
            import pandas as pd
            df = pd.read_csv(evaluator.results_dir / "summary_statistics.csv")
            
            # 按方法分组统计
            for method in ['canny', 'rgbd', 'alpha_shape']:
                if method in df['method'].values:
                    method_df = df[df['method'] == method]
                    if len(method_df) == 0:
                        continue
                        
                    print(f"\n{method.upper()}方法:")
                    
                    # 成功率统计
                    success_rate = method_df['success_rate'].iloc[0]
                    print(f"  成功率: {success_rate:.1%}")
                    
                    # 轮廓质量统计
                    avg_contours = method_df['avg_contours'].iloc[0]
                    avg_contour_area = method_df['avg_contour_area'].iloc[0]
                    print(f"  平均轮廓数量: {avg_contours:.1f}")
                    print(f"  平均轮廓面积: {avg_contour_area:.2f}")
                    
                    # 点云质量统计
                    avg_points = method_df['avg_points'].iloc[0]
                    print(f"  平均点云数量: {avg_points:.0f}")
                    
                    # 性能统计
                    avg_execution_time = method_df['avg_execution_time'].iloc[0]
                    std_execution_time = method_df['std_execution_time'].iloc[0]
                    print(f"  平均执行时间: {avg_execution_time:.3f}±{std_execution_time:.3f}s")
            
            # 找出最佳方法
            print(f"\n最佳方法分析:")
            print("-" * 30)
            
            best_success = df.loc[df['success_rate'].idxmax(), 'method']
            fastest = df.loc[df['avg_execution_time'].idxmin(), 'method']
            most_contours = df.loc[df['avg_contours'].idxmax(), 'method']
            
            print(f"最高成功率: {best_success}")
            print(f"最快执行速度: {fastest}")
            print(f"最多轮廓检测: {most_contours}")
            
            # 综合评分
            df['综合评分'] = (
                df['success_rate'] * 0.4 + 
                (1 - df['avg_execution_time'] / df['avg_execution_time'].max()) * 0.3 +
                (df['avg_contours'] / df['avg_contours'].max()) * 0.3
            )
            
            best_overall = df.loc[df['综合评分'].idxmax(), 'method']
            print(f"综合评分最高: {best_overall}")
            print(f"推荐用于您的pipeline: {best_overall}")
        
        except Exception as e:
            print(f"读取结果摘要时出错: {e}")
            print("请直接查看CSV文件了解详细结果")
        
        print()
        print("实验完成！请查看生成的文件了解详细结果。")
        print()
        print("后续分析建议：")
        print("1. 查看评估报告了解详细分析")
        print("2. 查看可视化图表了解性能对比")
        print("3. 根据您的具体需求选择最适合的方法")
        print("4. 考虑在您的pipeline中集成推荐的方法")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已安装：")
        print("pip install numpy matplotlib seaborn pandas scipy opencv-python open3d")
        if "shapely" in str(e):
            print("pip install shapely  # 用于Alpha-Shape方法")
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
