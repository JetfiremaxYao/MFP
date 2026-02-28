#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
物体检测方法对比实验快速启动脚本
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
    print("物体检测方法对比实验系统")
    print("="*60)
    print()
    print("本系统将对比以下两种物体检测方法：")
    print("1. HSV+Depth: 基于HSV颜色空间和深度信息的检测方法")
    print("2. Depth+Clustering: 基于深度聚类的方法")
    print()
    
    # 显示实验配置
    print("实验配置：")
    print("- 重复次数: 10次")
    print("- 位置扰动: ±2cm (x, y方向)")
    print("- 高度扰动: ±0.5cm (z方向)")
    print("- 成功阈值: 5cm")
    print("- 部分成功阈值: 10cm")
    print("- 物体尺寸: 0.105 × 0.18 × 0.022 m")
    print("- 基础位置: (0.35, 0, 0.02) m")
    print()
    
    # 询问用户是否继续
    while True:
        response = input("是否开始运行对比实验? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            break
        elif response in ['n', 'no', '否']:
            print("实验已取消")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    print("开始运行对比实验...")
    print("注意: 实验可能需要30-60分钟完成")
    print()
    
    try:
        # 导入并运行对比评估器
        from object_detection_comparison import ObjectDetectionComparison
        
        # 创建评估器
        evaluator = ObjectDetectionComparison(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0, 0.02]),
            experiment_name="hsv_depth_vs_clustering"
        )
        
        # 运行实验
        evaluator.run_comparison_experiments()
        
        print()
        print("="*60)
        print("对比实验完成！")
        print("="*60)
        print()
        
        # 询问是否进行统计分析
        while True:
            response = input("是否立即进行统计分析? (y/n): ").strip().lower()
            if response in ['y', 'yes', '是']:
                break
            elif response in ['n', 'no', '否']:
                print("您可以稍后手动运行统计分析:")
                print(f"python statistical_analysis.py {evaluator.results_dir}/experiment_results.json")
                return
            else:
                print("请输入 y 或 n")
        
        print()
        print("开始统计分析...")
        
        # 导入并运行统计分析器
        from statistical_analysis import StatisticalAnalyzer
        
        # 创建分析器
        results_file = evaluator.results_dir / "experiment_results.json"
        analyzer = StatisticalAnalyzer(str(results_file))
        
        # 执行统计检验
        analyzer.perform_statistical_tests()
        
        # 打印报告
        analyzer.print_statistical_report()
        
        # 生成图表
        analyzer.generate_statistical_plots()
        
        # 保存报告
        analyzer.save_statistical_report()
        
        print()
        print("="*60)
        print("统计分析完成！")
        print("="*60)
        print()
        
        # 显示结果文件位置
        print("结果文件位置：")
        print(f"实验数据: {evaluator.results_dir}/experiment_results.json")
        print(f"实验报告: {evaluator.results_dir}/experiment_report.txt")
        print(f"对比图表: {evaluator.results_dir}/comparison_charts.png")
        print(f"统计结果: {evaluator.results_dir}/statistical_analysis_results.json")
        print(f"统计报告: {evaluator.results_dir}/statistical_analysis_report.txt")
        print(f"统计图表: {evaluator.results_dir}/statistical_analysis_plots.png")
        print()
        
        # 显示结果摘要
        print("结果摘要：")
        try:
            # 读取实验结果
            with open(evaluator.results_dir / "experiment_results.json", 'r', encoding='utf-8') as f:
                import json
                results = json.load(f)
            
            # 统计成功率
            hsv_success = sum(1 for r in results['hsv_depth'] if r['status'] == 'Success')
            hsv_total = len(results['hsv_depth'])
            clustering_success = sum(1 for r in results['depth_clustering'] if r['status'] == 'Success')
            clustering_total = len(results['depth_clustering'])
            
            print(f"HSV+Depth方法: {hsv_success}/{hsv_total} 成功 ({hsv_success/hsv_total:.1%})")
            print(f"Depth+Clustering方法: {clustering_success}/{clustering_total} 成功 ({clustering_success/clustering_total:.1%})")
            
            # 读取统计结果
            if (evaluator.results_dir / "statistical_analysis_results.json").exists():
                with open(evaluator.results_dir / "statistical_analysis_results.json", 'r', encoding='utf-8') as f:
                    stats_results = json.load(f)
                
                if 'position_error' in stats_results:
                    pos_result = stats_results['position_error']
                    print(f"位置误差对比: p = {pos_result['p_value']:.4f}")
                    if pos_result['p_value'] < 0.05:
                        print("  → 差异具有统计学意义")
                    else:
                        print("  → 差异无统计学意义")
                
                if 'success_rate' in stats_results:
                    success_result = stats_results['success_rate']
                    print(f"成功率对比: p = {success_result['p_value']:.4f}")
                    if success_result['p_value'] < 0.05:
                        print("  → 差异具有统计学意义")
                    else:
                        print("  → 差异无统计学意义")
            
        except Exception as e:
            print(f"读取结果摘要时出错: {e}")
        
        print()
        print("实验完成！请查看生成的文件了解详细结果。")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已安装：")
        print("pip install numpy matplotlib seaborn pandas scipy")
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
