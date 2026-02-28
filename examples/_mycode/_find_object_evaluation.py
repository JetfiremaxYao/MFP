#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一物体检测方法评估系统快速启动脚本
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
    print("统一物体检测方法评估系统")
    print("="*60)
    print()
    print("本系统将按照标准化框架对比以下两种物体检测方法：")
    print("1. HSV+Depth: 基于HSV颜色空间和深度信息的检测方法")
    print("2. Depth+Clustering: 基于深度聚类的方法")
    print()
    
    # 显示实验配置
    print("实验配置：")
    print("- 重复次数: 10次/条件")
    print("- 光照条件: normal/bright/dim (3组)")
    print("- 背景条件: simple (1组)")
    print("- 总trial数: 60次 (3×1×10×2)")
    print("- 成功阈值: 2cm (严格), 5cm (宽松)")
    print("- 物体尺寸: 0.105 × 0.18 × 0.022 m")
    print("- 基础位置: (0.35, 0, 0.02) m")
    print()
    
    print("评估指标：")
    print("1. 准确度: 位置误差、范围误差")
    print("2. 稳定性: 标准差、中位绝对偏差")
    print("3. 召回/鲁棒性: 视角召回率、提前终止步数")
    print("4. 速度/成本: 检测时间、FPS")
    print("5. 诊断信息: 视角级详细信息")
    print()
    
    # 询问用户是否继续
    while True:
        response = input("是否开始运行统一评估实验? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            break
        elif response in ['n', 'no', '否']:
            print("实验已取消")
            return
        else:
            print("请输入 y 或 n")
    
    print()
    print("开始运行统一评估实验...")
    print("注意: 实验可能需要1-2小时完成")
    print()
    
    try:
        # 导入并运行统一评估器
        from unified_object_detection_evaluation import UnifiedObjectDetectionEvaluator
        
        # 创建评估器
        evaluator = UnifiedObjectDetectionEvaluator(
            cube_size=np.array([0.105, 0.18, 0.022]),
            base_cube_pos=np.array([0.35, 0, 0.02]),
            experiment_name="unified_object_detection"
        )
        
        # 运行实验
        evaluator.run_comprehensive_evaluation()
        
        print()
        print("="*60)
        print("统一评估实验完成！")
        print("="*60)
        print()
        
        # 显示结果文件位置
        print("结果文件位置：")
        print(f"CSV数据: {evaluator.results_dir}/trial_results.csv")
        print(f"JSON数据: {evaluator.results_dir}/experiment_results.json")
        print(f"实验报告: {evaluator.results_dir}/experiment_report.txt")
        print(f"可视化图表: {evaluator.results_dir}/evaluation_charts.png")
        print()
        
        # 显示结果摘要
        print("结果摘要：")
        try:
            # 读取实验结果
            import pandas as pd
            df = pd.read_csv(evaluator.results_dir / "trial_results.csv")
            
            # 按方法分组统计
            for method in ['hsv_depth', 'depth_clustering']:
                method_df = df[df['method'] == method]
                print(f"\n{method.upper()}方法:")
                
                # 成功率统计
                success_2cm = method_df['success_pos_2cm'].mean()
                success_5cm = method_df['success_pos_5cm'].mean()
                print(f"  成功率(2cm): {success_2cm:.1%}")
                print(f"  成功率(5cm): {success_5cm:.1%}")
                
                # 位置误差统计
                pos_err_mean = method_df['pos_err_m'].mean() * 1000  # 转换为mm
                pos_err_std = method_df['pos_err_m'].std() * 1000
                print(f"  位置误差: {pos_err_mean:.1f}±{pos_err_std:.1f}mm")
                
                # 检测时间统计
                time_mean = method_df['detection_time_s'].mean()
                time_std = method_df['detection_time_s'].std()
                print(f"  检测时间: {time_mean:.2f}±{time_std:.2f}s")
                
                # 视角召回率统计
                recall_mean = method_df['view_hit_rate'].mean()
                print(f"  视角召回率: {recall_mean:.1%}")
            
            # 按光照条件分组统计
            print(f"\n按光照条件分组统计:")
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = df[df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    print(f"\n  {lighting}光照条件:")
                    
                    for method in ['hsv_depth', 'depth_clustering']:
                        method_lighting_df = lighting_df[lighting_df['method'] == method]
                        if len(method_lighting_df) > 0:
                            pos_err = method_lighting_df['pos_err_m'].mean() * 1000
                            success_rate = method_lighting_df['success_pos_2cm'].mean()
                            print(f"    {method}: 位置误差={pos_err:.1f}mm, 成功率={success_rate:.1%}")
        
        except Exception as e:
            print(f"读取结果摘要时出错: {e}")
        
        print()
        print("实验完成！请查看生成的文件了解详细结果。")
        print()
        print("后续分析建议：")
        print("1. 使用statistical_analysis.py进行统计显著性检验")
        print("2. 查看CSV文件进行自定义分析")
        print("3. 查看可视化图表了解性能对比")
        
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
