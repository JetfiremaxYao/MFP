#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定性可视化运行示例
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from stability_visualization import StabilityVisualizer

def main():
    """主函数"""
    print("="*60)
    print("边界追踪算法稳定性指标可视化工具")
    print("="*60)
    
    # 查找最新的实验结果目录
    evaluation_dir = Path("evaluation_results")
    if not evaluation_dir.exists():
        print("❌ 未找到evaluation_results目录")
        print("请先运行边界追踪评估实验")
        return
    
    # 查找最新的实验目录
    experiment_dirs = [d for d in evaluation_dir.iterdir() if d.is_dir() and "canny_vs_rgbd" in d.name]
    if not experiment_dirs:
        print("❌ 未找到边界追踪实验结果目录")
        print("请先运行边界追踪评估实验")
        return
    
    # 选择最新的实验目录
    latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 找到实验结果目录: {latest_experiment}")
    
    # 检查CSV文件是否存在
    csv_files = [
        latest_experiment / "boundary_tracking_results.csv",
        latest_experiment / "boundary_tracking_summary.csv",
        latest_experiment / "boundary_tracking_grouped_summary.csv"
    ]
    
    existing_files = [f for f in csv_files if f.exists()]
    if not existing_files:
        print("❌ 未找到CSV结果文件")
        print("请先运行边界追踪评估实验")
        return
    
    print(f"✓ 找到 {len(existing_files)} 个CSV文件")
    for f in existing_files:
        print(f"  - {f.name}")
    
    print()
    print("开始稳定性分析...")
    
    # 创建可视化器
    visualizer = StabilityVisualizer(str(latest_experiment))
    
    # 打印稳定性摘要
    visualizer.print_stability_summary()
    
    # 生成可视化图表
    print("\n正在生成可视化图表...")
    visualizer.create_stability_comparison_plots()
    
    # 生成分析报告
    print("\n正在生成分析报告...")
    visualizer.generate_stability_report()
    
    print()
    print("="*60)
    print("稳定性分析完成！")
    print("="*60)
    print()
    print("生成的文件:")
    print(f"📊 可视化图表: {latest_experiment}/stability_analysis/*.png")
    print(f"📄 分析报告: {latest_experiment}/stability_analysis/stability_analysis_report.txt")
    print()
    print("图表说明:")
    print("1. std_residual_comparison.png - 残差标准差对比图")
    print("2. cv_residual_comparison.png - 变异系数对比图")
    print("3. stability_distribution.png - 稳定性分布图")
    print("4. stability_vs_accuracy.png - 稳定性vs准确性关系图")
    print("5. lighting_impact_on_stability.png - 光照条件对稳定性的影响")
    print("6. stability_radar_chart.png - 综合性能雷达图")

if __name__ == "__main__":
    main()
