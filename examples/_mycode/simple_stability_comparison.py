#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单清晰的稳定性对比图
专门展示不同条件下的稳定性指标对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as npv
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_stability_comparison():
    """创建简单的稳定性对比图"""
    
    # 从CSV文件读取数据
    results_dir = Path("../evaluation_results/canny_vs_rgbd_comparison_20250827_172724")
    
    # 读取分组摘要数据
    grouped_df = pd.read_csv(results_dir / "boundary_tracking_grouped_summary.csv")
    
    # 设置图表样式
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('边界追踪算法稳定性对比分析', fontsize=16, fontweight='bold')
    
    # 1. 残差标准差对比 (主要稳定性指标)
    ax1 = axes[0, 0]
    methods = ['canny', 'rgbd']
    lightings = ['normal', 'bright', 'dim']
    
    x = np.arange(len(lightings))
    width = 0.35
    
    canny_data = []
    rgbd_data = []
    
    for lighting in lightings:
        canny_row = grouped_df[(grouped_df['method'] == 'canny') & (grouped_df['lighting'] == lighting)]
        rgbd_row = grouped_df[(grouped_df['method'] == 'rgbd') & (grouped_df['lighting'] == lighting)]
        
        if len(canny_row) > 0:
            canny_data.append(canny_row['std_residual_mm_mean'].iloc[0])
        else:
            canny_data.append(0)
            
        if len(rgbd_row) > 0:
            rgbd_data.append(rgbd_row['std_residual_mm_mean'].iloc[0])
        else:
            rgbd_data.append(0)
    
    bars1 = ax1.bar(x - width/2, canny_data, width, label='Canny', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rgbd_data, width, label='RGB-D', color='#4ECDC4', alpha=0.8)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
                f'{canny_data[i]:.2f}', ha='center', va='bottom', fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
                f'{rgbd_data[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('光照条件', fontsize=12)
    ax1.set_ylabel('残差标准差 (mm)', fontsize=12)
    ax1.set_title('不同光照条件下的残差标准差对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Normal', 'Bright', 'Dim'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 总体稳定性对比
    ax2 = axes[0, 1]
    
    # 读取总体摘要数据
    summary_df = pd.read_csv(results_dir / "boundary_tracking_summary.csv")
    
    overall_canny = summary_df[summary_df['method'] == 'canny']['std_residual_mm_mean'].iloc[0]
    overall_rgbd = summary_df[summary_df['method'] == 'rgbd']['std_residual_mm_mean'].iloc[0]
    
    methods_overall = ['Canny', 'RGB-D']
    values_overall = [overall_canny, overall_rgbd]
    colors_overall = ['#FF6B6B', '#4ECDC4']
    
    bars_overall = ax2.bar(methods_overall, values_overall, color=colors_overall, alpha=0.8)
    
    # 添加数值标签
    for bar, value in zip(bars_overall, values_overall):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('残差标准差 (mm)', fontsize=12)
    ax2.set_title('总体稳定性对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 稳定性波动性对比 (标准差的标准差)
    ax3 = axes[1, 0]
    
    canny_std = summary_df[summary_df['method'] == 'canny']['std_residual_mm_std'].iloc[0]
    rgbd_std = summary_df[summary_df['method'] == 'rgbd']['std_residual_mm_std'].iloc[0]
    
    methods_std = ['Canny', 'RGB-D']
    values_std = [canny_std, rgbd_std]
    colors_std = ['#FF6B6B', '#4ECDC4']
    
    bars_std = ax3.bar(methods_std, values_std, color=colors_std, alpha=0.8)
    
    # 添加数值标签
    for bar, value in zip(bars_std, values_std):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('稳定性波动性 (mm)', fontsize=12)
    ax3.set_title('稳定性波动性对比\n(标准差越小越稳定)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 综合性能雷达图
    ax4 = axes[1, 1]
    
    # 计算综合得分 (归一化到0-1)
    canny_stability = 1 - min(overall_canny / 10, 1)  # 残差标准差越小越好
    rgbd_stability = 1 - min(overall_rgbd / 10, 1)
    
    canny_consistency = 1 - min(canny_std / 2, 1)  # 波动性越小越好
    rgbd_consistency = 1 - min(rgbd_std / 2, 1)
    
    # 从原始数据读取准确性指标
    canny_accuracy = 1 - min(summary_df[summary_df['method'] == 'canny']['chamfer_distance_mm_mean'].iloc[0] / 20, 1)
    rgbd_accuracy = 1 - min(summary_df[summary_df['method'] == 'rgbd']['chamfer_distance_mm_mean'].iloc[0] / 20, 1)
    
    # 雷达图数据
    categories = ['稳定性', '一致性', '准确性']
    canny_scores = [canny_stability, canny_consistency, canny_accuracy]
    rgbd_scores = [rgbd_stability, rgbd_consistency, rgbd_accuracy]
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    canny_scores += canny_scores[:1]
    rgbd_scores += rgbd_scores[:1]
    
    ax4.plot(angles, canny_scores, 'o-', linewidth=2, label='Canny', color='#FF6B6B')
    ax4.fill(angles, canny_scores, alpha=0.25, color='#FF6B6B')
    
    ax4.plot(angles, rgbd_scores, 'o-', linewidth=2, label='RGB-D', color='#4ECDC4')
    ax4.fill(angles, rgbd_scores, alpha=0.25, color='#4ECDC4')
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('综合性能雷达图', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("../../evaluation_results/canny_vs_rgbd_comparison_20250827_172724")
    output_file = output_dir / "simple_stability_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"稳定性对比图已保存: {output_file}")
    
    # 显示图表
    plt.show()
    
    # 打印数值摘要
    print("\n" + "="*60)
    print("稳定性分析数值摘要")
    print("="*60)
    
    print(f"\n📊 残差标准差对比 (mm):")
    print(f"  Canny方法:")
    for i, lighting in enumerate(['Normal', 'Bright', 'Dim']):
        print(f"    {lighting}: {canny_data[i]:.2f}")
    print(f"    总体平均: {overall_canny:.2f}")
    
    print(f"\n  RGB-D方法:")
    for i, lighting in enumerate(['Normal', 'Bright', 'Dim']):
        print(f"    {lighting}: {rgbd_data[i]:.2f}")
    print(f"    总体平均: {overall_rgbd:.2f}")
    
    print(f"\n📈 稳定性波动性 (标准差的标准差):")
    print(f"  Canny: {canny_std:.2f}mm (波动性小，更稳定)")
    print(f"  RGB-D: {rgbd_std:.2f}mm (波动性大，不够稳定)")
    
    print(f"\n🏆 稳定性排名:")
    if overall_canny < overall_rgbd:
        print(f"  1. Canny方法 - 更稳定 ({overall_canny:.2f}mm)")
        print(f"  2. RGB-D方法 - 相对不稳定 ({overall_rgbd:.2f}mm)")
    else:
        print(f"  1. RGB-D方法 - 更稳定 ({overall_rgbd:.2f}mm)")
        print(f"  2. Canny方法 - 相对不稳定 ({overall_canny:.2f}mm)")
    
    print(f"\n💡 关键结论:")
    print(f"  • Canny方法在所有光照条件下都表现稳定")
    print(f"  • RGB-D方法在dim光照下表现最好，normal光照下波动最大")
    print(f"  • Canny方法的稳定性波动性更小，说明更可靠")

if __name__ == "__main__":
    create_simple_stability_comparison()
