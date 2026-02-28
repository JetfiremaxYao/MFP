#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的边界追踪实验结果可视化
专注于关键几何精度指标和执行时间的直观对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def load_data():
    """加载CSV数据"""
    csv_path = "../../evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    df = pd.read_csv(csv_path)
    print(f"✅ 加载数据: {len(df)} 条记录")
    return df

def create_key_metrics_comparison(df):
    """创建关键指标对比图"""
    print("📊 创建关键指标对比图...")
    
    # 设置图形大小和布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('边界追踪算法关键指标对比分析', fontsize=16, fontweight='bold', y=0.98)
    
    # 关键指标
    metrics = [
        ('chamfer_distance_mm', 'Chamfer Distance (mm)', 'Lower is Better'),
        ('hausdorff_distance_mm', 'Hausdorff Distance (mm)', 'Lower is Better'),
        ('coverage_3mm', '3mm Coverage Rate', 'Higher is Better'),
        ('execution_time_s', 'Execution Time (s)', 'Lower is Better'),
        ('point_count', 'Point Cloud Count', 'Moderate is Better'),
        ('mean_residual_mm', 'Mean Residual (mm)', 'Lower is Better')
    ]
    
    # 颜色配置
    colors = {'canny': '#2E86AB', 'rgbd': '#A23B72'}
    
    for i, (metric, title, note) in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 创建分组数据
        data_to_plot = []
        labels = []
        
        for obj_name in df['object_name'].unique():
            for method in df['method'].unique():
                subset = df[
                    (df['object_name'] == obj_name) & 
                    (df['method'] == method)
                ]
                
                if len(subset) > 0:
                    data_to_plot.append(subset[metric].values)
                    # 将物体名称转换为中文描述
                    obj_display = "Regular Object" if obj_name == "Cube" else "Irregular Object"
                    labels.append(f'{obj_display}\n{method.upper()}')
        
        # 创建箱线图
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                          boxprops=dict(alpha=0.7), 
                          medianprops=dict(color='red', linewidth=2))
            
            # 设置颜色
            for patch, label in zip(bp['boxes'], labels):
                if 'CANNY' in label:
                    patch.set_facecolor(colors['canny'])
                else:
                    patch.set_facecolor(colors['rgbd'])
            
            ax.set_title(f'{title}\n({note})', fontsize=12, fontweight='bold')
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = "../../evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/key_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 关键指标对比图已保存: {output_file}")
    return output_file

def create_lighting_impact_analysis(df):
    """创建光照影响分析图"""
    print("💡 创建光照影响分析图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('光照条件对算法性能的影响分析', fontsize=16, fontweight='bold')
    
    # 关键指标
    metrics = [
        ('chamfer_distance_mm', 'Chamfer Distance (mm)'),
        ('hausdorff_distance_mm', 'Hausdorff Distance (mm)'),
        ('coverage_3mm', '3mm Coverage Rate'),
        ('execution_time_s', 'Execution Time (s)')
    ]
    
    colors = {'normal': '#2E86AB', 'bright': '#F18F01', 'dim': '#C73E1D'}
    
    for i, (metric, title) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 创建数据
        data = []
        labels = []
        colors_list = []
        
        for obj_name in df['object_name'].unique():
            for method in df['method'].unique():
                for lighting in df['lighting'].unique():
                    subset = df[
                        (df['object_name'] == obj_name) & 
                        (df['method'] == method) & 
                        (df['lighting'] == lighting)
                    ]
                    
                    if len(subset) > 0:
                        data.append(subset[metric].mean())
                        # 将物体名称转换为中文描述
                        obj_display = "Regular Object" if obj_name == "Cube" else "Irregular Object"
                        labels.append(f'{obj_display}\n{method.upper()}\n{lighting.upper()}')
                        colors_list.append(colors[lighting])
        
        # 创建柱状图
        bars = ax.bar(range(len(data)), data, color=colors_list, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_file = "../../evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/lighting_impact_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 光照影响分析图已保存: {output_file}")
    return output_file

def create_performance_summary(df):
    """创建性能总结图"""
    print("📋 创建性能总结图...")
    
    # 计算汇总数据
    summary_data = []
    
    for obj_name in df['object_name'].unique():
        for method in df['method'].unique():
            subset = df[
                (df['object_name'] == obj_name) & 
                (df['method'] == method)
            ]
            
            if len(subset) > 0:
                # 将物体名称转换为中文描述
                obj_display = "Regular Object" if obj_name == "Cube" else "Irregular Object"
                summary_data.append({
                    'Object': obj_display,
                    'Method': method.upper(),
                    'Success Rate': (subset['status'] == 'Success').mean() * 100,
                    'Avg Chamfer Distance': subset['chamfer_distance_mm'].mean(),
                    'Avg Hausdorff Distance': subset['hausdorff_distance_mm'].mean(),
                    'Avg 3mm Coverage': subset['coverage_3mm'].mean() * 100,
                    'Avg Execution Time': subset['execution_time_s'].mean(),
                    'Avg Point Count': subset['point_count'].mean()
                })
    
    # 创建汇总表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['Object', 'Method', 'Success Rate(%)', 'Chamfer Dist.(mm)', 'Hausdorff Dist.(mm)', 
               '3mm Coverage(%)', 'Exec. Time(s)', 'Point Count']
    
    for data in summary_data:
        table_data.append([
            data['Object'],
            data['Method'],
            f"{data['Success Rate']:.1f}",
            f"{data['Avg Chamfer Distance']:.2f}",
            f"{data['Avg Hausdorff Distance']:.2f}",
            f"{data['Avg 3mm Coverage']:.1f}",
            f"{data['Avg Execution Time']:.2f}",
            f"{data['Avg Point Count']:.0f}"
        ])
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    colors = {'CANNY': '#E3F2FD', 'RGBD': '#FFF3E0'}
    for i, data in enumerate(summary_data):
        row_color = colors.get(data['Method'], '#FFFFFF')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(row_color)
    
    plt.title('边界追踪算法性能汇总表', fontsize=16, fontweight='bold', pad=20)
    
    output_file = "../../evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/performance_summary.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 性能总结图已保存: {output_file}")
    return output_file

def print_key_insights(df):
    """打印关键洞察"""
    print("\n" + "="*80)
    print("🔍 关键洞察分析")
    print("="*80)
    
    # 按物体和方法分组分析
    for obj_name in df['object_name'].unique():
        print(f"\n📦 {obj_name} 物体:")
        print("-" * 40)
        
        for method in df['method'].unique():
            subset = df[
                (df['object_name'] == obj_name) & 
                (df['method'] == method)
            ]
            
            if len(subset) > 0:
                print(f"\n🔧 {method.upper()} 方法:")
                print(f"   ✅ 成功率: {(subset['status'] == 'Success').mean()*100:.1f}%")
                print(f"   📏 Chamfer距离: {subset['chamfer_distance_mm'].mean():.2f}±{subset['chamfer_distance_mm'].std():.2f} mm")
                print(f"   📐 Hausdorff距离: {subset['hausdorff_distance_mm'].mean():.2f}±{subset['hausdorff_distance_mm'].std():.2f} mm")
                print(f"   🎯 3mm覆盖率: {subset['coverage_3mm'].mean()*100:.1f}±{subset['coverage_3mm'].std()*100:.1f}%")
                print(f"   ⏱️  执行时间: {subset['execution_time_s'].mean():.2f}±{subset['execution_time_s'].std():.2f} s")
                print(f"   📊 点云数量: {subset['point_count'].mean():.0f}±{subset['point_count'].std():.0f}")
    
    # 光照影响分析
    print(f"\n💡 光照条件影响分析:")
    print("-" * 40)
    
    for lighting in df['lighting'].unique():
        subset = df[df['lighting'] == lighting]
        if len(subset) > 0:
            print(f"\n🌞 {lighting.upper()} 光照:")
            print(f"   📏 平均Chamfer距离: {subset['chamfer_distance_mm'].mean():.2f} mm")
            print(f"   📐 平均Hausdorff距离: {subset['hausdorff_distance_mm'].mean():.2f} mm")
            print(f"   🎯 平均3mm覆盖率: {subset['coverage_3mm'].mean()*100:.1f}%")
            print(f"   ⏱️  平均执行时间: {subset['execution_time_s'].mean():.2f} s")
    
    # 方法对比总结
    print(f"\n🏆 方法对比总结:")
    print("-" * 40)
    
    canny_data = df[df['method'] == 'canny']
    rgbd_data = df[df['method'] == 'rgbd']
    
    print(f"\n🔵 CANNY 方法:")
    print(f"   📏 平均Chamfer距离: {canny_data['chamfer_distance_mm'].mean():.2f} mm")
    print(f"   ⏱️  平均执行时间: {canny_data['execution_time_s'].mean():.2f} s")
    print(f"   📊 平均点云数量: {canny_data['point_count'].mean():.0f}")
    
    print(f"\n🔴 RGB-D 方法:")
    print(f"   📏 平均Chamfer距离: {rgbd_data['chamfer_distance_mm'].mean():.2f} mm")
    print(f"   ⏱️  平均执行时间: {rgbd_data['execution_time_s'].mean():.2f} s")
    print(f"   📊 平均点云数量: {rgbd_data['point_count'].mean():.0f}")
    
    # 推荐结论
    print(f"\n💡 推荐结论:")
    print("-" * 40)
    
    # 简单形状分析
    cube_data = df[df['object_name'] == 'Cube']
    cube_canny = cube_data[cube_data['method'] == 'canny']
    cube_rgbd = cube_data[cube_data['method'] == 'rgbd']
    
    if cube_canny['chamfer_distance_mm'].mean() < cube_rgbd['chamfer_distance_mm'].mean():
        print(f"   📦 简单形状 (Cube): 推荐使用 CANNY 方法")
        print(f"      - 精度更高: {cube_canny['chamfer_distance_mm'].mean():.2f} vs {cube_rgbd['chamfer_distance_mm'].mean():.2f} mm")
        print(f"      - 速度更快: {cube_canny['execution_time_s'].mean():.2f} vs {cube_rgbd['execution_time_s'].mean():.2f} s")
    else:
        print(f"   📦 简单形状 (Cube): 推荐使用 RGB-D 方法")
    
    # 复杂形状分析
    ctry_data = df[df['object_name'] == 'ctry.obj']
    ctry_canny = ctry_data[ctry_data['method'] == 'canny']
    ctry_rgbd = ctry_data[ctry_data['method'] == 'rgbd']
    
    if ctry_canny['chamfer_distance_mm'].mean() < ctry_rgbd['chamfer_distance_mm'].mean():
        print(f"   🔺 复杂形状 (ctry.obj): 推荐使用 CANNY 方法")
        print(f"      - 精度更高: {ctry_canny['chamfer_distance_mm'].mean():.2f} vs {ctry_rgbd['chamfer_distance_mm'].mean():.2f} mm")
    else:
        print(f"   🔺 复杂形状 (ctry.obj): 推荐使用 RGB-D 方法")
        print(f"      - 精度更高: {ctry_rgbd['chamfer_distance_mm'].mean():.2f} vs {ctry_canny['chamfer_distance_mm'].mean():.2f} mm")
        print(f"      - 覆盖率更好: {ctry_rgbd['coverage_3mm'].mean()*100:.1f}% vs {ctry_canny['coverage_3mm'].mean()*100:.1f}%")
    
    print("="*80)

def main():
    """主函数"""
    print("边界追踪实验结果简化可视化分析")
    print("="*50)
    
    # 加载数据
    df = load_data()
    
    # 生成可视化图表
    create_key_metrics_comparison(df)
    create_lighting_impact_analysis(df)
    create_performance_summary(df)
    
    # 打印关键洞察
    print_key_insights(df)
    
    print("\n🎉 可视化分析完成！")
    print("📂 图表已保存到: ../../evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/")

if __name__ == "__main__":
    main()