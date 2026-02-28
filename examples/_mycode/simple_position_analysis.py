#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版位置误差分析
直接读取CSV并生成与参考图片格式一致的图表
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_csv_data(csv_path):
    """加载CSV数据"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def create_position_error_chart(data, output_path):
    """创建位置误差对比图表 - 与参考图片格式一致"""
    
    # 提取数据
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['hsv_depth', 'depth_clustering']
    
    # 组织数据
    plot_data = {}
    for lighting in lighting_conditions:
        plot_data[lighting] = {}
        for method in methods:
            plot_data[lighting][method] = []
    
    for row in data:
        lighting = row['lighting']
        method = row['method']
        pos_err = float(row['pos_err_m']) * 1000  # 转换为毫米
        if lighting in plot_data and method in plot_data[lighting]:
            plot_data[lighting][method].append(pos_err)
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 准备箱线图数据
    box_data = []
    box_labels = []
    box_positions = []
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    
    x_pos = 0
    for lighting in lighting_conditions:
        for i, method in enumerate(methods):
            if plot_data[lighting][method]:
                box_data.append(plot_data[lighting][method])
                box_labels.append(f'{lighting}\n{method}')
                box_positions.append(x_pos)
                x_pos += 1
        x_pos += 0.5  # 光照条件之间的间距
    
    # 绘制箱线图
    bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True, 
                   widths=0.6, showfliers=True)
    
    # 设置颜色
    for i, patch in enumerate(bp['boxes']):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(0.7)
    
    # 设置标题和标签
    ax.set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Position Error (mm)', fontsize=12)
    ax.set_xlabel('Lighting Condition', fontsize=12)
    
    # 设置X轴标签
    lighting_positions = [1, 3.5, 6]  # 每个光照条件的中心位置
    ax.set_xticks(lighting_positions)
    ax.set_xticklabels(lighting_conditions)
    
    # 设置Y轴范围，与参考图片一致 (2.5-8.5mm)
    ax.set_ylim(2.5, 8.5)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], alpha=0.7, label='hsv_depth'),
                      Patch(facecolor=colors[1], alpha=0.7, label='depth_clustering')]
    ax.legend(handles=legend_elements, title='Detection Method', loc='upper right')
    
    # 在箱线图上添加中位数标注
    for i, (data_group, pos) in enumerate(zip(box_data, box_positions)):
        if data_group:
            median_val = np.median(data_group)
            ax.text(pos, median_val, f'{median_val:.1f}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def print_summary(data):
    """打印数据汇总"""
    print("\n" + "="*60)
    print("位置误差分析结果")
    print("="*60)
    
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['hsv_depth', 'depth_clustering']
    
    for lighting in lighting_conditions:
        print(f"\n{lighting.upper()}光照条件:")
        for method in methods:
            method_data = [float(row['pos_err_m']) * 1000 for row in data 
                          if row['lighting'] == lighting and row['method'] == method]
            if method_data:
                method_name = "HSV+Depth" if method == "hsv_depth" else "Depth+Clustering"
                mean_err = np.mean(method_data)
                std_err = np.std(method_data)
                median_err = np.median(method_data)
                print(f"  {method_name}: 平均={mean_err:.1f}±{std_err:.1f}mm, 中位数={median_err:.1f}mm")

def main():
    """主函数"""
    # CSV文件路径
    csv_file = Path("evaluation_results/unified_object_detection_20250827_152854/trial_results.csv")
    output_file = csv_file.parent / "position_error_comparison_fixed.png"
    
    if not csv_file.exists():
        print(f"未找到CSV文件: {csv_file}")
        return
    
    print("开始加载数据...")
    data = load_csv_data(csv_file)
    print(f"数据加载完成，共 {len(data)} 条记录")
    
    print("生成位置误差对比图表...")
    chart_path = create_position_error_chart(data, output_file)
    print(f"图表已保存: {chart_path}")
    
    # 打印汇总
    print_summary(data)

if __name__ == "__main__":
    main()

