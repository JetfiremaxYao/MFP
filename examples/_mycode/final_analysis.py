#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终版实验结果分析
专注于检测时间对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_final_data(csv_path):
    """加载最终版数据"""
    df = pd.read_csv(csv_path)
    
    # 检查数据情况
    print(f"原始数据: {len(df)} 条记录")
    print(f"光照条件分布: {df['lighting'].value_counts().to_dict()}")
    print(f"方法分布: {df['method'].value_counts().to_dict()}")
    
    # 过滤掉无效数据（inf值），但保留失败记录用于分析
    df_valid = df.replace([np.inf, -np.inf], np.nan)
    
    # 检查列名并适配不同的CSV格式
    if 'pos_err_m' in df_valid.columns and 'range_err_m' in df_valid.columns:
        # 统一版格式
        df_valid = df_valid.dropna(subset=['pos_err_m', 'range_err_m'])
    elif 'pos_error_mm' in df_valid.columns and 'range_error_mm' in df_valid.columns:
        # 最终版格式 - 需要转换单位
        df_valid = df_valid.dropna(subset=['pos_error_mm', 'range_error_mm'])
        # 将毫米转换为米
        df_valid['pos_err_m'] = df_valid['pos_error_mm'] / 1000
        df_valid['range_err_m'] = df_valid['range_error_mm'] / 1000
    else:
        print("警告：未找到位置误差和范围误差列")
        return df_valid
    
    print(f"有效数据: {len(df_valid)} 条记录")
    print(f"有效数据光照条件分布: {df_valid['lighting'].value_counts().to_dict()}")
    
    return df_valid

def create_final_comparison(df, output_dir):
    """创建最终版对比图表 - 包含position error、range error和detection time"""
    
    # 创建三个子图的综合对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Algorithm Performance Comparison Under Different Lighting Conditions', fontsize=16, fontweight='bold')
    
    # 定义一致的颜色方案和透明度
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色(hsv_depth)和橙色(depth_clustering)
    alpha = 0.8  # 统一的透明度
    method_order = ['hsv_depth', 'depth_clustering']  # 确保图例顺序一致
    
    # 1. Position Error Comparison
    sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', hue_order=method_order, ax=axes[0])
    # 设置颜色和透明度
    for i, patch in enumerate(axes[0].artists):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(alpha)
    axes[0].set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Position Error (mm)', fontsize=12)
    axes[0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title='Detection Method', loc='upper right')
    
    # 2. Range Error Comparison
    sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', hue_order=method_order, ax=axes[1])
    # 设置颜色和透明度
    for i, patch in enumerate(axes[1].artists):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(alpha)
    axes[1].set_title('Range Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Range Error (mm)', fontsize=12)
    axes[1].set_xlabel('Lighting Condition', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='Detection Method', loc='upper right')
    
    # 3. Detection Time Comparison (使用柱状图)
    avg_times = df.groupby(['method', 'lighting'])['detection_time_s'].mean().reset_index()
    sns.barplot(data=avg_times, x='lighting', y='detection_time_s', hue='method', hue_order=method_order, ax=axes[2])
    # 设置颜色和透明度
    for i, patch in enumerate(axes[2].patches):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(alpha)
    axes[2].set_title('Detection Time Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Average Detection Time (s)', fontsize=12)
    axes[2].set_xlabel('Lighting Condition', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(title='Detection Method', loc='upper right')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "final_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_detailed_time_analysis(df, output_dir):
    """创建详细的时间分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Time Performance Analysis', fontsize=16, fontweight='bold')
    
    # 定义一致的颜色方案和透明度
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色(hsv_depth)和橙色(depth_clustering)
    alpha = 0.8  # 统一的透明度
    method_order = ['hsv_depth', 'depth_clustering']  # 确保图例顺序一致
    method_colors = {'hsv_depth': colors[0], 'depth_clustering': colors[1]}
    
    # 1. Average Detection Time Comparison (按光照条件)
    avg_times = df.groupby(['method', 'lighting'])['detection_time_s'].mean().reset_index()
    sns.barplot(data=avg_times, x='lighting', y='detection_time_s', hue='method', hue_order=method_order, ax=axes[0, 0])
    # 设置颜色和透明度
    for i, patch in enumerate(axes[0, 0].patches):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(alpha)
    axes[0, 0].set_title('Average Detection Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Average Detection Time (s)', fontsize=12)
    axes[0, 0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(title='Detection Method', loc='upper right')
    
    # 2. Detection Time Distribution
    for method in method_order:
        if method in df['method'].unique():
            method_data = df[df['method'] == method]
            color = method_colors[method]
            axes[0, 1].hist(method_data['detection_time_s'], alpha=alpha, label=method, 
                           bins=20, color=color)
    axes[0, 1].set_title('Detection Time Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Detection Time (s)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend(title='Detection Method', loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Success Rate Comparison (按光照条件)
    success_rates = df.groupby(['method', 'lighting'])['success_pos_2cm'].mean().reset_index()
    sns.barplot(data=success_rates, x='lighting', y='success_pos_2cm', hue='method', hue_order=method_order, ax=axes[1, 0])
    # 设置颜色和透明度
    for i, patch in enumerate(axes[1, 0].patches):
        method_idx = i % 2
        patch.set_facecolor(colors[method_idx])
        patch.set_alpha(alpha)
    axes[1, 0].set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Success Rate', fontsize=12)
    axes[1, 0].set_xlabel('Lighting Condition', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(title='Detection Method', loc='upper right')
    
    # 4. Speed-Accuracy Trade-off
    for method in method_order:
        if method in df['method'].unique():
            method_data = df[df['method'] == method]
            color = method_colors[method]
            axes[1, 1].scatter(method_data['detection_time_s'], method_data['pos_err_m'] * 1000, 
                              alpha=alpha, label=method, s=60, color=color)
    axes[1, 1].set_title('Speed-Accuracy Trade-off', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Detection Time (s)', fontsize=12)
    axes[1, 1].set_ylabel('Position Error (mm)', fontsize=12)
    axes[1, 1].legend(title='Detection Method', loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "detailed_time_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_final_summary(df):
    """生成统一版汇总统计"""
    
    print("\n" + "="*60)
    print("统一版实验设计说明")
    print("="*60)
    
    print("1. 实验目标：对比HSV+Depth和Depth+Clustering两种检测方法在不同光照条件下的性能")
    print("2. 主要对比维度：光照条件（normal, bright, dim）")
    print("3. 物体类型：盒子（统一使用相同物体）")
    print("4. 位置范围：固定基础位置 + 小扰动")
    print("5. 实验条件：")
    print("   - 光照条件：normal, bright, dim (3种)")
    print("   - 每组条件重复：3次")
    print("   - 总试验次数：3×3×2 = 18次")
    print("6. 评估指标：检测时间、位置精度、范围精度")
    print("7. 成功标准：位置误差 < 2cm 且 范围误差 < 2cm")
    
    print("\n" + "="*60)
    print("核心指标对比结果")
    print("="*60)
    
    # 按方法分组统计
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        method_name = "HSV+Depth" if method == "hsv_depth" else "Depth+Clustering"
        
        print(f"\n{method_name}方法:")
        print(f"  平均检测时间: {method_data['detection_time_s'].mean():.2f}±{method_data['detection_time_s'].std():.2f}秒")
        print(f"  平均位置误差: {method_data['pos_err_m'].mean()*1000:.1f}±{method_data['pos_err_m'].std()*1000:.1f}mm")
        print(f"  平均范围误差: {method_data['range_err_m'].mean()*1000:.1f}±{method_data['range_err_m'].std()*1000:.1f}mm")
        print(f"  总体成功率: {method_data['success_pos_2cm'].mean():.1%}")
        
        # 按光照条件统计
        print(f"  按光照条件:")
        for lighting in df['lighting'].unique():
            lighting_data = method_data[method_data['lighting'] == lighting]
            if len(lighting_data) > 0:
                print(f"    {lighting}:")
                print(f"      检测时间: {lighting_data['detection_time_s'].mean():.2f}±{lighting_data['detection_time_s'].std():.2f}秒")
                print(f"      位置误差: {lighting_data['pos_err_m'].mean()*1000:.1f}±{lighting_data['pos_err_m'].std()*1000:.1f}mm")
                print(f"      范围误差: {lighting_data['range_err_m'].mean()*1000:.1f}±{lighting_data['range_err_m'].std()*1000:.1f}mm")
                print(f"      成功率: {lighting_data['success_pos_2cm'].mean():.1%}")
    
    print("\n" + "="*60)
    print("结论与建议")
    print("="*60)
    
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    print(f"1. 检测时间: HSV+Depth ({hsv_data['detection_time_s'].mean():.2f}s) vs Depth+Clustering ({clustering_data['detection_time_s'].mean():.2f}s)")
    print(f"2. 位置精度: HSV+Depth ({hsv_data['pos_err_m'].mean()*1000:.1f}mm) vs Depth+Clustering ({clustering_data['pos_err_m'].mean()*1000:.1f}mm)")
    print(f"3. 范围精度: HSV+Depth ({hsv_data['range_err_m'].mean()*1000:.1f}mm) vs Depth+Clustering ({clustering_data['range_err_m'].mean()*1000:.1f}mm)")
    print(f"4. 成功率: HSV+Depth ({hsv_data['success_pos_2cm'].mean():.1%}) vs Depth+Clustering ({clustering_data['success_pos_2cm'].mean():.1%})")
    
    print("\n推荐:")
    if hsv_data['detection_time_s'].mean() < clustering_data['detection_time_s'].mean():
        print("• HSV+Depth在检测速度方面更优")
    else:
        print("• Depth+Clustering在检测速度方面更优")
        
    if hsv_data['pos_err_m'].mean() < clustering_data['pos_err_m'].mean():
        print("• HSV+Depth在位置精度方面更优")
    else:
        print("• Depth+Clustering在位置精度方面更优")
        
    if hsv_data['range_err_m'].mean() < clustering_data['range_err_m'].mean():
        print("• HSV+Depth在范围精度方面更优")
    else:
        print("• Depth+Clustering在范围精度方面更优")
        
    if hsv_data['success_pos_2cm'].mean() > clustering_data['success_pos_2cm'].mean():
        print("• HSV+Depth在成功率方面更优")
    else:
        print("• Depth+Clustering在成功率方面更优")

def main():
    """主函数"""
    # 查找最新的CSV文件
    output_dir = Path("evaluation_results")
    
    # 查找所有可能的CSV文件
    csv_files = []
    for pattern in ["**/trial_results.csv", "**/final_trial_results.csv"]:
        csv_files.extend(output_dir.glob(pattern))
    
    if not csv_files:
        print("未找到任何CSV文件")
        return
    
    # 选择最新的文件
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"使用最新的CSV文件: {latest_csv}")
    
    # 设置输出目录
    output_dir = latest_csv.parent
    
    print("开始加载数据...")
    df = load_final_data(latest_csv)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    print("生成综合对比图表...")
    chart1 = create_final_comparison(df, output_dir)
    print(f"对比图表已保存: {chart1}")
    
    print("生成详细时间分析图表...")
    chart2 = create_detailed_time_analysis(df, output_dir)
    print(f"时间分析图表已保存: {chart2}")
    
    # 生成统计汇总
    generate_final_summary(df)

if __name__ == "__main__":
    main()
