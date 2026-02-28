#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁的算法对比分析
只对比位置精度、范围精度和检测时间三个核心指标
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

def load_data(csv_path):
    """加载数据"""
    df = pd.read_csv(csv_path)
    
    # 转换单位到毫米
    df['pos_err_mm'] = df['pos_err_m'] * 1000
    df['range_err_mm'] = df['range_err_m'] * 1000
    
    return df

def create_simple_comparison(df, output_dir):
    """创建简洁的对比图表"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('算法性能对比：位置精度 vs 范围精度 vs 检测时间', fontsize=16, fontweight='bold')
    
    # 1. 位置误差对比
    sns.boxplot(data=df, x='lighting', y='pos_err_mm', hue='method', ax=axes[0])
    axes[0].set_title('位置精度对比', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('位置误差 (mm)', fontsize=12)
    axes[0].set_xlabel('光照条件', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title='检测方法')
    
    # 2. 范围误差对比
    sns.boxplot(data=df, x='lighting', y='range_err_mm', hue='method', ax=axes[1])
    axes[1].set_title('范围精度对比', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('范围误差 (mm)', fontsize=12)
    axes[1].set_xlabel('光照条件', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='检测方法')
    
    # 3. 检测时间对比
    sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=axes[2])
    axes[2].set_title('检测时间对比', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('检测时间 (秒)', fontsize=12)
    axes[2].set_xlabel('光照条件', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(title='检测方法')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "simple_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_summary_statistics(df):
    """生成汇总统计"""
    
    print("\n" + "="*60)
    print("实验设计说明")
    print("="*60)
    
    print("1. 实验目标：对比HSV+Depth和Depth+Clustering两种物体检测方法")
    print("2. 评估指标：位置精度、范围精度、检测时间")
    print("3. 实验条件：")
    print("   - 光照条件：normal, bright, dim (3种)")
    print("   - 背景条件：simple (1种)")
    print("   - 每组条件重复：3次")
    print("   - 总试验次数：3×3×2 = 18次")
    print("4. 成功标准：位置误差 < 2cm 且 范围误差 < 2cm")
    
    print("\n" + "="*60)
    print("核心指标对比结果")
    print("="*60)
    
    # 按方法分组统计
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        method_name = "HSV+Depth" if method == "hsv_depth" else "Depth+Clustering"
        
        print(f"\n{method_name}方法:")
        print(f"  位置精度: {method_data['pos_err_mm'].mean():.1f}±{method_data['pos_err_mm'].std():.1f}mm")
        print(f"  范围精度: {method_data['range_err_mm'].mean():.1f}±{method_data['range_err_mm'].std():.1f}mm")
        print(f"  检测时间: {method_data['detection_time_s'].mean():.1f}±{method_data['detection_time_s'].std():.1f}秒")
        
        # 按光照条件统计
        for lighting in df['lighting'].unique():
            lighting_data = method_data[method_data['lighting'] == lighting]
            if len(lighting_data) > 0:
                print(f"    {lighting}光照:")
                print(f"      位置精度: {lighting_data['pos_err_mm'].mean():.1f}±{lighting_data['pos_err_mm'].std():.1f}mm")
                print(f"      范围精度: {lighting_data['range_err_mm'].mean():.1f}±{lighting_data['range_err_mm'].std():.1f}mm")
                print(f"      检测时间: {lighting_data['detection_time_s'].mean():.1f}±{lighting_data['detection_time_s'].std():.1f}秒")
    
    print("\n" + "="*60)
    print("结论与建议")
    print("="*60)
    
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    print(f"1. 位置精度: HSV+Depth ({hsv_data['pos_err_mm'].mean():.1f}mm) vs Depth+Clustering ({clustering_data['pos_err_mm'].mean():.1f}mm)")
    print(f"2. 范围精度: HSV+Depth ({hsv_data['range_err_mm'].mean():.1f}mm) vs Depth+Clustering ({clustering_data['range_err_mm'].mean():.1f}mm)")
    print(f"3. 检测时间: HSV+Depth ({hsv_data['detection_time_s'].mean():.1f}s) vs Depth+Clustering ({clustering_data['detection_time_s'].mean():.1f}s)")
    
    print("\n推荐:")
    if hsv_data['pos_err_mm'].mean() < clustering_data['pos_err_mm'].mean():
        print("• HSV+Depth在位置精度方面更优")
    else:
        print("• Depth+Clustering在位置精度方面更优")
        
    if hsv_data['range_err_mm'].mean() < clustering_data['range_err_mm'].mean():
        print("• HSV+Depth在范围精度方面更优")
    else:
        print("• Depth+Clustering在范围精度方面更优")
        
    if hsv_data['detection_time_s'].mean() < clustering_data['detection_time_s'].mean():
        print("• HSV+Depth在检测速度方面更优")
    else:
        print("• Depth+Clustering在检测速度方面更优")

def main():
    """主函数"""
    # 设置路径
    csv_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723/trial_results.csv"
    output_dir = Path("/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723")
    
    print("开始加载数据...")
    df = load_data(csv_path)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    print("生成简洁对比图表...")
    chart_file = create_simple_comparison(df, output_dir)
    print(f"图表已保存: {chart_file}")
    
    # 生成统计汇总
    generate_summary_statistics(df)

if __name__ == "__main__":
    main()

