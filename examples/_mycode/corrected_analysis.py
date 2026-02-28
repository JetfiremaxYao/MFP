#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版算法对比分析
修正召回率计算错误
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

def load_and_correct_data(csv_path):
    """加载数据并修正召回率计算"""
    df = pd.read_csv(csv_path)
    
    # 转换单位到毫米
    df['pos_err_mm'] = df['pos_err_m'] * 1000
    df['range_err_mm'] = df['range_err_m'] * 1000
    
    # 修正召回率计算
    # 正确的召回率 = 命中视角数 / 实际检测的视角数
    df['corrected_view_hit_rate'] = df['views_hit'] / (df['early_stop_index'] + 1)
    
    # 计算提前终止率 (提前终止的视角数 / 总视角数)
    df['early_termination_rate'] = 1 - (df['early_stop_index'] + 1) / df['views_total']
    
    return df

def create_corrected_comparison(df, output_dir):
    """创建修正后的对比图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('修正后的算法对比分析', fontsize=16, fontweight='bold')
    
    # 1. 位置误差对比
    sns.boxplot(data=df, x='lighting', y='pos_err_mm', hue='method', ax=axes[0, 0])
    axes[0, 0].set_title('位置误差对比 (mm)', fontsize=12)
    axes[0, 0].set_ylabel('位置误差 (mm)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 范围误差对比
    sns.boxplot(data=df, x='lighting', y='range_err_mm', hue='method', ax=axes[0, 1])
    axes[0, 1].set_title('范围误差对比 (mm)', fontsize=12)
    axes[0, 1].set_ylabel('范围误差 (mm)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 检测时间对比
    sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=axes[0, 2])
    axes[0, 2].set_title('检测时间对比 (秒)', fontsize=12)
    axes[0, 2].set_ylabel('检测时间 (秒)', fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 修正后的召回率对比
    recall_data = df.groupby(['method', 'lighting'])['corrected_view_hit_rate'].mean().reset_index()
    sns.barplot(data=recall_data, x='lighting', y='corrected_view_hit_rate', hue='method', ax=axes[1, 0])
    axes[1, 0].set_title('修正后的召回率对比', fontsize=12)
    axes[1, 0].set_ylabel('召回率', fontsize=10)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 提前终止率对比
    termination_data = df.groupby(['method', 'lighting'])['early_termination_rate'].mean().reset_index()
    sns.barplot(data=termination_data, x='lighting', y='early_termination_rate', hue='method', ax=axes[1, 1])
    axes[1, 1].set_title('提前终止率对比', fontsize=12)
    axes[1, 1].set_ylabel('提前终止率', fontsize=10)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 效率对比 (精度/时间)
    df['efficiency'] = 1 / (df['pos_err_mm'] * df['detection_time_s'])
    sns.boxplot(data=df, x='lighting', y='efficiency', hue='method', ax=axes[1, 2])
    axes[1, 2].set_title('效率对比 (精度/时间)', fontsize=12)
    axes[1, 2].set_ylabel('效率', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "corrected_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_detailed_analysis(df, output_dir):
    """创建详细分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('详细性能分析', fontsize=16, fontweight='bold')
    
    # 1. 检测步骤数对比
    df['steps_taken'] = df['early_stop_index'] + 1
    sns.boxplot(data=df, x='lighting', y='steps_taken', hue='method', ax=axes[0, 0])
    axes[0, 0].set_title('实际检测步骤数对比', fontsize=12)
    axes[0, 0].set_ylabel('检测步骤数', fontsize=10)
    axes[0, 0].axhline(y=24, color='red', linestyle='--', alpha=0.7, label='最大步骤数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 命中率对比 (在每个检测步骤中的成功率)
    sns.boxplot(data=df, x='lighting', y='corrected_view_hit_rate', hue='method', ax=axes[0, 1])
    axes[0, 1].set_title('命中率对比 (每步骤成功率)', fontsize=12)
    axes[0, 1].set_ylabel('命中率', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 速度-精度权衡 (修正版)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        axes[1, 0].scatter(method_data['detection_time_s'], method_data['pos_err_mm'], 
                          label=method, alpha=0.7, s=60)
    axes[1, 0].set_title('速度-精度权衡', fontsize=12)
    axes[1, 0].set_xlabel('检测时间 (秒)', fontsize=10)
    axes[1, 0].set_ylabel('位置误差 (mm)', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 综合性能评分
    # 权重: 精度40%, 速度30%, 命中率30%
    def calculate_score(row):
        # 精度分数 (误差越小分数越高)
        pos_score = max(0, 1 - row['pos_err_mm'] / 20)
        range_score = max(0, 1 - row['range_err_mm'] / 10)
        accuracy_score = (pos_score + range_score) / 2
        
        # 速度分数 (时间越短分数越高)
        speed_score = max(0, 1 - row['detection_time_s'] / 30)
        
        # 命中率分数
        hit_score = row['corrected_view_hit_rate']
        
        # 综合评分
        total_score = 0.4 * accuracy_score + 0.3 * speed_score + 0.3 * hit_score
        return total_score
    
    df['综合评分'] = df.apply(calculate_score, axis=1)
    score_data = df.groupby(['method', 'lighting'])['综合评分'].mean().reset_index()
    sns.barplot(data=score_data, x='lighting', y='综合评分', hue='method', ax=axes[1, 1])
    axes[1, 1].set_title('综合性能评分', fontsize=12)
    axes[1, 1].set_ylabel('综合评分', fontsize=10)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "detailed_corrected_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_corrected_recommendations(df):
    """生成修正后的建议"""
    
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    # 计算修正后的统计指标
    hsv_stats = {
        'pos_err_mm': hsv_data['pos_err_mm'].mean(),
        'range_err_mm': hsv_data['range_err_mm'].mean(),
        'detection_time_s': hsv_data['detection_time_s'].mean(),
        'corrected_hit_rate': hsv_data['corrected_view_hit_rate'].mean(),
        'early_termination_rate': hsv_data['early_termination_rate'].mean(),
        'steps_taken': hsv_data['early_stop_index'].mean() + 1
    }
    
    clustering_stats = {
        'pos_err_mm': clustering_data['pos_err_mm'].mean(),
        'range_err_mm': clustering_data['range_err_mm'].mean(),
        'detection_time_s': clustering_data['detection_time_s'].mean(),
        'corrected_hit_rate': clustering_data['corrected_view_hit_rate'].mean(),
        'early_termination_rate': clustering_data['early_termination_rate'].mean(),
        'steps_taken': clustering_data['early_stop_index'].mean() + 1
    }
    
    recommendations = {
        "HSV+Depth方法": {
            "优势": [
                f"检测速度快 (平均 {hsv_stats['detection_time_s']:.1f}秒)",
                f"提前终止能力强 (平均 {hsv_stats['early_termination_rate']:.1%} 提前终止)",
                f"实际命中率高 ({hsv_stats['corrected_hit_rate']:.1%})",
                f"平均只需 {hsv_stats['steps_taken']:.1f} 个步骤",
                "适合实时应用场景"
            ],
            "劣势": [
                f"范围检测精度较低 (平均误差 {hsv_stats['range_err_mm']:.1f}mm)",
                "可能受颜色特征影响"
            ],
            "适用场景": [
                "实时检测系统",
                "对速度要求高的应用",
                "需要快速响应的场合",
                "计算资源有限的环境"
            ]
        },
        "Depth+Clustering方法": {
            "优势": [
                f"范围检测精度高 (平均误差 {clustering_stats['range_err_mm']:.1f}mm)",
                f"命中率稳定 ({clustering_stats['corrected_hit_rate']:.1%})",
                "光照稳定性好",
                "适合高精度要求场景"
            ],
            "劣势": [
                f"检测时间较长 (平均 {clustering_stats['detection_time_s']:.1f}秒)",
                f"无法提前终止 (平均 {clustering_stats['early_termination_rate']:.1%} 提前终止)",
                f"需要完整 {clustering_stats['steps_taken']:.1f} 个步骤",
                "计算资源需求高"
            ],
            "适用场景": [
                "高精度检测需求",
                "离线处理系统",
                "复杂光照环境",
                "对稳定性要求高的应用"
            ]
        }
    }
    
    return recommendations, hsv_stats, clustering_stats

def main():
    """主函数"""
    # 设置路径
    csv_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723/trial_results.csv"
    output_dir = Path("/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723")
    
    print("开始加载和修正数据...")
    df = load_and_correct_data(csv_path)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    print("生成修正后的对比图表...")
    chart1 = create_corrected_comparison(df, output_dir)
    print(f"修正后的对比图表已保存: {chart1}")
    
    print("生成详细分析图表...")
    chart2 = create_detailed_analysis(df, output_dir)
    print(f"详细分析图表已保存: {chart2}")
    
    print("生成修正后的建议...")
    recommendations, hsv_stats, clustering_stats = generate_corrected_recommendations(df)
    
    # 保存修正后的建议
    with open(output_dir / "corrected_algorithm_recommendations.txt", 'w', encoding='utf-8') as f:
        f.write("修正后的算法选择建议报告\n")
        f.write("="*50 + "\n\n")
        
        f.write("重要说明：\n")
        f.write("原始召回率计算存在错误，已修正为：\n")
        f.write("修正召回率 = 命中视角数 / 实际检测的视角数\n")
        f.write("提前终止率 = (总视角数 - 实际检测视角数) / 总视角数\n\n")
        
        for method, info in recommendations.items():
            f.write(f"{method}\n")
            f.write("-" * 30 + "\n")
            
            f.write("优势:\n")
            for advantage in info["优势"]:
                f.write(f"  • {advantage}\n")
            
            f.write("\n劣势:\n")
            for disadvantage in info["劣势"]:
                f.write(f"  • {disadvantage}\n")
            
            f.write("\n适用场景:\n")
            for scenario in info["适用场景"]:
                f.write(f"  • {scenario}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print("修正后的建议已保存到文件")
    
    # 输出关键结论
    print("\n" + "="*60)
    print("修正后的关键结论:")
    print("="*60)
    
    print(f"1. 速度对比: HSV+Depth ({hsv_stats['detection_time_s']:.1f}s) vs Depth+Clustering ({clustering_stats['detection_time_s']:.1f}s)")
    print(f"2. 位置精度: HSV+Depth ({hsv_stats['pos_err_mm']:.1f}mm) vs Depth+Clustering ({clustering_stats['pos_err_mm']:.1f}mm)")
    print(f"3. 范围精度: HSV+Depth ({hsv_stats['range_err_mm']:.1f}mm) vs Depth+Clustering ({clustering_stats['range_err_mm']:.1f}mm)")
    print(f"4. 修正召回率: HSV+Depth ({hsv_stats['corrected_hit_rate']:.1%}) vs Depth+Clustering ({clustering_stats['corrected_hit_rate']:.1%})")
    print(f"5. 提前终止能力: HSV+Depth ({hsv_stats['early_termination_rate']:.1%}) vs Depth+Clustering ({clustering_stats['early_termination_rate']:.1%})")
    print(f"6. 平均检测步骤: HSV+Depth ({hsv_stats['steps_taken']:.1f}) vs Depth+Clustering ({clustering_stats['steps_taken']:.1f})")
    
    print("\n修正后的推荐:")
    print("• HSV+Depth方法实际上具有更好的提前终止能力和命中率")
    print("• 如果需要实时检测和快速响应 → 选择 HSV+Depth")
    print("• 如果需要高精度范围检测 → 选择 Depth+Clustering")
    print("• 如果需要在复杂光照环境下工作 → 选择 Depth+Clustering")
    print("• 如果计算资源有限 → 选择 HSV+Depth")

if __name__ == "__main__":
    main()

