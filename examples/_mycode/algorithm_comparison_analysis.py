#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法对比分析可视化脚本
基于实验结果数据进行深度分析和可视化
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

def load_and_analyze_data(csv_path):
    """加载和分析数据"""
    df = pd.read_csv(csv_path)
    
    # 转换单位到毫米
    df['pos_err_mm'] = df['pos_err_m'] * 1000
    df['range_err_mm'] = df['range_err_m'] * 1000
    
    return df

def create_comprehensive_comparison(df, output_dir):
    """创建综合对比图表"""
    
    # 创建大图表
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 位置误差对比 (箱线图)
    ax1 = plt.subplot(3, 4, 1)
    sns.boxplot(data=df, x='lighting', y='pos_err_mm', hue='method', ax=ax1)
    ax1.set_title('位置误差对比 (mm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('位置误差 (mm)', fontsize=12)
    ax1.set_xlabel('光照条件', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 范围误差对比 (箱线图)
    ax2 = plt.subplot(3, 4, 2)
    sns.boxplot(data=df, x='lighting', y='range_err_mm', hue='method', ax=ax2)
    ax2.set_title('范围误差对比 (mm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('范围误差 (mm)', fontsize=12)
    ax2.set_xlabel('光照条件', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. 检测时间对比 (箱线图)
    ax3 = plt.subplot(3, 4, 3)
    sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=ax3)
    ax3.set_title('检测时间对比 (秒)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('检测时间 (秒)', fontsize=12)
    ax3.set_xlabel('光照条件', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. 视角召回率对比 (柱状图)
    ax4 = plt.subplot(3, 4, 4)
    recall_data = df.groupby(['method', 'lighting'])['view_hit_rate'].mean().reset_index()
    sns.barplot(data=recall_data, x='lighting', y='view_hit_rate', hue='method', ax=ax4)
    ax4.set_title('视角召回率对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('召回率', fontsize=12)
    ax4.set_xlabel('光照条件', fontsize=12)
    ax4.set_ylim(0, 0.5)
    ax4.grid(True, alpha=0.3)
    
    # 5. 速度-精度权衡散点图
    ax5 = plt.subplot(3, 4, 5)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax5.scatter(method_data['detection_time_s'], method_data['pos_err_mm'], 
                   label=method, alpha=0.7, s=60)
    ax5.set_title('速度-精度权衡', fontsize=14, fontweight='bold')
    ax5.set_xlabel('检测时间 (秒)', fontsize=12)
    ax5.set_ylabel('位置误差 (mm)', fontsize=12)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 综合性能雷达图
    ax6 = plt.subplot(3, 4, 6, projection='polar')
    
    # 计算各方法的综合性能指标
    performance_data = {}
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        # 归一化各项指标 (0-1之间，1为最好)
        accuracy = 1.0 / (1.0 + method_df['pos_err_mm'].mean() / 20)  # 20mm为基准
        precision = 1.0 / (1.0 + method_df['range_err_mm'].mean() / 10)  # 10mm为基准
        speed = 1.0 / (1.0 + method_df['detection_time_s'].mean() / 30)  # 30s为基准
        recall = method_df['view_hit_rate'].mean()
        
        performance_data[method] = [accuracy, precision, speed, recall]
    
    categories = ['位置精度', '范围精度', '速度', '召回率']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = ['#1f77b4', '#ff7f0e']
    for i, (method, data) in enumerate(performance_data.items()):
        data += data[:1]  # 闭合
        ax6.plot(angles, data, 'o-', linewidth=2, label=method, color=colors[i], alpha=0.7)
        ax6.fill(angles, data, alpha=0.1, color=colors[i])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories)
    ax6.set_ylim(0, 1)
    ax6.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
    
    # 7. 成功率对比
    ax7 = plt.subplot(3, 4, 7)
    success_data = df.groupby(['method', 'lighting'])['success_pos_2cm'].mean().reset_index()
    sns.barplot(data=success_data, x='lighting', y='success_pos_2cm', hue='method', ax=ax7)
    ax7.set_title('成功率对比 (2cm阈值)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('成功率', fontsize=12)
    ax7.set_xlabel('光照条件', fontsize=12)
    ax7.set_ylim(0, 1)
    ax7.grid(True, alpha=0.3)
    
    # 8. 光照适应性分析
    ax8 = plt.subplot(3, 4, 8)
    lighting_stability = df.groupby(['method', 'lighting'])['pos_err_mm'].std().reset_index()
    lighting_stability = lighting_stability.groupby('method')['pos_err_mm'].mean().reset_index()
    sns.barplot(data=lighting_stability, x='method', y='pos_err_mm', ax=ax8)
    ax8.set_title('光照稳定性 (误差标准差)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('误差标准差 (mm)', fontsize=12)
    ax8.set_xlabel('方法', fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    # 9. FPS对比
    ax9 = plt.subplot(3, 4, 9)
    sns.boxplot(data=df, x='lighting', y='fps', hue='method', ax=ax9)
    ax9.set_title('FPS对比', fontsize=14, fontweight='bold')
    ax9.set_ylabel('FPS', fontsize=12)
    ax9.set_xlabel('光照条件', fontsize=12)
    ax9.grid(True, alpha=0.3)
    
    # 10. 误差分布直方图
    ax10 = plt.subplot(3, 4, 10)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax10.hist(method_data['pos_err_mm'], alpha=0.7, label=method, bins=8)
    ax10.set_title('位置误差分布', fontsize=14, fontweight='bold')
    ax10.set_xlabel('位置误差 (mm)', fontsize=12)
    ax10.set_ylabel('频次', fontsize=12)
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. 范围误差vs位置误差散点图
    ax11 = plt.subplot(3, 4, 11)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        ax11.scatter(method_data['pos_err_mm'], method_data['range_err_mm'], 
                    label=method, alpha=0.7, s=60)
    ax11.set_title('位置误差 vs 范围误差', fontsize=14, fontweight='bold')
    ax11.set_xlabel('位置误差 (mm)', fontsize=12)
    ax11.set_ylabel('范围误差 (mm)', fontsize=12)
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. 综合评分对比
    ax12 = plt.subplot(3, 4, 12)
    
    # 计算综合评分 (权重: 精度40%, 速度30%, 召回率30%)
    def calculate_score(row):
        # 精度分数 (误差越小分数越高)
        pos_score = max(0, 1 - row['pos_err_mm'] / 20)
        range_score = max(0, 1 - row['range_err_mm'] / 10)
        accuracy_score = (pos_score + range_score) / 2
        
        # 速度分数 (时间越短分数越高)
        speed_score = max(0, 1 - row['detection_time_s'] / 30)
        
        # 召回率分数
        recall_score = row['view_hit_rate']
        
        # 综合评分
        total_score = 0.4 * accuracy_score + 0.3 * speed_score + 0.3 * recall_score
        return total_score
    
    df['综合评分'] = df.apply(calculate_score, axis=1)
    score_data = df.groupby(['method', 'lighting'])['综合评分'].mean().reset_index()
    sns.barplot(data=score_data, x='lighting', y='综合评分', hue='method', ax=ax12)
    ax12.set_title('综合评分对比', fontsize=14, fontweight='bold')
    ax12.set_ylabel('综合评分', fontsize=12)
    ax12.set_xlabel('光照条件', fontsize=12)
    ax12.set_ylim(0, 1)
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "comprehensive_algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_decision_matrix(df, output_dir):
    """创建决策矩阵图表"""
    
    # 计算各方法的平均性能指标
    summary_stats = df.groupby('method').agg({
        'pos_err_mm': ['mean', 'std'],
        'range_err_mm': ['mean', 'std'],
        'detection_time_s': ['mean', 'std'],
        'view_hit_rate': ['mean', 'std'],
        'success_pos_2cm': 'mean'
    }).round(2)
    
    # 创建决策矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. 性能指标对比表
    metrics = ['位置误差(mm)', '范围误差(mm)', '检测时间(s)', '召回率', '成功率']
    hsv_values = [
        f"{summary_stats.loc['hsv_depth', ('pos_err_mm', 'mean')]:.1f}±{summary_stats.loc['hsv_depth', ('pos_err_mm', 'std')]:.1f}",
        f"{summary_stats.loc['hsv_depth', ('range_err_mm', 'mean')]:.1f}±{summary_stats.loc['hsv_depth', ('range_err_mm', 'std')]:.1f}",
        f"{summary_stats.loc['hsv_depth', ('detection_time_s', 'mean')]:.1f}±{summary_stats.loc['hsv_depth', ('detection_time_s', 'std')]:.1f}",
        f"{summary_stats.loc['hsv_depth', ('view_hit_rate', 'mean')]:.2f}±{summary_stats.loc['hsv_depth', ('view_hit_rate', 'std')]:.2f}",
        f"{summary_stats.loc['hsv_depth', ('success_pos_2cm', 'mean')]:.2f}"
    ]
    clustering_values = [
        f"{summary_stats.loc['depth_clustering', ('pos_err_mm', 'mean')]:.1f}±{summary_stats.loc['depth_clustering', ('pos_err_mm', 'std')]:.1f}",
        f"{summary_stats.loc['depth_clustering', ('range_err_mm', 'mean')]:.1f}±{summary_stats.loc['depth_clustering', ('range_err_mm', 'std')]:.1f}",
        f"{summary_stats.loc['depth_clustering', ('detection_time_s', 'mean')]:.1f}±{summary_stats.loc['depth_clustering', ('detection_time_s', 'std')]:.1f}",
        f"{summary_stats.loc['depth_clustering', ('view_hit_rate', 'mean')]:.2f}±{summary_stats.loc['depth_clustering', ('view_hit_rate', 'std')]:.2f}",
        f"{summary_stats.loc['depth_clustering', ('success_pos_2cm', 'mean')]:.2f}"
    ]
    
    # 创建表格
    table_data = list(zip(metrics, hsv_values, clustering_values))
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=table_data, 
                     colLabels=['指标', 'HSV+Depth', 'Depth+Clustering'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    ax1.set_title('性能指标对比表', fontsize=16, fontweight='bold', pad=20)
    
    # 2. 应用场景推荐
    scenarios = ['实时检测', '高精度要求', '复杂光照', '快速响应', '稳定性优先']
    hsv_scores = [9, 6, 7, 9, 6]  # HSV+Depth在各场景的评分
    clustering_scores = [4, 9, 9, 4, 9]  # Depth+Clustering在各场景的评分
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, hsv_scores, width, label='HSV+Depth', alpha=0.8)
    bars2 = ax2.bar(x + width/2, clustering_scores, width, label='Depth+Clustering', alpha=0.8)
    
    ax2.set_xlabel('应用场景', fontsize=12)
    ax2.set_ylabel('适用性评分 (1-10)', fontsize=12)
    ax2.set_title('应用场景适用性对比', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 10)
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_dir / "decision_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_recommendations(df):
    """生成算法选择建议"""
    
    # 计算各方法的平均性能
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    recommendations = {
        "HSV+Depth方法": {
            "优势": [
                f"检测速度快 (平均 {hsv_data['detection_time_s'].mean():.1f}秒)",
                f"计算资源需求低 (FPS: {hsv_data['fps'].mean():.1f})",
                "适合实时应用场景",
                "对光照变化有一定适应性"
            ],
            "劣势": [
                f"范围检测精度较低 (平均误差 {hsv_data['range_err_mm'].mean():.1f}mm)",
                f"视角召回率较低 ({hsv_data['view_hit_rate'].mean():.1%})",
                "可能受颜色特征影响"
            ],
            "适用场景": [
                "实时检测系统",
                "对速度要求高的应用",
                "计算资源有限的环境",
                "对精度要求不是特别严格的场合"
            ]
        },
        "Depth+Clustering方法": {
            "优势": [
                f"范围检测精度高 (平均误差 {clustering_data['range_err_mm'].mean():.1f}mm)",
                f"视角召回率高 ({clustering_data['view_hit_rate'].mean():.1%})",
                "光照稳定性好",
                "适合高精度要求场景"
            ],
            "劣势": [
                f"检测时间较长 (平均 {clustering_data['detection_time_s'].mean():.1f}秒)",
                f"计算资源需求高 (FPS: {clustering_data['fps'].mean():.1f})",
                "不适合实时应用"
            ],
            "适用场景": [
                "高精度检测需求",
                "离线处理系统",
                "复杂光照环境",
                "对稳定性要求高的应用"
            ]
        }
    }
    
    return recommendations

def main():
    """主函数"""
    # 设置路径
    csv_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723/trial_results.csv"
    output_dir = Path("/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/unified_object_detection_20250911_114723")
    
    print("开始加载和分析数据...")
    df = load_and_analyze_data(csv_path)
    print(f"数据加载完成，共 {len(df)} 条记录")
    
    print("生成综合对比图表...")
    chart1 = create_comprehensive_comparison(df, output_dir)
    print(f"综合对比图表已保存: {chart1}")
    
    print("生成决策矩阵...")
    chart2 = create_decision_matrix(df, output_dir)
    print(f"决策矩阵已保存: {chart2}")
    
    print("生成算法选择建议...")
    recommendations = generate_recommendations(df)
    
    # 保存建议到文件
    with open(output_dir / "algorithm_recommendations.txt", 'w', encoding='utf-8') as f:
        f.write("算法选择建议报告\n")
        f.write("="*50 + "\n\n")
        
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
    
    print("算法选择建议已保存到文件")
    
    # 输出关键结论
    print("\n" + "="*60)
    print("关键结论:")
    print("="*60)
    
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    print(f"1. 速度对比: HSV+Depth ({hsv_data['detection_time_s'].mean():.1f}s) vs Depth+Clustering ({clustering_data['detection_time_s'].mean():.1f}s)")
    print(f"2. 位置精度: HSV+Depth ({hsv_data['pos_err_mm'].mean():.1f}mm) vs Depth+Clustering ({clustering_data['pos_err_mm'].mean():.1f}mm)")
    print(f"3. 范围精度: HSV+Depth ({hsv_data['range_err_mm'].mean():.1f}mm) vs Depth+Clustering ({clustering_data['range_err_mm'].mean():.1f}mm)")
    print(f"4. 召回率: HSV+Depth ({hsv_data['view_hit_rate'].mean():.1%}) vs Depth+Clustering ({clustering_data['view_hit_rate'].mean():.1%})")
    
    print("\n推荐:")
    print("• 如果需要实时检测和快速响应 → 选择 HSV+Depth")
    print("• 如果需要高精度和稳定性 → 选择 Depth+Clustering")
    print("• 如果需要在复杂光照环境下工作 → 选择 Depth+Clustering")
    print("• 如果计算资源有限 → 选择 HSV+Depth")

if __name__ == "__main__":
    main()
