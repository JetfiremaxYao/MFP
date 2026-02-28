#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import ast

def load_and_clean_data(csv_file):
    """加载并清理CSV数据"""
    print(f"加载数据文件: {csv_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"原始数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    
    # 清理view_details列（移除numpy类型标记）
    if 'view_details' in df.columns:
        print("清理view_details列...")
        df['view_details'] = df['view_details'].apply(lambda x: 
            x.replace('np.float64(', '').replace('np.int64(', '').replace(')', '') 
            if isinstance(x, str) else x)
    
    # 确保数值列的类型正确
    numeric_columns = ['pos_err_m', 'dx_m', 'dy_m', 'range_err_m', 'detection_time_s', 
                      'frame_count', 'fps', 'views_total', 'views_hit', 'early_stop_index', 
                      'view_hit_rate', 'best_angle_deg', 'best_score']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 确保布尔列的类型正确
    boolean_columns = ['success_pos_2cm', 'success_range_2cm', 'success_pos_5cm', 'success_range_5cm']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    print(f"清理后数据形状: {df.shape}")
    print(f"方法类型: {df['method'].unique()}")
    print(f"光照条件: {df['lighting'].unique()}")
    
    return df

def generate_comprehensive_visualization(df, output_dir):
    """生成综合可视化图表"""
    print("开始生成综合可视化图表...")
    
    # 设置中文字体和样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('default')
    
    # 创建主图表 - 2x3布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('统一物体检测方法评估结果', fontsize=18, fontweight='bold', y=0.98)
    
    # 设置颜色方案
    colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
    method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
    
    # 1. 位置误差箱线图
    print("绘制位置误差箱线图...")
    sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', ax=axes[0, 0], palette=colors)
    axes[0, 0].set_title('位置误差对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('位置误差 (m)', fontsize=12)
    axes[0, 0].set_xlabel('光照条件', fontsize=12)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 0].legend(title='检测方法', title_fontsize=11, fontsize=10)
    
    # 添加均值标注
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['pos_err_m'].mean()
                axes[0, 0].text(j + i*0.4 - 0.2, mean_val, f'{mean_val*1000:.1f}mm', 
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. 范围误差箱线图
    print("绘制范围误差箱线图...")
    sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', ax=axes[0, 1], palette=colors)
    axes[0, 1].set_title('范围误差对比', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('范围误差 (m)', fontsize=12)
    axes[0, 1].set_xlabel('光照条件', fontsize=12)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 1].legend(title='检测方法', title_fontsize=11, fontsize=10)
    
    # 3. 检测时间箱线图
    print("绘制检测时间箱线图...")
    sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=axes[0, 2], palette=colors)
    axes[0, 2].set_title('检测时间对比', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('检测时间 (s)', fontsize=12)
    axes[0, 2].set_xlabel('光照条件', fontsize=12)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 2].legend(title='检测方法', title_fontsize=11, fontsize=10)
    
    # 4. 成功率柱状图
    print("绘制成功率柱状图...")
    success_rates = []
    methods = []
    lightings = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                success_rate = subset['success_pos_2cm'].mean()
                success_rates.append(success_rate)
                methods.append(method_names[method])
                lightings.append(lighting)
    
    if success_rates:
        success_df = pd.DataFrame({
            'method': methods,
            'lighting': lightings,
            'success_rate': success_rates
        })
        
        sns.barplot(data=success_df, x='lighting', y='success_rate', hue='method', ax=axes[1, 0], palette=colors)
        axes[1, 0].set_title('成功率对比 (2cm阈值)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('成功率', fontsize=12)
        axes[1, 0].set_xlabel('光照条件', fontsize=12)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 0].legend(title='检测方法', title_fontsize=11, fontsize=10)
        
        # 添加数值标注
        for i, p in enumerate(axes[1, 0].patches):
            axes[1, 0].annotate(f'{p.get_height():.1%}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. 视角召回率柱状图
    print("绘制视角召回率柱状图...")
    recall_rates = []
    methods = []
    lightings = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                recall_rate = subset['view_hit_rate'].mean()
                recall_rates.append(recall_rate)
                methods.append(method_names[method])
                lightings.append(lighting)
    
    if recall_rates:
        recall_df = pd.DataFrame({
            'method': methods,
            'lighting': lightings,
            'recall_rate': recall_rates
        })
        
        sns.barplot(data=recall_df, x='lighting', y='recall_rate', hue='method', ax=axes[1, 1], palette=colors)
        axes[1, 1].set_title('视角召回率对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('召回率', fontsize=12)
        axes[1, 1].set_xlabel('光照条件', fontsize=12)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 1].legend(title='检测方法', title_fontsize=11, fontsize=10)
        
        # 添加数值标注
        for i, p in enumerate(axes[1, 1].patches):
            axes[1, 1].annotate(f'{p.get_height():.1%}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. FPS对比
    print("绘制FPS对比图...")
    sns.boxplot(data=df, x='lighting', y='fps', hue='method', ax=axes[1, 2], palette=colors)
    axes[1, 2].set_title('FPS对比', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('FPS', fontsize=12)
    axes[1, 2].set_xlabel('光照条件', fontsize=12)
    axes[1, 2].tick_params(axis='both', which='major', labelsize=10)
    axes[1, 2].legend(title='检测方法', title_fontsize=11, fontsize=10)
    
    plt.tight_layout()
    
    # 保存主图表
    chart_file = output_dir / "comprehensive_evaluation_charts.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合图表已保存: {chart_file}")
    
    # 生成详细分析图表
    generate_detailed_analysis(df, output_dir)

def generate_detailed_analysis(df, output_dir):
    """生成详细分析图表"""
    try:
        print("生成详细分析图表...")
        
        # 创建新的图表
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('详细性能分析', fontsize=18, fontweight='bold', y=0.98)
        
        colors = ['#1f77b4', '#ff7f0e']
        method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
        
        # 1. 误差分布直方图
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[0, 0].hist(method_df['pos_err_m'] * 1000, alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[0, 0].set_title('位置误差分布', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('位置误差 (mm)', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 2. 检测时间vs误差散点图
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[0, 1].scatter(method_df['detection_time_s'], method_df['pos_err_m'] * 1000, 
                                 alpha=0.7, label=method_names[method], color=colors[i], s=50)
        
        axes[0, 1].set_title('检测时间 vs 位置误差', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('检测时间 (s)', fontsize=12)
        axes[0, 1].set_ylabel('位置误差 (mm)', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        
        # 3. 成功率热力图
        success_matrix = []
        for method in ['hsv_depth', 'depth_clustering']:
            row = []
            for lighting in ['normal', 'bright', 'dim']:
                subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
                if len(subset) > 0:
                    success_rate = subset['success_pos_2cm'].mean()
                    row.append(success_rate)
                else:
                    row.append(0.0)
            success_matrix.append(row)
        
        if success_matrix:
            sns.heatmap(success_matrix, 
                       xticklabels=['Normal', 'Bright', 'Dim'],
                       yticklabels=['HSV+Depth', 'Depth+Clustering'],
                       annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 0],
                       cbar_kws={'label': '成功率'})
            axes[1, 0].set_title('成功率热力图 (2cm阈值)', fontsize=14, fontweight='bold')
            axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 4. 综合性能雷达图
        # 计算各方法的综合性能指标
        performance_data = []
        method_labels = []
        
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                # 归一化各项指标 (0-1之间，1为最好)
                accuracy = 1.0 / (1.0 + method_df['pos_err_m'].mean() * 1000 / 50)  # 50mm为基准
                speed = 1.0 / (1.0 + method_df['detection_time_s'].mean() / 30)  # 30s为基准
                success_rate = method_df['success_pos_2cm'].mean()
                recall_rate = method_df['view_hit_rate'].mean()
                
                performance_data.append([accuracy, speed, success_rate, recall_rate])
                method_labels.append(method_names[method])
        
        if performance_data:
            # 雷达图
            categories = ['精度', '速度', '成功率', '召回率']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            ax = axes[1, 1]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            for i, (data, name) in enumerate(zip(performance_data, method_labels)):
                data += data[:1]  # 闭合
                ax.plot(angles, data, 'o-', linewidth=3, label=name, alpha=0.8, color=colors[i])
                ax.fill(angles, data, alpha=0.2, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_title('综合性能雷达图', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # 添加数值标注
            for i, (data, name) in enumerate(zip(performance_data, method_labels)):
                for j, (angle, value) in enumerate(zip(angles[:-1], data)):
                    ax.text(angle, value + 0.05, f'{value:.2f}', 
                           ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存详细图表
        detailed_chart_file = output_dir / "detailed_analysis_charts.png"
        plt.savefig(detailed_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"详细分析图表已保存: {detailed_chart_file}")
        
        # 生成统计摘要
        generate_statistical_summary(df, output_dir)
        
    except Exception as e:
        print(f"生成详细图表失败: {e}")
        import traceback
        traceback.print_exc()

def generate_statistical_summary(df, output_dir):
    """生成统计摘要"""
    try:
        print("生成统计摘要...")
        
        # 创建统计摘要图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('统计摘要分析', fontsize=18, fontweight='bold', y=0.98)
        
        colors = ['#1f77b4', '#ff7f0e']
        method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
        
        # 1. 各方法性能对比表
        summary_data = []
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                summary_data.append({
                    '方法': method_names[method],
                    '位置误差(mm)': f"{method_df['pos_err_m'].mean()*1000:.2f}±{method_df['pos_err_m'].std()*1000:.2f}",
                    '范围误差(mm)': f"{method_df['range_err_m'].mean()*1000:.2f}±{method_df['range_err_m'].std()*1000:.2f}",
                    '检测时间(s)': f"{method_df['detection_time_s'].mean():.2f}±{method_df['detection_time_s'].std():.2f}",
                    '成功率(%)': f"{method_df['success_pos_2cm'].mean()*100:.1f}",
                    '召回率(%)': f"{method_df['view_hit_rate'].mean()*100:.1f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            axes[0, 0].axis('tight')
            axes[0, 0].axis('off')
            table = axes[0, 0].table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[0, 0].set_title('性能对比表', fontsize=14, fontweight='bold')
        
        # 2. 光照条件影响分析
        lighting_impact = []
        for lighting in ['normal', 'bright', 'dim']:
            lighting_df = df[df['lighting'] == lighting]
            if len(lighting_df) > 0:
                lighting_impact.append({
                    '光照条件': lighting.capitalize(),
                    '平均位置误差(mm)': lighting_df['pos_err_m'].mean() * 1000,
                    '平均检测时间(s)': lighting_df['detection_time_s'].mean(),
                    '平均成功率(%)': lighting_df['success_pos_2cm'].mean() * 100
                })
        
        if lighting_impact:
            lighting_df = pd.DataFrame(lighting_impact)
            axes[0, 1].axis('tight')
            axes[0, 1].axis('off')
            table = axes[0, 1].table(cellText=lighting_df.values, colLabels=lighting_df.columns, 
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[0, 1].set_title('光照条件影响', fontsize=14, fontweight='bold')
        
        # 3. 误差分布对比
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 0].hist(method_df['range_err_m'] * 1000, alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[1, 0].set_title('范围误差分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('范围误差 (mm)', fontsize=12)
        axes[1, 0].set_ylabel('频次', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 4. 检测时间分布
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 1].hist(method_df['detection_time_s'], alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[1, 1].set_title('检测时间分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('检测时间 (s)', fontsize=12)
        axes[1, 1].set_ylabel('频次', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # 保存统计摘要图表
        summary_chart_file = output_dir / "statistical_summary.png"
        plt.savefig(summary_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计摘要图表已保存: {summary_chart_file}")
        
    except Exception as e:
        print(f"生成统计摘要失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("实验结果可视化工具")
    print("="*50)
    
    # 查找最新的实验结果目录
    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        print("错误：找不到evaluation_results目录")
        return
    
    # 查找最新的unified_object_detection实验目录
    experiment_dirs = [d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('unified_object_detection')]
    
    if not experiment_dirs:
        print("错误：找不到unified_object_detection实验目录")
        return
    
    # 选择最新的目录
    latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"使用最新实验目录: {latest_dir}")
    
    # 查找CSV文件
    csv_file = latest_dir / "trial_results.csv"
    if not csv_file.exists():
        print(f"错误：找不到CSV文件 {csv_file}")
        return
    
    # 加载数据
    df = load_and_clean_data(csv_file)
    
    # 创建可视化输出目录
    viz_dir = latest_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 生成可视化
    generate_comprehensive_visualization(df, viz_dir)
    
    print(f"\n可视化完成！结果保存在: {viz_dir}")
    print("生成的文件:")
    for file in viz_dir.glob("*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
