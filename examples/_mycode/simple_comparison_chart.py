#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁的关键数据对比图生成器
只显示最重要的指标：Chamfer距离、Hausdorff距离、执行时间、成功率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def create_simple_comparison_chart(csv_file: str, save_path: str = None):
    """
    创建简洁的关键数据对比图
    
    Parameters:
    -----------
    csv_file : str
        CSV文件路径
    save_path : str
        保存路径
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('边界追踪算法关键指标对比', fontsize=16, fontweight='bold')
    
    # 颜色设置
    colors = {'canny': '#1f77b4', 'rgbd': '#ff7f0e'}
    
    # 1. Chamfer距离对比 (越小越好)
    ax1 = axes[0, 0]
    plot_metric_comparison(ax1, df, 'chamfer_distance_mm', 'Chamfer距离 (mm)', 
                          '越小越好', colors, ylim=(0, 10))
    
    # 2. Hausdorff距离对比 (越小越好)
    ax2 = axes[0, 1]
    plot_metric_comparison(ax2, df, 'hausdorff_distance_mm', 'Hausdorff距离 (mm)', 
                          '越小越好', colors, ylim=(0, 25))
    
    # 3. 执行时间对比 (越小越好)
    ax3 = axes[1, 0]
    plot_metric_comparison(ax3, df, 'execution_time_s', '执行时间 (秒)', 
                          '越小越好', colors, ylim=(0, 300))
    
    # 4. 成功率对比 (越大越好)
    ax4 = axes[1, 1]
    plot_success_rate_comparison(ax4, df, colors)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path is None:
        save_path = Path(csv_file).parent / "关键指标对比.png"
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"关键指标对比图已保存: {save_path}")
    
    # 打印关键数据摘要
    print_key_summary(df)

def plot_metric_comparison(ax, df, metric, title, note, colors, ylim=None):
    """绘制指标对比图"""
    methods = df['method'].unique()
    
    # 计算平均值和标准差
    means = []
    stds = []
    labels = []
    
    for method in methods:
        method_df = df[df['method'] == method]
        if len(method_df) > 0:
            mean_val = method_df[metric].mean()
            std_val = method_df[metric].std()
            means.append(mean_val)
            stds.append(std_val)
            labels.append(method.upper())
    
    # 绘制柱状图
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                  color=[colors[method.lower()] for method in methods], 
                  alpha=0.8, width=0.6)
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02*max(means),
               f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(f'{title}\n({note})', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    
    if ylim:
        ax.set_ylim(ylim)
    
    # 添加优胜标记
    if '越小越好' in note:
        best_idx = np.argmin(means)
        ax.annotate('⭐ 更优', xy=(best_idx, means[best_idx]), 
                   xytext=(best_idx, means[best_idx] + 0.1*max(means)),
                   ha='center', va='bottom', fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red'))

def plot_success_rate_comparison(ax, df, colors):
    """绘制成功率对比图"""
    methods = df['method'].unique()
    
    success_rates = []
    labels = []
    
    for method in methods:
        method_df = df[df['method'] == method]
        if len(method_df) > 0:
            success_rate = len(method_df[method_df['status'] == 'Success']) / len(method_df)
            success_rates.append(success_rate)
            labels.append(method.upper())
    
    # 绘制柱状图
    x = np.arange(len(methods))
    bars = ax.bar(x, success_rates, 
                  color=[colors[method.lower()] for method in methods], 
                  alpha=0.8, width=0.6)
    
    # 添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('成功率\n(越大越好)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 添加优胜标记
    best_idx = np.argmax(success_rates)
    ax.annotate('⭐ 更优', xy=(best_idx, success_rates[best_idx]), 
               xytext=(best_idx, success_rates[best_idx] + 0.05),
               ha='center', va='bottom', fontweight='bold', color='red',
               arrowprops=dict(arrowstyle='->', color='red'))

def print_key_summary(df):
    """打印关键数据摘要"""
    print("\n" + "="*60)
    print("关键指标对比摘要")
    print("="*60)
    
    methods = df['method'].unique()
    
    for method in methods:
        method_df = df[df['method'] == method]
        if len(method_df) == 0:
            continue
        
        print(f"\n{method.upper()}方法:")
        
        # Chamfer距离
        chamfer_mean = method_df['chamfer_distance_mm'].mean()
        chamfer_std = method_df['chamfer_distance_mm'].std()
        print(f"  Chamfer距离: {chamfer_mean:.2f}±{chamfer_std:.2f}mm")
        
        # Hausdorff距离
        hausdorff_mean = method_df['hausdorff_distance_mm'].mean()
        hausdorff_std = method_df['hausdorff_distance_mm'].std()
        print(f"  Hausdorff距离: {hausdorff_mean:.2f}±{hausdorff_std:.2f}mm")
        
        # 执行时间
        time_mean = method_df['execution_time_s'].mean()
        time_std = method_df['execution_time_s'].std()
        print(f"  执行时间: {time_mean:.2f}±{time_std:.2f}秒")
        
        # 成功率
        success_rate = len(method_df[method_df['status'] == 'Success']) / len(method_df)
        print(f"  成功率: {success_rate:.1%}")
    
    # 综合推荐
    print(f"\n综合推荐:")
    
    # 找出各项指标的最优方法
    best_chamfer = min(methods, key=lambda m: df[df['method'] == m]['chamfer_distance_mm'].mean())
    best_hausdorff = min(methods, key=lambda m: df[df['method'] == m]['hausdorff_distance_mm'].mean())
    best_time = min(methods, key=lambda m: df[df['method'] == m]['execution_time_s'].mean())
    
    print(f"  准确性最优: {best_chamfer.upper()} (Chamfer距离最小)")
    print(f"  稳定性最优: {best_hausdorff.upper()} (Hausdorff距离最小)")
    print(f"  速度最优: {best_time.upper()} (执行时间最短)")
    
    # 总体推荐
    if best_chamfer == best_hausdorff:
        print(f"  🏆 总体推荐: {best_chamfer.upper()} (准确性和稳定性都最优)")
    else:
        print(f"  🏆 总体推荐: 根据应用需求选择")
        print(f"     - 精度优先: {best_chamfer.upper()}")
        print(f"     - 速度优先: {best_time.upper()}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成简洁的关键指标对比图')
    parser.add_argument('csv_file', help='CSV文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径')
    
    args = parser.parse_args()
    
    create_simple_comparison_chart(args.csv_file, args.output)

if __name__ == "__main__":
    main()
