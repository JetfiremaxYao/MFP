#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪算法稳定性指标可视化工具
专门用于分析和可视化CSV中存储的稳定性指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StabilityVisualizer:
    """稳定性指标可视化器"""
    
    def __init__(self, results_dir: str):
        """
        初始化可视化器
        
        Parameters:
        -----------
        results_dir : str
            结果目录路径
        """
        self.results_dir = Path(results_dir)
        self.df = None
        self.summary_df = None
        self.grouped_df = None
        
        # 加载数据
        self._load_data()
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _load_data(self):
        """加载CSV数据"""
        # 尝试加载主要结果文件
        main_csv = self.results_dir / "boundary_tracking_results.csv"
        if main_csv.exists():
            self.df = pd.read_csv(main_csv)
            print(f"✓ 已加载主要结果文件: {main_csv}")
        else:
            print(f"⚠ 未找到主要结果文件: {main_csv}")
        
        # 尝试加载摘要文件
        summary_csv = self.results_dir / "boundary_tracking_summary.csv"
        if summary_csv.exists():
            self.summary_df = pd.read_csv(summary_csv)
            print(f"✓ 已加载摘要文件: {summary_csv}")
        else:
            print(f"⚠ 未找到摘要文件: {summary_csv}")
        
        # 尝试加载分组文件
        grouped_csv = self.results_dir / "boundary_tracking_grouped_summary.csv"
        if grouped_csv.exists():
            self.grouped_df = pd.read_csv(grouped_csv)
            print(f"✓ 已加载分组文件: {grouped_csv}")
        else:
            print(f"⚠ 未找到分组文件: {grouped_csv}")
    
    def print_stability_summary(self):
        """打印稳定性指标摘要"""
        print("\n" + "="*80)
        print("稳定性指标摘要")
        print("="*80)
        
        if self.df is None:
            print("❌ 没有数据可分析")
            return
        
        # 检查是否有稳定性指标
        if 'std_residual_mm' not in self.df.columns:
            print("❌ 数据中没有稳定性指标字段")
            return
        
        # 按方法分组统计
        for method in ['canny', 'rgbd']:
            method_df = self.df[self.df['method'] == method]
            if len(method_df) == 0:
                continue
            
            print(f"\n📊 {method.upper()}方法稳定性分析:")
            
            # 基本统计
            std_residual_mean = method_df['std_residual_mm'].mean()
            std_residual_std = method_df['std_residual_mm'].std()
            std_residual_median = method_df['std_residual_mm'].median()
            std_residual_min = method_df['std_residual_mm'].min()
            std_residual_max = method_df['std_residual_mm'].max()
            
            print(f"  残差标准差:")
            print(f"    - 均值: {std_residual_mean:.2f}±{std_residual_std:.2f}mm")
            print(f"    - 中位数: {std_residual_median:.2f}mm")
            print(f"    - 范围: [{std_residual_min:.2f}, {std_residual_max:.2f}]mm")
            
            # 变异系数统计
            if 'cv_residual' in method_df.columns:
                cv_mean = method_df['cv_residual'].mean()
                cv_std = method_df['cv_residual'].std()
                cv_median = method_df['cv_residual'].median()
                
                print(f"  变异系数:")
                print(f"    - 均值: {cv_mean:.3f}±{cv_std:.3f}")
                print(f"    - 中位数: {cv_median:.3f}")
            
            # 按光照条件分析
            print(f"  按光照条件分析:")
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = method_df[method_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    lighting_std_mean = lighting_df['std_residual_mm'].mean()
                    lighting_std_std = lighting_df['std_residual_mm'].std()
                    print(f"    - {lighting}: {lighting_std_mean:.2f}±{lighting_std_std:.2f}mm")
    
    def create_stability_comparison_plots(self, save_dir: Optional[str] = None):
        """创建稳定性对比图表"""
        if self.df is None:
            print("❌ 没有数据可分析")
            return
        
        if save_dir is None:
            save_dir = self.results_dir / "stability_analysis"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📈 正在生成稳定性对比图表...")
        
        # 1. 残差标准差对比图
        self._plot_std_residual_comparison(save_path)
        
        # 2. 变异系数对比图
        self._plot_cv_residual_comparison(save_path)
        
        # 3. 稳定性分布图
        self._plot_stability_distribution(save_path)
        
        # 4. 稳定性vs准确性关系图
        self._plot_stability_vs_accuracy(save_path)
        
        # 5. 光照条件对稳定性的影响
        self._plot_lighting_impact_on_stability(save_path)
        
        # 6. 综合稳定性雷达图
        self._plot_stability_radar_chart(save_path)
        
        print(f"✅ 稳定性对比图表已保存到: {save_path}")
    
    def _plot_std_residual_comparison(self, save_path: Path):
        """绘制残差标准差对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 箱线图对比
        sns.boxplot(data=self.df, x='method', y='std_residual_mm', ax=axes[0,0])
        axes[0,0].set_title('残差标准差对比 (箱线图)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('检测方法', fontsize=12)
        axes[0,0].set_ylabel('残差标准差 (mm)', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 小提琴图对比
        sns.violinplot(data=self.df, x='method', y='std_residual_mm', ax=axes[0,1])
        axes[0,1].set_title('残差标准差分布 (小提琴图)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('检测方法', fontsize=12)
        axes[0,1].set_ylabel('残差标准差 (mm)', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 按光照条件分组对比
        sns.boxplot(data=self.df, x='method', y='std_residual_mm', hue='lighting', ax=axes[1,0])
        axes[1,0].set_title('按光照条件的残差标准差对比', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('检测方法', fontsize=12)
        axes[1,0].set_ylabel('残差标准差 (mm)', fontsize=12)
        axes[1,0].legend(title='光照条件')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 统计摘要
        methods = self.df['method'].unique()
        means = []
        stds = []
        
        for method in methods:
            method_df = self.df[self.df['method'] == method]
            means.append(method_df['std_residual_mm'].mean())
            stds.append(method_df['std_residual_mm'].std())
        
        x = np.arange(len(methods))
        bars = axes[1,1].bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1,1].set_title('残差标准差统计摘要', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('检测方法', fontsize=12)
        axes[1,1].set_ylabel('残差标准差 (mm)', fontsize=12)
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([m.upper() for m in methods])
        axes[1,1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                          f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'std_residual_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 残差标准差对比图已保存")
    
    def _plot_cv_residual_comparison(self, save_path: Path):
        """绘制变异系数对比图"""
        if 'cv_residual' not in self.df.columns:
            print("  ⚠ 数据中没有变异系数字段，跳过变异系数对比图")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 箱线图对比
        sns.boxplot(data=self.df, x='method', y='cv_residual', ax=axes[0,0])
        axes[0,0].set_title('变异系数对比 (箱线图)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('检测方法', fontsize=12)
        axes[0,0].set_ylabel('变异系数', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 小提琴图对比
        sns.violinplot(data=self.df, x='method', y='cv_residual', ax=axes[0,1])
        axes[0,1].set_title('变异系数分布 (小提琴图)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('检测方法', fontsize=12)
        axes[0,1].set_ylabel('变异系数', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 按光照条件分组对比
        sns.boxplot(data=self.df, x='method', y='cv_residual', hue='lighting', ax=axes[1,0])
        axes[1,0].set_title('按光照条件的变异系数对比', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('检测方法', fontsize=12)
        axes[1,0].set_ylabel('变异系数', fontsize=12)
        axes[1,0].legend(title='光照条件')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 统计摘要
        methods = self.df['method'].unique()
        means = []
        stds = []
        
        for method in methods:
            method_df = self.df[self.df['method'] == method]
            means.append(method_df['cv_residual'].mean())
            stds.append(method_df['cv_residual'].std())
        
        x = np.arange(len(methods))
        bars = axes[1,1].bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1,1].set_title('变异系数统计摘要', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('检测方法', fontsize=12)
        axes[1,1].set_ylabel('变异系数', fontsize=12)
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels([m.upper() for m in methods])
        axes[1,1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                          f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'cv_residual_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 变异系数对比图已保存")
    
    def _plot_stability_distribution(self, save_path: Path):
        """绘制稳定性分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 残差标准差直方图
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            axes[0,0].hist(method_df['std_residual_mm'], alpha=0.7, label=method.upper(), bins=15)
        
        axes[0,0].set_xlabel('残差标准差 (mm)', fontsize=12)
        axes[0,0].set_ylabel('频次', fontsize=12)
        axes[0,0].set_title('残差标准差分布', fontsize=14, fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 变异系数直方图
        if 'cv_residual' in self.df.columns:
            for method in self.df['method'].unique():
                method_df = self.df[self.df['method'] == method]
                axes[0,1].hist(method_df['cv_residual'], alpha=0.7, label=method.upper(), bins=15)
            
            axes[0,1].set_xlabel('变异系数', fontsize=12)
            axes[0,1].set_ylabel('频次', fontsize=12)
            axes[0,1].set_title('变异系数分布', fontsize=14, fontweight='bold')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 残差标准差密度图
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            sns.kdeplot(data=method_df['std_residual_mm'], label=method.upper(), ax=axes[1,0])
        
        axes[1,0].set_xlabel('残差标准差 (mm)', fontsize=12)
        axes[1,0].set_ylabel('密度', fontsize=12)
        axes[1,0].set_title('残差标准差密度分布', fontsize=14, fontweight='bold')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 稳定性指标相关性
        if 'cv_residual' in self.df.columns:
            sns.scatterplot(data=self.df, x='std_residual_mm', y='cv_residual', 
                          hue='method', style='lighting', ax=axes[1,1])
            axes[1,1].set_xlabel('残差标准差 (mm)', fontsize=12)
            axes[1,1].set_ylabel('变异系数', fontsize=12)
            axes[1,1].set_title('稳定性指标相关性', fontsize=14, fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'stability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 稳定性分布图已保存")
    
    def _plot_stability_vs_accuracy(self, save_path: Path):
        """绘制稳定性vs准确性关系图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 稳定性 vs Chamfer距离
        sns.scatterplot(data=self.df, x='std_residual_mm', y='chamfer_distance_mm', 
                       hue='method', style='lighting', ax=axes[0,0])
        axes[0,0].set_xlabel('残差标准差 (mm)', fontsize=12)
        axes[0,0].set_ylabel('Chamfer距离 (mm)', fontsize=12)
        axes[0,0].set_title('稳定性 vs 准确性 (Chamfer距离)', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 稳定性 vs 覆盖率
        sns.scatterplot(data=self.df, x='std_residual_mm', y='coverage_3mm', 
                       hue='method', style='lighting', ax=axes[0,1])
        axes[0,1].set_xlabel('残差标准差 (mm)', fontsize=12)
        axes[0,1].set_ylabel('3mm覆盖率', fontsize=12)
        axes[0,1].set_title('稳定性 vs 准确性 (覆盖率)', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 变异系数 vs Chamfer距离
        if 'cv_residual' in self.df.columns:
            sns.scatterplot(data=self.df, x='cv_residual', y='chamfer_distance_mm', 
                           hue='method', style='lighting', ax=axes[1,0])
            axes[1,0].set_xlabel('变异系数', fontsize=12)
            axes[1,0].set_ylabel('Chamfer距离 (mm)', fontsize=12)
            axes[1,0].set_title('相对稳定性 vs 准确性', fontsize=14, fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 综合性能散点图
        if 'cv_residual' in self.df.columns:
            # 计算综合性能指标
            self.df['stability_score'] = 1 / (1 + self.df['std_residual_mm'] / 50)  # 归一化稳定性
            self.df['accuracy_score'] = 1 / (1 + self.df['chamfer_distance_mm'] / 100)  # 归一化准确性
            
            sns.scatterplot(data=self.df, x='stability_score', y='accuracy_score', 
                           hue='method', style='lighting', ax=axes[1,1])
            axes[1,1].set_xlabel('稳定性得分 (归一化)', fontsize=12)
            axes[1,1].set_ylabel('准确性得分 (归一化)', fontsize=12)
            axes[1,1].set_title('综合性能分析', fontsize=14, fontweight='bold')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'stability_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 稳定性vs准确性关系图已保存")
    
    def _plot_lighting_impact_on_stability(self, save_path: Path):
        """绘制光照条件对稳定性的影响"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 光照条件对残差标准差的影响
        sns.boxplot(data=self.df, x='lighting', y='std_residual_mm', hue='method', ax=axes[0,0])
        axes[0,0].set_title('光照条件对残差标准差的影响', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('光照条件', fontsize=12)
        axes[0,0].set_ylabel('残差标准差 (mm)', fontsize=12)
        axes[0,0].legend(title='检测方法')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 光照条件对变异系数的影响
        if 'cv_residual' in self.df.columns:
            sns.boxplot(data=self.df, x='lighting', y='cv_residual', hue='method', ax=axes[0,1])
            axes[0,1].set_title('光照条件对变异系数的影响', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('光照条件', fontsize=12)
            axes[0,1].set_ylabel('变异系数', fontsize=12)
            axes[0,1].legend(title='检测方法')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 光照条件影响统计
        lighting_stats = []
        for method in self.df['method'].unique():
            for lighting in self.df['lighting'].unique():
                subset = self.df[(self.df['method'] == method) & (self.df['lighting'] == lighting)]
                if len(subset) > 0:
                    lighting_stats.append({
                        'method': method,
                        'lighting': lighting,
                        'mean_std': subset['std_residual_mm'].mean(),
                        'std_std': subset['std_residual_mm'].std()
                    })
        
        if lighting_stats:
            stats_df = pd.DataFrame(lighting_stats)
            sns.barplot(data=stats_df, x='lighting', y='mean_std', hue='method', ax=axes[1,0])
            axes[1,0].set_title('各光照条件下的平均残差标准差', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('光照条件', fontsize=12)
            axes[1,0].set_ylabel('平均残差标准差 (mm)', fontsize=12)
            axes[1,0].legend(title='检测方法')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 稳定性鲁棒性分析
        if self.grouped_df is not None:
            # 计算稳定性鲁棒性（标准差的标准差）
            robustness_data = []
            for method in ['canny', 'rgbd']:
                method_data = self.grouped_df[self.grouped_df['method'] == method]
                if len(method_data) > 0:
                    robustness = method_data['std_residual_mm_std'].mean()  # 稳定性指标的标准差
                    robustness_data.append({
                        'method': method,
                        'robustness': robustness
                    })
            
            if robustness_data:
                robustness_df = pd.DataFrame(robustness_data)
                sns.barplot(data=robustness_df, x='method', y='robustness', ax=axes[1,1])
                axes[1,1].set_title('稳定性鲁棒性分析', fontsize=14, fontweight='bold')
                axes[1,1].set_xlabel('检测方法', fontsize=12)
                axes[1,1].set_ylabel('稳定性鲁棒性 (标准差)', fontsize=12)
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'lighting_impact_on_stability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 光照条件对稳定性影响图已保存")
    
    def _plot_stability_radar_chart(self, save_path: Path):
        """绘制稳定性雷达图"""
        if self.summary_df is None:
            print("  ⚠ 没有摘要数据，跳过雷达图")
            return
        
        # 准备雷达图数据
        methods = self.summary_df['method'].unique()
        if len(methods) == 0:
            print("  ⚠ 没有方法数据，跳过雷达图")
            return
        
        # 定义评估维度
        categories = ['稳定性', '准确性', '覆盖率', '效率', '鲁棒性']
        
        # 计算各维度的得分
        radar_data = []
        for method in methods:
            method_df = self.summary_df[self.summary_df['method'] == method]
            if len(method_df) == 0:
                continue
            
            row = method_df.iloc[0]
            
            # 归一化各项指标 (0-1)
            stability_score = 1 - min(row.get('std_residual_mm_mean', 50) / 50, 1)  # 残差标准差越小越好
            accuracy_score = 1 - min(row.get('chamfer_distance_mm_mean', 100) / 100, 1)  # Chamfer距离越小越好
            coverage_score = min(row.get('coverage_3mm_mean', 0), 1)  # 覆盖率越大越好
            efficiency_score = 1 - min(row.get('execution_time_s_mean', 100) / 100, 1)  # 执行时间越短越好
            robustness_score = 1 - min(row.get('std_residual_mm_std', 20) / 20, 1)  # 标准差越小越稳定
            
            radar_data.append({
                'method': method,
                'scores': [stability_score, accuracy_score, coverage_score, efficiency_score, robustness_score]
            })
        
        if not radar_data:
            print("  ⚠ 没有有效的雷达图数据")
            return
        
        # 绘制雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, data in enumerate(radar_data):
            scores = data['scores'] + data['scores'][:1]  # 闭合
            ax.plot(angles, scores, 'o-', linewidth=2, label=data['method'].upper(), color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('边界追踪算法综合性能雷达图', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path / 'stability_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ 稳定性雷达图已保存")
    
    def generate_stability_report(self, save_dir: Optional[str] = None):
        """生成稳定性分析报告"""
        if save_dir is None:
            save_dir = self.results_dir / "stability_analysis"
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        report_file = save_path / "stability_analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("边界追踪算法稳定性分析报告\n")
            f.write("="*50 + "\n\n")
            f.write(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据来源: {self.results_dir}\n\n")
            
            if self.df is not None:
                f.write(f"总实验次数: {len(self.df)}\n")
                f.write(f"检测方法: {', '.join(self.df['method'].unique())}\n")
                f.write(f"光照条件: {', '.join(self.df['lighting'].unique())}\n\n")
                
                # 稳定性统计
                f.write("稳定性指标统计:\n")
                f.write("-"*30 + "\n")
                
                for method in self.df['method'].unique():
                    method_df = self.df[self.df['method'] == method]
                    f.write(f"\n{method.upper()}方法:\n")
                    
                    std_mean = method_df['std_residual_mm'].mean()
                    std_std = method_df['std_residual_mm'].std()
                    f.write(f"  残差标准差: {std_mean:.2f}±{std_std:.2f}mm\n")
                    
                    if 'cv_residual' in method_df.columns:
                        cv_mean = method_df['cv_residual'].mean()
                        cv_std = method_df['cv_residual'].std()
                        f.write(f"  变异系数: {cv_mean:.3f}±{cv_std:.3f}\n")
                
                # 稳定性评估
                f.write("\n稳定性评估:\n")
                f.write("-"*30 + "\n")
                
                for method in self.df['method'].unique():
                    method_df = self.df[self.df['method'] == method]
                    std_mean = method_df['std_residual_mm'].mean()
                    
                    if std_mean < 5:
                        stability_level = "非常稳定"
                    elif std_mean < 15:
                        stability_level = "稳定"
                    elif std_mean < 30:
                        stability_level = "中等稳定"
                    else:
                        stability_level = "不稳定"
                    
                    f.write(f"{method.upper()}方法: {stability_level} (残差标准差: {std_mean:.2f}mm)\n")
        
        print(f"✅ 稳定性分析报告已保存: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='边界追踪算法稳定性指标可视化工具')
    parser.add_argument('results_dir', help='结果目录路径')
    parser.add_argument('--save-dir', help='保存目录路径 (可选)')
    parser.add_argument('--report-only', action='store_true', help='仅生成报告，不生成图表')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not Path(args.results_dir).exists():
        print(f"❌ 错误: 目录不存在 {args.results_dir}")
        sys.exit(1)
    
    # 创建可视化器
    visualizer = StabilityVisualizer(args.results_dir)
    
    # 打印摘要
    visualizer.print_stability_summary()
    
    # 生成图表
    if not args.report_only:
        visualizer.create_stability_comparison_plots(args.save_dir)
    
    # 生成报告
    visualizer.generate_stability_report(args.save_dir)
    
    print("\n🎉 稳定性分析完成！")

if __name__ == "__main__":
    main()
