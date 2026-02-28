#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪评估CSV数据分析工具
用于分析边界追踪算法评估的CSV结果文件
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class BoundaryTrackingCSVAnalyzer:
    """边界追踪CSV数据分析器"""
    
    def __init__(self, results_dir: str):
        """
        初始化分析器
        
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
    
    def _load_data(self):
        """加载CSV数据"""
        try:
            # 加载主要结果数据
            main_csv = self.results_dir / "boundary_tracking_results.csv"
            if main_csv.exists():
                self.df = pd.read_csv(main_csv)
                print(f"已加载主要结果数据: {len(self.df)} 行")
            else:
                print(f"警告: 未找到主要结果文件 {main_csv}")
            
            # 加载摘要数据
            summary_csv = self.results_dir / "boundary_tracking_summary.csv"
            if summary_csv.exists():
                self.summary_df = pd.read_csv(summary_csv)
                print(f"已加载摘要数据: {len(self.summary_df)} 行")
            else:
                print(f"警告: 未找到摘要文件 {summary_csv}")
            
            # 加载分组数据
            grouped_csv = self.results_dir / "boundary_tracking_grouped_summary.csv"
            if grouped_csv.exists():
                self.grouped_df = pd.read_csv(grouped_csv)
                print(f"已加载分组数据: {len(self.grouped_df)} 行")
            else:
                print(f"警告: 未找到分组文件 {grouped_csv}")
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
    
    def print_basic_statistics(self):
        """打印基本统计信息"""
        if self.df is None:
            print("没有数据可分析")
            return
        
        print("\n" + "="*60)
        print("基本统计信息")
        print("="*60)
        
        print(f"总实验次数: {len(self.df)}")
        print(f"实验方法: {self.df['method'].unique()}")
        print(f"光照条件: {self.df['lighting'].unique()}")
        print(f"实验状态: {self.df['status'].value_counts().to_dict()}")
        
        # 按方法统计
        print("\n按方法统计:")
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            print(f"\n{method.upper()}方法:")
            print(f"  实验次数: {len(method_df)}")
            print(f"  状态分布: {method_df['status'].value_counts().to_dict()}")
            print(f"  平均点数: {method_df['point_count'].mean():.0f}")
            print(f"  平均执行时间: {method_df['execution_time_s'].mean():.2f}s")
    
    def print_accuracy_comparison(self):
        """打印准确性对比"""
        if self.df is None:
            print("没有数据可分析")
            return
        
        print("\n" + "="*60)
        print("准确性对比")
        print("="*60)
        
        # 只分析成功和部分成功的实验
        successful_df = self.df[self.df['status'].isin(['Success', 'Partial'])]
        
        if len(successful_df) == 0:
            print("没有成功的实验可分析")
            return
        
        for method in successful_df['method'].unique():
            method_df = successful_df[successful_df['method'] == method]
            print(f"\n{method.upper()}方法 (成功实验):")
            
            # Chamfer距离
            chamfer_mean = method_df['chamfer_distance_mm'].mean()
            chamfer_std = method_df['chamfer_distance_mm'].std()
            print(f"  Chamfer距离: {chamfer_mean:.2f}±{chamfer_std:.2f}mm")
            
            # Hausdorff距离
            hausdorff_mean = method_df['hausdorff_distance_mm'].mean()
            hausdorff_std = method_df['hausdorff_distance_mm'].std()
            print(f"  Hausdorff距离: {hausdorff_mean:.2f}±{hausdorff_std:.2f}mm")
            
            # 覆盖率
            coverage_mean = method_df['coverage_3mm'].mean()
            coverage_std = method_df['coverage_3mm'].std()
            print(f"  3mm覆盖率: {coverage_mean:.3f}±{coverage_std:.3f}")
            
            # 稳定性
            std_residual_mean = method_df['std_residual_mm'].mean()
            std_residual_std = method_df['std_residual_mm'].std()
            print(f"  残差标准差: {std_residual_mean:.2f}±{std_residual_std:.2f}mm")
    
    def print_lighting_analysis(self):
        """打印光照条件分析"""
        if self.grouped_df is None:
            print("没有分组数据可分析")
            return
        
        print("\n" + "="*60)
        print("光照条件分析")
        print("="*60)
        
        for lighting in ['normal', 'bright', 'dim']:
            lighting_df = self.grouped_df[self.grouped_df['lighting'] == lighting]
            if len(lighting_df) == 0:
                continue
            
            print(f"\n{lighting.upper()}光照条件:")
            
            for method in ['canny', 'rgbd']:
                method_df = lighting_df[lighting_df['method'] == method]
                if len(method_df) == 0:
                    continue
                
                row = method_df.iloc[0]
                print(f"  {method.upper()}:")
                print(f"    成功率: {row['success_rate']:.1%}")
                print(f"    总体成功率: {row['overall_success_rate']:.1%}")
                
                if not pd.isna(row['chamfer_distance_mm_mean']):
                    print(f"    Chamfer距离: {row['chamfer_distance_mm_mean']:.2f}±{row['chamfer_distance_mm_std']:.2f}mm")
                
                if not pd.isna(row['coverage_3mm_mean']):
                    print(f"    3mm覆盖率: {row['coverage_3mm_mean']:.3f}±{row['coverage_3mm_std']:.3f}")
    
    def create_comparison_plots(self, save_dir: Optional[str] = None):
        """创建对比图表"""
        if self.df is None:
            print("没有数据可分析")
            return
        
        if save_dir is None:
            save_dir = self.results_dir
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 成功率对比图
        self._plot_success_rate_comparison(save_path)
        
        # 2. 准确性指标对比图
        self._plot_accuracy_comparison(save_path)
        
        # 3. 性能指标对比图
        self._plot_performance_comparison(save_path)
        
        # 4. 光照条件影响图
        self._plot_lighting_impact(save_path)
        
        print(f"\n对比图表已保存到: {save_path}")
    
    def _plot_success_rate_comparison(self, save_path: Path):
        """绘制成功率对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 总体成功率
        if self.summary_df is not None:
            methods = self.summary_df['method'].unique()
            success_rates = []
            partial_rates = []
            
            for method in methods:
                method_df = self.summary_df[self.summary_df['method'] == method]
                if len(method_df) > 0:
                    success_rates.append(method_df['success_rate'].iloc[0])
                    partial_rates.append(method_df['partial_success_rate'].iloc[0])
                else:
                    success_rates.append(0)
                    partial_rates.append(0)
            
            x = np.arange(len(methods))
            width = 0.35
            
            ax1.bar(x - width/2, success_rates, width, label='Success', alpha=0.8)
            ax1.bar(x + width/2, partial_rates, width, label='Partial', alpha=0.6)
            
            ax1.set_xlabel('Method')
            ax1.set_ylabel('Success Rate')
            ax1.set_title('Overall Success Rate Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels([m.upper() for m in methods])
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 按光照条件的成功率
        if self.grouped_df is not None:
            lighting_data = {}
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = self.grouped_df[self.grouped_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    lighting_data[lighting] = {}
                    for method in ['canny', 'rgbd']:
                        method_df = lighting_df[lighting_df['method'] == method]
                        if len(method_df) > 0:
                            lighting_data[lighting][method] = method_df['success_rate'].iloc[0]
                        else:
                            lighting_data[lighting][method] = 0
            
            if lighting_data:
                x = np.arange(len(lighting_data))
                width = 0.35
                
                canny_rates = [lighting_data[light]['canny'] for light in lighting_data.keys()]
                rgbd_rates = [lighting_data[light]['rgbd'] for light in lighting_data.keys()]
                
                ax2.bar(x - width/2, canny_rates, width, label='Canny', alpha=0.8)
                ax2.bar(x + width/2, rgbd_rates, width, label='RGB-D', alpha=0.8)
                
                ax2.set_xlabel('Lighting Condition')
                ax2.set_ylabel('Success Rate')
                ax2.set_title('Success Rate by Lighting Condition')
                ax2.set_xticks(x)
                ax2.set_xticklabels([light.upper() for light in lighting_data.keys()])
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / "success_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_comparison(self, save_path: Path):
        """绘制准确性指标对比图"""
        successful_df = self.df[self.df['status'].isin(['Success', 'Partial'])]
        
        if len(successful_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Chamfer距离对比
        self._plot_boxplot(axes[0], successful_df, 'chamfer_distance_mm', 'Chamfer Distance (mm)', 'Method')
        
        # Hausdorff距离对比
        self._plot_boxplot(axes[1], successful_df, 'hausdorff_distance_mm', 'Hausdorff Distance (mm)', 'Method')
        
        # 覆盖率对比
        self._plot_boxplot(axes[2], successful_df, 'coverage_3mm', 'Coverage @3mm', 'Method')
        
        # 残差标准差对比
        self._plot_boxplot(axes[3], successful_df, 'std_residual_mm', 'Std Residual (mm)', 'Method')
        
        plt.tight_layout()
        plt.savefig(save_path / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, save_path: Path):
        """绘制性能指标对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 执行时间对比
        self._plot_boxplot(ax1, self.df, 'execution_time_s', 'Execution Time (s)', 'Method')
        
        # 点云数量对比
        self._plot_boxplot(ax2, self.df, 'point_count', 'Point Count', 'Method')
        
        plt.tight_layout()
        plt.savefig(save_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lighting_impact(self, save_path: Path):
        """绘制光照条件影响图"""
        if self.grouped_df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        metrics = ['chamfer_distance_mm_mean', 'coverage_3mm_mean', 'execution_time_s_mean', 'point_count_mean']
        titles = ['Chamfer Distance (mm)', 'Coverage @3mm', 'Execution Time (s)', 'Point Count']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # 提取数据
            data = []
            labels = []
            
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = self.grouped_df[self.grouped_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    for method in ['canny', 'rgbd']:
                        method_df = lighting_df[lighting_df['method'] == method]
                        if len(method_df) > 0 and not pd.isna(method_df[metric].iloc[0]):
                            data.append(method_df[metric].iloc[0])
                            labels.append(f"{method.upper()}\n({lighting})")
            
            if data:
                bars = ax.bar(range(len(data)), data, alpha=0.8)
                ax.set_title(title)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(labels, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for bar, value in zip(bars, data):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(data),
                           f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path / "lighting_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boxplot(self, ax, df, column, title, x_label):
        """绘制箱线图"""
        methods = df['method'].unique()
        data = [df[df['method'] == method][column].dropna() for method in methods]
        
        if any(len(d) > 0 for d in data):
            bp = ax.boxplot(data, labels=[m.upper() for m in methods], patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.grid(True, alpha=0.3)
    
    def export_statistical_summary(self, save_path: Optional[str] = None):
        """导出统计摘要"""
        if save_path is None:
            save_path = self.results_dir / "statistical_summary.txt"
        else:
            save_path = Path(save_path)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("边界追踪算法评估统计摘要\n")
            f.write("="*50 + "\n\n")
            
            if self.df is not None:
                f.write(f"总实验次数: {len(self.df)}\n")
                f.write(f"实验方法: {list(self.df['method'].unique())}\n")
                f.write(f"光照条件: {list(self.df['lighting'].unique())}\n\n")
                
                # 成功率统计
                f.write("成功率统计:\n")
                status_counts = self.df['status'].value_counts()
                for status, count in status_counts.items():
                    f.write(f"  {status}: {count} ({count/len(self.df)*100:.1f}%)\n")
                f.write("\n")
                
                # 按方法统计
                f.write("按方法统计:\n")
                for method in self.df['method'].unique():
                    method_df = self.df[self.df['method'] == method]
                    f.write(f"\n{method.upper()}方法:\n")
                    f.write(f"  实验次数: {len(method_df)}\n")
                    f.write(f"  成功率: {len(method_df[method_df['status']=='Success'])/len(method_df)*100:.1f}%\n")
                    f.write(f"  部分成功率: {len(method_df[method_df['status']=='Partial'])/len(method_df)*100:.1f}%\n")
                    f.write(f"  失败率: {len(method_df[method_df['status']=='Fail'])/len(method_df)*100:.1f}%\n")
                    
                    # 准确性指标（仅成功和部分成功的实验）
                    successful_df = method_df[method_df['status'].isin(['Success', 'Partial'])]
                    if len(successful_df) > 0:
                        f.write(f"  平均Chamfer距离: {successful_df['chamfer_distance_mm'].mean():.2f}mm\n")
                        f.write(f"  平均覆盖率: {successful_df['coverage_3mm'].mean():.3f}\n")
                        f.write(f"  平均执行时间: {successful_df['execution_time_s'].mean():.2f}s\n")
            
            if self.summary_df is not None:
                f.write("\n\n摘要统计:\n")
                f.write(self.summary_df.to_string())
            
            if self.grouped_df is not None:
                f.write("\n\n分组统计:\n")
                f.write(self.grouped_df.to_string())
        
        print(f"统计摘要已保存到: {save_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='边界追踪CSV数据分析工具')
    parser.add_argument('results_dir', help='结果目录路径')
    parser.add_argument('--plots', action='store_true', help='生成对比图表')
    parser.add_argument('--summary', action='store_true', help='导出统计摘要')
    parser.add_argument('--all', action='store_true', help='执行所有分析')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = BoundaryTrackingCSVAnalyzer(args.results_dir)
    
    # 执行分析
    if args.all or not (args.plots or args.summary):
        print("执行完整分析...")
        analyzer.print_basic_statistics()
        analyzer.print_accuracy_comparison()
        analyzer.print_lighting_analysis()
        analyzer.create_comparison_plots()
        analyzer.export_statistical_summary()
    else:
        if args.plots:
            print("生成对比图表...")
            analyzer.create_comparison_plots()
        
        if args.summary:
            print("导出统计摘要...")
            analyzer.export_statistical_summary()
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
