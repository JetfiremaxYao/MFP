#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边界追踪实验结果可视化分析
用于分析几何精度评估指标和执行时间的对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BoundaryTrackingVisualizer:
    """边界追踪实验结果可视化分析器"""
    
    def __init__(self, csv_file_path):
        """
        初始化可视化器
        
        Parameters:
        -----------
        csv_file_path : str
            CSV结果文件路径
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.output_dir = Path(csv_file_path).parent / "visualization_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        self.load_data()
        
        # 设置颜色主题
        self.colors = {
            'canny': '#1f77b4',  # 蓝色
            'rgbd': '#ff7f0e',   # 橙色
            'normal': '#2ca02c', # 绿色
            'bright': '#d62728', # 红色
            'dim': '#9467bd'     # 紫色
        }
    
    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"✅ 成功加载数据: {len(self.df)} 条记录")
            print(f"📊 数据列: {list(self.df.columns)}")
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return
    
    def create_geometric_accuracy_comparison(self):
        """创建几何精度对比图"""
        print("📈 创建几何精度对比图...")
        
        # 关键几何精度指标
        metrics = ['chamfer_distance_mm', 'hausdorff_distance_mm', 'coverage_3mm', 'execution_time_s']
        metric_names = ['Chamfer距离 (mm)', 'Hausdorff距离 (mm)', '3mm覆盖率', '执行时间 (s)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # 创建分组数据
            data_to_plot = []
            labels = []
            
            for obj_name in self.df['object_name'].unique():
                for method in self.df['method'].unique():
                    for lighting in self.df['lighting'].unique():
                        subset = self.df[
                            (self.df['object_name'] == obj_name) & 
                            (self.df['method'] == method) & 
                            (self.df['lighting'] == lighting)
                        ]
                        
                        if len(subset) > 0:
                            data_to_plot.append(subset[metric].values)
                            labels.append(f'{obj_name}\n{method.upper()}\n({lighting})')
            
            # 创建箱线图
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # 设置颜色
                for patch, label in zip(bp['boxes'], labels):
                    if 'CANNY' in label:
                        patch.set_facecolor(self.colors['canny'])
                    else:
                        patch.set_facecolor(self.colors['rgbd'])
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{metric_name} 对比', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_file = self.output_dir / "geometric_accuracy_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 几何精度对比图已保存: {output_file}")
    
    def create_method_performance_radar(self):
        """创建方法性能雷达图"""
        print("📊 创建方法性能雷达图...")
        
        # 计算平均指标
        radar_data = {}
        
        for obj_name in self.df['object_name'].unique():
            for method in self.df['method'].unique():
                subset = self.df[
                    (self.df['object_name'] == obj_name) & 
                    (self.df['method'] == method)
                ]
                
                if len(subset) > 0:
                    key = f"{obj_name}_{method}"
                    radar_data[key] = {
                        'chamfer_distance': subset['chamfer_distance_mm'].mean(),
                        'hausdorff_distance': subset['hausdorff_distance_mm'].mean(),
                        'coverage_3mm': subset['coverage_3mm'].mean(),
                        'execution_time': subset['execution_time_s'].mean(),
                        'point_count': subset['point_count'].mean()
                    }
        
        # 创建雷达图
        categories = ['Chamfer距离\n(mm)', 'Hausdorff距离\n(mm)', '3mm覆盖率\n(%)', 
                     '执行时间\n(s)', '点云数量\n(个)']
        
        # 归一化数据
        normalized_data = {}
        for key, data in radar_data.items():
            normalized_data[key] = {
                'chamfer_distance': 1.0 / (1.0 + data['chamfer_distance'] / 50),  # 归一化距离
                'hausdorff_distance': 1.0 / (1.0 + data['hausdorff_distance'] / 100),
                'coverage_3mm': data['coverage_3mm'],  # 已经是0-1
                'execution_time': 1.0 / (1.0 + data['execution_time'] / 15),  # 归一化时间
                'point_count': min(1.0, data['point_count'] / 15000)  # 归一化点数
            }
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
        
        for idx, obj_name in enumerate(self.df['object_name'].unique()):
            ax = axes[idx]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            for method in self.df['method'].unique():
                key = f"{obj_name}_{method}"
                if key in normalized_data:
                    # 映射类别名称到数据键
                    category_mapping = {
                        'Chamfer距离\n(mm)': 'chamfer_distance',
                        'Hausdorff距离\n(mm)': 'hausdorff_distance', 
                        '3mm覆盖率\n(%)': 'coverage_3mm',
                        '执行时间\n(s)': 'execution_time',
                        '点云数量\n(个)': 'point_count'
                    }
                    
                    values = [normalized_data[key][category_mapping[cat]] for cat in categories]
                    values += values[:1]  # 闭合
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=method.upper(), 
                           color=self.colors[method], alpha=0.8)
                    ax.fill(angles, values, alpha=0.1, color=self.colors[method])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            # 将物体名称转换为中文描述
            obj_display = "Regular Object Performance" if obj_name == "Cube" else "Irregular Object Performance"
            ax.set_title(f'{obj_display}', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
        
        plt.tight_layout()
        output_file = self.output_dir / "method_performance_radar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 方法性能雷达图已保存: {output_file}")
    
    def create_lighting_condition_analysis(self):
        """创建光照条件分析图"""
        print("💡 创建光照条件分析图...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['chamfer_distance_mm', 'hausdorff_distance_mm', 'coverage_3mm', 'execution_time_s']
        metric_names = ['Chamfer距离 (mm)', 'Hausdorff距离 (mm)', '3mm覆盖率', '执行时间 (s)']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            # 创建分组数据
            data_to_plot = []
            labels = []
            
            for obj_name in self.df['object_name'].unique():
                for method in self.df['method'].unique():
                    for lighting in self.df['lighting'].unique():
                        subset = self.df[
                            (self.df['object_name'] == obj_name) & 
                            (self.df['method'] == method) & 
                            (self.df['lighting'] == lighting)
                        ]
                        
                        if len(subset) > 0:
                            data_to_plot.append(subset[metric].values)
                            labels.append(f'{obj_name}\n{method.upper()}\n({lighting})')
            
            # 创建分组柱状图
            if data_to_plot:
                x = np.arange(len(labels))
                width = 0.25
                
                # 按光照条件分组
                lighting_conditions = self.df['lighting'].unique()
                
                for j, lighting in enumerate(lighting_conditions):
                    lighting_data = []
                    for label in labels:
                        if lighting in label:
                            # 找到对应的数据
                            for k, data in enumerate(data_to_plot):
                                if labels[k] == label:
                                    lighting_data.append(np.mean(data))
                                    break
                        else:
                            lighting_data.append(0)
                    
                    ax.bar(x + j * width, lighting_data, width, 
                          label=lighting.upper(), color=self.colors[lighting], alpha=0.8)
                
                ax.set_title(f'{metric_name} - 光照条件影响', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=12)
                ax.set_xticks(x + width)
                ax.set_xticklabels([label.replace('\n', ' ') for label in labels], rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "lighting_condition_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 光照条件分析图已保存: {output_file}")
    
    def create_success_rate_analysis(self):
        """创建成功率分析图"""
        print("📊 创建成功率分析图...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, obj_name in enumerate(self.df['object_name'].unique()):
            ax = axes[idx]
            
            # 计算成功率
            success_data = []
            method_labels = []
            lighting_labels = []
            
            for method in self.df['method'].unique():
                for lighting in self.df['lighting'].unique():
                    subset = self.df[
                        (self.df['object_name'] == obj_name) & 
                        (self.df['method'] == method) & 
                        (self.df['lighting'] == lighting)
                    ]
                    
                    if len(subset) > 0:
                        success_rate = (subset['status'] == 'Success').mean()
                        success_data.append(success_rate)
                        method_labels.append(method.upper())
                        lighting_labels.append(lighting.upper())
            
            # 创建柱状图
            x = np.arange(len(success_data))
            colors = [self.colors[method.lower()] for method in method_labels]
            
            bars = ax.bar(x, success_data, color=colors, alpha=0.8)
            ax.set_title(f'{obj_name} 成功率分析', fontsize=14, fontweight='bold')
            ax.set_ylabel('成功率', fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{method}\n({lighting})' for method, lighting in zip(method_labels, lighting_labels)])
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / "success_rate_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 成功率分析图已保存: {output_file}")
    
    def create_correlation_heatmap(self):
        """创建相关性热力图"""
        print("🔥 创建相关性热力图...")
        
        # 选择数值列
        numeric_cols = ['chamfer_distance_mm', 'hausdorff_distance_mm', 'coverage_3mm', 
                       'execution_time_s', 'point_count', 'mean_residual_mm', 'median_residual_mm']
        
        # 计算相关性矩阵
        corr_matrix = self.df[numeric_cols].corr()
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('几何精度指标相关性分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 相关性热力图已保存: {output_file}")
    
    def create_summary_statistics_table(self):
        """创建汇总统计表"""
        print("📋 创建汇总统计表...")
        
        # 按物体和方法分组计算统计
        summary_stats = []
        
        for obj_name in self.df['object_name'].unique():
            for method in self.df['method'].unique():
                subset = self.df[
                    (self.df['object_name'] == obj_name) & 
                    (self.df['method'] == method)
                ]
                
                if len(subset) > 0:
                    stats = {
                        '物体': obj_name,
                        '方法': method.upper(),
                        '测试次数': len(subset),
                        '成功率': f"{(subset['status'] == 'Success').mean():.1%}",
                        '平均Chamfer距离(mm)': f"{subset['chamfer_distance_mm'].mean():.2f}",
                        '平均Hausdorff距离(mm)': f"{subset['hausdorff_distance_mm'].mean():.2f}",
                        '平均3mm覆盖率': f"{subset['coverage_3mm'].mean():.3f}",
                        '平均执行时间(s)': f"{subset['execution_time_s'].mean():.2f}",
                        '平均点云数量': f"{subset['point_count'].mean():.0f}"
                    }
                    summary_stats.append(stats)
        
        # 创建DataFrame并保存
        summary_df = pd.DataFrame(summary_stats)
        summary_file = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"✅ 汇总统计表已保存: {summary_file}")
        
        # 打印到控制台
        print("\n📊 汇总统计:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print("=" * 80)
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("🎨 开始生成可视化图表...")
        print(f"📁 输出目录: {self.output_dir}")
        
        try:
            # 生成各种图表
            self.create_geometric_accuracy_comparison()
            self.create_method_performance_radar()
            self.create_lighting_condition_analysis()
            self.create_success_rate_analysis()
            self.create_correlation_heatmap()
            self.create_summary_statistics_table()
            
            print("\n🎉 所有可视化图表生成完成！")
            print(f"📂 请查看输出目录: {self.output_dir}")
            
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("边界追踪实验结果可视化分析")
    print("=" * 50)
    
    # 查找最新的CSV文件
    results_dir = Path("../../evaluation_results")
    csv_files = list(results_dir.glob("*/boundary_tracking_results_detailed.csv"))
    
    if not csv_files:
        print("❌ 未找到CSV结果文件")
        return
    
    # 选择最新的文件
    latest_csv = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 使用CSV文件: {latest_csv}")
    
    # 创建可视化器
    visualizer = BoundaryTrackingVisualizer(str(latest_csv))
    
    # 生成所有可视化图表
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()
