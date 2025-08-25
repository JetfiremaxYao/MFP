# 边界追踪算法结果分析模块
# 用于分析评估结果并生成对比图表
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BoundaryTrackingAnalyzer:
    """边界追踪结果分析器"""
    
    def __init__(self, results_file: str):
        """
        初始化分析器
        
        Parameters:
        -----------
        results_file : str
            实验结果文件路径
        """
        self.results_file = Path(results_file)
        self.results_dir = self.results_file.parent
        
        # 加载结果
        with open(self.results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        # 创建分析结果目录
        self.analysis_dir = self.results_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # 设置绘图样式
        self._setup_plotting_style()
        
        print(f"结果分析器已初始化")
        print(f"结果文件: {self.results_file}")
        print(f"分析目录: {self.analysis_dir}")
    
    def _setup_plotting_style(self):
        """设置绘图样式"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def _extract_metrics_data(self) -> pd.DataFrame:
        """提取指标数据为DataFrame格式"""
        data = []
        
        for method in ['canny', 'rgbd']:
            for result in self.results[method]:
                if 'accuracy_metrics' in result and result['status'] != 'Fail':
                    row = {
                        'method': method.upper(),
                        'run_id': result['run_id'] + 1,
                        'status': result['status'],
                        'point_count': result['point_count'],
                        'execution_time': result['execution_time'],
                        'chamfer_distance': result['accuracy_metrics']['chamfer_distance'],
                        'hausdorff_distance': result['accuracy_metrics']['hausdorff_distance'],
                        'mean_residual': result['accuracy_metrics']['mean_residual'],
                        'median_residual': result['accuracy_metrics']['median_residual'],
                        'coverage_3mm': result['accuracy_metrics']['coverage_3mm'],
                        'coverage_adaptive': result['accuracy_metrics']['coverage_adaptive'],
                        'std_residual': result['stability_metrics']['std_residual'],
                        'cv_residual': result['stability_metrics']['cv_residual']
                    }
                    data.append(row)
        
        return pd.DataFrame(data)
    
    def _extract_status_data(self) -> pd.DataFrame:
        """提取状态数据"""
        status_data = []
        
        for method in ['canny', 'rgbd']:
            success_count = sum(1 for r in self.results[method] if r['status'] == 'Success')
            partial_count = sum(1 for r in self.results[method] if r['status'] == 'Partial')
            fail_count = sum(1 for r in self.results[method] if r['status'] == 'Fail')
            total_count = len(self.results[method])
            
            status_data.extend([
                {'method': method.upper(), 'status': 'Success', 'count': success_count, 'percentage': success_count/total_count*100},
                {'method': method.upper(), 'status': 'Partial', 'count': partial_count, 'percentage': partial_count/total_count*100},
                {'method': method.upper(), 'status': 'Fail', 'count': fail_count, 'percentage': fail_count/total_count*100}
            ])
        
        return pd.DataFrame(status_data)
    
    def generate_comprehensive_analysis(self):
        """生成综合分析报告"""
        print("开始生成综合分析报告...")
        
        # 1. 成功率对比
        self._plot_success_rate_comparison()
        
        # 2. 准确性指标对比
        self._plot_accuracy_comparison()
        
        # 3. 稳定性指标对比
        self._plot_stability_comparison()
        
        # 4. 效率指标对比
        self._plot_efficiency_comparison()
        
        # 5. 综合性能雷达图
        self._plot_radar_chart()
        
        # 6. 统计显著性检验
        self._perform_statistical_tests()
        
        # 7. 生成详细分析报告
        self._generate_detailed_report()
        
        print("综合分析报告生成完成！")
    
    def _plot_success_rate_comparison(self):
        """绘制成功率对比图"""
        status_df = self._extract_status_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 柱状图
        methods = status_df['method'].unique()
        statuses = ['Success', 'Partial', 'Fail']
        colors = ['#2E8B57', '#FFD700', '#DC143C']
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, status in enumerate(statuses):
            data = status_df[status_df['status'] == status]
            values = [data[data['method'] == method]['count'].iloc[0] if len(data[data['method'] == method]) > 0 else 0 
                     for method in methods]
            ax1.bar(x + i*width, values, width, label=status, color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('检测方法')
        ax1.set_ylabel('实验次数')
        ax1.set_title('成功率对比 (次数)')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 饼图
        for i, method in enumerate(methods):
            method_data = status_df[status_df['method'] == method]
            counts = method_data['count'].values
            labels = method_data['status'].values
            
            ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
                   colors=colors[:len(counts)], alpha=0.8)
            ax2.set_title(f'{method}方法成功率分布')
            break  # 只显示第一个方法的饼图作为示例
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("成功率对比图已保存")
    
    def _plot_accuracy_comparison(self):
        """绘制准确性指标对比图"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效的准确性数据")
            return
        
        # 选择关键准确性指标
        accuracy_metrics = ['chamfer_distance', 'hausdorff_distance', 'coverage_3mm']
        metric_names = ['Chamfer距离 (mm)', 'Hausdorff距离 (mm)', '3mm覆盖率']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 1. 箱线图对比
        for i, (metric, name) in enumerate(zip(accuracy_metrics, metric_names)):
            if i >= 3:
                break
                
            sns.boxplot(data=metrics_df, x='method', y=metric, ax=axes[i])
            axes[i].set_title(f'{name}对比')
            axes[i].set_xlabel('检测方法')
            axes[i].set_ylabel(name)
            axes[i].grid(True, alpha=0.3)
        
        # 2. 散点图
        ax = axes[3]
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            ax.scatter(method_data['chamfer_distance'], method_data['coverage_3mm'], 
                      label=method, alpha=0.7, s=100)
        
        ax.set_xlabel('Chamfer距离 (mm)')
        ax.set_ylabel('3mm覆盖率')
        ax.set_title('Chamfer距离 vs 覆盖率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("准确性指标对比图已保存")
    
    def _plot_stability_comparison(self):
        """绘制稳定性指标对比图"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效的稳定性数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 残差标准差对比
        sns.boxplot(data=metrics_df, x='method', y='std_residual', ax=axes[0,0])
        axes[0,0].set_title('残差标准差对比')
        axes[0,0].set_xlabel('检测方法')
        axes[0,0].set_ylabel('残差标准差 (mm)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 变异系数对比
        sns.boxplot(data=metrics_df, x='method', y='cv_residual', ax=axes[0,1])
        axes[0,1].set_title('变异系数对比')
        axes[0,1].set_xlabel('检测方法')
        axes[0,1].set_ylabel('变异系数')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 残差分布直方图
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            axes[1,0].hist(method_data['std_residual'], alpha=0.7, label=method, bins=10)
        
        axes[1,0].set_xlabel('残差标准差 (mm)')
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title('残差标准差分布')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 稳定性vs准确性
        ax = axes[1,1]
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            ax.scatter(method_data['std_residual'], method_data['chamfer_distance'], 
                      label=method, alpha=0.7, s=100)
        
        ax.set_xlabel('残差标准差 (mm)')
        ax.set_ylabel('Chamfer距离 (mm)')
        ax.set_title('稳定性 vs 准确性')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'stability_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("稳定性指标对比图已保存")
    
    def _plot_efficiency_comparison(self):
        """绘制效率指标对比图"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效的效率数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 执行时间对比
        sns.boxplot(data=metrics_df, x='method', y='execution_time', ax=axes[0,0])
        axes[0,0].set_title('执行时间对比')
        axes[0,0].set_xlabel('检测方法')
        axes[0,0].set_ylabel('执行时间 (秒)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 采集点数对比
        sns.boxplot(data=metrics_df, x='method', y='point_count', ax=axes[0,1])
        axes[0,1].set_title('采集点数对比')
        axes[0,1].set_xlabel('检测方法')
        axes[0,1].set_ylabel('点数')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 效率vs准确性
        ax = axes[1,0]
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            ax.scatter(method_data['execution_time'], method_data['coverage_3mm'], 
                      label=method, alpha=0.7, s=100)
        
        ax.set_xlabel('执行时间 (秒)')
        ax.set_ylabel('3mm覆盖率')
        ax.set_title('效率 vs 准确性')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 点数vs覆盖率
        ax = axes[1,1]
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            ax.scatter(method_data['point_count'], method_data['coverage_3mm'], 
                      label=method, alpha=0.7, s=100)
        
        ax.set_xlabel('采集点数')
        ax.set_ylabel('3mm覆盖率')
        ax.set_title('点数 vs 覆盖率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("效率指标对比图已保存")
    
    def _plot_radar_chart(self):
        """绘制综合性能雷达图"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效数据用于雷达图")
            return
        
        # 计算综合性能指标
        radar_metrics = {}
        for method in metrics_df['method'].unique():
            method_data = metrics_df[metrics_df['method'] == method]
            
            # 归一化指标 (0-1, 越小越好)
            chamfer_norm = 1 - np.clip(method_data['chamfer_distance'].mean() / 100, 0, 1)  # 假设100mm为最大值
            hausdorff_norm = 1 - np.clip(method_data['hausdorff_distance'].mean() / 100, 0, 1)
            coverage_norm = method_data['coverage_3mm'].mean()  # 覆盖率本身就是0-1
            stability_norm = 1 - np.clip(method_data['std_residual'].mean() / 50, 0, 1)  # 假设50mm为最大值
            efficiency_norm = 1 - np.clip(method_data['execution_time'].mean() / 300, 0, 1)  # 假设300s为最大值
            
            radar_metrics[method] = [chamfer_norm, hausdorff_norm, coverage_norm, stability_norm, efficiency_norm]
        
        # 绘制雷达图
        categories = ['准确性\n(Chamfer)', '准确性\n(Hausdorff)', '覆盖率\n(3mm)', '稳定性\n(残差)', '效率\n(时间)']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        colors = ['#1f77b4', '#ff7f0e']
        
        for i, (method, values) in enumerate(radar_metrics.items()):
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('综合性能雷达图', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'performance_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("综合性能雷达图已保存")
    
    def _perform_statistical_tests(self):
        """执行统计显著性检验"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效数据用于统计检验")
            return
        
        from scipy import stats
        
        # 关键指标列表
        key_metrics = ['chamfer_distance', 'hausdorff_distance', 'coverage_3mm', 'execution_time', 'point_count']
        metric_names = ['Chamfer距离', 'Hausdorff距离', '3mm覆盖率', '执行时间', '采集点数']
        
        # 执行Wilcoxon秩和检验
        test_results = []
        
        for metric, name in zip(key_metrics, metric_names):
            canny_data = metrics_df[metrics_df['method'] == 'CANNY'][metric].values
            rgbd_data = metrics_df[metrics_df['method'] == 'RGBD'][metric].values
            
            if len(canny_data) > 0 and len(rgbd_data) > 0:
                # Wilcoxon秩和检验
                statistic, p_value = stats.ranksums(canny_data, rgbd_data)
                
                # 计算效应量 (Cohen's d)
                pooled_std = np.sqrt(((len(canny_data) - 1) * np.var(canny_data, ddof=1) + 
                                    (len(rgbd_data) - 1) * np.var(rgbd_data, ddof=1)) / 
                                   (len(canny_data) + len(rgbd_data) - 2))
                cohens_d = (np.mean(canny_data) - np.mean(rgbd_data)) / pooled_std if pooled_std > 0 else 0
                
                test_results.append({
                    'metric': name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'canny_mean': np.mean(canny_data),
                    'rgbd_mean': np.mean(rgbd_data),
                    'canny_std': np.std(canny_data),
                    'rgbd_std': np.std(rgbd_data)
                })
        
        # 保存统计检验结果
        stats_file = self.analysis_dir / "statistical_tests.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成统计检验报告
        self._generate_statistical_report(test_results)
        
        print("统计显著性检验完成")
    
    def _generate_statistical_report(self, test_results: List[Dict]):
        """生成统计检验报告"""
        report_file = self.analysis_dir / "statistical_analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("边界追踪算法统计显著性检验报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("检验方法: Wilcoxon秩和检验 (非参数检验)\n")
            f.write("显著性水平: α = 0.05\n")
            f.write("效应量: Cohen's d\n\n")
            
            f.write("详细检验结果:\n")
            f.write("-" * 60 + "\n")
            
            for result in test_results:
                f.write(f"\n指标: {result['metric']}\n")
                f.write(f"  Canny方法: 均值={result['canny_mean']:.4f}, 标准差={result['canny_std']:.4f}\n")
                f.write(f"  RGB-D方法: 均值={result['rgbd_mean']:.4f}, 标准差={result['rgbd_std']:.4f}\n")
                f.write(f"  检验统计量: {result['statistic']:.4f}\n")
                f.write(f"  p值: {result['p_value']:.6f}\n")
                f.write(f"  效应量 (Cohen's d): {result['cohens_d']:.4f}\n")
                f.write(f"  统计显著性: {'是' if result['significant'] else '否'}\n")
                
                # 效应量解释
                if abs(result['cohens_d']) < 0.2:
                    effect_size = "小"
                elif abs(result['cohens_d']) < 0.5:
                    effect_size = "中等"
                elif abs(result['cohens_d']) < 0.8:
                    effect_size = "大"
                else:
                    effect_size = "很大"
                
                f.write(f"  效应量大小: {effect_size}\n")
            
            # 总结
            f.write(f"\n总结:\n")
            f.write("-" * 60 + "\n")
            significant_tests = [r for r in test_results if r['significant']]
            f.write(f"在{len(test_results)}个指标中，有{len(significant_tests)}个指标显示统计显著性差异。\n")
            
            if significant_tests:
                f.write("\n具有统计显著性差异的指标:\n")
                for result in significant_tests:
                    f.write(f"  - {result['metric']}: p={result['p_value']:.6f}, d={result['cohens_d']:.4f}\n")
        
        print(f"统计检验报告已生成: {report_file}")
    
    def _generate_detailed_report(self):
        """生成详细分析报告"""
        report_file = self.analysis_dir / "detailed_analysis_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("边界追踪算法详细分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"实验名称: {self.results.get('experiment_name', 'Unknown')}\n")
            f.write(f"实验时间: {self.results.get('timestamp', 'Unknown')}\n")
            f.write(f"重复次数: {self.results.get('experiment_config', {}).get('n_runs', 'Unknown')}\n\n")
            
            # 实验配置
            f.write("实验配置:\n")
            f.write("-" * 30 + "\n")
            config = self.results.get('experiment_config', {})
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 硬件信息
            f.write("硬件信息:\n")
            f.write("-" * 30 + "\n")
            hardware = self.results.get('hardware_info', {})
            for key, value in hardware.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 方法对比总结
            f.write("方法对比总结:\n")
            f.write("-" * 30 + "\n")
            
            for method in ['canny', 'rgbd']:
                method_results = self.results[method]
                method_name = method.upper()
                
                f.write(f"\n{method_name}方法:\n")
                
                # 成功率统计
                success_count = sum(1 for r in method_results if r['status'] == 'Success')
                partial_count = sum(1 for r in method_results if r['status'] == 'Partial')
                fail_count = sum(1 for r in method_results if r['status'] == 'Fail')
                total_count = len(method_results)
                
                f.write(f"  成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)\n")
                f.write(f"  部分成功率: {partial_count}/{total_count} ({partial_count/total_count*100:.1f}%)\n")
                f.write(f"  失败率: {fail_count}/{total_count} ({fail_count/total_count*100:.1f}%)\n")
                
                # 平均指标
                successful_runs = [r for r in method_results if r['status'] in ['Success', 'Partial']]
                if successful_runs:
                    f.write(f"  成功运行平均指标:\n")
                    
                    # 准确性
                    if 'accuracy_metrics' in successful_runs[0]:
                        acc_metrics = [r['accuracy_metrics'] for r in successful_runs]
                        chamfer_avg = np.mean([m['chamfer_distance'] for m in acc_metrics])
                        hausdorff_avg = np.mean([m['hausdorff_distance'] for m in acc_metrics])
                        coverage_avg = np.mean([m['coverage_3mm'] for m in acc_metrics])
                        
                        f.write(f"    Chamfer距离: {chamfer_avg:.2f} ± {np.std([m['chamfer_distance'] for m in acc_metrics]):.2f} mm\n")
                        f.write(f"    Hausdorff距离: {hausdorff_avg:.2f} ± {np.std([m['hausdorff_distance'] for m in acc_metrics]):.2f} mm\n")
                        f.write(f"    3mm覆盖率: {coverage_avg:.3f} ± {np.std([m['coverage_3mm'] for m in acc_metrics]):.3f}\n")
                    
                    # 效率
                    exec_times = [r['execution_time'] for r in successful_runs]
                    point_counts = [r['point_count'] for r in successful_runs]
                    
                    f.write(f"    执行时间: {np.mean(exec_times):.2f} ± {np.std(exec_times):.2f} 秒\n")
                    f.write(f"    采集点数: {np.mean(point_counts):.0f} ± {np.std(point_counts):.0f}\n")
            
            # 建议
            f.write(f"\n\n建议:\n")
            f.write("-" * 30 + "\n")
            f.write("基于以上分析结果，建议:\n")
            f.write("1. 重点关注统计显著性差异的指标\n")
            f.write("2. 考虑算法参数优化以提高性能\n")
            f.write("3. 在实际应用中根据具体需求选择合适的方法\n")
        
        print(f"详细分析报告已生成: {report_file}")
    
    def generate_summary_table(self):
        """生成汇总表格"""
        metrics_df = self._extract_metrics_data()
        
        if metrics_df.empty:
            print("没有有效数据用于生成汇总表格")
            return
        
        # 按方法分组计算统计量
        summary_stats = metrics_df.groupby('method').agg({
            'chamfer_distance': ['mean', 'std', 'min', 'max'],
            'hausdorff_distance': ['mean', 'std', 'min', 'max'],
            'coverage_3mm': ['mean', 'std', 'min', 'max'],
            'execution_time': ['mean', 'std', 'min', 'max'],
            'point_count': ['mean', 'std', 'min', 'max'],
            'std_residual': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # 保存汇总表格
        summary_file = self.analysis_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        
        # 生成LaTeX格式的表格
        latex_file = self.analysis_dir / "summary_statistics.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{边界追踪算法性能对比汇总}\n")
            f.write("\\label{tab:boundary_tracking_summary}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\hline\n")
            f.write("指标 & 方法 & 均值 & 标准差 & 最小值 & 最大值 \\\\\n")
            f.write("\\hline\n")
            
            for metric in ['chamfer_distance', 'hausdorff_distance', 'coverage_3mm', 'execution_time', 'point_count']:
                metric_name = {
                    'chamfer_distance': 'Chamfer距离(mm)',
                    'hausdorff_distance': 'Hausdorff距离(mm)',
                    'coverage_3mm': '3mm覆盖率',
                    'execution_time': '执行时间(s)',
                    'point_count': '采集点数'
                }[metric]
                
                for method in summary_stats.index:
                    stats = summary_stats.loc[method, metric]
                    f.write(f"{metric_name} & {method} & {stats['mean']:.3f} & {stats['std']:.3f} & {stats['min']:.3f} & {stats['max']:.3f} \\\\\n")
                
                if metric != 'point_count':  # 最后一行不添加额外的分隔线
                    f.write("\\hline\n")
            
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"汇总表格已生成:")
        print(f"  CSV格式: {summary_file}")
        print(f"  LaTeX格式: {latex_file}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python boundary_tracking_analysis.py <results_file>")
        print("示例: python boundary_tracking_analysis.py evaluation_results/experiment_results.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"错误: 结果文件 {results_file} 不存在")
        return
    
    # 创建分析器
    analyzer = BoundaryTrackingAnalyzer(results_file)
    
    # 生成综合分析
    analyzer.generate_comprehensive_analysis()
    
    # 生成汇总表格
    analyzer.generate_summary_table()
    
    print(f"\n分析完成！结果保存在: {analyzer.analysis_dir}")

if __name__ == "__main__":
    main()
