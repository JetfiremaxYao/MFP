# 物体检测方法统计分析工具
# 用于对HSV+Depth和Depth+Clustering两种方法进行统计显著性检验
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, results_file: str):
        """
        初始化统计分析器
        
        Parameters:
        -----------
        results_file : str
            实验结果文件路径
        """
        self.results_file = Path(results_file)
        self.results = self._load_results()
        self.analysis_results = {}
        
    def _load_results(self) -> Dict:
        """加载实验结果"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_metrics(self, method: str) -> Dict[str, List[float]]:
        """提取指定方法的指标数据"""
        method_data = self.results[method]
        
        metrics = {
            'position_error': [],
            'range_error': [],
            'detection_time': [],
            'success_status': []
        }
        
        for result in method_data:
            if result['status'] != 'Fail':
                metrics['position_error'].append(result['position_error'] * 1000)  # 转换为mm
                metrics['range_error'].append(result['range_error'] * 1000)  # 转换为mm
                metrics['detection_time'].append(result['detection_time'])
                metrics['success_status'].append(1 if result['status'] == 'Success' else 0)
        
        return metrics
    
    def perform_statistical_tests(self):
        """执行统计检验"""
        print("开始执行统计检验...")
        
        # 提取数据
        hsv_metrics = self._extract_metrics('hsv_depth')
        clustering_metrics = self._extract_metrics('depth_clustering')
        
        # 统计检验结果
        test_results = {}
        
        # 1. 位置误差检验
        if len(hsv_metrics['position_error']) > 0 and len(clustering_metrics['position_error']) > 0:
            # Shapiro-Wilk正态性检验
            hsv_normality = stats.shapiro(hsv_metrics['position_error'])
            clustering_normality = stats.shapiro(clustering_metrics['position_error'])
            
            # 根据正态性选择检验方法
            if hsv_normality.pvalue > 0.05 and clustering_normality.pvalue > 0.05:
                # 正态分布，使用t检验
                t_stat, t_pvalue = stats.ttest_ind(hsv_metrics['position_error'], 
                                                 clustering_metrics['position_error'])
                test_type = 'Independent t-test'
            else:
                # 非正态分布，使用Mann-Whitney U检验
                u_stat, u_pvalue = mannwhitneyu(hsv_metrics['position_error'], 
                                              clustering_metrics['position_error'],
                                              alternative='two-sided')
                t_stat, t_pvalue = u_stat, u_pvalue
                test_type = 'Mann-Whitney U test'
            
            # 计算效应量 (Cohen's d)
            pooled_std = np.sqrt(((len(hsv_metrics['position_error']) - 1) * np.var(hsv_metrics['position_error']) +
                                 (len(clustering_metrics['position_error']) - 1) * np.var(clustering_metrics['position_error'])) /
                                (len(hsv_metrics['position_error']) + len(clustering_metrics['position_error']) - 2))
            
            cohens_d = (np.mean(hsv_metrics['position_error']) - np.mean(clustering_metrics['position_error'])) / pooled_std
            
            test_results['position_error'] = {
                'test_type': test_type,
                'statistic': t_stat,
                'p_value': t_pvalue,
                'cohens_d': cohens_d,
                'hsv_mean': np.mean(hsv_metrics['position_error']),
                'clustering_mean': np.mean(clustering_metrics['position_error']),
                'hsv_std': np.std(hsv_metrics['position_error']),
                'clustering_std': np.std(clustering_metrics['position_error']),
                'hsv_normality_p': hsv_normality.pvalue,
                'clustering_normality_p': clustering_normality.pvalue
            }
        
        # 2. 范围误差检验
        if len(hsv_metrics['range_error']) > 0 and len(clustering_metrics['range_error']) > 0:
            hsv_normality = stats.shapiro(hsv_metrics['range_error'])
            clustering_normality = stats.shapiro(clustering_metrics['range_error'])
            
            if hsv_normality.pvalue > 0.05 and clustering_normality.pvalue > 0.05:
                t_stat, t_pvalue = stats.ttest_ind(hsv_metrics['range_error'], 
                                                 clustering_metrics['range_error'])
                test_type = 'Independent t-test'
            else:
                u_stat, u_pvalue = mannwhitneyu(hsv_metrics['range_error'], 
                                              clustering_metrics['range_error'],
                                              alternative='two-sided')
                t_stat, t_pvalue = u_stat, u_pvalue
                test_type = 'Mann-Whitney U test'
            
            pooled_std = np.sqrt(((len(hsv_metrics['range_error']) - 1) * np.var(hsv_metrics['range_error']) +
                                 (len(clustering_metrics['range_error']) - 1) * np.var(clustering_metrics['range_error'])) /
                                (len(hsv_metrics['range_error']) + len(clustering_metrics['range_error']) - 2))
            
            cohens_d = (np.mean(hsv_metrics['range_error']) - np.mean(clustering_metrics['range_error'])) / pooled_std
            
            test_results['range_error'] = {
                'test_type': test_type,
                'statistic': t_stat,
                'p_value': t_pvalue,
                'cohens_d': cohens_d,
                'hsv_mean': np.mean(hsv_metrics['range_error']),
                'clustering_mean': np.mean(clustering_metrics['range_error']),
                'hsv_std': np.std(hsv_metrics['range_error']),
                'clustering_std': np.std(clustering_metrics['range_error']),
                'hsv_normality_p': hsv_normality.pvalue,
                'clustering_normality_p': clustering_normality.pvalue
            }
        
        # 3. 检测时间检验
        if len(hsv_metrics['detection_time']) > 0 and len(clustering_metrics['detection_time']) > 0:
            hsv_normality = stats.shapiro(hsv_metrics['detection_time'])
            clustering_normality = stats.shapiro(clustering_metrics['detection_time'])
            
            if hsv_normality.pvalue > 0.05 and clustering_normality.pvalue > 0.05:
                t_stat, t_pvalue = stats.ttest_ind(hsv_metrics['detection_time'], 
                                                 clustering_metrics['detection_time'])
                test_type = 'Independent t-test'
            else:
                u_stat, u_pvalue = mannwhitneyu(hsv_metrics['detection_time'], 
                                              clustering_metrics['detection_time'],
                                              alternative='two-sided')
                t_stat, t_pvalue = u_stat, u_pvalue
                test_type = 'Mann-Whitney U test'
            
            pooled_std = np.sqrt(((len(hsv_metrics['detection_time']) - 1) * np.var(hsv_metrics['detection_time']) +
                                 (len(clustering_metrics['detection_time']) - 1) * np.var(clustering_metrics['detection_time'])) /
                                (len(hsv_metrics['detection_time']) + len(clustering_metrics['detection_time']) - 2))
            
            cohens_d = (np.mean(hsv_metrics['detection_time']) - np.mean(clustering_metrics['detection_time'])) / pooled_std
            
            test_results['detection_time'] = {
                'test_type': test_type,
                'statistic': t_stat,
                'p_value': t_pvalue,
                'cohens_d': cohens_d,
                'hsv_mean': np.mean(hsv_metrics['detection_time']),
                'clustering_mean': np.mean(clustering_metrics['detection_time']),
                'hsv_std': np.std(hsv_metrics['detection_time']),
                'clustering_std': np.std(clustering_metrics['detection_time']),
                'hsv_normality_p': hsv_normality.pvalue,
                'clustering_normality_p': clustering_normality.pvalue
            }
        
        # 4. 成功率检验 (卡方检验)
        hsv_success = sum(hsv_metrics['success_status'])
        hsv_total = len(hsv_metrics['success_status'])
        clustering_success = sum(clustering_metrics['success_status'])
        clustering_total = len(clustering_metrics['success_status'])
        
        if hsv_total > 0 and clustering_total > 0:
            contingency_table = np.array([[hsv_success, hsv_total - hsv_success],
                                        [clustering_success, clustering_total - clustering_success]])
            
            chi2_stat, chi2_pvalue = stats.chi2_contingency(contingency_table)[:2]
            
            # 计算成功率差异的效应量 (Phi系数)
            phi = np.sqrt(chi2_stat / (hsv_total + clustering_total))
            
            test_results['success_rate'] = {
                'test_type': 'Chi-square test',
                'statistic': chi2_stat,
                'p_value': chi2_pvalue,
                'phi_coefficient': phi,
                'hsv_success_rate': hsv_success / hsv_total if hsv_total > 0 else 0,
                'clustering_success_rate': clustering_success / clustering_total if clustering_total > 0 else 0,
                'hsv_success_count': hsv_success,
                'hsv_total_count': hsv_total,
                'clustering_success_count': clustering_success,
                'clustering_total_count': clustering_total
            }
        
        self.analysis_results = test_results
        return test_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """解释效应量大小"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "小效应"
        elif abs_d < 0.5:
            return "中等效应"
        elif abs_d < 0.8:
            return "大效应"
        else:
            return "很大效应"
    
    def _interpret_phi_coefficient(self, phi: float) -> str:
        """解释Phi系数大小"""
        abs_phi = abs(phi)
        if abs_phi < 0.1:
            return "小效应"
        elif abs_phi < 0.3:
            return "中等效应"
        elif abs_phi < 0.5:
            return "大效应"
        else:
            return "很大效应"
    
    def print_statistical_report(self):
        """打印统计报告"""
        if not self.analysis_results:
            print("请先执行统计检验")
            return
        
        print("\n" + "="*80)
        print("统计检验报告")
        print("="*80)
        
        for metric, result in self.analysis_results.items():
            print(f"\n【{metric.upper()}】")
            print(f"检验方法: {result['test_type']}")
            print(f"统计量: {result['statistic']:.4f}")
            print(f"p值: {result['p_value']:.4f}")
            
            if result['p_value'] < 0.001:
                significance = "*** (p < 0.001)"
            elif result['p_value'] < 0.01:
                significance = "** (p < 0.01)"
            elif result['p_value'] < 0.05:
                significance = "* (p < 0.05)"
            else:
                significance = "不显著 (p ≥ 0.05)"
            
            print(f"显著性: {significance}")
            
            if 'cohens_d' in result:
                print(f"Cohen's d效应量: {result['cohens_d']:.4f} ({self._interpret_effect_size(result['cohens_d'])})")
                print(f"HSV+Depth均值: {result['hsv_mean']:.4f} ± {result['hsv_std']:.4f}")
                print(f"Depth+Clustering均值: {result['clustering_mean']:.4f} ± {result['clustering_std']:.4f}")
            elif 'phi_coefficient' in result:
                print(f"Phi系数效应量: {result['phi_coefficient']:.4f} ({self._interpret_phi_coefficient(result['phi_coefficient'])})")
                print(f"HSV+Depth成功率: {result['hsv_success_rate']:.2%} ({result['hsv_success_count']}/{result['hsv_total_count']})")
                print(f"Depth+Clustering成功率: {result['clustering_success_rate']:.2%} ({result['clustering_success_count']}/{result['clustering_total_count']})")
            
            if 'hsv_normality_p' in result:
                print(f"正态性检验 (HSV+Depth): p = {result['hsv_normality_p']:.4f}")
                print(f"正态性检验 (Depth+Clustering): p = {result['clustering_normality_p']:.4f}")
    
    def generate_statistical_plots(self, save_dir: str = None):
        """生成统计图表"""
        if not self.analysis_results:
            print("请先执行统计检验")
            return
        
        if save_dir is None:
            save_dir = self.results_file.parent
        
        # 提取数据
        hsv_metrics = self._extract_metrics('hsv_depth')
        clustering_metrics = self._extract_metrics('depth_clustering')
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('物体检测方法统计对比分析', fontsize=16, fontweight='bold')
        
        # 1. 位置误差分布对比
        if len(hsv_metrics['position_error']) > 0 and len(clustering_metrics['position_error']) > 0:
            axes[0, 0].hist(hsv_metrics['position_error'], alpha=0.7, label='HSV+Depth', bins=10)
            axes[0, 0].hist(clustering_metrics['position_error'], alpha=0.7, label='Depth+Clustering', bins=10)
            axes[0, 0].set_xlabel('位置误差 (mm)')
            axes[0, 0].set_ylabel('频次')
            axes[0, 0].set_title('位置误差分布对比')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 范围误差分布对比
        if len(hsv_metrics['range_error']) > 0 and len(clustering_metrics['range_error']) > 0:
            axes[0, 1].hist(hsv_metrics['range_error'], alpha=0.7, label='HSV+Depth', bins=10)
            axes[0, 1].hist(clustering_metrics['range_error'], alpha=0.7, label='Depth+Clustering', bins=10)
            axes[0, 1].set_xlabel('范围误差 (mm)')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].set_title('范围误差分布对比')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 检测时间分布对比
        if len(hsv_metrics['detection_time']) > 0 and len(clustering_metrics['detection_time']) > 0:
            axes[0, 2].hist(hsv_metrics['detection_time'], alpha=0.7, label='HSV+Depth', bins=10)
            axes[0, 2].hist(clustering_metrics['detection_time'], alpha=0.7, label='Depth+Clustering', bins=10)
            axes[0, 2].set_xlabel('检测时间 (s)')
            axes[0, 2].set_ylabel('频次')
            axes[0, 2].set_title('检测时间分布对比')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 箱线图对比
        if len(hsv_metrics['position_error']) > 0 and len(clustering_metrics['position_error']) > 0:
            data_for_boxplot = [hsv_metrics['position_error'], clustering_metrics['position_error']]
            axes[1, 0].boxplot(data_for_boxplot, labels=['HSV+Depth', 'Depth+Clustering'])
            axes[1, 0].set_ylabel('位置误差 (mm)')
            axes[1, 0].set_title('位置误差箱线图')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 成功率对比
        if 'success_rate' in self.analysis_results:
            success_rates = [self.analysis_results['success_rate']['hsv_success_rate'],
                           self.analysis_results['success_rate']['clustering_success_rate']]
            methods = ['HSV+Depth', 'Depth+Clustering']
            bars = axes[1, 1].bar(methods, success_rates, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_ylabel('成功率')
            axes[1, 1].set_title('成功率对比')
            axes[1, 1].set_ylim(0, 1)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f'{rate:.2%}', ha='center', va='bottom')
            
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 效应量对比
        effect_sizes = []
        effect_labels = []
        
        for metric, result in self.analysis_results.items():
            if 'cohens_d' in result:
                effect_sizes.append(abs(result['cohens_d']))
                effect_labels.append(f'{metric}\n(Cohen\'s d)')
            elif 'phi_coefficient' in result:
                effect_sizes.append(abs(result['phi_coefficient']))
                effect_labels.append(f'{metric}\n(Phi)')
        
        if effect_sizes:
            bars = axes[1, 2].bar(effect_labels, effect_sizes, color='lightgreen')
            axes[1, 2].set_ylabel('效应量大小')
            axes[1, 2].set_title('效应量对比')
            
            # 添加效应量解释
            for bar, effect_size in zip(bars, effect_sizes):
                if effect_size < 0.2:
                    interpretation = '小'
                elif effect_size < 0.5:
                    interpretation = '中'
                elif effect_size < 0.8:
                    interpretation = '大'
                else:
                    interpretation = '很大'
                
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               interpretation, ha='center', va='bottom')
            
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = Path(save_dir) / "statistical_analysis_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计图表已保存: {plot_file}")
    
    def save_statistical_report(self, save_dir: str = None):
        """保存统计报告"""
        if not self.analysis_results:
            print("请先执行统计检验")
            return
        
        if save_dir is None:
            save_dir = self.results_file.parent
        
        # 保存JSON格式的详细结果
        report_file = Path(save_dir) / "statistical_analysis_results.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        # 保存文本格式的报告
        text_report_file = Path(save_dir) / "statistical_analysis_report.txt"
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("物体检测方法统计检验报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"实验文件: {self.results_file.name}\n")
            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for metric, result in self.analysis_results.items():
                f.write(f"【{metric.upper()}】\n")
                f.write(f"检验方法: {result['test_type']}\n")
                f.write(f"统计量: {result['statistic']:.4f}\n")
                f.write(f"p值: {result['p_value']:.4f}\n")
                
                if result['p_value'] < 0.001:
                    significance = "*** (p < 0.001)"
                elif result['p_value'] < 0.01:
                    significance = "** (p < 0.01)"
                elif result['p_value'] < 0.05:
                    significance = "* (p < 0.05)"
                else:
                    significance = "不显著 (p ≥ 0.05)"
                
                f.write(f"显著性: {significance}\n")
                
                if 'cohens_d' in result:
                    f.write(f"Cohen's d效应量: {result['cohens_d']:.4f} ({self._interpret_effect_size(result['cohens_d'])})\n")
                    f.write(f"HSV+Depth均值: {result['hsv_mean']:.4f} ± {result['hsv_std']:.4f}\n")
                    f.write(f"Depth+Clustering均值: {result['clustering_mean']:.4f} ± {result['clustering_std']:.4f}\n")
                elif 'phi_coefficient' in result:
                    f.write(f"Phi系数效应量: {result['phi_coefficient']:.4f} ({self._interpret_phi_coefficient(result['phi_coefficient'])})\n")
                    f.write(f"HSV+Depth成功率: {result['hsv_success_rate']:.2%} ({result['hsv_success_count']}/{result['hsv_total_count']})\n")
                    f.write(f"Depth+Clustering成功率: {result['clustering_success_rate']:.2%} ({result['clustering_success_count']}/{result['clustering_total_count']})\n")
                
                if 'hsv_normality_p' in result:
                    f.write(f"正态性检验 (HSV+Depth): p = {result['hsv_normality_p']:.4f}\n")
                    f.write(f"正态性检验 (Depth+Clustering): p = {result['clustering_normality_p']:.4f}\n")
                
                f.write("\n")
        
        print(f"统计报告已保存:")
        print(f"  JSON格式: {report_file}")
        print(f"  文本格式: {text_report_file}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python statistical_analysis.py <experiment_results.json>")
        print("示例: python statistical_analysis.py evaluation_results/hsv_depth_vs_clustering_20250101_120000/experiment_results.json")
        return
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"错误: 文件 {results_file} 不存在")
        return
    
    print("物体检测方法统计分析工具")
    print("="*50)
    
    # 创建分析器
    analyzer = StatisticalAnalyzer(results_file)
    
    # 执行统计检验
    analyzer.perform_statistical_tests()
    
    # 打印报告
    analyzer.print_statistical_report()
    
    # 生成图表
    analyzer.generate_statistical_plots()
    
    # 保存报告
    analyzer.save_statistical_report()
    
    print("\n统计分析完成！")

if __name__ == "__main__":
    main()
