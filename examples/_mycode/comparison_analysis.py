"""
物体检测方法对比分析工具
用于比较不同检测方法的性能指标
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime

class MethodComparisonAnalyzer:
    """方法对比分析器"""
    
    def __init__(self):
        self.methods_data = {}
        self.comparison_results = {}
        
    def load_evaluation_results(self, results_dir: str = "evaluation_results"):
        """加载评估结果文件"""
        if not os.path.exists(results_dir):
            print(f"结果目录 {results_dir} 不存在")
            return
        
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    method_name = data.get('method_name', filename.split('_')[2])
                    self.methods_data[method_name] = data
                    print(f"已加载 {method_name} 方法的评估结果")
                    
                except Exception as e:
                    print(f"加载文件 {filename} 失败: {e}")
    
    def compare_methods(self):
        """比较不同方法的性能"""
        if len(self.methods_data) < 2:
            print("需要至少两种方法的评估结果才能进行对比")
            return
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'methods_compared': list(self.methods_data.keys()),
            'accuracy_comparison': {},
            'performance_comparison': {},
            'stability_comparison': {},
            'overall_ranking': {}
        }
        
        # 精度对比
        accuracy_metrics = {}
        for method_name, data in self.methods_data.items():
            if 'accuracy_analysis' in data:
                accuracy_metrics[method_name] = {
                    'position_error_mean': data['accuracy_analysis']['position_error_mean'],
                    'position_error_std': data['accuracy_analysis']['position_error_std'],
                    'range_error_mean': data['accuracy_analysis']['range_error_mean'],
                    'range_error_std': data['accuracy_analysis']['range_error_std']
                }
        
        comparison['accuracy_comparison'] = accuracy_metrics
        
        # 性能对比
        performance_metrics = {}
        for method_name, data in self.methods_data.items():
            if 'performance_metrics' in data:
                performance_metrics[method_name] = {
                    'avg_detection_time': data['performance_metrics']['avg_detection_time'],
                    'avg_fps': data['performance_metrics']['avg_fps'],
                    'success_rate': data['performance_metrics']['success_rate'],
                    'total_detections': data['performance_metrics']['total_detections']
                }
        
        comparison['performance_comparison'] = performance_metrics
        
        # 稳定性对比
        stability_metrics = {}
        for method_name, data in self.methods_data.items():
            if 'stability_analysis' in data:
                stability_metrics[method_name] = data['stability_analysis']
        
        comparison['stability_comparison'] = stability_metrics
        
        # 综合排名
        comparison['overall_ranking'] = self._calculate_overall_ranking(
            accuracy_metrics, performance_metrics
        )
        
        self.comparison_results = comparison
        return comparison
    
    def _calculate_overall_ranking(self, accuracy_metrics: Dict, performance_metrics: Dict) -> Dict:
        """计算综合排名"""
        ranking_scores = {}
        
        for method_name in accuracy_metrics.keys():
            if method_name not in performance_metrics:
                continue
            
            # 精度得分（误差越小越好）
            pos_error = accuracy_metrics[method_name]['position_error_mean']
            range_error = accuracy_metrics[method_name]['range_error_mean']
            
            # 性能得分（FPS越高越好，成功率越高越好）
            fps = performance_metrics[method_name]['avg_fps']
            success_rate = performance_metrics[method_name]['success_rate']
            
            # 归一化得分（假设理想值）
            pos_error_score = max(0, 1 - pos_error / 0.1)  # 10cm为基准
            range_error_score = max(0, 1 - range_error / 0.05)  # 5cm为基准
            fps_score = min(1, fps / 10)  # 10FPS为基准
            success_rate_score = success_rate
            
            # 综合得分（权重可调整）
            overall_score = (
                pos_error_score * 0.3 +
                range_error_score * 0.2 +
                fps_score * 0.2 +
                success_rate_score * 0.3
            )
            
            ranking_scores[method_name] = {
                'overall_score': overall_score,
                'position_accuracy_score': pos_error_score,
                'range_accuracy_score': range_error_score,
                'performance_score': fps_score,
                'reliability_score': success_rate_score
            }
        
        # 按综合得分排序
        sorted_methods = sorted(
            ranking_scores.items(),
            key=lambda x: x[1]['overall_score'],
            reverse=True
        )
        
        return {
            'ranking': [method for method, _ in sorted_methods],
            'scores': ranking_scores
        }
    
    def generate_comparison_report(self, output_dir: str = "comparison_reports"):
        """生成对比报告"""
        if not self.comparison_results:
            print("请先运行 compare_methods()")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON报告
        json_filename = f"method_comparison_{timestamp}.json"
        json_path = os.path.join(output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, indent=2, ensure_ascii=False)
        
        # 生成可视化图表
        self._generate_comparison_plots(output_dir, timestamp)
        
        # 生成文本报告
        self._generate_text_report(output_dir, timestamp)
        
        print(f"对比报告已生成到: {output_dir}")
        return json_path
    
    def _generate_comparison_plots(self, output_dir: str, timestamp: str):
        """生成对比图表"""
        if not self.comparison_results:
            return
        
        methods = list(self.methods_data.keys())
        
        # 1. 精度对比图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('物体检测方法性能对比', fontsize=16)
        
        # 位置误差对比
        pos_errors = [self.comparison_results['accuracy_comparison'][m]['position_error_mean'] 
                     for m in methods]
        pos_stds = [self.comparison_results['accuracy_comparison'][m]['position_error_std'] 
                   for m in methods]
        
        bars1 = ax1.bar(methods, pos_errors, yerr=pos_stds, capsize=5, alpha=0.7)
        ax1.set_title('位置检测误差对比')
        ax1.set_ylabel('误差 (米)')
        ax1.set_ylim(0, max(pos_errors) * 1.2)
        
        # 范围误差对比
        range_errors = [self.comparison_results['accuracy_comparison'][m]['range_error_mean'] 
                       for m in methods]
        range_stds = [self.comparison_results['accuracy_comparison'][m]['range_error_std'] 
                     for m in methods]
        
        bars2 = ax2.bar(methods, range_errors, yerr=range_stds, capsize=5, alpha=0.7)
        ax2.set_title('范围检测误差对比')
        ax2.set_ylabel('误差 (米)')
        ax2.set_ylim(0, max(range_errors) * 1.2)
        
        # 检测时间对比
        detection_times = [self.comparison_results['performance_comparison'][m]['avg_detection_time'] 
                          for m in methods]
        bars3 = ax3.bar(methods, detection_times, alpha=0.7)
        ax3.set_title('平均检测时间对比')
        ax3.set_ylabel('时间 (秒)')
        
        # 成功率对比
        success_rates = [self.comparison_results['performance_comparison'][m]['success_rate'] 
                        for m in methods]
        bars4 = ax4.bar(methods, success_rates, alpha=0.7)
        ax4.set_title('检测成功率对比')
        ax4.set_ylabel('成功率')
        ax4.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plot_filename = f"performance_comparison_{timestamp}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 综合得分雷达图
        if 'overall_ranking' in self.comparison_results:
            self._generate_radar_chart(output_dir, timestamp)
    
    def _generate_radar_chart(self, output_dir: str, timestamp: str):
        """生成雷达图"""
        if 'overall_ranking' not in self.comparison_results:
            return
        
        scores = self.comparison_results['overall_ranking']['scores']
        methods = list(scores.keys())
        
        # 雷达图数据
        categories = ['位置精度', '范围精度', '性能', '可靠性']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for method_name in methods:
            method_scores = scores[method_name]
            values = [
                method_scores['position_accuracy_score'],
                method_scores['range_accuracy_score'],
                method_scores['performance_score'],
                method_scores['reliability_score']
            ]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, alpha=0.7)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('方法综合性能雷达图', size=15, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        radar_filename = f"radar_chart_{timestamp}.png"
        radar_path = os.path.join(output_dir, radar_filename)
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_text_report(self, output_dir: str, timestamp: str):
        """生成文本报告"""
        report_lines = [
            "=" * 60,
            "物体检测方法对比分析报告",
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"对比方法: {', '.join(self.comparison_results['methods_compared'])}",
            "",
            "1. 精度对比",
            "-" * 30
        ]
        
        # 精度对比
        accuracy = self.comparison_results['accuracy_comparison']
        for method_name, metrics in accuracy.items():
            report_lines.extend([
                f"{method_name}:",
                f"  位置误差: {metrics['position_error_mean']:.4f} ± {metrics['position_error_std']:.4f} m",
                f"  范围误差: {metrics['range_error_mean']:.4f} ± {metrics['range_error_std']:.4f} m",
                ""
            ])
        
        report_lines.extend([
            "2. 性能对比",
            "-" * 30
        ])
        
        # 性能对比
        performance = self.comparison_results['performance_comparison']
        for method_name, metrics in performance.items():
            report_lines.extend([
                f"{method_name}:",
                f"  平均检测时间: {metrics['avg_detection_time']:.3f} s",
                f"  平均FPS: {metrics['avg_fps']:.1f}",
                f"  成功率: {metrics['success_rate']:.2%}",
                ""
            ])
        
        report_lines.extend([
            "3. 综合排名",
            "-" * 30
        ])
        
        # 综合排名
        ranking = self.comparison_results['overall_ranking']
        for i, method_name in enumerate(ranking['ranking'], 1):
            score = ranking['scores'][method_name]['overall_score']
            report_lines.append(f"第{i}名: {method_name} (综合得分: {score:.3f})")
        
        report_lines.extend([
            "",
            "4. 结论与建议",
            "-" * 30,
            "基于以上分析，建议选择综合得分最高的方法作为主要检测方案。",
            "同时可以考虑不同方法的优势，在特定场景下使用最适合的方法。"
        ])
        
        # 保存文本报告
        text_filename = f"comparison_report_{timestamp}.txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告
        print('\n'.join(report_lines))
    
    def print_quick_summary(self):
        """打印快速摘要"""
        if not self.comparison_results:
            print("请先运行 compare_methods()")
            return
        
        print("\n=== 方法对比快速摘要 ===")
        
        # 最佳方法
        ranking = self.comparison_results['overall_ranking']
        best_method = ranking['ranking'][0]
        best_score = ranking['scores'][best_method]['overall_score']
        
        print(f"最佳方法: {best_method} (综合得分: {best_score:.3f})")
        
        # 各方法关键指标
        print("\n关键指标对比:")
        for method_name in ranking['ranking']:
            accuracy = self.comparison_results['accuracy_comparison'][method_name]
            performance = self.comparison_results['performance_comparison'][method_name]
            
            print(f"\n{method_name}:")
            print(f"  位置误差: {accuracy['position_error_mean']:.4f}m")
            print(f"  检测时间: {performance['avg_detection_time']:.3f}s")
            print(f"  成功率: {performance['success_rate']:.2%}")


def main():
    """主函数 - 运行对比分析"""
    analyzer = MethodComparisonAnalyzer()
    
    print("=== 物体检测方法对比分析工具 ===")
    
    # 加载评估结果
    print("\n正在加载评估结果...")
    analyzer.load_evaluation_results()
    
    if len(analyzer.methods_data) == 0:
        print("未找到评估结果文件，请先运行各方法的评估测试")
        return
    
    print(f"已加载 {len(analyzer.methods_data)} 种方法的评估结果")
    
    # 运行对比分析
    print("\n正在运行对比分析...")
    comparison_results = analyzer.compare_methods()
    
    # 生成报告
    print("\n正在生成对比报告...")
    analyzer.generate_comparison_report()
    
    # 打印快速摘要
    analyzer.print_quick_summary()


if __name__ == "__main__":
    main() 