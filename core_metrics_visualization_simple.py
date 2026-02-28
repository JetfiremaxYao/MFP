#!/usr/bin/env python3
"""
Core Metrics Visualization for Boundary Tracking Evaluation (Simple Version)
Focus on: Chamfer Distance, Hausdorff Distance, Std Residual, CV Residual, Execution Time, Point Count
"""

import csv
import json
from pathlib import Path

class CoreMetricsVisualizer:
    """Core metrics visualization for boundary tracking evaluation"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.data = self.load_data()
        self.output_dir = Path("evaluation_results/core_metrics_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define core metrics
        self.core_metrics = {
            'chamfer_distance_mm': 'Chamfer Distance (mm)',
            'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
            'std_residual_mm': 'Standard Residual (mm)',
            'cv_residual': 'Coefficient of Variation',
            'execution_time_s': 'Execution Time (s)',
            'point_count': 'Point Count'
        }
        
        # Define objects and lighting conditions
        self.objects = ['Cube', 'ctry.obj']
        self.lighting_conditions = ['normal', 'bright', 'dim']
        self.methods = ['canny', 'rgbd']
    
    def load_data(self):
        """Load data from CSV file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def calculate_statistics(self, values):
        """Calculate basic statistics for a list of values"""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'count': 0}
        
        # Convert to float and filter out empty values
        float_values = [float(v) for v in values if v != '']
        if not float_values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'count': 0}
        
        # Sort for median calculation
        sorted_values = sorted(float_values)
        n = len(sorted_values)
        
        # Calculate statistics
        mean = sum(sorted_values) / n
        variance = sum((x - mean) ** 2 for x in sorted_values) / n
        std = variance ** 0.5
        
        min_val = sorted_values[0]
        max_val = sorted_values[-1]
        
        # Calculate median
        if n % 2 == 0:
            median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            median = sorted_values[n//2]
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median,
            'count': n
        }
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis for all core metrics"""
        print("Creating comprehensive core metrics analysis...")
        
        analysis_results = {}
        
        # Analyze each metric
        for metric, metric_name in self.core_metrics.items():
            print(f"  Analyzing {metric_name}...")
            
            metric_analysis = {
                'overall': {},
                'by_object': {},
                'by_lighting': {},
                'by_method': {}
            }
            
            # Overall statistics
            all_values = [row[metric] for row in self.data if row[metric] != '']
            metric_analysis['overall'] = self.calculate_statistics(all_values)
            
            # By object
            for obj in self.objects:
                obj_values = [row[metric] for row in self.data 
                             if row['object_name'] == obj and row[metric] != '']
                metric_analysis['by_object'][obj] = self.calculate_statistics(obj_values)
            
            # By lighting condition
            for lighting in self.lighting_conditions:
                lighting_values = [row[metric] for row in self.data 
                                 if row['lighting'] == lighting and row[metric] != '']
                metric_analysis['by_lighting'][lighting] = self.calculate_statistics(lighting_values)
            
            # By method
            for method in self.methods:
                method_values = [row[metric] for row in self.data 
                               if row['method'] == method and row[metric] != '']
                metric_analysis['by_method'][method] = self.calculate_statistics(method_values)
            
            analysis_results[metric] = metric_analysis
        
        # Save analysis results
        with open(self.output_dir / "core_metrics_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        return analysis_results
    
    def create_detailed_comparison_table(self, analysis_results):
        """Create detailed comparison table"""
        print("Creating detailed comparison table...")
        
        # Create detailed comparison for each object and lighting condition
        detailed_results = []
        
        for obj in self.objects:
            for lighting in self.lighting_conditions:
                for method in self.methods:
                    # Get data for this combination
                    obj_data = [row for row in self.data 
                              if row['object_name'] == obj and 
                                 row['lighting'] == lighting and 
                                 row['method'] == method]
                    
                    if obj_data:
                        result_row = {
                            'object': obj,
                            'lighting': lighting,
                            'method': method.upper(),
                            'n_trials': len(obj_data)
                        }
                        
                        # Add statistics for each metric
                        for metric in self.core_metrics.keys():
                            values = [row[metric] for row in obj_data if row[metric] != '']
                            stats = self.calculate_statistics(values)
                            
                            result_row[f'{metric}_mean'] = round(stats['mean'], 3)
                            result_row[f'{metric}_std'] = round(stats['std'], 3)
                            result_row[f'{metric}_min'] = round(stats['min'], 3)
                            result_row[f'{metric}_max'] = round(stats['max'], 3)
                            result_row[f'{metric}_median'] = round(stats['median'], 3)
                        
                        detailed_results.append(result_row)
        
        # Save detailed results
        with open(self.output_dir / "detailed_comparison.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return detailed_results
    
    def create_summary_report(self, analysis_results, detailed_results):
        """Create summary report"""
        print("Creating summary report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CORE METRICS ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall summary
        report_lines.append("OVERALL SUMMARY:")
        report_lines.append("-" * 40)
        for metric, metric_name in self.core_metrics.items():
            overall_stats = analysis_results[metric]['overall']
            report_lines.append(f"{metric_name}:")
            report_lines.append(f"  Mean: {overall_stats['mean']:.3f}")
            report_lines.append(f"  Std:  {overall_stats['std']:.3f}")
            report_lines.append(f"  Min:  {overall_stats['min']:.3f}")
            report_lines.append(f"  Max:  {overall_stats['max']:.3f}")
            report_lines.append(f"  Median: {overall_stats['median']:.3f}")
            report_lines.append(f"  Count: {overall_stats['count']}")
            report_lines.append("")
        
        # Method comparison
        report_lines.append("METHOD COMPARISON:")
        report_lines.append("-" * 40)
        for metric, metric_name in self.core_metrics.items():
            report_lines.append(f"\n{metric_name}:")
            canny_stats = analysis_results[metric]['by_method']['canny']
            rgbd_stats = analysis_results[metric]['by_method']['rgbd']
            
            report_lines.append(f"  Canny - Mean: {canny_stats['mean']:.3f} ± {canny_stats['std']:.3f}")
            report_lines.append(f"  RGBD  - Mean: {rgbd_stats['mean']:.3f} ± {rgbd_stats['std']:.3f}")
            
            # Determine winner
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                # Lower is better
                winner = "Canny" if canny_stats['mean'] < rgbd_stats['mean'] else "RGBD"
            else:
                # Higher is better (point_count)
                winner = "Canny" if canny_stats['mean'] > rgbd_stats['mean'] else "RGBD"
            
            report_lines.append(f"  Winner: {winner}")
        
        # Object comparison
        report_lines.append("\n\nOBJECT COMPARISON:")
        report_lines.append("-" * 40)
        for metric, metric_name in self.core_metrics.items():
            report_lines.append(f"\n{metric_name}:")
            cube_stats = analysis_results[metric]['by_object']['Cube']
            ctry_stats = analysis_results[metric]['by_object']['ctry.obj']
            
            report_lines.append(f"  Cube    - Mean: {cube_stats['mean']:.3f} ± {cube_stats['std']:.3f}")
            report_lines.append(f"  ctry.obj - Mean: {ctry_stats['mean']:.3f} ± {ctry_stats['std']:.3f}")
        
        # Lighting comparison
        report_lines.append("\n\nLIGHTING CONDITION COMPARISON:")
        report_lines.append("-" * 40)
        for metric, metric_name in self.core_metrics.items():
            report_lines.append(f"\n{metric_name}:")
            for lighting in self.lighting_conditions:
                lighting_stats = analysis_results[metric]['by_lighting'][lighting]
                report_lines.append(f"  {lighting.upper():<8} - Mean: {lighting_stats['mean']:.3f} ± {lighting_stats['std']:.3f}")
        
        # Detailed results table
        report_lines.append("\n\nDETAILED RESULTS TABLE:")
        report_lines.append("-" * 80)
        header_line = f"{'Object':<10} {'Lighting':<8} {'Method':<6} {'N':<3} "
        for metric in self.core_metrics.keys():
            header_line += f"{metric[:8]:<8} "
        report_lines.append(header_line)
        report_lines.append("-" * 80)
        
        for result in detailed_results:
            line = f"{result['object']:<10} {result['lighting']:<8} {result['method']:<6} {result['n_trials']:<3} "
            for metric in self.core_metrics.keys():
                line += f"{result[f'{metric}_mean']:<8.3f} "
            report_lines.append(line)
        
        # Save report
        with open(self.output_dir / "core_metrics_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
        print("✅ Summary report saved")
    
    def create_visualization_data(self, detailed_results):
        """Create data for visualization (CSV format)"""
        print("Creating visualization data...")
        
        # Create CSV file for visualization
        csv_file = self.output_dir / "core_metrics_visualization_data.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['object', 'lighting', 'method', 'n_trials']
            for metric in self.core_metrics.keys():
                header.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_max', f'{metric}_median'])
            writer.writerow(header)
            
            # Write data
            for result in detailed_results:
                row = [result['object'], result['lighting'], result['method'], result['n_trials']]
                for metric in self.core_metrics.keys():
                    row.extend([
                        result[f'{metric}_mean'],
                        result[f'{metric}_std'],
                        result[f'{metric}_min'],
                        result[f'{metric}_max'],
                        result[f'{metric}_median']
                    ])
                writer.writerow(row)
        
        print("✅ Visualization data saved")
    
    def create_performance_ranking(self, detailed_results):
        """Create performance ranking"""
        print("Creating performance ranking...")
        
        # Calculate performance scores for each combination
        rankings = []
        
        for result in detailed_results:
            # Calculate normalized scores (0-1 scale)
            scores = {}
            
            for metric in self.core_metrics.keys():
                mean_val = result[f'{metric}_mean']
                
                if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                             'cv_residual', 'execution_time_s']:
                    # Lower is better - find max value across all results for normalization
                    max_val = max([r[f'{metric}_mean'] for r in detailed_results])
                    scores[metric] = 1 - (mean_val / max_val) if max_val > 0 else 0
                else:
                    # Higher is better (point_count)
                    max_val = max([r[f'{metric}_mean'] for r in detailed_results])
                    scores[metric] = mean_val / max_val if max_val > 0 else 0
            
            # Calculate overall score (weighted average)
            weights = {
                'chamfer_distance_mm': 0.25,      # Accuracy
                'hausdorff_distance_mm': 0.20,    # Accuracy
                'std_residual_mm': 0.15,          # Stability
                'cv_residual': 0.10,              # Stability
                'execution_time_s': 0.15,         # Performance
                'point_count': 0.15               # Performance
            }
            
            overall_score = sum(scores[metric] * weights[metric] for metric in self.core_metrics.keys())
            
            rankings.append({
                'object': result['object'],
                'lighting': result['lighting'],
                'method': result['method'],
                'overall_score': round(overall_score, 3),
                **{f'{metric}_score': round(scores[metric], 3) for metric in self.core_metrics.keys()}
            })
        
        # Sort by overall score
        rankings.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Save rankings
        with open(self.output_dir / "performance_ranking.json", 'w') as f:
            json.dump(rankings, f, indent=2)
        
        # Create ranking report
        ranking_lines = []
        ranking_lines.append("PERFORMANCE RANKING")
        ranking_lines.append("=" * 50)
        ranking_lines.append("")
        
        for i, rank in enumerate(rankings, 1):
            ranking_lines.append(f"{i:2d}. {rank['method']:<6} - {rank['object']:<10} - {rank['lighting']:<8} - Score: {rank['overall_score']:.3f}")
        
        ranking_lines.append("")
        ranking_lines.append("SCORE BREAKDOWN:")
        ranking_lines.append("-" * 30)
        
        for rank in rankings:
            ranking_lines.append(f"\n{rank['method']} - {rank['object']} - {rank['lighting']}:")
            for metric in self.core_metrics.keys():
                ranking_lines.append(f"  {self.core_metrics[metric]}: {rank[f'{metric}_score']:.3f}")
            ranking_lines.append(f"  Overall Score: {rank['overall_score']:.3f}")
        
        with open(self.output_dir / "performance_ranking.txt", 'w') as f:
            f.write('\n'.join(ranking_lines))
        
        print("✅ Performance ranking saved")
    
    def run_all_analysis(self):
        """Run all analysis and create visualizations"""
        print("🚀 Starting core metrics visualization analysis...")
        print(f"📊 Analyzing {len(self.data)} data points")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Create all analysis
        analysis_results = self.create_comprehensive_analysis()
        detailed_results = self.create_detailed_comparison_table(analysis_results)
        self.create_summary_report(analysis_results, detailed_results)
        self.create_visualization_data(detailed_results)
        self.create_performance_ranking(detailed_results)
        
        print("\n✅ All core metrics visualizations completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        print("\n📋 Generated files:")
        print("  - core_metrics_analysis.json (detailed statistics)")
        print("  - detailed_comparison.json (comparison table)")
        print("  - core_metrics_report.txt (summary report)")
        print("  - core_metrics_visualization_data.csv (CSV for plotting)")
        print("  - performance_ranking.json (performance scores)")
        print("  - performance_ranking.txt (ranking report)")

def main():
    """Main function"""
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    # Create visualizer
    visualizer = CoreMetricsVisualizer(data_path)
    
    # Run analysis
    visualizer.run_all_analysis()

if __name__ == "__main__":
    main()
