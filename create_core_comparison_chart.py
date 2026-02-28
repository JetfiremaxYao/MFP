#!/usr/bin/env python3
"""
Create Core Comparison Chart - Canny vs RGBD across lighting conditions
Merged data from both objects (Cube + ctry.obj)
"""

import csv
import json
from pathlib import Path

def create_core_comparison_chart():
    """Create core comparison chart for Canny vs RGBD across lighting conditions"""
    
    # Load data
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Core metrics for comparison
    core_metrics = {
        'chamfer_distance_mm': 'Chamfer Distance (mm)',
        'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
        'std_residual_mm': 'Standard Residual (mm)',
        'cv_residual': 'Coefficient of Variation',
        'execution_time_s': 'Execution Time (s)',
        'point_count': 'Point Count'
    }
    
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    # Create output directory
    output_dir = Path("evaluation_results/core_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison chart for each metric
    for metric, metric_name in core_metrics.items():
        print(f"Creating chart for {metric_name}...")
        
        chart_lines = []
        chart_lines.append("=" * 80)
        chart_lines.append(f"CORE COMPARISON: {metric_name.upper()}")
        chart_lines.append("Canny vs RGBD across Lighting Conditions (Merged Data)")
        chart_lines.append("=" * 80)
        chart_lines.append("")
        
        # Calculate statistics for each lighting condition
        for lighting in lighting_conditions:
            chart_lines.append(f"{lighting.upper()} LIGHTING:")
            chart_lines.append("-" * 40)
            
            # Get data for both methods in this lighting condition
            canny_data = [float(row[metric]) for row in data 
                         if row['lighting'] == lighting and 
                            row['method'] == 'canny' and 
                            row[metric] != '']
            
            rgbd_data = [float(row[metric]) for row in data 
                        if row['lighting'] == lighting and 
                           row['method'] == 'rgbd' and 
                           row[metric] != '']
            
            if canny_data and rgbd_data:
                # Calculate statistics
                canny_mean = sum(canny_data) / len(canny_data)
                canny_std = (sum((x - canny_mean) ** 2 for x in canny_data) / len(canny_data)) ** 0.5
                
                rgbd_mean = sum(rgbd_data) / len(rgbd_data)
                rgbd_std = (sum((x - rgbd_mean) ** 2 for x in rgbd_data) / len(rgbd_data)) ** 0.5
                
                # Create bar chart
                max_val = max(canny_mean, rgbd_mean)
                scale = 50  # Scale for bar length
                
                canny_bar_length = int((canny_mean / max_val) * scale) if max_val > 0 else 0
                rgbd_bar_length = int((rgbd_mean / max_val) * scale) if max_val > 0 else 0
                
                chart_lines.append(f"  Canny: {canny_mean:.3f} ± {canny_std:.3f} {'█' * canny_bar_length}")
                chart_lines.append(f"  RGBD:  {rgbd_mean:.3f} ± {rgbd_std:.3f} {'█' * rgbd_bar_length}")
                
                # Determine winner
                if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                             'cv_residual', 'execution_time_s']:
                    winner = "Canny" if canny_mean < rgbd_mean else "RGBD"
                    better = "lower"
                else:  # point_count
                    winner = "Canny" if canny_mean > rgbd_mean else "RGBD"
                    better = "higher"
                
                chart_lines.append(f"  Winner: {winner} ({better} is better)")
                
                # Calculate improvement percentage
                if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                             'cv_residual', 'execution_time_s']:
                    if canny_mean < rgbd_mean:
                        improvement = ((rgbd_mean - canny_mean) / rgbd_mean) * 100 if rgbd_mean > 0 else 0
                        chart_lines.append(f"  Canny improvement: {improvement:.1f}%")
                    else:
                        improvement = ((canny_mean - rgbd_mean) / canny_mean) * 100 if canny_mean > 0 else 0
                        chart_lines.append(f"  RGBD improvement: {improvement:.1f}%")
                else:  # point_count - higher is better
                    if rgbd_mean > canny_mean:
                        improvement = ((rgbd_mean - canny_mean) / canny_mean) * 100 if canny_mean > 0 else 0
                        chart_lines.append(f"  RGBD improvement: {improvement:.1f}%")
                    else:
                        improvement = ((canny_mean - rgbd_mean) / rgbd_mean) * 100 if rgbd_mean > 0 else 0
                        chart_lines.append(f"  Canny improvement: {improvement:.1f}%")
            else:
                chart_lines.append("  No data available")
            
            chart_lines.append("")
        
        # Add overall summary
        chart_lines.append("=" * 80)
        chart_lines.append("OVERALL SUMMARY (All Lighting Conditions)")
        chart_lines.append("=" * 80)
        
        # Calculate overall statistics
        canny_all = [float(row[metric]) for row in data 
                    if row['method'] == 'canny' and row[metric] != '']
        rgbd_all = [float(row[metric]) for row in data 
                   if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_all and rgbd_all:
            canny_overall = sum(canny_all) / len(canny_all)
            canny_std_all = (sum((x - canny_overall) ** 2 for x in canny_all) / len(canny_all)) ** 0.5
            
            rgbd_overall = sum(rgbd_all) / len(rgbd_all)
            rgbd_std_all = (sum((x - rgbd_overall) ** 2 for x in rgbd_all) / len(rgbd_all)) ** 0.5
            
            chart_lines.append(f"Canny: {canny_overall:.3f} ± {canny_std_all:.3f}")
            chart_lines.append(f"RGBD:  {rgbd_overall:.3f} ± {rgbd_std_all:.3f}")
            
            # Determine overall winner
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                overall_winner = "Canny" if canny_overall < rgbd_overall else "RGBD"
                overall_better = "lower"
            else:  # point_count
                overall_winner = "Canny" if canny_overall > rgbd_overall else "RGBD"
                overall_better = "higher"
            
            chart_lines.append(f"Overall Winner: {overall_winner} ({overall_better} is better)")
            
            # Calculate overall improvement
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                if canny_overall < rgbd_overall:
                    improvement = ((rgbd_overall - canny_overall) / rgbd_overall) * 100 if rgbd_overall > 0 else 0
                    chart_lines.append(f"Canny overall improvement: {improvement:.1f}%")
                else:
                    improvement = ((canny_overall - rgbd_overall) / canny_overall) * 100 if canny_overall > 0 else 0
                    chart_lines.append(f"RGBD overall improvement: {improvement:.1f}%")
            else:  # point_count
                if rgbd_overall > canny_overall:
                    improvement = ((rgbd_overall - canny_overall) / canny_overall) * 100 if canny_overall > 0 else 0
                    chart_lines.append(f"RGBD overall improvement: {improvement:.1f}%")
                else:
                    improvement = ((canny_overall - rgbd_overall) / rgbd_overall) * 100 if rgbd_overall > 0 else 0
                    chart_lines.append(f"Canny overall improvement: {improvement:.1f}%")
        
        # Save chart
        chart_file = output_dir / f"{metric}_comparison_chart.txt"
        with open(chart_file, 'w') as f:
            f.write('\n'.join(chart_lines))
        
        print(f"  ✅ Chart saved: {chart_file}")
    
    # Create combined summary chart
    print("Creating combined summary chart...")
    create_combined_summary_chart(data, core_metrics, output_dir)
    
    print(f"\n✅ All core comparison charts created in: {output_dir}")

def create_combined_summary_chart(data, core_metrics, output_dir):
    """Create a combined summary chart showing all metrics"""
    
    chart_lines = []
    chart_lines.append("=" * 100)
    chart_lines.append("CORE METRICS COMPARISON SUMMARY")
    chart_lines.append("Canny vs RGBD across All Lighting Conditions (Merged Data)")
    chart_lines.append("=" * 100)
    chart_lines.append("")
    
    # Create summary table
    chart_lines.append("SUMMARY TABLE (Mean ± Std)")
    chart_lines.append("-" * 100)
    
    # Header
    header = f"{'Metric':<25} {'Canny':<20} {'RGBD':<20} {'Winner':<10} {'Improvement':<15}"
    chart_lines.append(header)
    chart_lines.append("-" * 100)
    
    # Data rows
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    for metric, metric_name in core_metrics.items():
        # Calculate overall statistics
        canny_all = [float(row[metric]) for row in data 
                    if row['method'] == 'canny' and row[metric] != '']
        rgbd_all = [float(row[metric]) for row in data 
                   if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_all and rgbd_all:
            canny_mean = sum(canny_all) / len(canny_all)
            canny_std = (sum((x - canny_mean) ** 2 for x in canny_all) / len(canny_all)) ** 0.5
            
            rgbd_mean = sum(rgbd_all) / len(rgbd_all)
            rgbd_std = (sum((x - rgbd_mean) ** 2 for x in rgbd_all) / len(rgbd_all)) ** 0.5
            
            # Determine winner
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                winner = "Canny" if canny_mean < rgbd_mean else "RGBD"
            else:  # point_count
                winner = "Canny" if canny_mean > rgbd_mean else "RGBD"
            
            # Calculate improvement
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                if canny_mean < rgbd_mean:
                    improvement = ((rgbd_mean - canny_mean) / rgbd_mean) * 100 if rgbd_mean > 0 else 0
                    improvement_str = f"Canny +{improvement:.1f}%"
                else:
                    improvement = ((canny_mean - rgbd_mean) / canny_mean) * 100 if canny_mean > 0 else 0
                    improvement_str = f"RGBD +{improvement:.1f}%"
            else:  # point_count
                if rgbd_mean > canny_mean:
                    improvement = ((rgbd_mean - canny_mean) / canny_mean) * 100 if canny_mean > 0 else 0
                    improvement_str = f"RGBD +{improvement:.1f}%"
                else:
                    improvement = ((canny_mean - rgbd_mean) / rgbd_mean) * 100 if rgbd_mean > 0 else 0
                    improvement_str = f"Canny +{improvement:.1f}%"
            
            row = f"{metric_name:<25} {canny_mean:.3f}±{canny_std:.3f}    {rgbd_mean:.3f}±{rgbd_std:.3f}    {winner:<10} {improvement_str:<15}"
            chart_lines.append(row)
    
    # Add performance summary
    chart_lines.append("\n" + "=" * 100)
    chart_lines.append("PERFORMANCE SUMMARY")
    chart_lines.append("=" * 100)
    
    # Count wins
    canny_wins = 0
    rgbd_wins = 0
    
    for metric in core_metrics.keys():
        canny_all = [float(row[metric]) for row in data 
                    if row['method'] == 'canny' and row[metric] != '']
        rgbd_all = [float(row[metric]) for row in data 
                   if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_all and rgbd_all:
            canny_mean = sum(canny_all) / len(canny_all)
            rgbd_mean = sum(rgbd_all) / len(rgbd_all)
            
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                if canny_mean < rgbd_mean:
                    canny_wins += 1
                else:
                    rgbd_wins += 1
            else:  # point_count
                if canny_mean > rgbd_mean:
                    canny_wins += 1
                else:
                    rgbd_wins += 1
    
    chart_lines.append(f"Canny wins: {canny_wins} metrics")
    chart_lines.append(f"RGBD wins:  {rgbd_wins} metrics")
    
    if rgbd_wins > canny_wins:
        chart_lines.append("\n🏆 RECOMMENDATION: RGBD method")
        chart_lines.append("   RGBD performs better in most core metrics")
    elif canny_wins > rgbd_wins:
        chart_lines.append("\n🏆 RECOMMENDATION: Canny method")
        chart_lines.append("   Canny performs better in most core metrics")
    else:
        chart_lines.append("\n🤝 RECOMMENDATION: Both methods are comparable")
        chart_lines.append("   Consider specific use case requirements")
    
    # Save combined chart
    combined_file = output_dir / "core_comparison_summary.txt"
    with open(combined_file, 'w') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"  ✅ Combined summary saved: {combined_file}")

if __name__ == "__main__":
    create_core_comparison_chart()
