#!/usr/bin/env python3
"""
Create Core Metrics Charts for Boundary Tracking Evaluation
Focus on: Chamfer Distance, Hausdorff Distance, Std Residual, CV Residual, Execution Time, Point Count
"""

import csv
import json
from pathlib import Path

def create_simple_charts():
    """Create simple text-based charts for core metrics"""
    
    # Load data
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Core metrics
    core_metrics = {
        'chamfer_distance_mm': 'Chamfer Distance (mm)',
        'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
        'std_residual_mm': 'Standard Residual (mm)',
        'cv_residual': 'Coefficient of Variation',
        'execution_time_s': 'Execution Time (s)',
        'point_count': 'Point Count'
    }
    
    objects = ['Cube', 'ctry.obj']
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    # Create output directory
    output_dir = Path("evaluation_results/core_metrics_charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create charts for each metric
    for metric, metric_name in core_metrics.items():
        print(f"Creating chart for {metric_name}...")
        
        chart_lines = []
        chart_lines.append("=" * 80)
        chart_lines.append(f"CHART: {metric_name.upper()}")
        chart_lines.append("=" * 80)
        chart_lines.append("")
        
        # Create bar chart for each object and lighting condition
        for obj in objects:
            chart_lines.append(f"OBJECT: {obj}")
            chart_lines.append("-" * 40)
            
            for lighting in lighting_conditions:
                chart_lines.append(f"\n{lighting.upper()} Lighting:")
                
                # Get data for this combination
                canny_data = [float(row[metric]) for row in data 
                             if row['object_name'] == obj and 
                                row['lighting'] == lighting and 
                                row['method'] == 'canny' and 
                                row[metric] != '']
                
                rgbd_data = [float(row[metric]) for row in data 
                            if row['object_name'] == obj and 
                               row['lighting'] == lighting and 
                               row['method'] == 'rgbd' and 
                               row[metric] != '']
                
                if canny_data and rgbd_data:
                    canny_mean = sum(canny_data) / len(canny_data)
                    rgbd_mean = sum(rgbd_data) / len(rgbd_data)
                    
                    # Create simple bar chart
                    max_val = max(canny_mean, rgbd_mean)
                    scale = 50  # Scale for bar length
                    
                    canny_bar_length = int((canny_mean / max_val) * scale) if max_val > 0 else 0
                    rgbd_bar_length = int((rgbd_mean / max_val) * scale) if max_val > 0 else 0
                    
                    chart_lines.append(f"  Canny: {canny_mean:.3f} {'█' * canny_bar_length}")
                    chart_lines.append(f"  RGBD:  {rgbd_mean:.3f} {'█' * rgbd_bar_length}")
                    
                    # Show difference
                    diff = rgbd_mean - canny_mean
                    if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                                 'cv_residual', 'execution_time_s']:
                        winner = "Canny" if canny_mean < rgbd_mean else "RGBD"
                        better = "lower" if canny_mean < rgbd_mean else "higher"
                    else:  # point_count
                        winner = "Canny" if canny_mean > rgbd_mean else "RGBD"
                        better = "higher" if canny_mean > rgbd_mean else "lower"
                    
                    chart_lines.append(f"  Winner: {winner} ({better} is better)")
                    chart_lines.append(f"  Difference: {abs(diff):.3f}")
                else:
                    chart_lines.append("  No data available")
        
        # Add summary
        chart_lines.append("\n" + "=" * 80)
        chart_lines.append("SUMMARY")
        chart_lines.append("=" * 80)
        
        # Calculate overall statistics
        canny_all = [float(row[metric]) for row in data 
                    if row['method'] == 'canny' and row[metric] != '']
        rgbd_all = [float(row[metric]) for row in data 
                   if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_all and rgbd_all:
            canny_overall = sum(canny_all) / len(canny_all)
            rgbd_overall = sum(rgbd_all) / len(rgbd_all)
            
            chart_lines.append(f"Overall Canny: {canny_overall:.3f}")
            chart_lines.append(f"Overall RGBD:  {rgbd_overall:.3f}")
            
            # Determine overall winner
            if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                         'cv_residual', 'execution_time_s']:
                overall_winner = "Canny" if canny_overall < rgbd_overall else "RGBD"
                overall_better = "lower" if canny_overall < rgbd_overall else "higher"
            else:  # point_count
                overall_winner = "Canny" if canny_overall > rgbd_overall else "RGBD"
                overall_better = "higher" if canny_overall > rgbd_overall else "lower"
            
            chart_lines.append(f"Overall Winner: {overall_winner} ({overall_better} is better)")
            
            overall_diff = rgbd_overall - canny_overall
            chart_lines.append(f"Overall Difference: {abs(overall_diff):.3f}")
        
        # Save chart
        chart_file = output_dir / f"{metric}_chart.txt"
        with open(chart_file, 'w') as f:
            f.write('\n'.join(chart_lines))
        
        print(f"  ✅ Chart saved: {chart_file}")
    
    # Create combined comparison chart
    print("Creating combined comparison chart...")
    create_combined_comparison_chart(data, core_metrics, output_dir)
    
    print(f"\n✅ All charts created in: {output_dir}")

def create_combined_comparison_chart(data, core_metrics, output_dir):
    """Create a combined comparison chart"""
    
    chart_lines = []
    chart_lines.append("=" * 100)
    chart_lines.append("COMBINED CORE METRICS COMPARISON")
    chart_lines.append("=" * 100)
    chart_lines.append("")
    
    # Create comparison table
    chart_lines.append("COMPARISON TABLE (Mean Values)")
    chart_lines.append("-" * 100)
    
    # Header
    header = f"{'Object':<10} {'Lighting':<8} {'Method':<6} "
    for metric in core_metrics.keys():
        header += f"{metric[:8]:<8} "
    chart_lines.append(header)
    chart_lines.append("-" * 100)
    
    # Data rows
    objects = ['Cube', 'ctry.obj']
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    for obj in objects:
        for lighting in lighting_conditions:
            for method in methods:
                # Get data for this combination
                obj_data = [row for row in data 
                          if row['object_name'] == obj and 
                             row['lighting'] == lighting and 
                             row['method'] == method]
                
                if obj_data:
                    row = f"{obj:<10} {lighting:<8} {method.upper():<6} "
                    
                    for metric in core_metrics.keys():
                        values = [float(row[metric]) for row in obj_data if row[metric] != '']
                        if values:
                            mean_val = sum(values) / len(values)
                            row += f"{mean_val:<8.3f} "
                        else:
                            row += f"{'N/A':<8} "
                    
                    chart_lines.append(row)
    
    # Add performance summary
    chart_lines.append("\n" + "=" * 100)
    chart_lines.append("PERFORMANCE SUMMARY")
    chart_lines.append("=" * 100)
    
    # Calculate overall performance
    for metric, metric_name in core_metrics.items():
        chart_lines.append(f"\n{metric_name}:")
        
        canny_data = [float(row[metric]) for row in data 
                     if row['method'] == 'canny' and row[metric] != '']
        rgbd_data = [float(row[metric]) for row in data 
                    if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_data and rgbd_data:
            canny_mean = sum(canny_data) / len(canny_data)
            rgbd_mean = sum(rgbd_data) / len(rgbd_data)
            
            chart_lines.append(f"  Canny: {canny_mean:.3f}")
            chart_lines.append(f"  RGBD:  {rgbd_mean:.3f}")
            
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
                # For these metrics, lower is better
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
    
    # Overall recommendation
    chart_lines.append("\n" + "=" * 100)
    chart_lines.append("OVERALL RECOMMENDATION")
    chart_lines.append("=" * 100)
    
    # Count wins for each method
    canny_wins = 0
    rgbd_wins = 0
    
    for metric in core_metrics.keys():
        canny_data = [float(row[metric]) for row in data 
                     if row['method'] == 'canny' and row[metric] != '']
        rgbd_data = [float(row[metric]) for row in data 
                    if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_data and rgbd_data:
            canny_mean = sum(canny_data) / len(canny_data)
            rgbd_mean = sum(rgbd_data) / len(rgbd_data)
            
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
    combined_file = output_dir / "combined_comparison_chart.txt"
    with open(combined_file, 'w') as f:
        f.write('\n'.join(chart_lines))
    
    print(f"  ✅ Combined chart saved: {combined_file}")

if __name__ == "__main__":
    create_simple_charts()
