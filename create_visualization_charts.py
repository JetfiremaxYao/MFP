#!/usr/bin/env python3
"""
Create Visualization Charts for Core Metrics Comparison
Canny vs RGBD across lighting conditions (merged data)
"""

import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def create_visualization_charts():
    """Create visualization charts for core metrics comparison"""
    
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
    output_dir = Path("evaluation_results/visualization_charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colors for methods
    colors = {'canny': '#1f77b4', 'rgbd': '#ff7f0e'}
    
    # Create individual charts for each metric
    for metric, metric_name in core_metrics.items():
        print(f"Creating visualization for {metric_name}...")
        
        # Prepare data for plotting
        canny_data = []
        rgbd_data = []
        canny_std = []
        rgbd_std = []
        
        for lighting in lighting_conditions:
            # Get data for this lighting condition
            canny_values = [float(row[metric]) for row in data 
                           if row['lighting'] == lighting and 
                              row['method'] == 'canny' and 
                              row[metric] != '']
            
            rgbd_values = [float(row[metric]) for row in data 
                          if row['lighting'] == lighting and 
                             row['method'] == 'rgbd' and 
                             row[metric] != '']
            
            if canny_values and rgbd_values:
                canny_mean = np.mean(canny_values)
                canny_std_val = np.std(canny_values)
                rgbd_mean = np.mean(rgbd_values)
                rgbd_std_val = np.std(rgbd_values)
                
                canny_data.append(canny_mean)
                rgbd_data.append(rgbd_mean)
                canny_std.append(canny_std_val)
                rgbd_std.append(rgbd_std_val)
            else:
                canny_data.append(0)
                rgbd_data.append(0)
                canny_std.append(0)
                rgbd_std.append(0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(lighting_conditions))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, canny_data, width, label='Canny', 
                      color=colors['canny'], alpha=0.8, yerr=canny_std, 
                      capsize=5, error_kw={'linewidth': 2})
        
        bars2 = ax.bar(x + width/2, rgbd_data, width, label='RGBD', 
                      color=colors['rgbd'], alpha=0.8, yerr=rgbd_std, 
                      capsize=5, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        max_val = max(max(canny_data) if canny_data else 0, max(rgbd_data) if rgbd_data else 0)
        for i, (canny_val, rgbd_val) in enumerate(zip(canny_data, rgbd_data)):
            if canny_val > 0:
                ax.text(i - width/2, canny_val + canny_std[i] + max_val*0.01, 
                       f'{canny_val:.2f}', ha='center', va='bottom', fontsize=8)
            if rgbd_val > 0:
                ax.text(i + width/2, rgbd_val + rgbd_std[i] + max_val*0.01, 
                       f'{rgbd_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Customize the plot
        ax.set_xlabel('Lighting Conditions', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'{metric_name} vs Lighting Conditions', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in lighting_conditions])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Determine if lower or higher is better
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                     'cv_residual', 'execution_time_s']:
            better_text = "Lower is better"
        else:  # point_count
            better_text = "Higher is better"
        
        # Add better/worse indicator
        ax.text(0.02, 0.98, better_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add winner information
        canny_overall = np.mean([float(row[metric]) for row in data 
                               if row['method'] == 'canny' and row[metric] != ''])
        rgbd_overall = np.mean([float(row[metric]) for row in data 
                              if row['method'] == 'rgbd' and row[metric] != ''])
        
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                     'cv_residual', 'execution_time_s']:
            winner = "Canny" if canny_overall < rgbd_overall else "RGBD"
        else:
            winner = "Canny" if canny_overall > rgbd_overall else "RGBD"
        
        ax.text(0.98, 0.98, f'Winner: {winner}', transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        chart_file = output_dir / f"{metric}_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Chart saved: {chart_file}")
    
    # Create combined comparison chart
    print("Creating combined comparison chart...")
    create_combined_chart(data, core_metrics, output_dir, colors)
    
    print(f"\n✅ All visualization charts created in: {output_dir}")

def create_combined_chart(data, core_metrics, output_dir, colors):
    """Create a combined chart showing all metrics"""
    
    # Prepare data for all metrics
    metrics_data = {}
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    for metric, metric_name in core_metrics.items():
        metrics_data[metric] = {
            'canny': {'means': [], 'stds': []},
            'rgbd': {'means': [], 'stds': []}
        }
        
        for lighting in lighting_conditions:
            canny_values = [float(row[metric]) for row in data 
                           if row['lighting'] == lighting and 
                              row['method'] == 'canny' and 
                              row[metric] != '']
            
            rgbd_values = [float(row[metric]) for row in data 
                          if row['lighting'] == lighting and 
                             row['method'] == 'rgbd' and 
                             row[metric] != '']
            
            if canny_values and rgbd_values:
                canny_mean = np.mean(canny_values)
                canny_std = np.std(canny_values)
                rgbd_mean = np.mean(rgbd_values)
                rgbd_std = np.std(rgbd_values)
            else:
                canny_mean = canny_std = rgbd_mean = rgbd_std = 0
            
            metrics_data[metric]['canny']['means'].append(canny_mean)
            metrics_data[metric]['canny']['stds'].append(canny_std)
            metrics_data[metric]['rgbd']['means'].append(rgbd_mean)
            metrics_data[metric]['rgbd']['stds'].append(rgbd_std)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(core_metrics.items()):
        ax = axes[i]
        
        x = np.arange(len(lighting_conditions))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, metrics_data[metric]['canny']['means'], width, 
                      label='Canny', color=colors['canny'], alpha=0.8, 
                      yerr=metrics_data[metric]['canny']['stds'], capsize=3)
        
        bars2 = ax.bar(x + width/2, metrics_data[metric]['rgbd']['means'], width, 
                      label='RGBD', color=colors['rgbd'], alpha=0.8, 
                      yerr=metrics_data[metric]['rgbd']['stds'], capsize=3)
        
        # Customize subplot
        ax.set_title(metric_name, fontweight='bold')
        ax.set_xlabel('Lighting Conditions')
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in lighting_conditions])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add better/worse indicator
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'std_residual_mm', 
                     'cv_residual', 'execution_time_s']:
            better_text = "Lower is better"
        else:
            better_text = "Higher is better"
        
        ax.text(0.02, 0.98, better_text, transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Core Metrics Comparison: Canny vs RGBD across Lighting Conditions', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save combined chart
    combined_file = output_dir / "combined_comparison.png"
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Combined chart saved: {combined_file}")

if __name__ == "__main__":
    create_visualization_charts()
