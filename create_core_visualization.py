#!/usr/bin/env python3
"""
Create Core Visualization Charts - Focus on Key Metrics
Chamfer Distance, Hausdorff Distance, Std Residual, CV Residual, Execution Time, Point Count
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set matplotlib style
plt.style.use('default')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11

def create_core_visualization():
    """Create core visualization charts for key metrics"""
    
    # Load data
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Core metrics - focus on the most important ones (reordered)
    core_metrics = {
        'chamfer_distance_mm': 'Chamfer Distance (mm)',
        'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
        'execution_time_s': 'Execution Time (s)',
        'std_residual_mm': 'Standard Residual (mm)',
        'cv_residual': 'Coefficient of Variation',
        'point_count': 'Point Count'
    }
    
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    # Create output directory
    output_dir = Path("evaluation_results/core_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Colors for methods - updated color scheme
    colors = {'canny': '#FF983E', 'rgbd': '#4C92C3'}  # Orange and Blue
    
    # Create the main comparison chart
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(core_metrics.items()):
        ax = axes[i]
        
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
        
        # Create bars
        x = np.arange(len(lighting_conditions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, canny_data, width, label='Canny', 
                      color=colors['canny'], alpha=0.8, yerr=canny_std, 
                      capsize=5, error_kw={'linewidth': 2})
        
        bars2 = ax.bar(x + width/2, rgbd_data, width, label='RGBD', 
                      color=colors['rgbd'], alpha=0.8, yerr=rgbd_std, 
                      capsize=5, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        max_val = max(max(canny_data) if canny_data else 0, max(rgbd_data) if rgbd_data else 0)
        for j, (canny_val, rgbd_val) in enumerate(zip(canny_data, rgbd_data)):
            if canny_val > 0:
                ax.text(j - width/2, canny_val + canny_std[j] + max_val*0.02, 
                       f'{canny_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            if rgbd_val > 0:
                ax.text(j + width/2, rgbd_val + rgbd_std[j] + max_val*0.02, 
                       f'{rgbd_val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize the plot
        ax.set_xlabel('Lighting Conditions', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in lighting_conditions])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Clean up the plot - remove annotations for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Core Metrics Comparison: Canny vs RGBD across Lighting Conditions', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the main chart
    main_chart_file = output_dir / "core_metrics_comparison.png"
    plt.savefig(main_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Main comparison chart saved: {main_chart_file}")
    
    # Create individual charts for the most important metrics
    important_metrics = ['chamfer_distance_mm', 'hausdorff_distance_mm', 'execution_time_s']
    
    for metric in important_metrics:
        metric_name = core_metrics[metric]
        print(f"Creating individual chart for {metric_name}...")
        
        # Create individual chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        canny_data = []
        rgbd_data = []
        canny_std = []
        rgbd_std = []
        
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
        
        # Create bars
        x = np.arange(len(lighting_conditions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, canny_data, width, label='Canny', 
                      color=colors['canny'], alpha=0.8, yerr=canny_std, 
                      capsize=8, error_kw={'linewidth': 3})
        
        bars2 = ax.bar(x + width/2, rgbd_data, width, label='RGBD', 
                      color=colors['rgbd'], alpha=0.8, yerr=rgbd_std, 
                      capsize=8, error_kw={'linewidth': 3})
        
        # Add value labels
        max_val = max(max(canny_data) if canny_data else 0, max(rgbd_data) if rgbd_data else 0)
        for j, (canny_val, rgbd_val) in enumerate(zip(canny_data, rgbd_data)):
            if canny_val > 0:
                ax.text(j - width/2, canny_val + canny_std[j] + max_val*0.03, 
                       f'{canny_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            if rgbd_val > 0:
                ax.text(j + width/2, rgbd_val + rgbd_std[j] + max_val*0.03, 
                       f'{rgbd_val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Customize
        ax.set_xlabel('Lighting Conditions', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in lighting_conditions], fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Clean up the plot - remove annotations for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual chart
        individual_file = output_dir / f"{metric}_detailed.png"
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Individual chart saved: {individual_file}")
    
    print(f"\n✅ All core visualization charts created in: {output_dir}")

if __name__ == "__main__":
    create_core_visualization()
