#!/usr/bin/env python3
"""
Create Final Comparison Chart - Clean and Professional
Focus on the most important metrics for decision making
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set professional style
plt.style.use('default')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def create_final_comparison():
    """Create the final, clean comparison chart"""
    
    # Load data
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # Focus on the most critical metrics
    critical_metrics = {
        'chamfer_distance_mm': 'Chamfer Distance (mm)',
        'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
        'execution_time_s': 'Execution Time (s)',
        'point_count': 'Point Count'
    }
    
    lighting_conditions = ['normal', 'bright', 'dim']
    methods = ['canny', 'rgbd']
    
    # Create output directory
    output_dir = Path("evaluation_results/final_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Professional colors
    colors = {'canny': '#1f77b4', 'rgbd': '#ff7f0e'}
    
    # Create the main comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(critical_metrics.items()):
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
                      capsize=6, error_kw={'linewidth': 2})
        
        bars2 = ax.bar(x + width/2, rgbd_data, width, label='RGBD', 
                      color=colors['rgbd'], alpha=0.8, yerr=rgbd_std, 
                      capsize=6, error_kw={'linewidth': 2})
        
        # Add value labels on bars
        max_val = max(max(canny_data) if canny_data else 0, max(rgbd_data) if rgbd_data else 0)
        for j, (canny_val, rgbd_val) in enumerate(zip(canny_data, rgbd_data)):
            if canny_val > 0:
                ax.text(j - width/2, canny_val + canny_std[j] + max_val*0.02, 
                       f'{canny_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            if rgbd_val > 0:
                ax.text(j + width/2, rgbd_val + rgbd_std[j] + max_val*0.02, 
                       f'{rgbd_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Customize the plot
        ax.set_xlabel('Lighting Conditions', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontweight='bold', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in lighting_conditions])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Determine if lower or higher is better
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'execution_time_s']:
            better_text = "Lower is better"
            better_color = 'lightgreen'
        else:  # point_count
            better_text = "Higher is better"
            better_color = 'lightblue'
        
        # Add better/worse indicator
        ax.text(0.02, 0.98, better_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=better_color, alpha=0.8))
        
        # Add winner information
        canny_overall = np.mean([float(row[metric]) for row in data 
                               if row['method'] == 'canny' and row[metric] != ''])
        rgbd_overall = np.mean([float(row[metric]) for row in data 
                              if row['method'] == 'rgbd' and row[metric] != ''])
        
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'execution_time_s']:
            winner = "Canny" if canny_overall < rgbd_overall else "RGBD"
        else:
            winner = "Canny" if canny_overall > rgbd_overall else "RGBD"
        
        winner_color = 'lightgreen' if winner == 'RGBD' else 'lightcoral'
        ax.text(0.98, 0.98, f'Winner: {winner}', transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', horizontalalignment='right', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.8))
    
    plt.suptitle('Boundary Tracking Method Comparison: Canny vs RGBD', 
                fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # Save the main chart
    main_chart_file = output_dir / "final_comparison.png"
    plt.savefig(main_chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Final comparison chart saved: {main_chart_file}")
    
    # Create a summary chart with just the key metrics
    create_summary_chart(data, critical_metrics, output_dir, colors)
    
    print(f"\n✅ All final comparison charts created in: {output_dir}")

def create_summary_chart(data, critical_metrics, output_dir, colors):
    """Create a summary chart with key insights"""
    
    # Calculate overall statistics
    summary_data = {}
    
    for metric, metric_name in critical_metrics.items():
        canny_all = [float(row[metric]) for row in data 
                    if row['method'] == 'canny' and row[metric] != '']
        rgbd_all = [float(row[metric]) for row in data 
                   if row['method'] == 'rgbd' and row[metric] != '']
        
        if canny_all and rgbd_all:
            canny_mean = np.mean(canny_all)
            canny_std = np.std(canny_all)
            rgbd_mean = np.mean(rgbd_all)
            rgbd_std = np.std(rgbd_all)
            
            summary_data[metric] = {
                'canny_mean': canny_mean,
                'canny_std': canny_std,
                'rgbd_mean': rgbd_mean,
                'rgbd_std': rgbd_std,
                'name': metric_name
            }
    
    # Create summary chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics = list(critical_metrics.keys())
    canny_means = [summary_data[metric]['canny_mean'] for metric in metrics]
    rgbd_means = [summary_data[metric]['rgbd_mean'] for metric in metrics]
    canny_stds = [summary_data[metric]['canny_std'] for metric in metrics]
    rgbd_stds = [summary_data[metric]['rgbd_std'] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, canny_means, width, label='Canny', 
                  color=colors['canny'], alpha=0.8, yerr=canny_stds, 
                  capsize=6, error_kw={'linewidth': 2})
    
    bars2 = ax.bar(x + width/2, rgbd_means, width, label='RGBD', 
                  color=colors['rgbd'], alpha=0.8, yerr=rgbd_stds, 
                  capsize=6, error_kw={'linewidth': 2})
    
    # Add value labels
    max_val = max(max(canny_means), max(rgbd_means))
    for i, (canny_val, rgbd_val) in enumerate(zip(canny_means, rgbd_means)):
        ax.text(i - width/2, canny_val + canny_stds[i] + max_val*0.02, 
               f'{canny_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i + width/2, rgbd_val + rgbd_stds[i] + max_val*0.02, 
               f'{rgbd_val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Customize
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Values', fontweight='bold')
    ax.set_title('Overall Performance Comparison: Canny vs RGBD', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([critical_metrics[metric] for metric in metrics], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add winner summary
    canny_wins = 0
    rgbd_wins = 0
    
    for metric in critical_metrics.keys():
        if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 'execution_time_s']:
            if summary_data[metric]['canny_mean'] < summary_data[metric]['rgbd_mean']:
                canny_wins += 1
            else:
                rgbd_wins += 1
        else:  # point_count
            if summary_data[metric]['canny_mean'] > summary_data[metric]['rgbd_mean']:
                canny_wins += 1
            else:
                rgbd_wins += 1
    
    winner_text = f"RGBD wins {rgbd_wins} metrics, Canny wins {canny_wins} metrics"
    winner_color = 'lightgreen' if rgbd_wins > canny_wins else 'lightcoral'
    
    ax.text(0.5, 0.98, winner_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', horizontalalignment='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save summary chart
    summary_file = output_dir / "summary_comparison.png"
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Summary chart saved: {summary_file}")

if __name__ == "__main__":
    create_final_comparison()

