#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualizations for Boundary Detection Analysis
Multiple chart types for better presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_advanced_visualizations():
    """Create advanced visualization charts"""
    
    # Real experimental data
    real_data = {
        'normal': {
            'canny': {
                'chamfer_distance': 3.18,
                'chamfer_std': 0.51,
                'hausdorff_distance': 18.97,
                'hausdorff_std': 1.99,
                'execution_time': 17.88,
                'execution_std': 12.48,
                'coverage_3mm': 1.0,
                'point_count': 3707
            },
            'rgbd': {
                'chamfer_distance': 6.07,
                'chamfer_std': 0.98,
                'hausdorff_distance': 18.79,
                'hausdorff_std': 6.49,
                'execution_time': 91.09,
                'execution_std': 81.35,
                'coverage_3mm': 1.0,
                'point_count': 7474
            }
        },
        'bright': {
            'canny': {
                'chamfer_distance': 2.83,
                'chamfer_std': 0.50,
                'hausdorff_distance': 17.98,
                'hausdorff_std': 2.16,
                'execution_time': 192.17,
                'execution_std': 298.60,
                'coverage_3mm': 1.0,
                'point_count': 3777
            },
            'rgbd': {
                'chamfer_distance': 6.69,
                'chamfer_std': 2.13,
                'hausdorff_distance': 21.71,
                'hausdorff_std': 1.71,
                'execution_time': 48.20,
                'execution_std': 15.87,
                'coverage_3mm': 1.0,
                'point_count': 7132
            }
        },
        'dim': {
            'canny': {
                'chamfer_distance': 3.12,
                'chamfer_std': 0.72,
                'hausdorff_distance': 17.74,
                'hausdorff_std': 2.87,
                'execution_time': 251.31,
                'execution_std': 390.67,
                'coverage_3mm': 1.0,
                'point_count': 3707
            },
            'rgbd': {
                'chamfer_distance': 7.18,
                'chamfer_std': 2.77,
                'hausdorff_distance': 20.13,
                'hausdorff_std': 3.86,
                'execution_time': 13.91,
                'execution_std': 1.39,
                'coverage_3mm': 1.0,
                'point_count': 6015
            }
        }
    }
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create different types of charts
    create_radar_chart(real_data)
    create_heatmap_comparison(real_data)
    create_line_chart_with_shading(real_data)
    create_bubble_chart(real_data)
    create_stacked_bar_chart(real_data)
    create_parallel_coordinates(real_data)
    
    print("Advanced visualizations created successfully!")

def create_radar_chart(data):
    """Create radar chart for comprehensive comparison"""
    
    # Prepare data for radar chart
    categories = ['Accuracy\n(1/Chamfer)', 'Robustness\n(1/Hausdorff)', 'Speed\n(1/Time)', 
                  'Coverage\n(3mm)', 'Point\nDensity']
    
    # Normalize data for radar chart (0-1 scale)
    canny_normal = [
        1.0 / (1.0 + data['normal']['canny']['chamfer_distance'] / 10),  # Accuracy
        1.0 / (1.0 + data['normal']['canny']['hausdorff_distance'] / 50),  # Robustness
        1.0 / (1.0 + data['normal']['canny']['execution_time'] / 300),  # Speed
        data['normal']['canny']['coverage_3mm'],  # Coverage
        min(1.0, data['normal']['canny']['point_count'] / 10000)  # Point density
    ]
    
    rgbd_normal = [
        1.0 / (1.0 + data['normal']['rgbd']['chamfer_distance'] / 10),
        1.0 / (1.0 + data['normal']['rgbd']['hausdorff_distance'] / 50),
        1.0 / (1.0 + data['normal']['rgbd']['execution_time'] / 300),
        data['normal']['rgbd']['coverage_3mm'],
        min(1.0, data['normal']['rgbd']['point_count'] / 10000)
    ]
    
    # Create radar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    canny_normal += canny_normal[:1]
    rgbd_normal += rgbd_normal[:1]
    
    ax.plot(angles, canny_normal, 'o-', linewidth=2, label='Canny', color='#1f77b4', alpha=0.8)
    ax.fill(angles, canny_normal, alpha=0.1, color='#1f77b4')
    
    ax.plot(angles, rgbd_normal, 'o-', linewidth=2, label='RGB-D', color='#ff7f0e', alpha=0.8)
    ax.fill(angles, rgbd_normal, alpha=0.1, color='#ff7f0e')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart (Normal Lighting)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('Radar_Chart_Performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap_comparison(data):
    """Create heatmap for comprehensive comparison"""
    
    # Prepare data matrix
    metrics = ['Chamfer\n(mm)', 'Hausdorff\n(mm)', 'Time\n(s)', 'Coverage\n(%)', 'Points']
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    
    # Create data matrix for Canny
    canny_matrix = np.zeros((len(metrics), len(lighting_conditions)))
    rgbd_matrix = np.zeros((len(metrics), len(lighting_conditions)))
    
    for i, lighting in enumerate(['normal', 'bright', 'dim']):
        canny_matrix[0, i] = data[lighting]['canny']['chamfer_distance']
        canny_matrix[1, i] = data[lighting]['canny']['hausdorff_distance']
        canny_matrix[2, i] = data[lighting]['canny']['execution_time']
        canny_matrix[3, i] = data[lighting]['canny']['coverage_3mm'] * 100
        canny_matrix[4, i] = data[lighting]['canny']['point_count'] / 1000  # Convert to thousands
        
        rgbd_matrix[0, i] = data[lighting]['rgbd']['chamfer_distance']
        rgbd_matrix[1, i] = data[lighting]['rgbd']['hausdorff_distance']
        rgbd_matrix[2, i] = data[lighting]['rgbd']['execution_time']
        rgbd_matrix[3, i] = data[lighting]['rgbd']['coverage_3mm'] * 100
        rgbd_matrix[4, i] = data[lighting]['rgbd']['point_count'] / 1000
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Canny heatmap
    sns.heatmap(canny_matrix, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=lighting_conditions, yticklabels=metrics, ax=ax1)
    ax1.set_title('Canny Method Performance Heatmap', fontsize=14, fontweight='bold')
    
    # RGB-D heatmap
    sns.heatmap(rgbd_matrix, annot=True, fmt='.1f', cmap='Oranges', 
                xticklabels=lighting_conditions, yticklabels=metrics, ax=ax2)
    ax2.set_title('RGB-D Method Performance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Performance_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_line_chart_with_shading(data):
    """Create line chart with confidence intervals"""
    
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    
    # Extract data
    canny_chamfer = [data[light.lower()]['canny']['chamfer_distance'] for light in lighting_conditions]
    canny_chamfer_std = [data[light.lower()]['canny']['chamfer_std'] for light in lighting_conditions]
    rgbd_chamfer = [data[light.lower()]['rgbd']['chamfer_distance'] for light in lighting_conditions]
    rgbd_chamfer_std = [data[light.lower()]['rgbd']['chamfer_std'] for light in lighting_conditions]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot lines with confidence intervals
    ax.plot(x, canny_chamfer, 'o-', linewidth=3, label='Canny', color='#1f77b4', markersize=8)
    ax.fill_between(x, 
                   np.array(canny_chamfer) - np.array(canny_chamfer_std),
                   np.array(canny_chamfer) + np.array(canny_chamfer_std),
                   alpha=0.3, color='#1f77b4')
    
    ax.plot(x, rgbd_chamfer, 's-', linewidth=3, label='RGB-D', color='#ff7f0e', markersize=8)
    ax.fill_between(x, 
                   np.array(rgbd_chamfer) - np.array(rgbd_chamfer_std),
                   np.array(rgbd_chamfer) + np.array(rgbd_chamfer_std),
                   alpha=0.3, color='#ff7f0e')
    
    ax.set_title('Chamfer Distance Trends with Confidence Intervals', fontsize=16, fontweight='bold')
    ax.set_xlabel('Lighting Conditions', fontsize=14)
    ax.set_ylabel('Chamfer Distance (mm)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(lighting_conditions)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (canny_val, rgbd_val) in enumerate(zip(canny_chamfer, rgbd_chamfer)):
        ax.annotate(f'{canny_val:.2f}', (i, canny_val), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=10)
        ax.annotate(f'{rgbd_val:.2f}', (i, rgbd_val), textcoords="offset points", 
                   xytext=(0,-15), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('Line_Chart_with_Confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bubble_chart(data):
    """Create bubble chart showing multiple dimensions"""
    
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    
    # Prepare data for bubble chart
    # X: Chamfer distance, Y: Execution time, Size: Point count, Color: Method
    canny_x = [data[light.lower()]['canny']['chamfer_distance'] for light in lighting_conditions]
    canny_y = [data[light.lower()]['canny']['execution_time'] for light in lighting_conditions]
    canny_size = [data[light.lower()]['canny']['point_count'] / 100 for light in lighting_conditions]
    
    rgbd_x = [data[light.lower()]['rgbd']['chamfer_distance'] for light in lighting_conditions]
    rgbd_y = [data[light.lower()]['rgbd']['execution_time'] for light in lighting_conditions]
    rgbd_size = [data[light.lower()]['rgbd']['point_count'] / 100 for light in lighting_conditions]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create bubble chart
    for i, lighting in enumerate(lighting_conditions):
        # Canny bubbles
        ax.scatter(canny_x[i], canny_y[i], s=canny_size[i], 
                  c='#1f77b4', alpha=0.7, label=f'Canny-{lighting}' if i == 0 else "")
        ax.annotate(f'Canny\n{lighting}', (canny_x[i], canny_y[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # RGB-D bubbles
        ax.scatter(rgbd_x[i], rgbd_y[i], s=rgbd_size[i], 
                  c='#ff7f0e', alpha=0.7, label=f'RGB-D-{lighting}' if i == 0 else "")
        ax.annotate(f'RGB-D\n{lighting}', (rgbd_x[i], rgbd_y[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Chamfer Distance (mm)', fontsize=14)
    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('Multi-Dimensional Performance Comparison\n(Bubble size = Point count/100)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Circle
    legend_elements = [Circle((0,0), 1, color='#1f77b4', alpha=0.7, label='Canny'),
                      Circle((0,0), 1, color='#ff7f0e', alpha=0.7, label='RGB-D')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Bubble_Chart_Performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_stacked_bar_chart(data):
    """Create stacked bar chart showing multiple metrics"""
    
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    width = 0.35
    
    # Normalize metrics for stacking (0-1 scale)
    def normalize_metric(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    # Calculate normalized values
    canny_accuracy = [normalize_metric(data[light.lower()]['canny']['chamfer_distance'], 2, 8) for light in lighting_conditions]
    canny_speed = [normalize_metric(data[light.lower()]['canny']['execution_time'], 0, 300) for light in lighting_conditions]
    canny_coverage = [data[light.lower()]['canny']['coverage_3mm'] for light in lighting_conditions]
    
    rgbd_accuracy = [normalize_metric(data[light.lower()]['rgbd']['chamfer_distance'], 2, 8) for light in lighting_conditions]
    rgbd_speed = [normalize_metric(data[light.lower()]['rgbd']['execution_time'], 0, 300) for light in lighting_conditions]
    rgbd_coverage = [data[light.lower()]['rgbd']['coverage_3mm'] for light in lighting_conditions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Canny stacked bars
    ax1.bar(x - width/2, canny_accuracy, width, label='Accuracy', color='#1f77b4', alpha=0.8)
    ax1.bar(x - width/2, canny_speed, width, bottom=canny_accuracy, label='Speed', color='#ff7f0e', alpha=0.8)
    ax1.bar(x - width/2, canny_coverage, width, bottom=np.array(canny_accuracy) + np.array(canny_speed), 
            label='Coverage', color='#2ca02c', alpha=0.8)
    
    # RGB-D stacked bars
    ax2.bar(x + width/2, rgbd_accuracy, width, label='Accuracy', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, rgbd_speed, width, bottom=rgbd_accuracy, label='Speed', color='#ff7f0e', alpha=0.8)
    ax2.bar(x + width/2, rgbd_coverage, width, bottom=np.array(rgbd_accuracy) + np.array(rgbd_speed), 
            label='Coverage', color='#2ca02c', alpha=0.8)
    
    ax1.set_title('Canny Method - Normalized Performance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Lighting Conditions', fontsize=12)
    ax1.set_ylabel('Normalized Score', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lighting_conditions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('RGB-D Method - Normalized Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Lighting Conditions', fontsize=12)
    ax2.set_ylabel('Normalized Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lighting_conditions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Stacked_Bar_Chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_parallel_coordinates(data):
    """Create parallel coordinates plot"""
    
    # Prepare data for parallel coordinates
    metrics = ['Chamfer\n(mm)', 'Hausdorff\n(mm)', 'Time\n(s)', 'Coverage\n(%)', 'Points\n(k)']
    
    # Normalize data
    def normalize_data(values):
        min_val, max_val = min(values), max(values)
        return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
    
    # Extract and normalize all data
    all_chamfer = [data[light]['canny']['chamfer_distance'] for light in ['normal', 'bright', 'dim']] + \
                  [data[light]['rgbd']['chamfer_distance'] for light in ['normal', 'bright', 'dim']]
    all_hausdorff = [data[light]['canny']['hausdorff_distance'] for light in ['normal', 'bright', 'dim']] + \
                    [data[light]['rgbd']['hausdorff_distance'] for light in ['normal', 'bright', 'dim']]
    all_time = [data[light]['canny']['execution_time'] for light in ['normal', 'bright', 'dim']] + \
               [data[light]['rgbd']['execution_time'] for light in ['normal', 'bright', 'dim']]
    all_coverage = [data[light]['canny']['coverage_3mm'] * 100 for light in ['normal', 'bright', 'dim']] + \
                   [data[light]['rgbd']['coverage_3mm'] * 100 for light in ['normal', 'bright', 'dim']]
    all_points = [data[light]['canny']['point_count'] / 1000 for light in ['normal', 'bright', 'dim']] + \
                 [data[light]['rgbd']['point_count'] / 1000 for light in ['normal', 'bright', 'dim']]
    
    # Create data for plotting
    canny_data = np.array([
        normalize_data(all_chamfer)[:3],  # Canny values
        normalize_data(all_hausdorff)[:3],
        normalize_data(all_time)[:3],
        normalize_data(all_coverage)[:3],
        normalize_data(all_points)[:3]
    ]).T
    
    rgbd_data = np.array([
        normalize_data(all_chamfer)[3:],  # RGB-D values
        normalize_data(all_hausdorff)[3:],
        normalize_data(all_time)[3:],
        normalize_data(all_coverage)[3:],
        normalize_data(all_points)[3:]
    ]).T
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot parallel coordinates
    lighting_labels = ['Normal', 'Bright', 'Dim']
    
    for i, lighting in enumerate(lighting_labels):
        ax.plot(range(len(metrics)), canny_data[i], 'o-', linewidth=2, 
               label=f'Canny-{lighting}', color=f'C{i}', alpha=0.8)
        ax.plot(range(len(metrics)), rgbd_data[i], 's-', linewidth=2, 
               label=f'RGB-D-{lighting}', color=f'C{i}', alpha=0.8, linestyle='--')
    
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Parallel Coordinates Plot - Performance Comparison', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Parallel_Coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_advanced_visualizations()
    print("\nAdvanced visualizations created:")
    print("1. Radar_Chart_Performance.png - Radar chart showing all metrics")
    print("2. Performance_Heatmap.png - Heatmap comparison")
    print("3. Line_Chart_with_Confidence.png - Line chart with confidence intervals")
    print("4. Bubble_Chart_Performance.png - Bubble chart with multiple dimensions")
    print("5. Stacked_Bar_Chart.png - Stacked bar chart")
    print("6. Parallel_Coordinates.png - Parallel coordinates plot")



