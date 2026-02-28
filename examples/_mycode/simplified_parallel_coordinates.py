#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Parallel Coordinates Plot
Showing Time, Coverage, Point Count, and Residual (Stability)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_simplified_parallel_coordinates():
    """Create simplified parallel coordinates plot with 4 key metrics including stability"""
    
    # Real experimental data with residual information
    real_data = {
        'normal': {
            'canny': {
                'execution_time': 17.88,
                'coverage_3mm': 1.0,
                'point_count': 3707,
                'std_residual': 4.27  # Standard deviation of residuals
            },
            'rgbd': {
                'execution_time': 91.09,
                'coverage_3mm': 1.0,
                'point_count': 7474,
                'std_residual': 4.46  # Standard deviation of residuals
            }
        },
        'bright': {
            'canny': {
                'execution_time': 192.17,
                'coverage_3mm': 1.0,
                'point_count': 3777,
                'std_residual': 3.87  # Standard deviation of residuals
            },
            'rgbd': {
                'execution_time': 48.20,
                'coverage_3mm': 1.0,
                'point_count': 7132,
                'std_residual': 5.08  # Standard deviation of residuals
            }
        },
        'dim': {
            'canny': {
                'execution_time': 251.31,
                'coverage_3mm': 1.0,
                'point_count': 3707,
                'std_residual': 4.28  # Standard deviation of residuals
            },
            'rgbd': {
                'execution_time': 13.91,
                'coverage_3mm': 1.0,
                'point_count': 6015,
                'std_residual': 4.49  # Standard deviation of residuals
            }
        }
    }
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create simplified parallel coordinates
    create_clean_parallel_coordinates(real_data)
    
    print("Simplified parallel coordinates with stability metrics created successfully!")

def create_clean_parallel_coordinates(data):
    """Create clean parallel coordinates plot with 4 metrics including stability"""
    
    # Prepare data for parallel coordinates - now with 4 metrics
    metrics = ['Execution\nTime (s)', 'Coverage\n(%)', 'Point\nCount (k)', 'Stability\n(1/Residual)']
    
    # Normalize data (0-1 scale)
    def normalize_data(values):
        min_val, max_val = min(values), max(values)
        return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
    
    # Extract data for the 4 metrics
    all_time = [data[light]['canny']['execution_time'] for light in ['normal', 'bright', 'dim']] + \
               [data[light]['rgbd']['execution_time'] for light in ['normal', 'bright', 'dim']]
    
    all_coverage = [data[light]['canny']['coverage_3mm'] * 100 for light in ['normal', 'bright', 'dim']] + \
                   [data[light]['rgbd']['coverage_3mm'] * 100 for light in ['normal', 'bright', 'dim']]
    
    all_points = [data[light]['canny']['point_count'] / 1000 for light in ['normal', 'bright', 'dim']] + \
                 [data[light]['rgbd']['point_count'] / 1000 for light in ['normal', 'bright', 'dim']]
    
    # For stability, we use 1/residual (higher is better, more stable)
    all_residuals = [data[light]['canny']['std_residual'] for light in ['normal', 'bright', 'dim']] + \
                    [data[light]['rgbd']['std_residual'] for light in ['normal', 'bright', 'dim']]
    all_stability = [1.0 / (1.0 + residual) for residual in all_residuals]  # Convert to 0-1 scale
    
    # Normalize all data
    normalized_time = normalize_data(all_time)
    normalized_coverage = normalize_data(all_coverage)
    normalized_points = normalize_data(all_points)
    normalized_stability = normalize_data(all_stability)
    
    # Create data arrays for plotting
    canny_data = np.array([
        normalized_time[:3],      # Canny time values
        normalized_coverage[:3],  # Canny coverage values
        normalized_points[:3],    # Canny point count values
        normalized_stability[:3]  # Canny stability values
    ]).T
    
    rgbd_data = np.array([
        normalized_time[3:],      # RGB-D time values
        normalized_coverage[3:],  # RGB-D coverage values
        normalized_points[3:],    # RGB-D point count values
        normalized_stability[3:]  # RGB-D stability values
    ]).T
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # Increased width for 4 metrics
    
    # Plot parallel coordinates with clear styling
    lighting_labels = ['Normal', 'Bright', 'Dim']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Plot Canny lines (solid lines)
    for i, lighting in enumerate(lighting_labels):
        ax.plot(range(len(metrics)), canny_data[i], 'o-', linewidth=3, 
               label=f'Canny-{lighting}', color=colors[i], alpha=0.8, markersize=8)
    
    # Plot RGB-D lines (dashed lines)
    for i, lighting in enumerate(lighting_labels):
        ax.plot(range(len(metrics)), rgbd_data[i], 's--', linewidth=3, 
               label=f'RGB-D-{lighting}', color=colors[i], alpha=0.8, markersize=8)
    
    # Customize the plot
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Value (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison\n(Time, Coverage, Point Count, Stability)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.set_ylim(-0.1, 1.1)
    
    # Add legend with better positioning
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, 
             frameon=True, fancybox=True, shadow=True)
    
    # Add key insights as text box below the title
    #     insight_text = """Key Insights:
    # • RGB-D: Generally faster execution
    # • Both methods: High coverage (100%)
    # • RGB-D: More points collected
    # • Canny: More consistent across conditions"""
    #     
    #     # Position the text box below the title, not overlapping data
    #     ax.text(0.5, 0.95, insight_text, transform=ax.transAxes, fontsize=11, 
    #             verticalalignment='top', horizontalalignment='center',
    #             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig('Simplified_Parallel_Coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_alternative_3d_scatter():
    """Create 3D scatter plot as alternative visualization"""
    
    # Real experimental data
    real_data = {
        'normal': {
            'canny': {'execution_time': 17.88, 'coverage_3mm': 1.0, 'point_count': 3707},
            'rgbd': {'execution_time': 91.09, 'coverage_3mm': 1.0, 'point_count': 7474}
        },
        'bright': {
            'canny': {'execution_time': 192.17, 'coverage_3mm': 1.0, 'point_count': 3777},
            'rgbd': {'execution_time': 48.20, 'coverage_3mm': 1.0, 'point_count': 7132}
        },
        'dim': {
            'canny': {'execution_time': 251.31, 'coverage_3mm': 1.0, 'point_count': 3707},
            'rgbd': {'execution_time': 13.91, 'coverage_3mm': 1.0, 'point_count': 6015}
        }
    }
    
    # Prepare data
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    for i, lighting in enumerate(['normal', 'bright', 'dim']):
        # Canny points
        ax.scatter(real_data[lighting]['canny']['execution_time'],
                  real_data[lighting]['canny']['coverage_3mm'] * 100,
                  real_data[lighting]['canny']['point_count'] / 1000,
                  c=colors[i], marker='o', s=100, alpha=0.8, 
                  label=f'Canny-{lighting_conditions[i]}')
        
        # RGB-D points
        ax.scatter(real_data[lighting]['rgbd']['execution_time'],
                  real_data[lighting]['rgbd']['coverage_3mm'] * 100,
                  real_data[lighting]['rgbd']['point_count'] / 1000,
                  c=colors[i], marker='s', s=100, alpha=0.8,
                  label=f'RGB-D-{lighting_conditions[i]}')
    
    ax.set_xlabel('Execution Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Point Count (k)', fontsize=12, fontweight='bold')
    ax.set_title('3D Performance Comparison\n(Time, Coverage, Point Count)', 
                fontsize=16, fontweight='bold')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('3D_Scatter_Performance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_simplified_parallel_coordinates()
    print("\nSimplified visualizations created:")
    print("1. Simplified_Parallel_Coordinates.png - Clean 4-metric comparison (including stability)")
    print("2. 3D_Scatter_Performance.png - 3D scatter plot alternative")
