#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presentation Charts for Boundary Detection Analysis
Chamfer Distance vs Lighting Conditions
Hausdorff Distance vs Lighting Conditions
Execution Time vs Lighting Conditions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_presentation_charts():
    """Create presentation charts for Chamfer and Hausdorff distance analysis"""
    
    # Real experimental data from boundary_tracking_grouped_summary.csv
    real_data = {
        'normal': {
            'canny': {
                'chamfer_distance': 3.18,
                'chamfer_std': 0.51,
                'hausdorff_distance': 18.97,
                'hausdorff_std': 1.99,
                'execution_time': 17.88,
                'execution_std': 12.48
            },
            'rgbd': {
                'chamfer_distance': 6.07,
                'chamfer_std': 0.98,
                'hausdorff_distance': 18.79,
                'hausdorff_std': 6.49,
                'execution_time': 91.09,
                'execution_std': 81.35
            }
        },
        'bright': {
            'canny': {
                'chamfer_distance': 2.83,
                'chamfer_std': 0.50,
                'hausdorff_distance': 17.98,
                'hausdorff_std': 2.16,
                'execution_time': 192.17,
                'execution_std': 298.60
            },
            'rgbd': {
                'chamfer_distance': 6.69,
                'chamfer_std': 2.13,
                'hausdorff_distance': 21.71,
                'hausdorff_std': 1.71,
                'execution_time': 48.20,
                'execution_std': 15.87
            }
        },
        'dim': {
            'canny': {
                'chamfer_distance': 3.12,
                'chamfer_std': 0.72,
                'hausdorff_distance': 17.74,
                'hausdorff_std': 2.87,
                'execution_time': 251.31,
                'execution_std': 390.67
            },
            'rgbd': {
                'chamfer_distance': 7.18,
                'chamfer_std': 2.77,
                'hausdorff_distance': 20.13,
                'hausdorff_std': 3.86,
                'execution_time': 13.91,
                'execution_std': 1.39
            }
        }
    }
    
    # Set style for professional presentation
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the charts
    create_chamfer_distance_chart(real_data)
    create_hausdorff_distance_chart(real_data)
    create_execution_time_chart(real_data)
    create_combined_comparison_chart(real_data)
    
    print("Presentation charts created successfully!")
    print("- Chamfer_Distance_vs_Lighting.png")
    print("- Hausdorff_Distance_vs_Lighting.png")
    print("- Execution_Time_vs_Lighting.png")
    print("- Combined_Distance_Comparison.png")

def create_chamfer_distance_chart(data):
    """Create Chamfer Distance vs Lighting Conditions chart"""
    
    # Prepare data
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    width = 0.35
    
    # Extract Chamfer distance data
    canny_chamfer = [data[light.lower()]['canny']['chamfer_distance'] for light in lighting_conditions]
    canny_chamfer_std = [data[light.lower()]['canny']['chamfer_std'] for light in lighting_conditions]
    rgbd_chamfer = [data[light.lower()]['rgbd']['chamfer_distance'] for light in lighting_conditions]
    rgbd_chamfer_std = [data[light.lower()]['rgbd']['chamfer_std'] for light in lighting_conditions]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, canny_chamfer, width, label='Canny', 
                   color='#1f77b4', alpha=0.8, yerr=canny_chamfer_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2 = ax.bar(x + width/2, rgbd_chamfer, width, label='RGB-D', 
                   color='#ff7f0e', alpha=0.8, yerr=rgbd_chamfer_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize the chart
    ax.set_title('Chamfer Distance vs Lighting Conditions', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Lighting Conditions', fontsize=16, fontweight='bold')
    ax.set_ylabel('Chamfer Distance (mm)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lighting_conditions, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, value, std in zip(bars1, canny_chamfer, canny_chamfer_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.1,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    for bar, value, std in zip(bars2, rgbd_chamfer, rgbd_chamfer_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.1,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add key insights as text
    ax.text(0.02, 0.98, 'Key Insights:\n• Canny: Stable ~3mm across all conditions\n• RGB-D: Higher values with more variation\n• Canny shows better accuracy', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Chamfer_Distance_vs_Lighting.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_hausdorff_distance_chart(data):
    """Create Hausdorff Distance vs Lighting Conditions chart"""
    
    # Prepare data
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    width = 0.35
    
    # Extract Hausdorff distance data
    canny_hausdorff = [data[light.lower()]['canny']['hausdorff_distance'] for light in lighting_conditions]
    canny_hausdorff_std = [data[light.lower()]['canny']['hausdorff_std'] for light in lighting_conditions]
    rgbd_hausdorff = [data[light.lower()]['rgbd']['hausdorff_distance'] for light in lighting_conditions]
    rgbd_hausdorff_std = [data[light.lower()]['rgbd']['hausdorff_std'] for light in lighting_conditions]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, canny_hausdorff, width, label='Canny', 
                   color='#1f77b4', alpha=0.8, yerr=canny_hausdorff_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2 = ax.bar(x + width/2, rgbd_hausdorff, width, label='RGB-D', 
                   color='#ff7f0e', alpha=0.8, yerr=rgbd_hausdorff_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize the chart
    ax.set_title('Hausdorff Distance vs Lighting Conditions', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Lighting Conditions', fontsize=16, fontweight='bold')
    ax.set_ylabel('Hausdorff Distance (mm)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lighting_conditions, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, 30)
    
    # Add value labels on bars
    for bar, value, std in zip(bars1, canny_hausdorff, canny_hausdorff_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.5,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    for bar, value, std in zip(bars2, rgbd_hausdorff, rgbd_hausdorff_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 0.5,
                f'{value:.2f}±{std:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add key insights as text
    ax.text(0.02, 0.98, 'Key Insights:\n• Both methods show similar worst-case performance\n• Hausdorff distance indicates comparable robustness\n• Lighting conditions have minimal impact on worst-case errors', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Hausdorff_Distance_vs_Lighting.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_execution_time_chart(data):
    """Create Execution Time vs Lighting Conditions chart"""
    
    # Prepare data
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    width = 0.35
    
    # Extract execution time data
    canny_time = [data[light.lower()]['canny']['execution_time'] for light in lighting_conditions]
    canny_time_std = [data[light.lower()]['canny']['execution_std'] for light in lighting_conditions]
    rgbd_time = [data[light.lower()]['rgbd']['execution_time'] for light in lighting_conditions]
    rgbd_time_std = [data[light.lower()]['rgbd']['execution_std'] for light in lighting_conditions]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, canny_time, width, label='Canny', 
                   color='#1f77b4', alpha=0.8, yerr=canny_time_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    bars2 = ax.bar(x + width/2, rgbd_time, width, label='RGB-D', 
                   color='#ff7f0e', alpha=0.8, yerr=rgbd_time_std, 
                   capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Customize the chart
    ax.set_title('Execution Time vs Lighting Conditions', fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Lighting Conditions', fontsize=16, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lighting_conditions, fontsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits for better visualization
    ax.set_ylim(0, 300)
    
    # Add value labels on bars
    for bar, value, std in zip(bars1, canny_time, canny_time_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 5,
                f'{value:.1f}±{std:.1f}s', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    for bar, value, std in zip(bars2, rgbd_time, rgbd_time_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + std + 5,
                f'{value:.1f}±{std:.1f}s', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    # Add key insights as text
    ax.text(0.02, 0.98, 'Key Insights:\n• RGB-D: Generally faster (13-91s)\n• Canny: Slower but more stable in normal lighting\n• Dim lighting affects Canny performance significantly\n• RGB-D shows better efficiency in challenging conditions', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Execution_Time_vs_Lighting.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_comparison_chart(data):
    """Create a combined comparison chart showing both metrics"""
    
    # Prepare data
    lighting_conditions = ['Normal', 'Bright', 'Dim']
    x = np.arange(len(lighting_conditions))
    width = 0.35
    
    # Extract data
    canny_chamfer = [data[light.lower()]['canny']['chamfer_distance'] for light in lighting_conditions]
    rgbd_chamfer = [data[light.lower()]['rgbd']['chamfer_distance'] for light in lighting_conditions]
    canny_hausdorff = [data[light.lower()]['canny']['hausdorff_distance'] for light in lighting_conditions]
    rgbd_hausdorff = [data[light.lower()]['rgbd']['hausdorff_distance'] for light in lighting_conditions]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Chamfer Distance subplot
    bars1 = ax1.bar(x - width/2, canny_chamfer, width, label='Canny', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, rgbd_chamfer, width, label='RGB-D', color='#ff7f0e', alpha=0.8)
    
    ax1.set_title('Chamfer Distance (mm)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Lighting Conditions', fontsize=14)
    ax1.set_ylabel('Distance (mm)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lighting_conditions)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)
    
    # Add value labels
    for bar, value in zip(bars1, canny_chamfer):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    for bar, value in zip(bars2, rgbd_chamfer):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Hausdorff Distance subplot
    bars3 = ax2.bar(x - width/2, canny_hausdorff, width, label='Canny', color='#1f77b4', alpha=0.8)
    bars4 = ax2.bar(x + width/2, rgbd_hausdorff, width, label='RGB-D', color='#ff7f0e', alpha=0.8)
    
    ax2.set_title('Hausdorff Distance (mm)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Lighting Conditions', fontsize=14)
    ax2.set_ylabel('Distance (mm)', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lighting_conditions)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 30)
    
    # Add value labels
    for bar, value in zip(bars3, canny_hausdorff):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    for bar, value in zip(bars4, rgbd_hausdorff):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add main title
    fig.suptitle('Boundary Detection Performance: Chamfer vs Hausdorff Distance', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig('Combined_Distance_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_presentation_charts()
    print("\nCharts created for presentation:")
    print("1. Chamfer_Distance_vs_Lighting.png - Shows Canny stability vs RGB-D variation")
    print("2. Hausdorff_Distance_vs_Lighting.png - Shows similar worst-case performance")
    print("3. Execution_Time_vs_Lighting.png - Shows efficiency comparison")
    print("4. Combined_Distance_Comparison.png - Side-by-side comparison")
