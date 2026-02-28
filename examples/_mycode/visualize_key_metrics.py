#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Key Metrics Visualization Tool
专注于三个关键指标的可视化：位置误差、范围误差、视角召回率
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

def load_and_clean_data(csv_file):
    """Load and clean CSV data"""
    print(f"Loading data file: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    print(f"Original data shape: {df.shape}")
    print(f"Data columns: {list(df.columns)}")
    
    # Clean view_details column (remove numpy type markers)
    if 'view_details' in df.columns:
        print("Cleaning view_details column...")
        df['view_details'] = df['view_details'].apply(lambda x: 
            x.replace('np.float64(', '').replace('np.int64(', '').replace(')', '') 
            if isinstance(x, str) else x)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Method types: {df['method'].unique()}")
    print(f"Lighting conditions: {df['lighting'].unique()}")
    
    return df

def find_latest_experiment():
    """Find the latest experiment directory"""
    pattern = "evaluation_results/unified_object_detection_*"
    directories = glob.glob(pattern)
    if not directories:
        raise FileNotFoundError("No experiment directories found")
    
    latest_dir = max(directories, key=os.path.getctime)
    print(f"Using latest experiment directory: {latest_dir}")
    return latest_dir

def generate_key_metrics_visualization(df, output_dir):
    """Generate visualization for the three key metrics"""
    print("Generating key metrics visualization...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Key Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 1. Position Error Comparison
    print("Drawing position error comparison...")
    sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', ax=axes[0])
    axes[0].set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Add mean values as text
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['pos_err_m'].mean() * 1000  # Convert to mm
                axes[0].text(j + (i-0.5)*0.4, mean_val + 0.001, f'{mean_val:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Range Error Comparison
    print("Drawing range error comparison...")
    sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', ax=axes[1])
    axes[1].set_title('Range Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Range Error (m)', fontsize=12)
    axes[1].set_xlabel('Lighting Condition', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    # Add mean values as text
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['range_err_m'].mean() * 1000  # Convert to mm
                axes[1].text(j + (i-0.5)*0.4, mean_val + 0.001, f'{mean_val:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. View Recall Rate Comparison
    print("Drawing view recall rate comparison...")
    sns.boxplot(data=df, x='lighting', y='view_hit_rate', hue='method', ax=axes[2])
    axes[2].set_title('View Recall Rate Comparison', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('View Recall Rate', fontsize=12)
    axes[2].set_xlabel('Lighting Condition', fontsize=12)
    axes[2].tick_params(axis='both', which='major', labelsize=10)
    
    # Add mean values as text
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['view_hit_rate'].mean() * 100  # Convert to percentage
                axes[2].text(j + (i-0.5)*0.4, mean_val/100 + 0.01, f'{mean_val:.1f}%', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / "key_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Key metrics comparison chart saved: {output_file}")
    return output_file

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics table"""
    print("Generating summary statistics...")
    
    # Calculate summary statistics
    summary_data = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                summary_data.append({
                    'Method': method.replace('_', '+').upper(),
                    'Lighting': lighting.capitalize(),
                    'Position Error (mm)': f"{subset['pos_err_m'].mean()*1000:.2f} ± {subset['pos_err_m'].std()*1000:.2f}",
                    'Range Error (mm)': f"{subset['range_err_m'].mean()*1000:.2f} ± {subset['range_err_m'].std()*1000:.2f}",
                    'Recall Rate (%)': f"{subset['view_hit_rate'].mean()*100:.1f} ± {subset['view_hit_rate'].std()*100:.1f}",
                    'Success Rate (%)': f"{subset['success_pos_2cm'].mean()*100:.0f}",
                    'Avg Time (s)': f"{subset['detection_time_s'].mean():.2f} ± {subset['detection_time_s'].std():.2f}"
                })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=summary_df.values, 
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternate rows
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary Statistics', fontsize=16, fontweight='bold', pad=20)
    
    # Save the table
    output_file = output_dir / "summary_statistics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary statistics saved: {output_file}")
    return output_file

def print_key_insights(df):
    """Print key insights from the data"""
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    # Overall comparison
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    print(f"HSV+Depth Method:")
    print(f"  • Average Position Error: {hsv_data['pos_err_m'].mean()*1000:.2f} ± {hsv_data['pos_err_m'].std()*1000:.2f} mm")
    print(f"  • Average Range Error: {hsv_data['range_err_m'].mean()*1000:.2f} ± {hsv_data['range_err_m'].std()*1000:.2f} mm")
    print(f"  • Average Recall Rate: {hsv_data['view_hit_rate'].mean()*100:.1f} ± {hsv_data['view_hit_rate'].std()*100:.1f}%")
    
    print(f"\nDepth+Clustering Method:")
    print(f"  • Average Position Error: {clustering_data['pos_err_m'].mean()*1000:.2f} ± {clustering_data['pos_err_m'].std()*1000:.2f} mm")
    print(f"  • Average Range Error: {clustering_data['range_err_m'].mean()*1000:.2f} ± {clustering_data['range_err_m'].std()*1000:.2f} mm")
    print(f"  • Average Recall Rate: {clustering_data['view_hit_rate'].mean()*100:.1f} ± {clustering_data['view_hit_rate'].std()*100:.1f}%")
    
    print(f"\nCOMPARISON:")
    pos_diff = (hsv_data['pos_err_m'].mean() - clustering_data['pos_err_m'].mean()) * 1000
    range_diff = (hsv_data['range_err_m'].mean() - clustering_data['range_err_m'].mean()) * 1000
    recall_diff = (hsv_data['view_hit_rate'].mean() - clustering_data['view_hit_rate'].mean()) * 100
    
    print(f"  • Position Error: HSV+Depth is {'better' if pos_diff < 0 else 'worse'} by {abs(pos_diff):.2f} mm")
    print(f"  • Range Error: HSV+Depth is {'better' if range_diff < 0 else 'worse'} by {abs(range_diff):.2f} mm")
    print(f"  • Recall Rate: HSV+Depth is {'better' if recall_diff > 0 else 'worse'} by {abs(recall_diff):.1f}%")
    
    print(f"\nRECOMMENDATION:")
    if pos_diff < 0 and range_diff < 0:
        print("  • HSV+Depth is better for both position and range accuracy")
    elif pos_diff < 0 and recall_diff > 0:
        print("  • HSV+Depth is better for position accuracy and recall rate")
    elif range_diff < 0 and recall_diff > 0:
        print("  • HSV+Depth is better for range accuracy and recall rate")
    elif clustering_data['pos_err_m'].mean() < hsv_data['pos_err_m'].mean() and clustering_data['range_err_m'].mean() < hsv_data['range_err_m'].mean():
        print("  • Depth+Clustering is better for both position and range accuracy")
    else:
        print("  • Consider specific requirements: HSV+Depth for accuracy, Depth+Clustering for robustness")

def main():
    """Main function"""
    print("Key Metrics Visualization Tool")
    print("专注于三个关键指标的可视化：位置误差、范围误差、视角召回率")
    print("="*60)
    
    try:
        # Find latest experiment directory
        experiment_dir = find_latest_experiment()
        
        # Load data
        csv_file = Path(experiment_dir) / "trial_results.csv"
        df = load_and_clean_data(csv_file)
        
        # Create visualizations directory
        vis_dir = Path(experiment_dir) / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        generate_key_metrics_visualization(df, vis_dir)
        generate_summary_statistics(df, vis_dir)
        
        # Print insights
        print_key_insights(df)
        
        print(f"\nVisualization completed! Results saved in: {vis_dir}")
        print(f"\nGenerated files:")
        print(f"  - key_metrics_comparison.png")
        print(f"  - summary_statistics.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
