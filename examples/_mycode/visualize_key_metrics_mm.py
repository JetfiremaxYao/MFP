#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Key Metrics Visualization Tool with Millimeter Units
关键指标可视化工具 - 毫米单位
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

def generate_mm_visualization(df, output_dir):
    """Generate visualization with millimeter units"""
    print("Generating key metrics visualization with mm units...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Key Performance Metrics', fontsize=18, fontweight='bold', y=1.0)
    
    # 1. Position Error Comparison (in mm)
    print("Drawing position error comparison...")
    # Convert position error to mm for display
    df_pos_mm = df.copy()
    df_pos_mm['pos_err_mm'] = df_pos_mm['pos_err_m'] * 1000
    
    sns.boxplot(data=df_pos_mm, x='lighting', y='pos_err_mm', hue='method', ax=axes[0])
    axes[0].set_title('Position Error', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Position Error (mm)', fontsize=12)
    axes[0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    
    # 2. Range Error Comparison (in mm)
    print("Drawing range error comparison...")
    # Convert range error to mm for display
    df_range_mm = df.copy()
    df_range_mm['range_err_mm'] = df_range_mm['range_err_m'] * 1000
    
    sns.boxplot(data=df_range_mm, x='lighting', y='range_err_mm', hue='method', ax=axes[1])
    axes[1].set_title('Range Error', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Range Error (mm)', fontsize=12)
    axes[1].set_xlabel('Lighting Condition', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    
    # 3. View Recall Rate Comparison
    print("Drawing view recall rate comparison...")
    sns.boxplot(data=df, x='lighting', y='view_hit_rate', hue='method', ax=axes[2])
    axes[2].set_title('View Recall Rate', fontsize=14, fontweight='bold', pad=15)
    axes[2].set_ylabel('View Recall Rate', fontsize=12)
    axes[2].set_xlabel('Lighting Condition', fontsize=12)
    axes[2].tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout with more space for titles
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Give more space for the main title
    
    # Save the plot with lower DPI to reduce file size
    output_file = output_dir / "key_metrics_mm.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"Key metrics chart with mm units saved: {output_file}")
    return output_file

def generate_simple_summary(df, output_dir):
    """Generate simple summary statistics"""
    print("Generating simple summary...")
    
    # Calculate summary statistics
    summary_data = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                summary_data.append({
                    'Method': method.replace('_', '+').upper(),
                    'Lighting': lighting.capitalize(),
                    'Pos Error (mm)': f"{subset['pos_err_m'].mean()*1000:.1f}±{subset['pos_err_m'].std()*1000:.1f}",
                    'Range Error (mm)': f"{subset['range_err_m'].mean()*1000:.1f}±{subset['range_err_m'].std()*1000:.1f}",
                    'Recall (%)': f"{subset['view_hit_rate'].mean()*100:.1f}±{subset['view_hit_rate'].std()*100:.1f}"
                })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Create a simple figure for the table
    fig, ax = plt.subplots(figsize=(12, 6))
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
    table.set_fontsize(9)
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
    
    plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Save the table with lower DPI
    output_file = output_dir / "summary_mm.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"Simple summary saved: {output_file}")
    return output_file

def print_simple_insights(df):
    """Print simple insights from the data"""
    print("\n" + "="*50)
    print("SIMPLE INSIGHTS")
    print("="*50)
    
    # Overall comparison
    hsv_data = df[df['method'] == 'hsv_depth']
    clustering_data = df[df['method'] == 'depth_clustering']
    
    print(f"HSV+Depth:")
    print(f"  • Position Error: {hsv_data['pos_err_m'].mean()*1000:.1f} ± {hsv_data['pos_err_m'].std()*1000:.1f} mm")
    print(f"  • Range Error: {hsv_data['range_err_m'].mean()*1000:.1f} ± {hsv_data['range_err_m'].std()*1000:.1f} mm")
    print(f"  • Recall Rate: {hsv_data['view_hit_rate'].mean()*100:.1f} ± {hsv_data['view_hit_rate'].std()*100:.1f}%")
    
    print(f"\nDepth+Clustering:")
    print(f"  • Position Error: {clustering_data['pos_err_m'].mean()*1000:.1f} ± {clustering_data['pos_err_m'].std()*1000:.1f} mm")
    print(f"  • Range Error: {clustering_data['range_err_m'].mean()*1000:.1f} ± {clustering_data['range_err_m'].std()*1000:.1f} mm")
    print(f"  • Recall Rate: {clustering_data['view_hit_rate'].mean()*100:.1f} ± {clustering_data['view_hit_rate'].std()*100:.1f}%")
    
    print(f"\nCOMPARISON:")
    pos_diff = (hsv_data['pos_err_m'].mean() - clustering_data['pos_err_m'].mean()) * 1000
    range_diff = (hsv_data['range_err_m'].mean() - clustering_data['range_err_m'].mean()) * 1000
    recall_diff = (hsv_data['view_hit_rate'].mean() - clustering_data['view_hit_rate'].mean()) * 100
    
    print(f"  • Position: HSV+Depth {'better' if pos_diff < 0 else 'worse'} by {abs(pos_diff):.1f} mm")
    print(f"  • Range: HSV+Depth {'better' if range_diff < 0 else 'worse'} by {abs(range_diff):.1f} mm")
    print(f"  • Recall: HSV+Depth {'better' if recall_diff > 0 else 'worse'} by {abs(recall_diff):.1f}%")

def main():
    """Main function"""
    print("Key Metrics Visualization Tool with Millimeter Units")
    print("关键指标可视化工具 - 毫米单位")
    print("="*50)
    
    try:
        # Find latest experiment directory
        experiment_dir = find_latest_experiment()
        
        # Load data
        csv_file = Path(experiment_dir) / "trial_results.csv"
        df = load_and_clean_data(csv_file)
        
        # Create visualizations directory
        vis_dir = Path(experiment_dir) / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Generate visualizations with mm units
        generate_mm_visualization(df, vis_dir)
        generate_simple_summary(df, vis_dir)
        
        # Print insights
        print_simple_insights(df)
        
        print(f"\nVisualization with mm units completed!")
        print(f"Results saved in: {vis_dir}")
        print(f"\nGenerated files:")
        print(f"  - key_metrics_mm.png")
        print(f"  - summary_mm.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
