#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Recall Rate Results Visualization Script
专门用于可视化修复后召回率数据的脚本
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

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
    
    # Ensure numeric columns have correct types
    numeric_columns = ['pos_err_m', 'dx_m', 'dy_m', 'range_err_m', 'detection_time_s', 
                      'frame_count', 'fps', 'views_total', 'views_hit', 'early_stop_index', 
                      'view_hit_rate', 'best_angle_deg', 'best_score']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure boolean columns have correct types
    boolean_columns = ['success_pos_2cm', 'success_range_2cm', 'success_pos_5cm', 'success_range_5cm']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    print(f"Cleaned data shape: {df.shape}")
    print(f"Method types: {df['method'].unique()}")
    print(f"Lighting conditions: {df['lighting'].unique()}")
    
    return df

def generate_fixed_recall_visualization(df, output_dir):
    """Generate visualization focusing on fixed recall rate data"""
    print("Generating fixed recall rate visualization...")
    
    # Set style
    plt.style.use('default')
    
    # Create figure with 1x3 layout (removed heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Key Performance Metrics', fontsize=18, fontweight='bold', y=1.)
    
    # Set color scheme
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
    
    # 1. Position Error Comparison (Left)
    print("Drawing position error comparison...")
    # Convert position error to mm for display
    df_pos_mm = df.copy()
    df_pos_mm['pos_err_mm'] = df_pos_mm['pos_err_m'] * 1000
    
    sns.boxplot(data=df_pos_mm, x='lighting', y='pos_err_mm', hue='method', ax=axes[0], palette=colors)
    axes[0].set_title('Position Error Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Position Error (mm)', fontsize=12)
    axes[0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # Add mean annotations for position error (in mm)
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df_pos_mm[(df_pos_mm['method'] == method) & (df_pos_mm['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['pos_err_mm'].mean()
                axes[0].text(j + i*0.4 - 0.2, mean_val, f'{mean_val:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Range Error Comparison (Middle)
    print("Drawing range error comparison...")
    # Convert range error to mm for display
    df_range_mm = df.copy()
    df_range_mm['range_err_mm'] = df_range_mm['range_err_m'] * 1000
    
    sns.boxplot(data=df_range_mm, x='lighting', y='range_err_mm', hue='method', ax=axes[1], palette=colors)
    axes[1].set_title('Range Error Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Range Error (mm)', fontsize=12)
    axes[1].set_xlabel('Lighting Condition', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # Add mean annotations for range error (in mm)
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df_range_mm[(df_range_mm['method'] == method) & (df_range_mm['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['range_err_mm'].mean()
                axes[1].text(j + i*0.4 - 0.2, mean_val, f'{mean_val:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Fixed View Recall Rate Comparison (Right)
    print("Drawing fixed view recall rate comparison...")
    recall_rates = []
    methods = []
    lightings = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                recall_rate = subset['view_hit_rate'].mean()
                recall_rates.append(recall_rate)
                methods.append(method_names[method])
                lightings.append(lighting)
    
    if recall_rates:
        recall_df = pd.DataFrame({
            'method': methods,
            'lighting': lightings,
            'recall_rate': recall_rates
        })
        
        sns.barplot(data=recall_df, x='lighting', y='recall_rate', hue='method', ax=axes[2], palette=colors)
        axes[2].set_title('Fixed View Recall Rate Comparison', fontsize=14, fontweight='bold', pad=15)
        axes[2].set_ylabel('Recall Rate', fontsize=12)
        axes[2].set_xlabel('Lighting Condition', fontsize=12)
        axes[2].tick_params(axis='both', which='major', labelsize=10)
        axes[2].legend(title='Detection Method', title_fontsize=11, fontsize=10)
        
        # Add value annotations
        for i, p in enumerate(axes[2].patches):
            axes[2].annotate(f'{p.get_height():.1%}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adjust layout with more space for titles
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Give more space for the main title
    
    # Save fixed recall chart
    chart_file = output_dir / "fixed_recall_analysis.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Fixed recall analysis chart saved: {chart_file}")
    
    # Generate detailed analysis
    generate_detailed_fixed_analysis(df, output_dir)

def generate_detailed_fixed_analysis(df, output_dir):
    """Generate detailed analysis of fixed recall rate data"""
    try:
        print("Generating detailed fixed analysis...")
        
        # Create detailed analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Fixed Recall Rate Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Method Performance Comparison (Top Left)
        method_performance = []
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                method_performance.append({
                    'Method': 'HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering',
                    'Avg Position Error (mm)': method_df['pos_err_m'].mean() * 1000,
                    'Avg Range Error (mm)': method_df['range_err_m'].mean() * 1000,
                    'Avg Recall Rate (%)': method_df['view_hit_rate'].mean() * 100,
                    'Success Rate (%)': method_df['success_pos_2cm'].mean() * 100
                })
        
        if method_performance:
            perf_df = pd.DataFrame(method_performance)
            ax = axes[0, 0]
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=perf_df.values, colLabels=perf_df.columns, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the header row
            for i in range(len(perf_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color alternate rows
            for i in range(1, len(perf_df) + 1):
                for j in range(len(perf_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            ax.set_title('Overall Method Performance', fontsize=12, fontweight='bold')
        
        # 2. Lighting Condition Impact (Top Right)
        lighting_impact = []
        for lighting in ['normal', 'bright', 'dim']:
            lighting_df = df[df['lighting'] == lighting]
            if len(lighting_df) > 0:
                lighting_impact.append({
                    'Lighting': lighting.capitalize(),
                    'Avg Position Error (mm)': lighting_df['pos_err_m'].mean() * 1000,
                    'Avg Range Error (mm)': lighting_df['range_err_m'].mean() * 1000,
                    'Avg Recall Rate (%)': lighting_df['view_hit_rate'].mean() * 100
                })
        
        if lighting_impact:
            lighting_df = pd.DataFrame(lighting_impact)
            ax = axes[0, 1]
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=lighting_df.values, colLabels=lighting_df.columns, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Color the header row
            for i in range(len(lighting_df.columns)):
                table[(0, i)].set_facecolor('#2196F3')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color alternate rows
            for i in range(1, len(lighting_df) + 1):
                for j in range(len(lighting_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
            
            ax.set_title('Lighting Condition Impact', fontsize=12, fontweight='bold')
        
        # 3. Recall Rate Distribution (Bottom Left)
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 0].hist(method_df['view_hit_rate'] * 100, alpha=0.7, 
                               label='HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering', 
                               bins=10)
        
        axes[1, 0].set_title('Recall Rate Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Recall Rate (%)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Position Error vs Recall Rate Scatter (Bottom Right)
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 1].scatter(method_df['pos_err_m'] * 1000, method_df['view_hit_rate'] * 100, 
                                 alpha=0.7, label='HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering')
        
        axes[1, 1].set_title('Position Error vs Recall Rate', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Position Error (mm)', fontsize=10)
        axes[1, 1].set_ylabel('Recall Rate (%)', fontsize=10)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save detailed analysis
        detailed_file = output_dir / "detailed_fixed_analysis.png"
        plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed fixed analysis saved: {detailed_file}")
        
        # Print key insights
        print_fixed_insights(df)
        
    except Exception as e:
        print(f"Failed to generate detailed fixed analysis: {e}")
        import traceback
        traceback.print_exc()

def print_fixed_insights(df):
    """Print key insights from fixed recall rate data"""
    print("\n" + "="*70)
    print("FIXED RECALL RATE ANALYSIS INSIGHTS")
    print("="*70)
    
    for method in ['hsv_depth', 'depth_clustering']:
        method_df = df[df['method'] == method]
        method_name = 'HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering'
        
        if len(method_df) > 0:
            pos_err_mean = method_df['pos_err_m'].mean() * 1000
            range_err_mean = method_df['range_err_m'].mean() * 1000
            recall_mean = method_df['view_hit_rate'].mean() * 100
            
            print(f"\n{method_name} (Fixed Recall Rate):")
            print(f"  • Average Position Error: {pos_err_mean:.2f} mm")
            print(f"  • Average Range Error: {range_err_mean:.2f} mm")
            print(f"  • Average Recall Rate: {recall_mean:.1f}%")
            
            # Analyze lighting impact
            for lighting in ['normal', 'bright', 'dim']:
                lighting_df = method_df[method_df['lighting'] == lighting]
                if len(lighting_df) > 0:
                    lighting_recall = lighting_df['view_hit_rate'].mean() * 100
                    print(f"    - {lighting.capitalize()} lighting: {lighting_recall:.1f}% recall")
    
    # Compare methods with fixed recall rates
    hsv_df = df[df['method'] == 'hsv_depth']
    clustering_df = df[df['method'] == 'depth_clustering']
    
    if len(hsv_df) > 0 and len(clustering_df) > 0:
        print(f"\nFIXED RECALL RATE COMPARISON:")
        
        # Position error comparison
        hsv_pos = hsv_df['pos_err_m'].mean() * 1000
        clustering_pos = clustering_df['pos_err_m'].mean() * 1000
        pos_diff = hsv_pos - clustering_pos
        pos_better = "HSV+Depth" if hsv_pos < clustering_pos else "Depth+Clustering"
        print(f"  • Position Error: {pos_better} is better by {abs(pos_diff):.2f} mm")
        
        # Range error comparison
        hsv_range = hsv_df['range_err_m'].mean() * 1000
        clustering_range = clustering_df['range_err_m'].mean() * 1000
        range_diff = hsv_range - clustering_range
        range_better = "HSV+Depth" if hsv_range < clustering_range else "Depth+Clustering"
        print(f"  • Range Error: {range_better} is better by {abs(range_diff):.2f} mm")
        
        # Fixed recall rate comparison
        hsv_recall = hsv_df['view_hit_rate'].mean() * 100
        clustering_recall = clustering_df['view_hit_rate'].mean() * 100
        recall_diff = hsv_recall - clustering_recall
        recall_better = "HSV+Depth" if hsv_recall > clustering_recall else "Depth+Clustering"
        print(f"  • Fixed Recall Rate: {recall_better} is better by {abs(recall_diff):.1f}%")
        
        # Overall recommendation
        print(f"\nFIXED ANALYSIS RECOMMENDATION:")
        if pos_diff < 0 and range_diff < 0 and recall_diff > 0:
            print("  • HSV+Depth method is recommended for better overall performance")
        elif pos_diff > 0 and range_diff > 0 and recall_diff < 0:
            print("  • Depth+Clustering method is recommended for better overall performance")
        else:
            print("  • Consider specific requirements: HSV+Depth for accuracy, Depth+Clustering for robustness")

def main():
    """Main function"""
    print("Fixed Recall Rate Results Visualization Tool")
    print("专门用于可视化修复后召回率数据")
    print("="*70)
    
    # Find latest experimental results directory
    results_dir = Path("evaluation_results")
    if not results_dir.exists():
        print("Error: evaluation_results directory not found")
        return
    
    # Find latest unified_object_detection experiment directory
    experiment_dirs = [d for d in results_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('unified_object_detection')]
    
    if not experiment_dirs:
        print("Error: unified_object_detection experiment directory not found")
        return
    
    # Select latest directory
    latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"Using latest experiment directory: {latest_dir}")
    
    # Find CSV file
    csv_file = latest_dir / "trial_results.csv"
    if not csv_file.exists():
        print(f"Error: CSV file not found {csv_file}")
        return
    
    # Load data
    df = load_and_clean_data(csv_file)
    
    # Create visualization output directory
    viz_dir = latest_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate fixed recall visualizations
    generate_fixed_recall_visualization(df, viz_dir)
    
    print(f"\nFixed recall visualization completed! Results saved in: {viz_dir}")
    print("Generated files:")
    for file in viz_dir.glob("*fixed*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
