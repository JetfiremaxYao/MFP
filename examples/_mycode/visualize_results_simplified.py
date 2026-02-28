#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Experimental Results Visualization Script
Focus on key metrics: Position Error, Range Error, and View Recall Rate
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

def generate_simplified_visualization(df, output_dir):
    """Generate simplified visualization focusing on key metrics"""
    print("Generating simplified visualization...")
    
    # Set style
    plt.style.use('default')
    
    # Create figure with 1x3 layout for the three key metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Key Performance Metrics Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    # Set color scheme
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
    
    # 1. Position Error Comparison
    print("Drawing position error comparison...")
    sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', ax=axes[0], palette=colors)
    axes[0].set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # Add mean annotations for position error
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['pos_err_m'].mean()
                axes[0].text(j + i*0.4 - 0.2, mean_val, f'{mean_val*1000:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Range Error Comparison
    print("Drawing range error comparison...")
    sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', ax=axes[1], palette=colors)
    axes[1].set_title('Range Error Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Range Error (m)', fontsize=12)
    axes[1].set_xlabel('Lighting Condition', fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # Add mean annotations for range error
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['range_err_m'].mean()
                axes[1].text(j + i*0.4 - 0.2, mean_val, f'{mean_val*1000:.1f}mm', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. View Recall Rate Comparison
    print("Drawing view recall rate comparison...")
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
        axes[2].set_title('View Recall Rate Comparison', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Recall Rate', fontsize=12)
        axes[2].set_xlabel('Lighting Condition', fontsize=12)
        axes[2].tick_params(axis='both', which='major', labelsize=10)
        axes[2].legend(title='Detection Method', title_fontsize=11, fontsize=10)
        
        # Add value annotations
        for i, p in enumerate(axes[2].patches):
            axes[2].annotate(f'{p.get_height():.1%}', 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save simplified chart
    chart_file = output_dir / "key_metrics_comparison.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Simplified chart saved: {chart_file}")
    
    # Generate summary statistics
    generate_summary_statistics(df, output_dir)

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics for the key metrics"""
    try:
        print("Generating summary statistics...")
        
        # Create summary table
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Summary Statistics - Key Performance Metrics', fontsize=16, fontweight='bold', y=0.95)
        
        # Calculate summary statistics
        summary_data = []
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                # Position error statistics
                pos_err_mean = method_df['pos_err_m'].mean() * 1000  # Convert to mm
                pos_err_std = method_df['pos_err_m'].std() * 1000
                
                # Range error statistics
                range_err_mean = method_df['range_err_m'].mean() * 1000  # Convert to mm
                range_err_std = method_df['range_err_m'].std() * 1000
                
                # View recall rate statistics
                recall_mean = method_df['view_hit_rate'].mean() * 100  # Convert to percentage
                recall_std = method_df['view_hit_rate'].std() * 100
                
                summary_data.append({
                    'Method': 'HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering',
                    'Position Error (mm)': f"{pos_err_mean:.2f} ± {pos_err_std:.2f}",
                    'Range Error (mm)': f"{range_err_mean:.2f} ± {range_err_std:.2f}",
                    'View Recall Rate (%)': f"{recall_mean:.1f} ± {recall_std:.1f}",
                    'Success Rate (%)': f"{method_df['success_pos_2cm'].mean()*100:.1f}",
                    'Avg Detection Time (s)': f"{method_df['detection_time_s'].mean():.2f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            
            # Color the header row
            for i in range(len(summary_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color alternate rows for better readability
            for i in range(1, len(summary_df) + 1):
                for j in range(len(summary_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.tight_layout()
        
        # Save summary statistics
        summary_file = output_dir / "summary_statistics.png"
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary statistics saved: {summary_file}")
        
        # Print key insights
        print_key_insights(df)
        
    except Exception as e:
        print(f"Failed to generate summary statistics: {e}")
        import traceback
        traceback.print_exc()

def print_key_insights(df):
    """Print key insights from the data"""
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    for method in ['hsv_depth', 'depth_clustering']:
        method_df = df[df['method'] == method]
        method_name = 'HSV+Depth' if method == 'hsv_depth' else 'Depth+Clustering'
        
        if len(method_df) > 0:
            pos_err_mean = method_df['pos_err_m'].mean() * 1000
            range_err_mean = method_df['range_err_m'].mean() * 1000
            recall_mean = method_df['view_hit_rate'].mean() * 100
            
            print(f"\n{method_name}:")
            print(f"  • Average Position Error: {pos_err_mean:.2f} mm")
            print(f"  • Average Range Error: {range_err_mean:.2f} mm")
            print(f"  • Average View Recall Rate: {recall_mean:.1f}%")
    
    # Compare methods
    hsv_df = df[df['method'] == 'hsv_depth']
    clustering_df = df[df['method'] == 'depth_clustering']
    
    if len(hsv_df) > 0 and len(clustering_df) > 0:
        print(f"\nCOMPARISON:")
        
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
        
        # Recall rate comparison
        hsv_recall = hsv_df['view_hit_rate'].mean() * 100
        clustering_recall = clustering_df['view_hit_rate'].mean() * 100
        recall_diff = hsv_recall - clustering_recall
        recall_better = "HSV+Depth" if hsv_recall > clustering_recall else "Depth+Clustering"
        print(f"  • View Recall Rate: {recall_better} is better by {abs(recall_diff):.1f}%")
        
        # Overall recommendation
        print(f"\nRECOMMENDATION:")
        if pos_diff < 0 and range_diff < 0 and recall_diff > 0:
            print("  • HSV+Depth method is recommended for better overall performance")
        elif pos_diff > 0 and range_diff > 0 and recall_diff < 0:
            print("  • Depth+Clustering method is recommended for better overall performance")
        else:
            print("  • Consider specific requirements: HSV+Depth for accuracy, Depth+Clustering for robustness")

def main():
    """Main function"""
    print("Simplified Experimental Results Visualization Tool")
    print("Focusing on Key Metrics: Position Error, Range Error, View Recall Rate")
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
    
    # Generate simplified visualizations
    generate_simplified_visualization(df, viz_dir)
    
    print(f"\nSimplified visualization completed! Results saved in: {viz_dir}")
    print("Generated files:")
    for file in viz_dir.glob("*key_metrics*"):
        print(f"  - {file.name}")
    for file in viz_dir.glob("*summary_statistics*"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
