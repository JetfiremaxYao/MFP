#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English Version of Experimental Results Visualization Script
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import ast

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

def generate_comprehensive_visualization(df, output_dir):
    """Generate comprehensive visualization charts"""
    print("Starting comprehensive visualization generation...")
    
    # Set style
    plt.style.use('default')
    
    # Create main chart - 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('Unified Object Detection Method Evaluation Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Set color scheme
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
    
    # 1. Position Error Boxplot
    print("Drawing position error boxplot...")
    sns.boxplot(data=df, x='lighting', y='pos_err_m', hue='method', ax=axes[0, 0], palette=colors)
    axes[0, 0].set_title('Position Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Position Error (m)', fontsize=12)
    axes[0, 0].set_xlabel('Lighting Condition', fontsize=12)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 0].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # Add mean annotations
    for i, method in enumerate(['hsv_depth', 'depth_clustering']):
        for j, lighting in enumerate(['normal', 'bright', 'dim']):
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                mean_val = subset['pos_err_m'].mean()
                axes[0, 0].text(j + i*0.4 - 0.2, mean_val, f'{mean_val*1000:.1f}mm', 
                               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Range Error Boxplot
    print("Drawing range error boxplot...")
    sns.boxplot(data=df, x='lighting', y='range_err_m', hue='method', ax=axes[0, 1], palette=colors)
    axes[0, 1].set_title('Range Error Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Range Error (m)', fontsize=12)
    axes[0, 1].set_xlabel('Lighting Condition', fontsize=12)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 1].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # 3. Detection Time Boxplot
    print("Drawing detection time boxplot...")
    sns.boxplot(data=df, x='lighting', y='detection_time_s', hue='method', ax=axes[0, 2], palette=colors)
    axes[0, 2].set_title('Detection Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Detection Time (s)', fontsize=12)
    axes[0, 2].set_xlabel('Lighting Condition', fontsize=12)
    axes[0, 2].tick_params(axis='both', which='major', labelsize=10)
    axes[0, 2].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    # 4. Success Rate Bar Chart
    print("Drawing success rate bar chart...")
    success_rates = []
    methods = []
    lightings = []
    
    for method in ['hsv_depth', 'depth_clustering']:
        for lighting in ['normal', 'bright', 'dim']:
            subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
            if len(subset) > 0:
                success_rate = subset['success_pos_2cm'].mean()
                success_rates.append(success_rate)
                methods.append(method_names[method])
                lightings.append(lighting)
    
    if success_rates:
        success_df = pd.DataFrame({
            'method': methods,
            'lighting': lightings,
            'success_rate': success_rates
        })
        
        sns.barplot(data=success_df, x='lighting', y='success_rate', hue='method', ax=axes[1, 0], palette=colors)
        axes[1, 0].set_title('Success Rate Comparison (2cm threshold)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Success Rate', fontsize=12)
        axes[1, 0].set_xlabel('Lighting Condition', fontsize=12)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 0].legend(title='Detection Method', title_fontsize=11, fontsize=10)
        
        # Add value annotations
        for i, p in enumerate(axes[1, 0].patches):
            axes[1, 0].annotate(f'{p.get_height():.1%}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. View Recall Rate Bar Chart
    print("Drawing view recall rate bar chart...")
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
        
        sns.barplot(data=recall_df, x='lighting', y='recall_rate', hue='method', ax=axes[1, 1], palette=colors)
        axes[1, 1].set_title('View Recall Rate Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Recall Rate', fontsize=12)
        axes[1, 1].set_xlabel('Lighting Condition', fontsize=12)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        axes[1, 1].legend(title='Detection Method', title_fontsize=11, fontsize=10)
        
        # Add value annotations
        for i, p in enumerate(axes[1, 1].patches):
            axes[1, 1].annotate(f'{p.get_height():.1%}', 
                              (p.get_x() + p.get_width() / 2., p.get_height()), 
                              ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. FPS Comparison
    print("Drawing FPS comparison...")
    sns.boxplot(data=df, x='lighting', y='fps', hue='method', ax=axes[1, 2], palette=colors)
    axes[1, 2].set_title('FPS Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('FPS', fontsize=12)
    axes[1, 2].set_xlabel('Lighting Condition', fontsize=12)
    axes[1, 2].tick_params(axis='both', which='major', labelsize=10)
    axes[1, 2].legend(title='Detection Method', title_fontsize=11, fontsize=10)
    
    plt.tight_layout()
    
    # Save main chart
    chart_file = output_dir / "comprehensive_evaluation_charts_english.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive chart saved: {chart_file}")
    
    # Generate detailed analysis charts
    generate_detailed_analysis(df, output_dir)

def generate_detailed_analysis(df, output_dir):
    """Generate detailed analysis charts"""
    try:
        print("Generating detailed analysis charts...")
        
        # Create new charts
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Detailed Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        colors = ['#1f77b4', '#ff7f0e']
        method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
        
        # 1. Error Distribution Histogram
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[0, 0].hist(method_df['pos_err_m'] * 1000, alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[0, 0].set_title('Position Error Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Position Error (mm)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 2. Detection Time vs Error Scatter Plot
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[0, 1].scatter(method_df['detection_time_s'], method_df['pos_err_m'] * 1000, 
                                 alpha=0.7, label=method_names[method], color=colors[i], s=50)
        
        axes[0, 1].set_title('Detection Time vs Position Error', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Detection Time (s)', fontsize=12)
        axes[0, 1].set_ylabel('Position Error (mm)', fontsize=12)
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=10)
        
        # 3. Success Rate Heatmap
        success_matrix = []
        for method in ['hsv_depth', 'depth_clustering']:
            row = []
            for lighting in ['normal', 'bright', 'dim']:
                subset = df[(df['method'] == method) & (df['lighting'] == lighting)]
                if len(subset) > 0:
                    success_rate = subset['success_pos_2cm'].mean()
                    row.append(success_rate)
                else:
                    row.append(0.0)
            success_matrix.append(row)
        
        if success_matrix:
            sns.heatmap(success_matrix, 
                       xticklabels=['Normal', 'Bright', 'Dim'],
                       yticklabels=['HSV+Depth', 'Depth+Clustering'],
                       annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 0],
                       cbar_kws={'label': 'Success Rate'})
            axes[1, 0].set_title('Success Rate Heatmap (2cm threshold)', fontsize=14, fontweight='bold')
            axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 4. Performance Comparison Bar Chart (instead of radar chart)
        # Calculate comprehensive performance metrics
        performance_data = []
        method_labels = []
        
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                # Normalize metrics (0-1 scale, 1 is best)
                accuracy = 1.0 / (1.0 + method_df['pos_err_m'].mean() * 1000 / 50)  # 50mm baseline
                speed = 1.0 / (1.0 + method_df['detection_time_s'].mean() / 30)  # 30s baseline
                success_rate = method_df['success_pos_2cm'].mean()
                recall_rate = method_df['view_hit_rate'].mean()
                
                performance_data.append([accuracy, speed, success_rate, recall_rate])
                method_labels.append(method_names[method])
        
        if performance_data:
            # Create bar chart instead of radar chart
            categories = ['Accuracy', 'Speed', 'Success Rate', 'Recall Rate']
            x = np.arange(len(categories))
            width = 0.35
            
            for i, (data, name) in enumerate(zip(performance_data, method_labels)):
                axes[1, 1].bar(x + i*width, data, width, label=name, alpha=0.8, color=colors[i])
            
            axes[1, 1].set_title('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Performance Metrics', fontsize=12)
            axes[1, 1].set_ylabel('Normalized Score', fontsize=12)
            axes[1, 1].set_xticks(x + width/2)
            axes[1, 1].set_xticklabels(categories, fontsize=10)
            axes[1, 1].legend(fontsize=11)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
            axes[1, 1].set_ylim(0, 1)
            
            # Add value annotations
            for i, (data, name) in enumerate(zip(performance_data, method_labels)):
                for j, value in enumerate(data):
                    axes[1, 1].text(j + i*width, value + 0.02, f'{value:.2f}', 
                                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save detailed charts
        detailed_chart_file = output_dir / "detailed_analysis_charts_english.png"
        plt.savefig(detailed_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed analysis charts saved: {detailed_chart_file}")
        
        # Generate statistical summary
        generate_statistical_summary(df, output_dir)
        
    except Exception as e:
        print(f"Failed to generate detailed charts: {e}")
        import traceback
        traceback.print_exc()

def generate_statistical_summary(df, output_dir):
    """Generate statistical summary"""
    try:
        print("Generating statistical summary...")
        
        # Create statistical summary charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Summary Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        colors = ['#1f77b4', '#ff7f0e']
        method_names = {'hsv_depth': 'HSV+Depth', 'depth_clustering': 'Depth+Clustering'}
        
        # 1. Performance comparison table
        summary_data = []
        for method in ['hsv_depth', 'depth_clustering']:
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                summary_data.append({
                    'Method': method_names[method],
                    'Pos Error(mm)': f"{method_df['pos_err_m'].mean()*1000:.2f}±{method_df['pos_err_m'].std()*1000:.2f}",
                    'Range Error(mm)': f"{method_df['range_err_m'].mean()*1000:.2f}±{method_df['range_err_m'].std()*1000:.2f}",
                    'Detection Time(s)': f"{method_df['detection_time_s'].mean():.2f}±{method_df['detection_time_s'].std():.2f}",
                    'Success Rate(%)': f"{method_df['success_pos_2cm'].mean()*100:.1f}",
                    'Recall Rate(%)': f"{method_df['view_hit_rate'].mean()*100:.1f}"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            axes[0, 0].axis('tight')
            axes[0, 0].axis('off')
            table = axes[0, 0].table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[0, 0].set_title('Performance Comparison Table', fontsize=14, fontweight='bold')
        
        # 2. Lighting condition impact analysis
        lighting_impact = []
        for lighting in ['normal', 'bright', 'dim']:
            lighting_df = df[df['lighting'] == lighting]
            if len(lighting_df) > 0:
                lighting_impact.append({
                    'Lighting': lighting.capitalize(),
                    'Avg Pos Error(mm)': lighting_df['pos_err_m'].mean() * 1000,
                    'Avg Detection Time(s)': lighting_df['detection_time_s'].mean(),
                    'Avg Success Rate(%)': lighting_df['success_pos_2cm'].mean() * 100
                })
        
        if lighting_impact:
            lighting_df = pd.DataFrame(lighting_impact)
            axes[0, 1].axis('tight')
            axes[0, 1].axis('off')
            table = axes[0, 1].table(cellText=lighting_df.values, colLabels=lighting_df.columns, 
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            axes[0, 1].set_title('Lighting Condition Impact', fontsize=14, fontweight='bold')
        
        # 3. Range error distribution comparison
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 0].hist(method_df['range_err_m'] * 1000, alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[1, 0].set_title('Range Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Range Error (mm)', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
        
        # 4. Detection time distribution
        for i, method in enumerate(['hsv_depth', 'depth_clustering']):
            method_df = df[df['method'] == method]
            if len(method_df) > 0:
                axes[1, 1].hist(method_df['detection_time_s'], alpha=0.7, 
                               label=method_names[method], bins=15, color=colors[i])
        
        axes[1, 1].set_title('Detection Time Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Detection Time (s)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        # Save statistical summary chart
        summary_chart_file = output_dir / "statistical_summary_english.png"
        plt.savefig(summary_chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical summary chart saved: {summary_chart_file}")
        
    except Exception as e:
        print(f"Failed to generate statistical summary: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("Experimental Results Visualization Tool (English Version)")
    print("="*60)
    
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
    
    # Generate visualizations
    generate_comprehensive_visualization(df, viz_dir)
    
    print(f"\nVisualization completed! Results saved in: {viz_dir}")
    print("Generated files:")
    for file in viz_dir.glob("*english*.png"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
