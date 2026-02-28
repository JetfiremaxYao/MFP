#!/usr/bin/env python3
"""
Core Metrics Visualization for Boundary Tracking Evaluation
Focus on: Chamfer Distance, Hausdorff Distance, Std Residual, CV Residual, Execution Time, Point Count
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoreMetricsVisualizer:
    """Core metrics visualization for boundary tracking evaluation"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.data = self.load_data()
        self.output_dir = Path("evaluation_results/core_metrics_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define core metrics
        self.core_metrics = {
            'chamfer_distance_mm': 'Chamfer Distance (mm)',
            'hausdorff_distance_mm': 'Hausdorff Distance (mm)', 
            'std_residual_mm': 'Standard Residual (mm)',
            'cv_residual': 'Coefficient of Variation',
            'execution_time_s': 'Execution Time (s)',
            'point_count': 'Point Count'
        }
        
        # Define objects and lighting conditions
        self.objects = ['Cube', 'ctry.obj']
        self.lighting_conditions = ['normal', 'bright', 'dim']
        self.methods = ['canny', 'rgbd']
        
        # Color scheme
        self.colors = {
            'canny': '#1f77b4',
            'rgbd': '#ff7f0e'
        }
    
    def load_data(self):
        """Load data from CSV file"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison plots for all core metrics"""
        print("Creating comprehensive core metrics visualization...")
        
        # Create main comparison figure
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplots for each metric
        for i, (metric, metric_name) in enumerate(self.core_metrics.items()):
            ax = plt.subplot(3, 2, i+1)
            self._plot_metric_comparison(ax, metric, metric_name)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "core_metrics_comprehensive_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Comprehensive comparison saved")
    
    def create_object_specific_analysis(self):
        """Create object-specific analysis (Cube vs ctry.obj)"""
        print("Creating object-specific analysis...")
        
        for obj in self.objects:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (metric, metric_name) in enumerate(self.core_metrics.items()):
                ax = axes[i]
                self._plot_object_metric(ax, obj, metric, metric_name)
            
            plt.suptitle(f'Core Metrics Analysis - {obj}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"core_metrics_{obj.lower()}_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {obj} analysis saved")
    
    def create_lighting_condition_analysis(self):
        """Create lighting condition specific analysis"""
        print("Creating lighting condition analysis...")
        
        for lighting in self.lighting_conditions:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, (metric, metric_name) in enumerate(self.core_metrics.items()):
                ax = axes[i]
                self._plot_lighting_metric(ax, lighting, metric, metric_name)
            
            plt.suptitle(f'Core Metrics Analysis - {lighting.upper()} Lighting', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"core_metrics_{lighting}_lighting_analysis.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ {lighting} lighting analysis saved")
    
    def create_radar_chart_comparison(self):
        """Create radar chart comparison for all conditions"""
        print("Creating radar chart comparison...")
        
        # Create radar charts for each object and lighting condition
        for obj in self.objects:
            for lighting in self.lighting_conditions:
                self._create_radar_chart(obj, lighting)
        
        print("✅ Radar charts saved")
    
    def create_performance_heatmap(self):
        """Create performance heatmap"""
        print("Creating performance heatmap...")
        
        # Prepare data for heatmap
        heatmap_data = []
        
        for obj in self.objects:
            for lighting in self.lighting_conditions:
                for method in self.methods:
                    # Get data for this combination
                    obj_data = [row for row in self.data 
                              if row['object_name'] == obj and 
                                 row['lighting'] == lighting and 
                                 row['method'] == method]
                    
                    if obj_data:
                        # Calculate normalized scores (lower is better for most metrics)
                        scores = {}
                        for metric in self.core_metrics.keys():
                            values = [float(row[metric]) for row in obj_data if row[metric] != '']
                            if values:
                                if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 
                                           'std_residual_mm', 'cv_residual', 'execution_time_s']:
                                    # Lower is better - invert score
                                    max_val = max([float(row[metric]) for row in self.data 
                                                 if row[metric] != ''])
                                    scores[metric] = 1 - (np.mean(values) / max_val)
                                else:  # point_count - higher is better
                                    max_val = max([float(row[metric]) for row in self.data 
                                                 if row[metric] != ''])
                                    scores[metric] = np.mean(values) / max_val
                            else:
                                scores[metric] = 0
                        
                        heatmap_data.append({
                            'Object': obj,
                            'Lighting': lighting,
                            'Method': method.upper(),
                            **scores
                        })
        
        # Create heatmap
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Pivot for heatmap
        pivot_data = df_heatmap.pivot_table(
            index=['Object', 'Lighting'], 
            columns='Method', 
            values=list(self.core_metrics.keys())
        )
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.core_metrics.keys()):
            ax = axes[i]
            
            # Get data for this metric
            metric_data = pivot_data[metric]
            
            # Create heatmap
            sns.heatmap(metric_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       ax=ax, cbar_kws={'label': 'Normalized Score'})
            ax.set_title(f'{self.core_metrics[metric]}', fontweight='bold')
            ax.set_xlabel('Method')
            ax.set_ylabel('Object & Lighting')
        
        plt.suptitle('Performance Heatmap - Normalized Scores', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Performance heatmap saved")
    
    def create_statistical_summary(self):
        """Create statistical summary table"""
        print("Creating statistical summary...")
        
        summary_data = []
        
        for obj in self.objects:
            for lighting in self.lighting_conditions:
                for method in self.methods:
                    obj_data = [row for row in self.data 
                              if row['object_name'] == obj and 
                                 row['lighting'] == lighting and 
                                 row['method'] == method]
                    
                    if obj_data:
                        summary_row = {
                            'Object': obj,
                            'Lighting': lighting,
                            'Method': method.upper(),
                            'N': len(obj_data)
                        }
                        
                        # Calculate statistics for each metric
                        for metric in self.core_metrics.keys():
                            values = [float(row[metric]) for row in obj_data if row[metric] != '']
                            if values:
                                summary_row[f'{metric}_mean'] = np.mean(values)
                                summary_row[f'{metric}_std'] = np.std(values)
                                summary_row[f'{metric}_min'] = np.min(values)
                                summary_row[f'{metric}_max'] = np.max(values)
                                summary_row[f'{metric}_median'] = np.median(values)
                            else:
                                summary_row[f'{metric}_mean'] = np.nan
                                summary_row[f'{metric}_std'] = np.nan
                                summary_row[f'{metric}_min'] = np.nan
                                summary_row[f'{metric}_max'] = np.nan
                                summary_row[f'{metric}_median'] = np.nan
                        
                        summary_data.append(summary_row)
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "statistical_summary.csv", index=False)
        print("✅ Statistical summary saved")
        
        # Create summary visualization
        self._create_summary_visualization(summary_df)
    
    def _plot_metric_comparison(self, ax, metric, metric_name):
        """Plot metric comparison across all conditions"""
        data_to_plot = []
        labels = []
        
        for obj in self.objects:
            for lighting in self.lighting_conditions:
                for method in self.methods:
                    obj_data = [row for row in self.data 
                              if row['object_name'] == obj and 
                                 row['lighting'] == lighting and 
                                 row['method'] == method]
                    
                    if obj_data:
                        values = [float(row[metric]) for row in obj_data if row[metric] != '']
                        if values:
                            data_to_plot.append(values)
                            labels.append(f'{method.upper()}\n{obj}\n{lighting}')
        
        if data_to_plot:
            # Create box plot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            for i, patch in enumerate(bp['boxes']):
                if 'CANNY' in labels[i]:
                    patch.set_facecolor(self.colors['canny'])
                else:
                    patch.set_facecolor(self.colors['rgbd'])
                patch.set_alpha(0.7)
            
            ax.set_title(metric_name, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _plot_object_metric(self, ax, obj, metric, metric_name):
        """Plot metric for specific object"""
        data_to_plot = []
        labels = []
        
        for lighting in self.lighting_conditions:
            for method in self.methods:
                obj_data = [row for row in self.data 
                          if row['object_name'] == obj and 
                             row['lighting'] == lighting and 
                             row['method'] == method]
                
                if obj_data:
                    values = [float(row[metric]) for row in obj_data if row[metric] != '']
                    if values:
                        data_to_plot.append(values)
                        labels.append(f'{method.upper()}\n{lighting}')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for i, patch in enumerate(bp['boxes']):
                if 'CANNY' in labels[i]:
                    patch.set_facecolor(self.colors['canny'])
                else:
                    patch.set_facecolor(self.colors['rgbd'])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric_name} - {obj}', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _plot_lighting_metric(self, ax, lighting, metric, metric_name):
        """Plot metric for specific lighting condition"""
        data_to_plot = []
        labels = []
        
        for obj in self.objects:
            for method in self.methods:
                obj_data = [row for row in self.data 
                          if row['object_name'] == obj and 
                             row['lighting'] == lighting and 
                             row['method'] == method]
                
                if obj_data:
                    values = [float(row[metric]) for row in obj_data if row[metric] != '']
                    if values:
                        data_to_plot.append(values)
                        labels.append(f'{method.upper()}\n{obj}')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for i, patch in enumerate(bp['boxes']):
                if 'CANNY' in labels[i]:
                    patch.set_facecolor(self.colors['canny'])
                else:
                    patch.set_facecolor(self.colors['rgbd'])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{metric_name} - {lighting.upper()} Lighting', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
    
    def _create_radar_chart(self, obj, lighting):
        """Create radar chart for specific object and lighting condition"""
        # Get data for this combination
        obj_data = [row for row in self.data 
                   if row['object_name'] == obj and row['lighting'] == lighting]
        
        if not obj_data:
            return
        
        # Calculate average metrics for each method
        methods_data = {}
        for method in self.methods:
            method_data = [row for row in obj_data if row['method'] == method]
            if method_data:
                metrics = {}
                for metric in self.core_metrics.keys():
                    values = [float(row[metric]) for row in method_data if row[metric] != '']
                    if values:
                        metrics[metric] = np.mean(values)
                    else:
                        metrics[metric] = 0
                methods_data[method] = metrics
        
        if not methods_data:
            return
        
        # Normalize metrics for radar chart
        normalized_data = {}
        for method, metrics in methods_data.items():
            normalized_metrics = {}
            for metric, value in metrics.items():
                if metric in ['chamfer_distance_mm', 'hausdorff_distance_mm', 
                            'std_residual_mm', 'cv_residual', 'execution_time_s']:
                    # Lower is better - normalize to 0-1 (inverted)
                    max_val = max([float(row[metric]) for row in self.data 
                                 if row[metric] != ''])
                    normalized_metrics[metric] = 1 - (value / max_val) if max_val > 0 else 0
                else:  # point_count - higher is better
                    max_val = max([float(row[metric]) for row in self.data 
                                 if row[metric] != ''])
                    normalized_metrics[metric] = value / max_val if max_val > 0 else 0
            normalized_data[method] = normalized_metrics
        
        # Create radar chart
        categories = list(self.core_metrics.keys())
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for method, metrics in normalized_data.items():
            values = [metrics[cat] for cat in categories]
            values += values[:1]  # Close the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method.upper(), 
                   color=self.colors[method], alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=self.colors[method])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.core_metrics[cat] for cat in categories])
        ax.set_ylim(0, 1)
        ax.set_title(f'Performance Radar - {obj} ({lighting.upper()} Lighting)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"radar_{obj.lower()}_{lighting}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_visualization(self, summary_df):
        """Create summary visualization"""
        # Create bar chart comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.core_metrics.keys()):
            ax = axes[i]
            
            # Pivot data for plotting
            pivot_data = summary_df.pivot_table(
                index=['Object', 'Lighting'], 
                columns='Method', 
                values=f'{metric}_mean'
            )
            
            # Create bar plot
            pivot_data.plot(kind='bar', ax=ax, color=[self.colors['canny'], self.colors['rgbd']])
            ax.set_title(f'{self.core_metrics[metric]} - Mean Values', fontweight='bold')
            ax.set_xlabel('Object & Lighting')
            ax.set_ylabel(self.core_metrics[metric])
            ax.legend(title='Method')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Core Metrics Summary - Mean Values Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_all_analysis(self):
        """Run all analysis and create visualizations"""
        print("🚀 Starting core metrics visualization analysis...")
        print(f"📊 Analyzing {len(self.data)} data points")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Create all visualizations
        self.create_comprehensive_comparison()
        self.create_object_specific_analysis()
        self.create_lighting_condition_analysis()
        self.create_radar_chart_comparison()
        self.create_performance_heatmap()
        self.create_statistical_summary()
        
        print("\n✅ All core metrics visualizations completed!")
        print(f"📁 Results saved in: {self.output_dir}")

def main():
    """Main function"""
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    # Create visualizer
    visualizer = CoreMetricsVisualizer(data_path)
    
    # Run analysis
    visualizer.run_all_analysis()

if __name__ == "__main__":
    main()
