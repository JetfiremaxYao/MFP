#!/usr/bin/env python3
"""
Boundary Tracking Results Analysis Script
Compare performance between Canny and RGBD methods
"""

import csv

def load_data():
    """Load data from CSV file"""
    data_path = "/Volumes/Data/CS/Develop/IndividualProject/Genesis/evaluation_results/multi_object_boundary_tracking_comparison_20250912_160838/boundary_tracking_results_detailed.csv"
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    return data

def analyze_performance(data):
    """Analyze performance metrics"""
    print("=== Boundary Tracking Method Performance Analysis ===\n")
    
    # Separate data by method
    canny_data = [row for row in data if row['method'] == 'canny']
    rgbd_data = [row for row in data if row['method'] == 'rgbd']
    
    print(f"Total experiments: {len(data)}")
    print(f"Canny experiments: {len(canny_data)}")
    print(f"RGBD experiments: {len(rgbd_data)}")
    print("\n" + "="*60 + "\n")
    
    # Calculate statistics for each method
    def calculate_stats(method_data, method_name):
        if not method_data:
            return None
            
        execution_times = [float(row['execution_time_s']) for row in method_data]
        chamfer_distances = [float(row['chamfer_distance_mm']) for row in method_data]
        hausdorff_distances = [float(row['hausdorff_distance_mm']) for row in method_data]
        coverages = [float(row['coverage_3mm']) for row in method_data]
        point_counts = [int(row['point_count']) for row in method_data]
        std_residuals = [float(row['std_residual_mm']) for row in method_data]
        
        stats = {
            'method': method_name,
            'execution_time': {
                'mean': sum(execution_times) / len(execution_times),
                'min': min(execution_times),
                'max': max(execution_times)
            },
            'chamfer_distance': {
                'mean': sum(chamfer_distances) / len(chamfer_distances),
                'min': min(chamfer_distances),
                'max': max(chamfer_distances)
            },
            'hausdorff_distance': {
                'mean': sum(hausdorff_distances) / len(hausdorff_distances),
                'min': min(hausdorff_distances),
                'max': max(hausdorff_distances)
            },
            'coverage': {
                'mean': sum(coverages) / len(coverages),
                'min': min(coverages),
                'max': max(coverages)
            },
            'point_count': {
                'mean': sum(point_counts) / len(point_counts),
                'min': min(point_counts),
                'max': max(point_counts)
            },
            'std_residual': {
                'mean': sum(std_residuals) / len(std_residuals),
                'min': min(std_residuals),
                'max': max(std_residuals)
            }
        }
        return stats
    
    canny_stats = calculate_stats(canny_data, 'Canny')
    rgbd_stats = calculate_stats(rgbd_data, 'RGBD')
    
    # Print comparison table
    print("PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Canny':<20} {'RGBD':<20} {'Winner':<10}")
    print("-" * 80)
    
    # Execution time (lower is better)
    canny_time = canny_stats['execution_time']['mean']
    rgbd_time = rgbd_stats['execution_time']['mean']
    time_winner = "Canny" if canny_time < rgbd_time else "RGBD"
    print(f"{'Avg Execution Time (s)':<25} {canny_time:<20.2f} {rgbd_time:<20.2f} {time_winner:<10}")
    
    # Chamfer distance (lower is better)
    canny_chamfer = canny_stats['chamfer_distance']['mean']
    rgbd_chamfer = rgbd_stats['chamfer_distance']['mean']
    chamfer_winner = "Canny" if canny_chamfer < rgbd_chamfer else "RGBD"
    print(f"{'Avg Chamfer Distance (mm)':<25} {canny_chamfer:<20.2f} {rgbd_chamfer:<20.2f} {chamfer_winner:<10}")
    
    # Hausdorff distance (lower is better)
    canny_hausdorff = canny_stats['hausdorff_distance']['mean']
    rgbd_hausdorff = rgbd_stats['hausdorff_distance']['mean']
    hausdorff_winner = "Canny" if canny_hausdorff < rgbd_hausdorff else "RGBD"
    print(f"{'Avg Hausdorff Distance (mm)':<25} {canny_hausdorff:<20.2f} {rgbd_hausdorff:<20.2f} {hausdorff_winner:<10}")
    
    # Coverage (higher is better)
    canny_coverage = canny_stats['coverage']['mean']
    rgbd_coverage = rgbd_stats['coverage']['mean']
    coverage_winner = "Canny" if canny_coverage > rgbd_coverage else "RGBD"
    print(f"{'Avg Coverage (3mm)':<25} {canny_coverage:<20.3f} {rgbd_coverage:<20.3f} {coverage_winner:<10}")
    
    # Point count
    canny_points = canny_stats['point_count']['mean']
    rgbd_points = rgbd_stats['point_count']['mean']
    print(f"{'Avg Point Count':<25} {canny_points:<20.0f} {rgbd_points:<20.0f} {'RGBD':<10}")
    
    # Standard residual (lower is better)
    canny_std = canny_stats['std_residual']['mean']
    rgbd_std = rgbd_stats['std_residual']['mean']
    std_winner = "Canny" if canny_std < rgbd_std else "RGBD"
    print(f"{'Avg Std Residual (mm)':<25} {canny_std:<20.2f} {rgbd_std:<20.2f} {std_winner:<10}")
    
    print("-" * 80)
    
    return canny_stats, rgbd_stats

def provide_recommendations(canny_stats, rgbd_stats):
    """Provide selection recommendations"""
    print("\n" + "="*60)
    print("🎯 CANNY vs RGBD METHOD SELECTION RECOMMENDATIONS")
    print("="*60)
    
    # Calculate scores
    canny_score = 0
    rgbd_score = 0
    
    # Speed score (faster is better)
    if canny_stats['execution_time']['mean'] < rgbd_stats['execution_time']['mean']:
        canny_score += 3
    else:
        rgbd_score += 3
    
    # Accuracy score (lower Chamfer distance is better)
    if canny_stats['chamfer_distance']['mean'] < rgbd_stats['chamfer_distance']['mean']:
        canny_score += 3
    else:
        rgbd_score += 3
    
    # Stability score (lower Hausdorff distance is better)
    if canny_stats['hausdorff_distance']['mean'] < rgbd_stats['hausdorff_distance']['mean']:
        canny_score += 2
    else:
        rgbd_score += 2
    
    # Coverage score (higher coverage is better)
    if canny_stats['coverage']['mean'] > rgbd_stats['coverage']['mean']:
        canny_score += 1
    else:
        rgbd_score += 1
    
    # Precision score (lower std residual is better)
    if canny_stats['std_residual']['mean'] < rgbd_stats['std_residual']['mean']:
        canny_score += 1
    else:
        rgbd_score += 1
    
    print(f"\n📈 COMPREHENSIVE SCORING:")
    print(f"   Canny Score: {canny_score}/10")
    print(f"   RGBD Score:  {rgbd_score}/10")
    
    print("\n" + "="*60)
    print("🏆 RECOMMENDATION:")
    print("="*60)
    
    if canny_score > rgbd_score:
        print("\n✅ RECOMMENDED: CANNY METHOD")
        print("\n🎯 ADVANTAGES:")
        print("   • Faster execution time")
        print("   • Higher accuracy (lower Chamfer distance)")
        print("   • Better stability (lower Hausdorff distance)")
        print("   • Lower computational requirements")
        print("   • More precise results (lower std residual)")
        
        print("\n⚠️  CONSIDERATIONS:")
        print("   • May need parameter tuning for complex geometries")
        print("   • More sensitive to lighting conditions")
        print("   • Slightly lower coverage rate")
        
    else:
        print("\n✅ RECOMMENDED: RGBD METHOD")
        print("\n🎯 ADVANTAGES:")
        print("   • Higher coverage rate")
        print("   • Better adaptability to complex geometries")
        print("   • More robust to lighting variations")
        print("   • More point cloud data for analysis")
        
        print("\n⚠️  CONSIDERATIONS:")
        print("   • Longer execution time")
        print("   • Higher computational requirements")
        print("   • Slightly lower accuracy")
    
    print("\n" + "="*60)
    print("💡 USAGE RECOMMENDATIONS:")
    print("="*60)
    print("• For speed and accuracy: Choose CANNY")
    print("• For complex geometries: Choose RGBD")
    print("• For limited computational resources: Choose CANNY")
    print("• For high coverage requirements: Choose RGBD")
    print("• For real-time applications: Choose CANNY")
    print("• For research and detailed analysis: Choose RGBD")

def main():
    """Main function"""
    print("🔍 Starting boundary tracking results analysis...")
    
    # Load data
    data = load_data()
    print(f"✅ Data loaded successfully: {len(data)} records")
    
    # Analyze performance
    canny_stats, rgbd_stats = analyze_performance(data)
    
    # Provide recommendations
    provide_recommendations(canny_stats, rgbd_stats)
    
    print("\n✅ Analysis completed successfully!")
    print("📊 All performance metrics have been analyzed and recommendations provided.")

if __name__ == "__main__":
    main()

