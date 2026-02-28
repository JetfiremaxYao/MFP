#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单稳定性分析 - 不依赖外部库
直接从CSV文件读取数据并分析
"""

import csv
from pathlib import Path

def read_csv_data(file_path):
    """读取CSV文件数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

def analyze_stability():
    """分析稳定性数据"""
    
    # 读取数据
    results_dir = Path("../evaluation_results/canny_vs_rgbd_comparison_20250827_172724")
    
    # 读取分组摘要数据
    grouped_file = results_dir / "boundary_tracking_grouped_summary.csv"
    grouped_data = read_csv_data(grouped_file)
    
    if not grouped_data:
        print("无法读取分组数据")
        return
    
    # 读取总体摘要数据
    summary_file = results_dir / "boundary_tracking_summary.csv"
    summary_data = read_csv_data(summary_file)
    
    if not summary_data:
        print("无法读取总体摘要数据")
        return
    
    print("="*80)
    print("边界追踪算法稳定性分析")
    print("="*80)
    
    # 分析不同光照条件下的稳定性
    print("\n📊 不同光照条件下的残差标准差对比 (mm):")
    print("-" * 60)
    
    lightings = ['normal', 'bright', 'dim']
    canny_results = {}
    rgbd_results = {}
    
    for lighting in lightings:
        # 查找Canny数据
        canny_row = None
        rgbd_row = None
        
        for row in grouped_data:
            if row['lighting'] == lighting:
                if row['method'] == 'canny':
                    canny_row = row
                elif row['method'] == 'rgbd':
                    rgbd_row = row
        
        if canny_row and rgbd_row:
            canny_std = float(canny_row['std_residual_mm_mean'])
            rgbd_std = float(rgbd_row['std_residual_mm_mean'])
            
            canny_results[lighting] = canny_std
            rgbd_results[lighting] = rgbd_std
            
            # 判断优势方法
            if canny_std < rgbd_std:
                winner = "🏆 Canny"
            elif rgbd_std < canny_std:
                winner = "🏆 RGB-D"
            else:
                winner = "⚖️ 平手"
            
            print(f"{lighting.upper():>8}: Canny={canny_std:>6.2f} | RGB-D={rgbd_std:>6.2f} | {winner}")
    
    # 分析总体稳定性
    print("\n📈 总体稳定性对比:")
    print("-" * 60)
    
    canny_overall = None
    rgbd_overall = None
    
    for row in summary_data:
        if row['method'] == 'canny':
            canny_overall = {
                'std_mean': float(row['std_residual_mm_mean']),
                'std_std': float(row['std_residual_mm_std']),
                'chamfer_mean': float(row['chamfer_distance_mm_mean'])
            }
        elif row['method'] == 'rgbd':
            rgbd_overall = {
                'std_mean': float(row['std_residual_mm_mean']),
                'std_std': float(row['std_residual_mm_std']),
                'chamfer_mean': float(row['chamfer_distance_mm_mean'])
            }
    
    if canny_overall and rgbd_overall:
        print(f"Canny方法:  {canny_overall['std_mean']:>6.2f}±{canny_overall['std_std']:>5.2f}mm")
        print(f"RGB-D方法:  {rgbd_overall['std_mean']:>6.2f}±{rgbd_overall['std_std']:>5.2f}mm")
        
        # 稳定性排名
        print(f"\n🏆 稳定性排名:")
        if canny_overall['std_mean'] < rgbd_overall['std_mean']:
            print(f"  1. Canny方法 - 更稳定 ({canny_overall['std_mean']:.2f}mm)")
            print(f"  2. RGB-D方法 - 相对不稳定 ({rgbd_overall['std_mean']:.2f}mm)")
        else:
            print(f"  1. RGB-D方法 - 更稳定 ({rgbd_overall['std_mean']:.2f}mm)")
            print(f"  2. Canny方法 - 相对不稳定 ({canny_overall['std_mean']:.2f}mm)")
    
    # 稳定性波动性分析
    print(f"\n📊 稳定性波动性分析:")
    print("-" * 60)
    
    if canny_overall and rgbd_overall:
        print(f"Canny方法波动性: {canny_overall['std_std']:.2f}mm")
        print(f"RGB-D方法波动性: {rgbd_overall['std_std']:.2f}mm")
        
        if canny_overall['std_std'] < rgbd_overall['std_std']:
            print("✅ Canny方法波动性更小，稳定性更一致")
        else:
            print("✅ RGB-D方法波动性更小，稳定性更一致")
    
    # 光照条件影响分析
    print(f"\n💡 光照条件影响分析:")
    print("-" * 60)
    
    if canny_results and rgbd_results:
        # Canny方法在不同光照下的表现
        canny_variation = max(canny_results.values()) - min(canny_results.values())
        rgbd_variation = max(rgbd_results.values()) - min(rgbd_results.values())
        
        print(f"Canny方法在不同光照下的变化范围: {canny_variation:.2f}mm")
        print(f"RGB-D方法在不同光照下的变化范围: {rgbd_variation:.2f}mm")
        
        if canny_variation < rgbd_variation:
            print("✅ Canny方法对光照条件变化不敏感，更稳定")
        else:
            print("✅ RGB-D方法对光照条件变化不敏感，更稳定")
    
    # 综合性能分析
    print(f"\n🎯 综合性能分析:")
    print("-" * 60)
    
    if canny_overall and rgbd_overall:
        # 计算综合得分 (残差标准差越小越好，Chamfer距离越小越好)
        canny_score = 1 / (1 + canny_overall['std_mean'] / 10)  # 稳定性得分
        rgbd_score = 1 / (1 + rgbd_overall['std_mean'] / 10)
        
        canny_accuracy = 1 / (1 + canny_overall['chamfer_mean'] / 20)  # 准确性得分
        rgbd_accuracy = 1 / (1 + rgbd_overall['chamfer_mean'] / 20)
        
        print(f"Canny方法综合得分: 稳定性={canny_score:.3f}, 准确性={canny_accuracy:.3f}")
        print(f"RGB-D方法综合得分: 稳定性={rgbd_score:.3f}, 准确性={rgbd_accuracy:.3f}")
        
        canny_total = (canny_score + canny_accuracy) / 2
        rgbd_total = (rgbd_score + rgbd_accuracy) / 2
        
        print(f"\n🏆 综合推荐:")
        if canny_total > rgbd_total:
            print(f"  Canny方法 ({canny_total:.3f}) > RGB-D方法 ({rgbd_total:.3f})")
            print("  ✅ 推荐使用Canny方法")
        else:
            print(f"  RGB-D方法 ({rgbd_total:.3f}) > Canny方法 ({canny_total:.3f})")
            print("  ✅ 推荐使用RGB-D方法")
    
    # 关键结论
    print(f"\n🔍 关键结论:")
    print("-" * 60)
    
    if canny_overall and rgbd_overall:
        if canny_overall['std_mean'] < rgbd_overall['std_mean']:
            print("• Canny方法在稳定性方面表现更优")
            print("• Canny方法适合需要高稳定性的应用场景")
        else:
            print("• RGB-D方法在稳定性方面表现更优")
            print("• RGB-D方法适合需要高稳定性的应用场景")
        
        if canny_overall['std_std'] < rgbd_overall['std_std']:
            print("• Canny方法稳定性更一致，波动性更小")
        else:
            print("• RGB-D方法稳定性更一致，波动性更小")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_stability()



