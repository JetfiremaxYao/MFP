#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速数据对比脚本
直接显示关键指标对比
"""

import pandas as pd
import numpy as np

def quick_comparison():
    """快速对比分析"""
    
    # 基于您提供的数据
    print("="*60)
    print("边界追踪算法关键指标对比")
    print("="*60)
    
    # 数据摘要（基于您的实验结果）
    data = {
        'Canny': {
            'Chamfer距离(mm)': 3.04,
            'Hausdorff距离(mm)': 18.23,
            '执行时间(秒)': 153.78,
            '成功率': '100%',
            '平均点数': 3765
        },
        'RGB-D': {
            'Chamfer距离(mm)': 6.65,
            'Hausdorff距离(mm)': 20.21,
            '执行时间(秒)': 51.06,
            '成功率': '100%',
            '平均点数': 6874
        }
    }
    
    # 打印对比表格
    print("\n关键指标对比表:")
    print("-" * 60)
    print(f"{'指标':<15} {'Canny':<12} {'RGB-D':<12} {'更优':<8}")
    print("-" * 60)
    
    for metric in ['Chamfer距离(mm)', 'Hausdorff距离(mm)', '执行时间(秒)', '成功率', '平均点数']:
        canny_val = data['Canny'][metric]
        rgbd_val = data['RGB-D'][metric]
        
        if metric in ['Chamfer距离(mm)', 'Hausdorff距离(mm)', '执行时间(秒)']:
            if canny_val < rgbd_val:
                better = 'Canny'
            else:
                better = 'RGB-D'
        else:  # 成功率、点数
            if canny_val > rgbd_val:
                better = 'Canny'
            else:
                better = 'RGB-D'
        
        print(f"{metric:<15} {canny_val:<12} {rgbd_val:<12} {better:<8}")
    
    print("-" * 60)
    
    # 详细分析
    print("\n详细分析:")
    print("-" * 60)
    
    # Chamfer距离分析
    chamfer_diff = data['RGB-D']['Chamfer距离(mm)'] - data['Canny']['Chamfer距离(mm)']
    chamfer_ratio = data['RGB-D']['Chamfer距离(mm)'] / data['Canny']['Chamfer距离(mm)']
    print(f"1. 准确性 (Chamfer距离):")
    print(f"   - Canny: {data['Canny']['Chamfer距离(mm)']:.2f}mm")
    print(f"   - RGB-D: {data['RGB-D']['Chamfer距离(mm)']:.2f}mm")
    print(f"   - 差异: {chamfer_diff:.2f}mm (RGB-D比Canny大{chamfer_ratio:.1f}倍)")
    print(f"   - 结论: Canny方法在准确性上显著优于RGB-D方法")
    
    # Hausdorff距离分析
    hausdorff_diff = data['RGB-D']['Hausdorff距离(mm)'] - data['Canny']['Hausdorff距离(mm)']
    print(f"\n2. 稳定性 (Hausdorff距离):")
    print(f"   - Canny: {data['Canny']['Hausdorff距离(mm)']:.2f}mm")
    print(f"   - RGB-D: {data['RGB-D']['Hausdorff距离(mm)']:.2f}mm")
    print(f"   - 差异: {hausdorff_diff:.2f}mm")
    print(f"   - 结论: Canny方法在稳定性上略优于RGB-D方法")
    
    # 执行时间分析
    time_ratio = data['Canny']['执行时间(秒)'] / data['RGB-D']['执行时间(秒)']
    print(f"\n3. 执行速度:")
    print(f"   - Canny: {data['Canny']['执行时间(秒)']:.2f}秒")
    print(f"   - RGB-D: {data['RGB-D']['执行时间(秒)']:.2f}秒")
    print(f"   - 差异: Canny比RGB-D慢{time_ratio:.1f}倍")
    print(f"   - 结论: RGB-D方法在执行速度上显著优于Canny方法")
    
    # 成功率分析
    print(f"\n4. 成功率:")
    print(f"   - Canny: {data['Canny']['成功率']}")
    print(f"   - RGB-D: {data['RGB-D']['成功率']}")
    print(f"   - 结论: 两种方法都达到了100%的成功率")
    
    # 点云数量分析
    point_ratio = data['RGB-D']['平均点数'] / data['Canny']['平均点数']
    print(f"\n5. 数据量:")
    print(f"   - Canny: {data['Canny']['平均点数']:.0f}个点")
    print(f"   - RGB-D: {data['RGB-D']['平均点数']:.0f}个点")
    print(f"   - 差异: RGB-D采集的点云数量是Canny的{point_ratio:.1f}倍")
    print(f"   - 结论: RGB-D方法采集的数据量更多")
    
    # 综合推荐
    print(f"\n综合推荐:")
    print("-" * 60)
    print(f"🏆 总体评价:")
    print(f"   - 准确性: Canny方法更优 (Chamfer距离小{chamfer_ratio:.1f}倍)")
    print(f"   - 稳定性: Canny方法更优 (Hausdorff距离小{hausdorff_diff:.2f}mm)")
    print(f"   - 速度: RGB-D方法更优 (快{time_ratio:.1f}倍)")
    print(f"   - 数据量: RGB-D方法更优 (多{point_ratio:.1f}倍)")
    
    print(f"\n🎯 应用建议:")
    print(f"   - 精度优先场景: 选择Canny方法")
    print(f"     (适用于需要高精度边界检测的应用)")
    print(f"   - 速度优先场景: 选择RGB-D方法")
    print(f"     (适用于需要快速检测的应用)")
    print(f"   - 平衡场景: 根据具体需求权衡")
    print(f"     (如果精度要求不是特别高，RGB-D可能是更好的选择)")
    
    print(f"\n📊 关键结论:")
    print(f"   ✅ Canny方法: 精度高、稳定性好，但速度较慢")
    print(f"   ✅ RGB-D方法: 速度快、数据量大，但精度略低")
    print(f"   ✅ 两种方法都达到了100%的成功率")

if __name__ == "__main__":
    quick_comparison()



