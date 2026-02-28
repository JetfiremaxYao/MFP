#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试成功判定标准的修复
"""

import numpy as np
from boundary_tracking_evaluation import BoundaryTrackingEvaluator

def test_success_criteria():
    """测试成功判定标准"""
    print("=== 测试成功判定标准 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    
    # 测试用例1：大量点云（应该成功）
    print("\n测试用例1：大量点云")
    test_points_1 = np.random.rand(1500, 3)  # 1500个点
    result_1 = evaluator._evaluate_single_run(
        test_points_1, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        10.0
    )
    print(f"点数: {result_1['point_count']}")
    print(f"状态: {result_1['status']}")
    print(f"闭环检测: {result_1['evaluation_details']['closure_detected']}")
    
    # 测试用例2：中等点云（应该部分成功）
    print("\n测试用例2：中等点云")
    test_points_2 = np.random.rand(800, 3)  # 800个点
    result_2 = evaluator._evaluate_single_run(
        test_points_2, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        8.0
    )
    print(f"点数: {result_2['point_count']}")
    print(f"状态: {result_2['status']}")
    print(f"闭环检测: {result_2['evaluation_details']['closure_detected']}")
    
    # 测试用例3：少量点云（应该部分成功）
    print("\n测试用例3：少量点云")
    test_points_3 = np.random.rand(200, 3)  # 200个点
    result_3 = evaluator._evaluate_single_run(
        test_points_3, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        5.0
    )
    print(f"点数: {result_3['point_count']}")
    print(f"状态: {result_3['status']}")
    print(f"闭环检测: {result_3['evaluation_details']['closure_detected']}")
    
    # 测试用例4：极少点云（应该失败）
    print("\n测试用例4：极少点云")
    test_points_4 = np.random.rand(30, 3)  # 30个点
    result_4 = evaluator._evaluate_single_run(
        test_points_4, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        2.0
    )
    print(f"点数: {result_4['point_count']}")
    print(f"状态: {result_4['status']}")
    print(f"闭环检测: {result_4['evaluation_details']['closure_detected']}")
    
    # 总结
    print("\n=== 测试总结 ===")
    success_count = sum(1 for r in [result_1, result_2, result_3, result_4] if r['status'] in ['Success', 'Partial'])
    print(f"成功/部分成功: {success_count}/4")
    
    if success_count >= 3:
        print("✅ 成功判定标准修复成功！")
    else:
        print("❌ 成功判定标准仍有问题")

def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 使用真实数据测试 ===")
    
    # 从您的日志中看到的数据
    # 第5步：3636个点，内陆/目标比例: 2.126
    # 第16步：13850个点，内陆/目标比例: 0.001
    
    evaluator = BoundaryTrackingEvaluator()
    
    # 模拟第5步的数据
    print("\n模拟第5步数据（3636个点）:")
    test_points_5 = np.random.rand(3636, 3)
    result_5 = evaluator._evaluate_single_run(
        test_points_5, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        15.0
    )
    print(f"点数: {result_5['point_count']}")
    print(f"状态: {result_5['status']}")
    
    # 模拟第16步的数据
    print("\n模拟第16步数据（13850个点）:")
    test_points_16 = np.random.rand(13850, 3)
    result_16 = evaluator._evaluate_single_run(
        test_points_16, 
        np.array([0.35, 0, 0.02]), 
        'canny', 
        0, 
        25.0
    )
    print(f"点数: {result_16['point_count']}")
    print(f"状态: {result_16['status']}")

if __name__ == "__main__":
    test_success_criteria()
    test_with_real_data()
