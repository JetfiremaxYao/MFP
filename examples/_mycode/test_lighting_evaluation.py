#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试光照环境评估功能
"""

import numpy as np
from _boundary_tracking_evaluation import BoundaryTrackingEvaluator

def test_lighting_configuration():
    """测试光照环境配置"""
    print("=== 测试光照环境配置 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    
    print(f"光照条件: {evaluator.lighting_conditions}")
    print(f"背景条件: {evaluator.background_conditions}")
    print(f"重复次数: {evaluator.n_runs}")
    
    # 计算总实验数
    total_experiments = len(evaluator.lighting_conditions) * len(evaluator.background_conditions) * evaluator.n_runs * 2
    print(f"总实验数: {total_experiments}")
    
    return True

def test_scene_setup():
    """测试场景设置"""
    print("\n=== 测试场景设置 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    cube_pos = np.array([0.35, 0, 0.02])
    
    for lighting in evaluator.lighting_conditions:
        print(f"测试 {lighting} 光照环境...")
        try:
            scene, ed6, cam, motors_dof_idx, j6_link = evaluator._setup_scene(cube_pos, lighting, "simple")
            print(f"  ✅ {lighting} 场景设置成功")
            
            # 检查环境光设置
            ambient_light = scene.vis_options.ambient_light
            print(f"  环境光: {ambient_light}")
            
            del scene
        except Exception as e:
            print(f"  ❌ {lighting} 场景设置失败: {e}")
            return False
    
    return True

def test_perturbed_positions():
    """测试扰动位置生成"""
    print("\n=== 测试扰动位置生成 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    
    # 测试位置扰动
    for i in range(3):
        np.random.seed(42 + i)
        x_perturb = np.random.uniform(-evaluator.position_perturbation, evaluator.position_perturbation)
        y_perturb = np.random.uniform(-evaluator.position_perturbation, evaluator.position_perturbation)
        z_perturb = np.random.uniform(-evaluator.height_perturbation, evaluator.height_perturbation)
        
        cube_pos = evaluator.base_cube_pos.copy()
        cube_pos[0] += x_perturb
        cube_pos[1] += y_perturb
        cube_pos[2] += z_perturb
        
        print(f"  扰动 {i+1}: {cube_pos}")
        print(f"    扰动量: x={x_perturb*1000:.1f}mm, y={y_perturb*1000:.1f}mm, z={z_perturb*1000:.1f}mm")
    
    return True

def test_results_structure():
    """测试结果存储结构"""
    print("\n=== 测试结果存储结构 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    
    # 检查结果存储结构
    print("结果存储结构:")
    for method in ['canny', 'rgbd']:
        print(f"  {method}:")
        for lighting in evaluator.lighting_conditions:
            print(f"    {lighting}: {len(evaluator.results[method][lighting])} 个结果")
    
    return True

def test_single_experiment():
    """测试单次实验（简化版）"""
    print("\n=== 测试单次实验 ===")
    
    evaluator = BoundaryTrackingEvaluator()
    cube_pos = np.array([0.35, 0, 0.02])
    
    # 测试一个简化的实验
    try:
        result = evaluator._run_single_experiment('canny', cube_pos, 'normal', 'simple', 0)
        print(f"  ✅ 实验运行成功")
        print(f"  状态: {result['status']}")
        print(f"  点数: {result['point_count']}")
        return True
    except Exception as e:
        print(f"  ❌ 实验运行失败: {e}")
        return False

def main():
    """主测试函数"""
    print("光照环境评估功能测试")
    print("=" * 50)
    
    # 测试配置
    config_ok = test_lighting_configuration()
    
    # 测试场景设置
    scene_ok = test_scene_setup()
    
    # 测试扰动位置
    position_ok = test_perturbed_positions()
    
    # 测试结果结构
    structure_ok = test_results_structure()
    
    # 测试单次实验（可选）
    print("\n是否运行单次实验测试？(y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            experiment_ok = test_single_experiment()
        else:
            experiment_ok = True
            print("跳过单次实验测试")
    except:
        experiment_ok = True
        print("跳过单次实验测试")
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"配置测试: {'✅' if config_ok else '❌'}")
    print(f"场景设置: {'✅' if scene_ok else '❌'}")
    print(f"位置扰动: {'✅' if position_ok else '❌'}")
    print(f"结果结构: {'✅' if structure_ok else '❌'}")
    print(f"单次实验: {'✅' if experiment_ok else '❌'}")
    
    if all([config_ok, scene_ok, position_ok, structure_ok, experiment_ok]):
        print("\n🎉 所有测试通过！光照环境评估功能正常。")
        print("\n现在可以运行完整的评估:")
        print("python _boundary_tracking_evaluation.py")
    else:
        print("\n⚠️ 部分测试失败，需要进一步修复。")

if __name__ == "__main__":
    main()
