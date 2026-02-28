#!/usr/bin/env python3
"""
测试直接基于最高点的触诊建议方法
"""

import numpy as np
import json
import time

def create_test_surface_points():
    """创建测试用的表面点云数据"""
    # 创建一个包含明显最高点的测试表面
    np.random.seed(42)  # 固定随机种子
    
    # 基础点云
    n_points = 1000
    x = np.random.uniform(0.2, 0.4, n_points)
    y = np.random.uniform(-0.02, 0.12, n_points)
    
    # 基础高度（轻微倾斜）
    z_base = 0.02 + 0.001 * x + 0.0005 * y
    
    # 添加几个明显的最高点
    # 最高点1：圆形凸起
    center1 = (0.26, 0.01)
    dist1 = np.sqrt((x - center1[0])**2 + (y - center1[1])**2)
    z_anomaly1 = 0.008 * np.exp(-(dist1**2) / (2 * 0.03**2))  # 8mm高的高斯凸起
    
    # 最高点2：椭圆形凸起
    center2 = (0.34, 0.04)
    dist2 = np.sqrt(((x - center2[0])/0.04)**2 + ((y - center2[1])/0.03)**2)
    z_anomaly2 = 0.006 * np.exp(-(dist2**2) / 2)  # 6mm高的椭圆凸起
    
    # 最高点3：矩形凸起
    mask3 = (x >= 0.38) & (x <= 0.42) & (y >= 0.02) & (y <= 0.06)
    z_anomaly3 = np.where(mask3, 0.004, 0)  # 4mm高的矩形凸起
    
    # 组合所有高度
    z = z_base + z_anomaly1 + z_anomaly2 + z_anomaly3
    
    # 添加噪声
    noise = np.random.normal(0, 0.0005, z.shape)
    z += noise
    
    # 组合成点云
    surface_points = np.column_stack([x, y, z])
    
    return surface_points

def generate_palpation_suggestions_simple(surface_points, cube_pos, max_suggestions=3):
    """
    简化的触诊建议生成函数（直接基于最高点）
    """
    print(f"\n生成触诊建议（直接基于最高点）...")
    
    suggestions = []
    
    if surface_points is None or len(surface_points) == 0:
        print("表面点云为空，使用默认中心位置")
        palpation_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.15])
        center_suggestion = {
            'position': palpation_pos,
            'type': 'center',
            'priority': 1,
            'reason': '表面点云为空，使用默认中心位置',
            'confidence': 0.5
        }
        suggestions.append(center_suggestion)
        return suggestions
    
    # 直接基于表面点云生成触诊建议
    print(f"表面点云数量: {len(surface_points)}")
    
    # 1. 找到最高点
    heights = surface_points[:, 2]
    max_height_idx = np.argmax(heights)
    highest_point = surface_points[max_height_idx]
    
    print(f"最高点位置: ({highest_point[0]:.4f}, {highest_point[1]:.4f}, {highest_point[2]:.4f})")
    print(f"最高点高度: {highest_point[2]:.4f}m")
    
    # 2. 计算高度统计信息
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    height_range = np.max(heights) - np.min(heights)
    
    print(f"高度统计: 均值={mean_height:.4f}, 标准差={std_height:.4f}, 范围={height_range:.4f}")
    
    # 3. 判断是否为平面表面
    is_flat_surface = height_range < 0.002  # 2mm阈值
    
    if is_flat_surface:
        print(f"检测到平面表面（高度差={height_range*1000:.1f}mm），建议触诊中心位置")
        # 平面表面：使用点云质心
        centroid = np.mean(surface_points, axis=0)
        palpation_pos = np.array([centroid[0], centroid[1], centroid[2] + 0.15])
        
        center_suggestion = {
            'position': palpation_pos,
            'type': 'center',
            'priority': 1,
            'reason': f'平面表面，高度差={height_range*1000:.1f}mm',
            'confidence': 0.8
        }
        suggestions.append(center_suggestion)
    else:
        print(f"检测到非平面表面（高度差={height_range*1000:.1f}mm），建议触诊最高点")
        
        # 非平面表面：直接使用最高点
        palpation_pos = np.array([highest_point[0], highest_point[1], highest_point[2] + 0.15])
        
        # 计算置信度：基于高度差和相对高度
        height_above_mean = highest_point[2] - mean_height
        relative_height = height_above_mean / std_height if std_height > 0 else 0
        
        # 简化的置信度计算
        if height_above_mean > 0.005:  # 5mm以上
            confidence = 0.9
        elif height_above_mean > 0.002:  # 2-5mm
            confidence = 0.7
        elif height_above_mean > 0.001:  # 1-2mm
            confidence = 0.5
        else:  # 1mm以下
            confidence = 0.3
        
        # 根据相对高度调整置信度
        if relative_height > 2.0:  # 超过2个标准差
            confidence = min(0.95, confidence + 0.1)
        elif relative_height > 1.0:  # 超过1个标准差
            confidence = min(0.9, confidence + 0.05)
        
        highest_suggestion = {
            'position': palpation_pos,
            'type': 'highest_point',
            'priority': 1,
            'height': highest_point[2],
            'height_above_mean': height_above_mean,
            'relative_height': relative_height,
            'reason': f'表面最高点，高度={highest_point[2]:.4f}m，超出均值{height_above_mean*1000:.1f}mm',
            'confidence': confidence
        }
        suggestions.append(highest_suggestion)
        
        # 如果需要多个建议，可以添加其他高点
        if max_suggestions > 1:
            # 找到其他高点（排除最高点周围一定范围内的点）
            remaining_points = surface_points.copy()
            min_distance = 0.01  # 1cm最小距离
            
            for i in range(1, min(max_suggestions, 4)):  # 最多3个额外建议
                # 计算到已选择点的距离
                selected_positions = [s['position'][:2] for s in suggestions]
                if len(selected_positions) > 0:
                    distances = np.linalg.norm(remaining_points[:, :2] - np.array(selected_positions), axis=1)
                    valid_mask = distances > min_distance
                    if not np.any(valid_mask):
                        break
                    remaining_points = remaining_points[valid_mask]
                
                if len(remaining_points) == 0:
                    break
                
                # 找到剩余点中的最高点
                remaining_heights = remaining_points[:, 2]
                next_highest_idx = np.argmax(remaining_heights)
                next_highest_point = remaining_points[next_highest_idx]
                
                # 计算置信度
                next_height_above_mean = next_highest_point[2] - mean_height
                if next_height_above_mean > 0.003:  # 3mm以上
                    next_confidence = 0.6
                elif next_height_above_mean > 0.001:  # 1-3mm
                    next_confidence = 0.4
                else:
                    next_confidence = 0.2
                
                next_palpation_pos = np.array([next_highest_point[0], next_highest_point[1], next_highest_point[2] + 0.15])
                
                next_suggestion = {
                    'position': next_palpation_pos,
                    'type': 'high_point',
                    'priority': i + 1,
                    'height': next_highest_point[2],
                    'height_above_mean': next_height_above_mean,
                    'reason': f'第{i+1}高点，高度={next_highest_point[2]:.4f}m，超出均值{next_height_above_mean*1000:.1f}mm',
                    'confidence': next_confidence
                }
                suggestions.append(next_suggestion)
                
                print(f"建议{i+1}: 第{i+1}高点, "
                      f"位置=({next_palpation_pos[0]:.3f}, {next_palpation_pos[1]:.3f}, {next_palpation_pos[2]:.3f}), "
                      f"高度={next_highest_point[2]:.4f}m, 置信度={next_confidence:.2f}")
    
    print(f"触诊建议生成完成，共 {len(suggestions)} 个建议")
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        print(f"  建议{i+1}: 位置=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"类型={suggestion['type']}, 置信度={suggestion['confidence']:.2f}")
    
    return suggestions

def test_highest_point_method():
    """测试最高点方法"""
    print("="*60)
    print("测试直接基于最高点的触诊建议方法")
    print("="*60)
    
    # 1. 创建测试数据
    print("\n1. 创建测试表面点云...")
    surface_points = create_test_surface_points()
    print(f"生成 {len(surface_points)} 个表面点")
    
    # 2. 生成触诊建议
    print("\n2. 生成触诊建议...")
    cube_pos = np.array([0.3, 0.05, 0.02])
    suggestions = generate_palpation_suggestions_simple(surface_points, cube_pos, max_suggestions=3)
    
    # 3. 保存结果
    print("\n3. 保存结果...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_highest_point_suggestions_{timestamp}.json"
    
    suggestions_data = []
    for suggestion in suggestions:
        suggestion_data = {
            'position': suggestion['position'].tolist(),
            'type': suggestion['type'],
            'priority': suggestion['priority'],
            'confidence': suggestion['confidence'],
            'reason': suggestion['reason']
        }
        # 添加高度相关信息
        if 'height' in suggestion:
            suggestion_data['height'] = suggestion['height']
        if 'height_above_mean' in suggestion:
            suggestion_data['height_above_mean'] = suggestion['height_above_mean']
        if 'relative_height' in suggestion:
            suggestion_data['relative_height'] = suggestion['relative_height']
        suggestions_data.append(suggestion_data)
    
    with open(filename, 'w') as f:
        json.dump(suggestions_data, f, indent=2)
    
    print(f"结果已保存到: {filename}")
    
    # 4. 显示详细结果
    print("\n4. 详细结果分析:")
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        print(f"\n建议{i+1}:")
        print(f"  位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  类型: {suggestion['type']}")
        print(f"  置信度: {suggestion['confidence']:.3f}")
        print(f"  原因: {suggestion['reason']}")
        if 'height' in suggestion:
            print(f"  高度: {suggestion['height']:.4f}m")
        if 'height_above_mean' in suggestion:
            print(f"  超出均值: {suggestion['height_above_mean']*1000:.1f}mm")
        if 'relative_height' in suggestion:
            print(f"  相对高度: {suggestion['relative_height']:.2f}个标准差")
    
    return suggestions

if __name__ == "__main__":
    test_highest_point_method()

