#!/usr/bin/env python3
"""
测试改进后的触诊建议计算算法
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
import time

def create_test_surface_data():
    """创建测试用的表面数据"""
    # 创建一个包含明显异常的测试表面
    x = np.linspace(0.2, 0.4, 50)
    y = np.linspace(-0.02, 0.12, 50)
    X, Y = np.meshgrid(x, y)
    
    # 基础表面（轻微倾斜）
    Z_base = 0.02 + 0.001 * X + 0.0005 * Y
    
    # 添加几个明显的异常区域
    # 异常1：圆形凸起
    center1 = (0.26, 0.01)
    dist1 = np.sqrt((X - center1[0])**2 + (Y - center1[1])**2)
    Z_anomaly1 = 0.003 * np.exp(-(dist1**2) / (2 * 0.02**2))  # 3mm高的高斯凸起
    
    # 异常2：椭圆形凸起
    center2 = (0.34, 0.04)
    dist2 = np.sqrt(((X - center2[0])/0.03)**2 + ((Y - center2[1])/0.02)**2)
    Z_anomaly2 = 0.004 * np.exp(-(dist2**2) / 2)  # 4mm高的椭圆凸起
    
    # 异常3：矩形凸起
    mask3 = (X >= 0.38) & (X <= 0.42) & (Y >= 0.02) & (Y <= 0.06)
    Z_anomaly3 = np.where(mask3, 0.002, 0)  # 2mm高的矩形凸起
    
    # 组合所有表面
    Z = Z_base + Z_anomaly1 + Z_anomaly2 + Z_anomaly3
    
    # 添加一些噪声
    noise = np.random.normal(0, 0.0002, Z.shape)
    Z += noise
    
    return X, Y, Z

def test_improved_algorithm():
    """测试改进后的算法"""
    print("="*60)
    print("测试改进后的触诊建议计算算法")
    print("="*60)
    
    # 1. 创建测试数据
    print("\n1. 创建测试表面数据...")
    X, Y, Z = create_test_surface_data()
    
    # 2. 模拟表面重建过程
    print("\n2. 模拟表面重建...")
    surface_points = []
    for i in range(0, Z.shape[0], 2):  # 采样
        for j in range(0, Z.shape[1], 2):
            if not np.isnan(Z[i, j]):
                surface_points.append([X[i, j], Y[i, j], Z[i, j]])
    
    surface_points = np.array(surface_points)
    print(f"采样点云数量: {len(surface_points)}")
    
    # 3. 创建扫描网格信息
    scan_grid = {
        'min_coords': np.array([0.2, -0.02]),
        'max_coords': np.array([0.4, 0.12]),
        'scan_height': np.mean(surface_points[:, 2])
    }
    
    # 4. 表面重建
    print("\n3. 表面重建...")
    resolution = 0.002  # 2mm分辨率
    x_coords = np.arange(scan_grid['min_coords'][0], scan_grid['max_coords'][0] + resolution, resolution)
    y_coords = np.arange(scan_grid['min_coords'][1], scan_grid['max_coords'][1] + resolution, resolution)
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    # 插值
    surface_heights = griddata(
        surface_points[:, :2], surface_points[:, 2], (X_grid, Y_grid), 
        method='linear', fill_value=np.nan
    )
    
    print(f"重建网格尺寸: {surface_heights.shape}")
    
    # 5. 异常检测（使用改进的算法）
    print("\n4. 异常检测...")
    anomalies, anomaly_map = detect_surface_anomalies_improved(
        surface_heights, (x_coords, y_coords)
    )
    
    # 6. 生成触诊建议
    print("\n5. 生成触诊建议...")
    cube_pos = np.array([0.3, 0.05, 0.02])
    suggestions = generate_palpation_suggestions_improved(
        anomalies, cube_pos, surface_points
    )
    
    # 7. 显示结果
    print("\n6. 结果分析:")
    print(f"检测到 {len(anomalies)} 个异常区域")
    print(f"生成 {len(suggestions)} 个触诊建议")
    
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        print(f"\n建议{i+1}:")
        print(f"  位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  类型: {suggestion['type']}")
        print(f"  置信度: {suggestion['confidence']:.3f}")
        print(f"  原因: {suggestion['reason']}")
        if suggestion['type'] == 'anomaly':
            print(f"  异常强度: {suggestion['strength']:.4f}")
            print(f"  高度差: {suggestion['height_elevation']*1000:.1f}mm")
            print(f"  面积: {suggestion['area_mm2']:.1f}mm²")
    
    # 8. 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_palpation_suggestions_{timestamp}.json"
    
    suggestions_data = []
    for suggestion in suggestions:
        suggestion_data = {
            'position': suggestion['position'].tolist(),
            'type': suggestion['type'],
            'priority': suggestion['priority'],
            'confidence': suggestion['confidence'],
            'reason': suggestion['reason']
        }
        if suggestion['type'] == 'anomaly':
            suggestion_data.update({
                'anomaly_id': suggestion['anomaly_id'],
                'strength': suggestion['strength'],
                'height_elevation': suggestion['height_elevation'],
                'area_mm2': suggestion['area_mm2']
            })
        suggestions_data.append(suggestion_data)
    
    with open(filename, 'w') as f:
        json.dump(suggestions_data, f, indent=2)
    
    print(f"\n结果已保存到: {filename}")
    
    # 9. 可视化
    print("\n7. 生成可视化...")
    visualize_test_results(X, Y, Z, anomalies, suggestions)
    
    return suggestions

def detect_surface_anomalies_improved(surface_grid, grid_coords, threshold_percentile=75):
    """改进的异常检测算法（简化版本用于测试）"""
    if surface_grid is None:
        return [], None
    
    print(f"开始异常检测（阈值百分位: {threshold_percentile}%）...")
    
    # 处理NaN值
    valid_mask = ~np.isnan(surface_grid)
    if not np.any(valid_mask):
        return [], None
    
    surface_filled = surface_grid.copy()
    surface_filled[~valid_mask] = np.nanmedian(surface_grid[valid_mask])
    
    # 计算统计信息
    valid_heights = surface_filled[valid_mask]
    mean_height = np.mean(valid_heights)
    std_height = np.std(valid_heights)
    max_height = np.max(valid_heights)
    min_height = np.min(valid_heights)
    
    print(f"高度统计: 均值={mean_height:.4f}, 标准差={std_height:.4f}")
    print(f"高度范围: {min_height:.4f} - {max_height:.4f} (差值: {(max_height-min_height)*1000:.1f}mm)")
    
    # 改进的阈值计算
    std_threshold = mean_height + 1.5 * std_height
    percentile_threshold = np.percentile(valid_heights, threshold_percentile)
    top_height_threshold = min(std_threshold, percentile_threshold)
    
    print(f"标准差阈值: {std_threshold:.4f}")
    print(f"百分位阈值: {percentile_threshold:.4f}")
    print(f"使用阈值: {top_height_threshold:.4f}")
    
    # 创建异常图
    height_diff_map = np.zeros_like(surface_filled)
    height_diff_map[surface_filled >= top_height_threshold] = surface_filled[surface_filled >= top_height_threshold] - mean_height
    
    # 异常检测阈值
    valid_diff = height_diff_map[valid_mask]
    if len(valid_diff) > 0:
        threshold = np.percentile(valid_diff, 60)
    else:
        threshold = 0.001
    
    print(f"异常检测阈值: {threshold:.4f} ({threshold*1000:.1f}mm)")
    
    # 检测异常区域
    anomaly_mask = (height_diff_map > threshold) & valid_mask
    
    # 连通组件分析
    from scipy import ndimage
    labeled_anomalies, num_anomalies = ndimage.label(anomaly_mask)
    
    print(f"检测到 {num_anomalies} 个异常区域")
    
    # 提取异常信息
    x_coords, y_coords = grid_coords
    resolution = x_coords[1] - x_coords[0]
    
    anomalies = []
    for i in range(1, num_anomalies + 1):
        anomaly_region = (labeled_anomalies == i)
        y_indices, x_indices = np.where(anomaly_region)
        
        anomaly_heights = surface_filled[anomaly_region]
        max_height_idx = np.argmax(anomaly_heights)
        center_x = x_coords[x_indices[max_height_idx]]
        center_y = y_coords[y_indices[max_height_idx]]
        
        anomaly_strength = np.mean(height_diff_map[anomaly_region])
        max_h = np.max(anomaly_heights)
        min_h = np.min(anomaly_heights)
        height_elevation = max_h - min_h
        
        relative_strength = anomaly_strength / (max_height - mean_height) if (max_height - mean_height) > 0 else 0
        
        anomaly_info = {
            'id': i,
            'center': np.array([center_x, center_y]),
            'strength': anomaly_strength,
            'relative_strength': relative_strength,
            'height_elevation': height_elevation,
            'area_pixels': np.sum(anomaly_region),
            'area_mm2': np.sum(anomaly_region) * (resolution ** 2) * 1e6,
            'region_mask': anomaly_region
        }
        anomalies.append(anomaly_info)
    
    anomalies.sort(key=lambda x: x['strength'], reverse=True)
    
    print(f"异常检测完成，发现 {len(anomalies)} 个异常区域")
    for i, anomaly in enumerate(anomalies):
        print(f"  异常{i+1}: 强度={anomaly['strength']:.4f}, 相对强度={anomaly['relative_strength']:.3f}, "
              f"高度差={anomaly['height_elevation']*1000:.1f}mm, "
              f"面积={anomaly['area_mm2']:.1f}mm²")
    
    return anomalies, height_diff_map

def generate_palpation_suggestions_improved(anomalies, cube_pos, surface_points=None, max_suggestions=3):
    """改进的触诊建议生成算法（简化版本用于测试）"""
    print(f"\n生成触诊建议...")
    
    suggestions = []
    
    # 检查是否为平面
    is_flat_surface = False
    if len(anomalies) > 0:
        avg_height_diff = np.mean([anomaly['height_elevation'] for anomaly in anomalies])
        max_height_diff = np.max([anomaly['height_elevation'] for anomaly in anomalies])
        
        flat_threshold = 0.002  # 2mm
        if avg_height_diff < flat_threshold and max_height_diff < flat_threshold:
            is_flat_surface = True
            print(f"检测到平面表面：平均高度差={avg_height_diff*1000:.1f}mm, 最大高度差={max_height_diff*1000:.1f}mm")
    
    if len(anomalies) == 0 or is_flat_surface:
        # 平面或无异常见议
        if is_flat_surface:
            object_pos = np.array([0.3, 0.05, 0.02])
            palpation_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + 0.15])
        else:
            if surface_points is not None and len(surface_points) > 0:
                centroid = np.mean(surface_points, axis=0)
                palpation_pos = centroid
            else:
                palpation_pos = cube_pos.copy()
        
        center_suggestion = {
            'position': palpation_pos,
            'type': 'center',
            'priority': 1,
            'reason': '表面平滑，建议触诊中心位置',
            'confidence': 0.8 if is_flat_surface else 0.5
        }
        suggestions.append(center_suggestion)
    else:
        # 异常区域建议
        for i, anomaly in enumerate(anomalies[:max_suggestions]):
            palpation_pos = np.array([
                anomaly['center'][0],
                anomaly['center'][1], 
                cube_pos[2] + 0.15
            ])
            
            priority = i + 1
            
            # 改进的置信度计算
            base_confidence = min(0.8, anomaly.get('relative_strength', 0.1))
            height_factor = min(1.0, anomaly['height_elevation'] * 1000 / 5.0)
            
            area_mm2 = anomaly['area_mm2']
            if 50 <= area_mm2 <= 500:
                area_factor = 1.0
            elif area_mm2 < 50:
                area_factor = area_mm2 / 50.0
            else:
                area_factor = max(0.5, 500.0 / area_mm2)
            
            confidence = base_confidence * height_factor * area_factor
            confidence = max(0.1, min(0.95, confidence))
            
            suggestion = {
                'position': palpation_pos,
                'type': 'anomaly',
                'priority': priority,
                'anomaly_id': anomaly['id'],
                'strength': anomaly['strength'],
                'height_elevation': anomaly['height_elevation'],
                'area_mm2': anomaly['area_mm2'],
                'reason': f'检测到表面异常{i+1}，高度差={anomaly["height_elevation"]*1000:.1f}mm',
                'confidence': confidence
            }
            suggestions.append(suggestion)
    
    print(f"触诊建议生成完成，共 {len(suggestions)} 个建议")
    return suggestions

def visualize_test_results(X, Y, Z, anomalies, suggestions):
    """可视化测试结果"""
    plt.figure(figsize=(15, 5))
    
    # 1. 原始表面
    plt.subplot(1, 3, 1)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.title('Original Surface')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 2. 异常检测结果
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    # 标记异常区域
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    markers = ['X', '^', 's', 'D', 'v']
    
    for i, anomaly in enumerate(anomalies):
        center = anomaly['center']
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.scatter(center[0], center[1], c=color, s=100, marker=marker, 
                   label=f'Anomaly {i+1}' if i < 5 else '')
    
    plt.legend()
    plt.title('Anomaly Detection')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # 3. 触诊建议
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    for i, suggestion in enumerate(suggestions):
        pos = suggestion['position']
        if suggestion['type'] == 'anomaly':
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.scatter(pos[0], pos[1], c=color, s=100, marker=marker, 
                       label=f'Suggestion {i+1}' if i < 5 else '')
        else:
            plt.scatter(pos[0], pos[1], c='blue', s=100, marker='*', 
                       label='Center Suggestion')
    
    plt.legend()
    plt.title('Palpation Suggestions')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"test_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存为: {filename}")
    
    plt.show()

if __name__ == "__main__":
    test_improved_algorithm()

