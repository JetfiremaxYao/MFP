# Alpha-Shape边界追踪方法使用说明

## 概述

本项目在原有的Canny和RGB-D边界追踪方法基础上，新增了Alpha-Shape（凹包）边界检测方法，现在支持三种边界追踪方法的比较和评估。

## 新增功能

### 1. Alpha-Shape边界检测方法
- **原理**: 基于Delaunay三角剖分和Alpha参数控制，生成比凸包更精确的凹包边界
- **优势**: 能够检测到物体表面的凹陷和复杂形状
- **适用场景**: 形状复杂、有凹陷的物体边界检测

### 2. 三种方法比较
- **Canny边缘检测**: 基于图像梯度的传统边缘检测
- **RGB-D边界检测**: 结合RGB和深度信息的边界检测
- **Alpha-Shape凹包**: 基于点云几何的凹包检测

## 文件说明

### 主要文件
- `_boudary_track_alpha.py`: 集成了三种边界检测方法的主程序
- `_boundary_tracking_evaluation_alpha.py`: 三种方法的性能评估脚本
- `_test_alpha_shape.py`: Alpha-Shape方法的独立测试脚本

### 依赖库
```bash
# 必需库
pip install numpy opencv-python scipy matplotlib open3d scikit-learn

# Alpha-Shape相关库（可选）
pip install shapely
```

## 使用方法

### 1. 运行主程序
```bash
cd examples/tutorials
python _boudary_track_alpha.py
```

程序会提示选择边界检测方法：
```
请选择边界检测方法:
1. Canny边缘检测
2. RGB-D边界检测
3. Alpha-Shape凹包
4. 三种方法比较
请输入选择 (1-4，默认为1):
```

### 2. 运行评估脚本
```bash
python _boundary_tracking_evaluation_alpha.py
```

这将自动运行三种方法的性能比较，生成详细的评估报告。

### 3. 测试Alpha-Shape方法
```bash
python _test_alpha_shape.py
```

这将运行Alpha-Shape方法的独立测试，包括不同Alpha参数的效果比较。

## 方法特点对比

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| Canny | 速度快，对噪声鲁棒 | 可能丢失细节 | 简单形状，实时应用 |
| RGB-D | 结合深度信息，精度高 | 计算复杂，依赖深度质量 | 复杂场景，高精度要求 |
| Alpha-Shape | 几何精确，能检测凹陷 | 计算量大，参数敏感 | 复杂形状，几何分析 |

## 参数说明

### Alpha-Shape参数
- `alpha_value`: Alpha参数，控制凹包的复杂度
  - 值越大：形状越接近凸包
  - 值越小：形状越复杂，能检测更多凹陷
  - 自动计算：基于点云密度自动选择最优值

### 评估参数
- `num_runs`: 每个方法的运行次数（默认3次）
- `num_positions`: 测试位置数量（默认2个）
- `cube_size`: 物体尺寸
- `base_cube_pos`: 基础物体位置

## 输出结果

### 1. 点云文件
- `boundary_cloud_{method}.ply`: 边界点云
- `surface_cloud_{method}.ply`: 表面点云

### 2. 评估报告
- `detailed_results.json`: 详细评估结果
- `results_summary.csv`: 汇总统计结果
- `summary_statistics.csv`: 方法对比统计
- `comparison_charts.png`: 可视化对比图表
- `evaluation_report.txt`: 文本评估报告

### 3. 可视化结果
- 实时边界检测可视化窗口
- 三种方法并排比较图像
- 点云3D可视化

## 性能指标

评估脚本会计算以下性能指标：

1. **成功率**: 成功检测到边界的比例
2. **轮廓数量**: 检测到的轮廓数量
3. **轮廓面积**: 检测到的轮廓总面积
4. **点云质量**: 点云数量和密度
5. **执行时间**: 方法执行耗时
6. **稳定性**: 多次运行的标准差

## 故障排除

### 1. Alpha-Shape库未安装
```
警告: Alpha-Shape相关库未安装，将跳过Alpha-Shape方法
```
**解决方案**: 安装shapely库
```bash
pip install shapely
```

### 2. 点云数据不足
```
有效点云数量不足，回退到Canny方法
```
**解决方案**: 调整摄像机位置或参数，确保能获取到足够的点云数据

### 3. Delaunay三角剖分失败
```
Delaunay三角剖分失败，返回凸包
```
**解决方案**: 检查点云质量，可能需要预处理或降噪

## 扩展功能

### 1. 自定义Alpha参数
```python
# 在detect_boundary_alpha_shape函数中指定alpha值
contours, rgb, depth = detect_boundary_alpha_shape(cam, alpha_value=2.0)
```

### 2. 添加新的检测方法
在`compare_detection_methods`函数中添加新的方法分支：
```python
elif method == "new_method":
    contours, rgb, depth = detect_boundary_new_method(cam)
```

### 3. 自定义评估指标
在`BoundaryTrackingEvaluator`类中添加新的评估指标计算。

## 注意事项

1. **内存使用**: Alpha-Shape方法计算量较大，注意内存使用
2. **参数调优**: 不同场景可能需要调整Alpha参数
3. **实时性**: 三种方法中Canny最快，Alpha-Shape最慢
4. **精度权衡**: 在精度和速度之间需要根据应用需求选择

## 联系信息

如有问题或建议，请联系开发团队。
