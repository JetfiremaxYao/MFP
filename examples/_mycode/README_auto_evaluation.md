# 自动化边界追踪评估系统

## 概述

这是一个自动化版本的边界追踪算法评估系统，基于原有的评估系统进行了以下改进：

1. **去掉实时可视化**：移除了所有 `cv2.imshow()` 和 `o3d.visualization.draw_geometries()` 调用，实现完全自动化运行
2. **添加结果可视化**：实验完成后自动生成图表和统计报告
3. **保持功能完整性**：保留了所有原有的评估指标和功能

## 文件结构

```
examples/tutorials/
├── _boundary_tracking_evaluation_auto.py    # 主评估系统
├── _boundary_track_canny_auto.py            # Canny方法自动化版本
├── _boundary_track_rgbd_auto.py             # RGB-D方法自动化版本
├── test_auto_evaluation.py                  # 测试脚本
└── README_auto_evaluation.md                # 本说明文档
```

## 主要特性

### 1. 自动化运行
- 无需人工干预，可批量运行实验
- 支持多种光照条件测试
- 自动保存点云数据和实验结果

### 2. 完整的评估指标
- **准确性指标**：Chamfer距离、Hausdorff距离、平均残差、中位数残差
- **覆盖率指标**：3mm覆盖率、自适应覆盖率
- **成功标准**：基于点云数量、闭环检测、返回起始点等
- **执行效率**：执行时间统计

### 3. 结果可视化
- 自动生成对比图表
- 统计报告（JSON格式）
- 硬件信息记录

## 使用方法

### 1. 快速测试

运行测试脚本验证系统是否正常工作：

```bash
cd examples/tutorials
python test_auto_evaluation.py
```

### 2. 运行完整评估

直接运行主评估脚本：

```bash
cd examples/tutorials
python _boundary_tracking_evaluation_auto.py
```

### 3. 自定义评估

```python
from _boundary_tracking_evaluation_auto import BoundaryTrackingEvaluatorAuto

# 创建评估器
evaluator = BoundaryTrackingEvaluatorAuto(
    cube_size=[0.105, 0.18, 0.022],
    base_cube_pos=[0.35, 0, 0.02],
    experiment_name="my_experiment"
)

# 运行评估
evaluator.run_evaluation()
```

## 配置参数

### 实验配置
- `n_runs`: 重复实验次数（默认：3）
- `position_perturbation`: 位置扰动范围（默认：0.02m）
- `height_perturbation`: 高度扰动范围（默认：0.005m）

### 光照条件
- `normal`: 正常光照
- `bright`: 明亮光照
- `dim`: 昏暗光照

### 评估参数
- `coverage_threshold`: 3mm覆盖率阈值（默认：0.03m）

## 输出结果

### 1. 结果目录结构
```
evaluation_results/
└── boundary_tracking_comparison_auto_YYYYMMDD_HHMMSS/
    ├── evaluation_results.json          # 详细结果数据
    ├── hardware_info.json               # 硬件信息
    ├── statistical_report.txt           # 统计报告
    ├── evaluation_plots.png             # 可视化图表
    ├── canny_run_1_pointcloud.ply       # Canny方法点云
    ├── canny_run_2_pointcloud.ply
    ├── rgbd_run_1_pointcloud.ply        # RGB-D方法点云
    └── rgbd_run_2_pointcloud.ply
```

### 2. 评估指标说明

#### 准确性指标
- **Chamfer距离**：点云到真值边界的平均距离（mm）
- **Hausdorff距离**：点云到真值边界的最大距离（mm）
- **平均残差**：点云到真值边界的平均距离（mm）
- **中位数残差**：点云到真值边界距离的中位数（mm）

#### 覆盖率指标
- **3mm覆盖率**：距离真值边界3mm以内的点云比例
- **自适应覆盖率**：基于点云密度的自适应覆盖率

#### 成功标准
- **Success**：形成闭环且点云质量良好
- **Partial**：部分成功，点云数量足够但未形成闭环
- **Fail**：失败，点云数量不足或质量差

### 3. 可视化图表

自动生成6个对比图表：
1. **成功率对比**：不同方法在不同光照条件下的成功率
2. **执行时间对比**：平均执行时间对比
3. **点云数量对比**：采集到的点云数量对比
4. **Chamfer距离对比**：准确性指标对比
5. **覆盖率对比**：3mm覆盖率对比
6. **闭环检测成功率**：闭环检测成功率对比

## 与原版本的区别

### 移除的功能
- 实时图像显示（`cv2.imshow`）
- 实时点云可视化（`o3d.visualization.draw_geometries`）
- 交互式用户输入

### 新增的功能
- 自动化批量实验
- 结果可视化报告
- 统计分析和图表生成
- 硬件信息记录
- 测试脚本

### 保留的功能
- 所有边界检测算法
- 所有评估指标计算
- 点云保存功能
- 闭环检测算法

## 故障排除

### 1. 导入错误
确保所有依赖包已安装：
```bash
pip install genesis opencv-python open3d scikit-learn scipy matplotlib seaborn psutil
```

### 2. 路径错误
确保在正确的目录下运行：
```bash
cd examples/tutorials
```

### 3. 内存不足
如果遇到内存问题，可以减少实验次数：
```python
evaluator.n_runs = 1  # 减少重复次数
```

### 4. 执行时间过长
可以调整参数加快执行：
```python
evaluator.n_runs = 1  # 减少重复次数
evaluator.lighting_conditions = ["normal"]  # 只测试一种光照条件
```

## 扩展功能

### 1. 添加新的评估指标
在 `_calculate_evaluation_metrics` 方法中添加新的指标计算。

### 2. 添加新的可视化图表
在 `_generate_visualization_report` 方法中添加新的图表生成代码。

### 3. 添加新的边界检测方法
创建新的检测模块并集成到主评估系统中。

## 注意事项

1. **自动化模式**：系统运行期间无需人工干预，但可以通过终端输入 'esc' 或 'q' 来中断实验
2. **资源消耗**：长时间运行可能消耗较多CPU和内存资源
3. **结果保存**：所有结果会自动保存，建议定期备份重要数据
4. **可重复性**：使用固定随机种子确保实验结果可重复

## 联系信息

如有问题或建议，请查看原始评估系统的文档或联系开发团队。
