# 边界追踪算法评估系统

## 概述

本系统用于对比Canny边缘检测和RGB-D边界检测两种方法在边界追踪任务中的性能。系统实现了完整的评估流水线，包括实验执行、指标计算、结果分析和可视化。

## 系统架构

```
boundary_tracking_evaluation.py  # 主评估系统
├── BoundaryTrackingEvaluator   # 评估器主类
├── 实验执行模块
├── 指标计算模块
└── 结果保存模块

boundary_tracking_analysis.py   # 结果分析模块
├── BoundaryTrackingAnalyzer    # 分析器主类
├── 图表生成模块
├── 统计分析模块
└── 报告生成模块
```

## 主要功能

### 1. 统一评估流水线
- **输入**: 每次run的边界点云 (N×3)
- **投影**: 到扫描主平面 (z=0.031m)
- **真值**: 生成理论矩形边界和采样点
- **指标**: 计算准确性、稳定性、效率指标

### 2. 评估指标

#### 准确性 (Accuracy)
- **Chamfer距离**: `mean(r_i) + mean(s_j)` (mm)
- **Hausdorff距离**: `max(max(r_i), max(s_j))` (mm)
- **覆盖率**: Coverage@3mm, Coverage@自适应阈值

#### 稳定性 (Stability)
- **Within-run**: 残差标准差
- **Cross-run**: 多次实验的均值和标准差

#### 效率 (Efficiency)
- **执行时间**: 算法运行时间 (秒)
- **采集点数**: 边界点总数
- **步数**: 执行到闭环/终止的步数

### 3. 成功判定标准
- **Success**: 闭环成功 OR (回起点 AND Coverage@3mm ≥ 70%)
- **Partial**: 点数≥80 AND Coverage@3mm ≥ 30%
- **Fail**: MaxStep/NoContour/IKFail/PlanFail/TimeBudget/Safety

## 使用方法

### 1. 运行评估实验

```bash
# 进入tutorials目录
cd examples/tutorials

# 运行评估实验
python boundary_tracking_evaluation.py
```

### 2. 分析实验结果

```bash
# 分析实验结果 (需要先运行评估实验)
python boundary_tracking_analysis.py evaluation_results/experiment_name_timestamp/experiment_results.json
```

## 实验配置

### 实验参数
- **重复次数**: 5次
- **位置扰动**: ±2cm (x, y方向)
- **高度扰动**: ±0.5cm (z方向)
- **覆盖率阈值**: 3mm
- **部分成功阈值**: 30%
- **成功阈值**: 70%

### 物体配置
- **尺寸**: 0.105 × 0.18 × 0.022 m
- **基础位置**: (0.35, 0, 0.02) m
- **扫描平面**: z = 0.031 m (顶面中心)

### 机械臂配置
- **型号**: ED6 (6自由度)
- **控制器**: 位置控制 + IK逆解
- **工作空间**: 保持固定，通过扰动物体位置实现重复性

## 输出结果

### 1. 实验数据
```
evaluation_results/
└── experiment_name_timestamp/
    ├── experiment_results.json      # 完整实验结果
    ├── experiment_report.txt        # 实验报告
    ├── canny_run_1_pointcloud.ply  # Canny方法点云
    ├── canny_run_1_visualization.png # 可视化图像
    ├── rgbd_run_1_pointcloud.ply   # RGB-D方法点云
    └── rgbd_run_1_visualization.png # 可视化图像
```

### 2. 分析结果
```
evaluation_results/
└── experiment_name_timestamp/
    └── analysis/
        ├── success_rate_comparison.png      # 成功率对比
        ├── accuracy_comparison.png          # 准确性对比
        ├── stability_comparison.png         # 稳定性对比
        ├── efficiency_comparison.png        # 效率对比
        ├── performance_radar_chart.png      # 综合性能雷达图
        ├── summary_statistics.csv           # 汇总统计 (CSV)
        ├── summary_statistics.tex           # 汇总统计 (LaTeX)
        ├── statistical_tests.json           # 统计检验结果
        ├── statistical_analysis_report.txt  # 统计检验报告
        └── detailed_analysis_report.txt     # 详细分析报告
```

## 评估指标详解

### 1. Chamfer距离
计算点云与真值边界的双向距离：
- **点→边**: 每个点到最近边的距离
- **边→点**: 每个边采样点到最近点的距离
- **总距离**: `mean(点→边) + mean(边→点)`

### 2. Hausdorff距离
衡量点云与真值边界的最大偏差：
- **单向Hausdorff**: `max(点→边)` 或 `max(边→点)`
- **双向Hausdorff**: `max(max(点→边), max(边→点))`

### 3. 覆盖率
在指定阈值内的点云比例：
- **固定阈值**: Coverage@3mm (3mm内点的比例)
- **自适应阈值**: 基于点云密度自动调整

### 4. 稳定性指标
- **残差标准差**: 同次扫描内残差的离散程度
- **变异系数**: 标准差/均值，无量纲稳定性指标

## 统计分析方法

### 1. 非参数检验
- **检验方法**: Wilcoxon秩和检验
- **适用性**: 不要求数据正态分布
- **显著性水平**: α = 0.05

### 2. 效应量
- **Cohen's d**: 标准化均值差异
- **解释标准**: 
  - |d| < 0.2: 小效应
  - 0.2 ≤ |d| < 0.5: 中等效应
  - 0.5 ≤ |d| < 0.8: 大效应
  - |d| ≥ 0.8: 很大效应

## 可视化图表

### 1. 成功率对比
- **柱状图**: 各方法成功/部分成功/失败次数
- **饼图**: 成功率分布比例

### 2. 指标对比
- **箱线图**: 各指标的中位数、四分位数、异常值
- **散点图**: 指标间的相关性分析

### 3. 综合性能雷达图
- **多维度**: 准确性、稳定性、效率等指标
- **归一化**: 0-1标准化，便于对比

## 注意事项

### 1. 运行环境
- **Python版本**: 3.7+
- **依赖包**: numpy, matplotlib, seaborn, pandas, scipy, open3d
- **Genesis版本**: 最新稳定版

### 2. 硬件要求
- **内存**: 建议8GB+
- **GPU**: 可选，用于加速计算
- **存储**: 每次实验约100-500MB

### 3. 实验时间
- **单次实验**: 约5-15分钟 (取决于算法复杂度)
- **完整评估**: 约1-2小时 (5次重复 × 2种方法)

## 故障排除

### 1. 常见问题
- **导入错误**: 检查Genesis安装和路径配置
- **内存不足**: 减少max_steps或关闭viewer
- **IK失败**: 检查机械臂工作空间和目标位置

### 2. 调试建议
- 启用详细日志: `logging_level="debug"`
- 检查点云质量: 可视化中间结果
- 验证参数设置: 确认阈值和配置

## 扩展功能

### 1. 自定义指标
可以在`_calculate_accuracy_metrics`中添加新的评估指标

### 2. 新算法集成
继承`BoundaryTrackingEvaluator`类，实现新的检测方法

### 3. 参数优化
实现网格搜索或贝叶斯优化来自动调参

## 引用

如果您在研究中使用了本评估系统，请引用：

```bibtex
@software{boundary_tracking_evaluation,
  title={边界追踪算法评估系统},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系：
- 邮箱: your.email@example.com
- GitHub: https://github.com/your-username
- 项目主页: https://github.com/your-repo

---

**版本**: 1.0.0  
**最后更新**: 2024年12月  
**维护者**: Your Name
