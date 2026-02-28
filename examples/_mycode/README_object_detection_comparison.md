# 物体检测方法对比评估系统使用指南

## 概述

本系统专门用于对比HSV+Depth和Depth+Clustering两种物体检测方法的性能。系统提供完整的评估流水线，包括实验执行、数据收集、统计分析和可视化报告。

## 系统架构

```
object_detection_comparison.py     # 主对比评估系统
├── ObjectDetectionComparison     # 对比评估器主类
├── 实验执行模块
├── 数据收集模块
└── 结果分析模块

statistical_analysis.py           # 统计分析工具
├── StatisticalAnalyzer          # 统计分析器主类
├── 统计检验模块
├── 效应量分析模块
└── 可视化模块

_find_the_object_hsv+depth.py     # HSV+Depth检测方法
_find_the_object_depth_clustering.py  # Depth+Clustering检测方法
```

## 主要功能

### 1. 统一评估框架
- **输入**: 两种检测方法的实现
- **输出**: 定量性能指标和统计显著性检验
- **评估指标**: 位置精度、范围精度、检测时间、成功率

### 2. 评估指标详解

#### 位置精度 (Position Accuracy)
- **定义**: 检测位置与真实位置的欧几里得距离
- **单位**: 毫米 (mm)
- **阈值**: 成功 < 50mm, 部分成功 < 100mm

#### 范围精度 (Range Accuracy)
- **定义**: 检测范围与真实范围的差异
- **单位**: 毫米 (mm)
- **计算**: 基于X和Y方向尺寸差异的欧几里得距离

#### 检测时间 (Detection Time)
- **定义**: 从开始检测到获得结果的时间
- **单位**: 秒 (s)
- **包含**: 环视扫描时间 + 数据处理时间

#### 成功率 (Success Rate)
- **定义**: 成功检测次数占总检测次数的比例
- **分类**: Success (位置误差 < 50mm), Partial (50-100mm), Fail (>100mm)

### 3. 统计分析方法

#### 正态性检验
- **方法**: Shapiro-Wilk检验
- **目的**: 确定数据分布类型，选择适当的统计检验方法

#### 参数检验 vs 非参数检验
- **正态分布**: 独立样本t检验 (Independent t-test)
- **非正态分布**: Mann-Whitney U检验

#### 效应量分析
- **连续变量**: Cohen's d效应量
- **分类变量**: Phi系数
- **解释标准**: 小效应 (<0.2), 中等效应 (0.2-0.5), 大效应 (0.5-0.8), 很大效应 (>0.8)

## 使用方法

### 第一步：运行对比实验

```bash
# 进入tutorials目录
cd examples/tutorials

# 运行对比实验
python object_detection_comparison.py
```

系统会：
1. 自动初始化Genesis环境
2. 生成10个扰动位置（±2cm位置扰动，±0.5cm高度扰动）
3. 对每个位置运行两种检测方法
4. 收集性能指标数据
5. 生成对比报告和可视化图表

### 第二步：统计分析

```bash
# 对实验结果进行统计分析
python statistical_analysis.py evaluation_results/hsv_depth_vs_clustering_YYYYMMDD_HHMMSS/experiment_results.json
```

系统会：
1. 执行统计显著性检验
2. 计算效应量
3. 生成统计报告
4. 创建可视化图表

## 输出文件说明

### 实验数据文件
```
evaluation_results/
└── hsv_depth_vs_clustering_YYYYMMDD_HHMMSS/
    ├── experiment_results.json      # 完整实验结果
    ├── experiment_report.txt        # 实验报告
    └── comparison_charts.png        # 对比图表
```

### 统计分析文件
```
evaluation_results/
└── hsv_depth_vs_clustering_YYYYMMDD_HHMMSS/
    ├── statistical_analysis_results.json    # 统计检验结果
    ├── statistical_analysis_report.txt      # 统计报告
    └── statistical_analysis_plots.png       # 统计图表
```

## 实验配置

### 默认参数
- **重复次数**: 10次
- **位置扰动**: ±2cm (x, y方向)
- **高度扰动**: ±0.5cm (z方向)
- **成功阈值**: 50mm
- **部分成功阈值**: 100mm
- **物体尺寸**: 0.105 × 0.18 × 0.022 m
- **基础位置**: (0.35, 0, 0.02) m

### 自定义配置

在 `object_detection_comparison.py` 中修改：

```python
# 修改实验参数
self.n_runs = 15  # 增加重复次数
self.position_perturbation = 0.03  # 增加位置扰动
self.success_threshold = 0.03  # 更严格的成功阈值
```

## 统计检验解释

### 显著性水平
- **p < 0.001**: *** (极显著)
- **p < 0.01**: ** (高度显著)
- **p < 0.05**: * (显著)
- **p ≥ 0.05**: 不显著

### 效应量解释

#### Cohen's d效应量
- **|d| < 0.2**: 小效应
- **0.2 ≤ |d| < 0.5**: 中等效应
- **0.5 ≤ |d| < 0.8**: 大效应
- **|d| ≥ 0.8**: 很大效应

#### Phi系数
- **|φ| < 0.1**: 小效应
- **0.1 ≤ |φ| < 0.3**: 中等效应
- **0.3 ≤ |φ| < 0.5**: 大效应
- **|φ| ≥ 0.5**: 很大效应

## 结果解读示例

### 典型输出
```
【POSITION_ERROR】
检验方法: Mann-Whitney U test
统计量: 15.0000
p值: 0.0234
显著性: * (p < 0.05)
Cohen's d效应量: 0.8234 (大效应)
HSV+Depth均值: 12.3456 ± 3.4567
Depth+Clustering均值: 8.9012 ± 2.3456
```

### 解读说明
1. **检验方法**: 使用Mann-Whitney U检验（非参数检验）
2. **显著性**: p < 0.05，差异具有统计学意义
3. **效应量**: Cohen's d = 0.82，大效应，实际意义显著
4. **均值比较**: Depth+Clustering方法位置误差更小

## 学术写作建议

### 1. 方法对比表格
```latex
\begin{table}[h]
\centering
\begin{tabular}{lccc}
\hline
指标 & HSV+Depth & Depth+Clustering & p值 \\
\hline
位置误差 (mm) & 12.3±3.5 & 8.9±2.3 & 0.023* \\
范围误差 (mm) & 15.6±4.2 & 11.2±3.1 & 0.045* \\
检测时间 (s) & 28.2±0.4 & 25.1±0.6 & 0.156 \\
成功率 (\%) & 90.0 & 95.0 & 0.234 \\
\hline
\end{tabular}
\end{table}
```

### 2. 统计检验报告
```latex
位置误差方面，Depth+Clustering方法显著优于HSV+Depth方法
(Mann-Whitney U = 15.0, p = 0.023, Cohen's d = 0.82, 大效应)。
```

### 3. 可视化图表
- **箱线图**: 显示数据分布和异常值
- **直方图**: 显示数据分布形状
- **柱状图**: 显示成功率对比
- **效应量图**: 显示各指标的效应量大小

## 性能优化建议

### 1. 实验设计
- **样本量**: 建议至少10次重复，理想情况15-20次
- **随机化**: 使用固定随机种子确保可重复性
- **平衡设计**: 确保两种方法在相同条件下测试

### 2. 统计分析
- **正态性检验**: 自动选择适当的检验方法
- **效应量**: 同时报告统计显著性和实际意义
- **多重比较**: 如需要，考虑Bonferroni校正

### 3. 结果解释
- **统计显著性**: 关注p值但不过度依赖
- **实际意义**: 重点考虑效应量和均值差异
- **置信区间**: 如需要，计算95%置信区间

## 故障排除

### 1. 常见问题

#### 导入错误
```bash
# 确保在正确的目录
cd examples/tutorials

# 检查依赖包
pip install scipy pandas matplotlib seaborn
```

#### 内存不足
```python
# 减少重复次数
self.n_runs = 5

# 关闭viewer
show_viewer=False
```

#### 检测失败
```python
# 调整成功阈值
self.success_threshold = 0.10  # 放宽到10cm

# 检查物体位置是否在工作空间内
```

### 2. 调试建议
- 启用详细日志: `logging_level="debug"`
- 单次测试: 先运行单次实验验证
- 检查数据: 查看生成的JSON文件确认数据正确性

## 扩展功能

### 1. 添加新的检测方法
1. 创建新的检测方法文件
2. 实现相同的接口
3. 在对比系统中添加新方法

### 2. 添加新的评估指标
1. 在 `ObjectDetectionComparison` 类中添加新指标
2. 在统计分析器中添加相应的检验
3. 更新可视化模块

### 3. 自定义统计检验
1. 在 `StatisticalAnalyzer` 中添加新的检验方法
2. 实现相应的效应量计算
3. 更新报告生成逻辑

## 联系支持

如果在使用过程中遇到问题，请：
1. 检查错误日志
2. 确认依赖包版本
3. 验证实验配置
4. 查看示例输出文件

---

**注意**: 本系统专为学术研究设计，请确保在论文中正确引用相关方法和工具。
