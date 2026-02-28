# 实验结果可视化工具使用说明

## 概述

本目录包含了用于生成统一物体检测方法评估结果可视化图表的工具。这些工具可以将实验数据转换为直观的图表，帮助分析HSV+Depth和Depth+Clustering两种检测方法的性能差异。

## 文件说明

### 主要脚本

1. **`_unified_object_detection_evaluation.py`** - 主要的评估脚本
   - 运行完整的实验评估
   - 自动生成CSV和JSON格式的结果文件
   - 包含内置的可视化功能（但可能有字体问题）

2. **`visualize_results.py`** - 中文版可视化脚本
   - 处理真实的实验数据
   - 生成中文标签的图表
   - 可能在某些系统上出现中文字体警告

3. **`visualize_results_english.py`** - 英文版可视化脚本（推荐）
   - 处理真实的实验数据
   - 生成英文标签的图表
   - 避免字体问题，兼容性更好

4. **`test_visualization.py`** - 测试脚本
   - 使用模拟数据测试可视化功能
   - 验证图表生成是否正常工作

## 使用方法

### 1. 运行实验评估

```bash
# 激活conda环境
conda activate mfp

# 运行评估实验
python examples/tutorials/_unified_object_detection_evaluation.py
```

### 2. 生成可视化图表

```bash
# 使用英文版可视化脚本（推荐）
python examples/tutorials/visualize_results_english.py

# 或使用中文版可视化脚本
python examples/tutorials/visualize_results.py
```

### 3. 测试可视化功能

```bash
# 使用模拟数据测试
python examples/tutorials/test_visualization.py
```

## 生成的图表类型

### 综合评估图表 (`comprehensive_evaluation_charts_english.png`)

包含6个子图：
1. **位置误差对比** - 箱线图显示两种方法在不同光照条件下的位置误差
2. **范围误差对比** - 箱线图显示范围检测误差
3. **检测时间对比** - 箱线图显示检测耗时
4. **成功率对比** - 柱状图显示2cm阈值下的成功率
5. **视角召回率对比** - 柱状图显示视角检测召回率
6. **FPS对比** - 箱线图显示处理帧率

### 详细分析图表 (`detailed_analysis_charts_english.png`)

包含4个子图：
1. **位置误差分布** - 直方图显示误差分布情况
2. **检测时间vs位置误差** - 散点图显示时间-精度关系
3. **成功率热力图** - 热力图显示不同条件下的成功率
4. **综合性能对比** - 柱状图显示归一化的综合性能指标

### 统计摘要图表 (`statistical_summary_english.png`)

包含4个子图：
1. **性能对比表** - 表格形式显示各方法的详细性能指标
2. **光照条件影响** - 表格显示不同光照条件的影响
3. **范围误差分布** - 直方图显示范围误差分布
4. **检测时间分布** - 直方图显示检测时间分布

## 结果文件结构

```
evaluation_results/
└── unified_object_detection_YYYYMMDD_HHMMSS/
    ├── trial_results.csv              # 原始实验数据
    ├── experiment_results.json        # 实验配置和结果
    ├── experiment_report.txt          # 文本报告
    └── visualizations/                # 可视化图表目录
        ├── comprehensive_evaluation_charts_english.png
        ├── detailed_analysis_charts_english.png
        └── statistical_summary_english.png
```

## 数据说明

### 主要指标

- **位置误差 (Position Error)**: 检测位置与真实位置的欧几里得距离
- **范围误差 (Range Error)**: 检测范围与真实范围的差异
- **检测时间 (Detection Time)**: 完成一次检测所需的时间
- **成功率 (Success Rate)**: 位置误差小于2cm的比例
- **视角召回率 (View Recall Rate)**: 成功检测到物体的视角比例
- **FPS**: 每秒处理的帧数

### 实验条件

- **检测方法**: HSV+Depth, Depth+Clustering
- **光照条件**: normal, bright, dim
- **背景条件**: simple
- **重复次数**: 每种条件3次

## 故障排除

### 字体问题
如果遇到中文字体警告，建议使用英文版可视化脚本：
```bash
python examples/tutorials/visualize_results_english.py
```

### 依赖问题
确保安装了必要的Python包：
```bash
conda install numpy pandas matplotlib seaborn
```

### 数据文件问题
确保实验已经运行并生成了CSV文件：
```bash
ls evaluation_results/unified_object_detection_*/trial_results.csv
```

## 自定义修改

### 修改图表样式
在可视化脚本中修改以下参数：
- `figsize`: 图表大小
- `colors`: 颜色方案
- `fontsize`: 字体大小
- `dpi`: 输出分辨率

### 添加新的图表类型
在相应的函数中添加新的matplotlib/seaborn图表代码。

### 修改数据处理
在`load_and_clean_data`函数中修改数据清理逻辑。

## 注意事项

1. 确保在运行可视化脚本前已经完成了实验评估
2. 英文版脚本兼容性更好，推荐使用
3. 生成的PNG文件分辨率较高，适合用于论文或报告
4. 可以根据需要调整图表大小和样式
5. 如果数据量很大，可能需要较长的处理时间
