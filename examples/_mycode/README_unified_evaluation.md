# 统一物体检测方法评估系统使用指南

## 概述

本系统按照标准化框架对比HSV+Depth和Depth+Clustering两种物体检测方法，提供全面的定量分析功能。

## 系统特点

### 1. 统一数据口径
- **标准化指标**: 位置误差、范围误差、检测时间、视角召回率
- **统一阈值**: 2cm严格阈值，5cm宽松阈值
- **一致条件**: 相同光照、背景、角度序列、终止规则

### 2. 全面评估指标

#### 准确度 (Accuracy)
- **位置误差**: ‖p_detect − p_true‖ (米)
- **范围误差**: √(ΔX²+ΔY²) (米)
  - ΔX = |(xmax−xmin)_det − (xmax−xmin)_true|
  - ΔY = |(ymax−ymin)_det − (ymax−ymin)_true|

#### 稳定性 (Stability)
- **标准差**: 位置误差的标准差
- **中位绝对偏差**: 位置误差的MAD

#### 召回/鲁棒性 (Recall/Robustness)
- **视角召回率**: 命中视角数 / 总视角数 (24)
- **提前终止步数**: 终止角度索引 / 总视角数

#### 速度/成本 (Speed/Cost)
- **检测时间**: 单次trial总时间 (秒)
- **FPS**: frame_count / detection_time
- **单帧处理时延**: detection_time / frame_count

#### 诊断信息 (Diagnostic)
- **视角级数据**: 每个视角的命中状态、得分、坐标
- **最佳角度**: 得分最高的视角角度
- **有效深度点数**: 每个视角的有效深度数据量

## 实验设计

### 1. 条件矩阵
- **光照条件**: normal / bright / dim (3组)
- **背景条件**: simple (1组)
- **总条件数**: 3 × 1 = 3组

### 2. 重复设计
- **每组条件**: 10次trial
- **总trial数**: 3 × 10 × 2 = 60次
- **随机种子**: 固定种子确保可重复性

### 3. 公平性控制
- **相同条件**: 两算法在完全相同的条件下运行
- **相同序列**: 24个视角角度序列一致
- **相同规则**: 提前终止规则一致 (miss_count >= 2)
- **关闭显示**: 避免计时污染

## 使用方法

### 快速开始

```bash
# 进入tutorials目录
cd examples/tutorials

# 运行统一评估实验
python run_unified_evaluation.py
```

### 分步执行

```bash
# 1. 直接运行统一评估系统
python unified_object_detection_evaluation.py

# 2. 统计分析 (可选)
python statistical_analysis.py evaluation_results/unified_object_detection_YYYYMMDD_HHMMSS/experiment_results.json
```

## 输出文件说明

### 1. 主要数据文件

#### CSV格式 (trial_results.csv)
```csv
method,lighting,background,trial_id,seed,pos_err_m,dx_m,dy_m,range_err_m,
success_pos_2cm,success_range_2cm,success_pos_5cm,success_range_5cm,
detection_time_s,frame_count,fps,views_total,views_hit,early_stop_index,
view_hit_rate,best_angle_deg,best_score
```

#### JSON格式 (experiment_results.json)
```json
{
  "experiment_config": {...},
  "hardware_info": {...},
  "timestamp": "...",
  "trial_results": [...]
}
```

### 2. 报告文件

#### 实验报告 (experiment_report.txt)
- 实验配置信息
- 硬件环境信息
- 详细统计结果
- 按方法和条件分组的数据

#### 可视化图表 (evaluation_charts.png)
- 位置误差箱线图
- 范围误差箱线图
- 检测时间箱线图
- 成功率柱状图
- 视角召回率柱状图
- FPS对比图

## 数据结构详解

### TrialResult数据类

```python
@dataclass
class TrialResult:
    # 基本信息
    method: str              # 检测方法
    lighting: str            # 光照条件
    background: str          # 背景条件
    trial_id: int           # trial ID
    seed: int               # 随机种子
    
    # 准确度指标
    pos_err_m: float        # 位置误差（米）
    dx_m: float            # X方向范围误差（米）
    dy_m: float            # Y方向范围误差（米）
    range_err_m: float     # 合成范围误差（米）
    
    # 成功判定
    success_pos_2cm: bool   # 位置误差 < 2cm
    success_range_2cm: bool # 范围误差 < 2cm
    success_pos_5cm: bool   # 位置误差 < 5cm
    success_range_5cm: bool # 范围误差 < 5cm
    
    # 速度/成本指标
    detection_time_s: float # 检测时间（秒）
    frame_count: int        # 处理帧数
    fps: float             # 平均帧率
    
    # 召回/鲁棒性指标
    views_total: int       # 总视角数（24）
    views_hit: int         # 命中视角数
    early_stop_index: int  # 提前终止的视角索引
    view_hit_rate: float   # 视角召回率
    
    # 诊断信息
    best_angle_deg: float  # 最佳角度（度）
    best_score: float      # 最佳得分
    
    # 视角级详细信息（可选）
    view_details: Optional[List[Dict]] = None
```

### 视角级详细信息

```python
view_details = [
    {
        'angle_deg': 0.0,           # 视角角度（度）
        'hit': 1,                   # 是否命中（0/1）
        'score': 1000.0,            # 视角得分
        'cx': 320,                  # 检测中心X坐标
        'cy': 240,                  # 检测中心Y坐标
        'depth_valid_count': 5000   # 有效深度点数
    },
    # ... 24个视角的数据
]
```

## 结果解读

### 1. 成功率分析
- **严格标准**: 位置误差 < 2cm 且 范围误差 < 2cm
- **宽松标准**: 位置误差 < 5cm 且 范围误差 < 5cm
- **对比方法**: 两算法在相同条件下的成功率对比

### 2. 精度分析
- **位置精度**: 检测位置与真实位置的欧几里得距离
- **范围精度**: 检测范围与真实范围的差异
- **稳定性**: 多次实验的标准差和中位绝对偏差

### 3. 效率分析
- **检测时间**: 从开始到完成的总时间
- **FPS**: 每秒处理的帧数
- **视角召回率**: 成功检测的视角比例

### 4. 鲁棒性分析
- **光照适应性**: 不同光照条件下的性能变化
- **提前终止**: 算法在遇到困难时的行为
- **最佳角度**: 算法最擅长检测的角度

## 统计分析方法

### 1. 描述性统计
- **均值±标准差**: 反映中心趋势和离散程度
- **中位数(MAD)**: 反映稳健的中心趋势
- **成功率**: 成功trial的比例

### 2. 推断性统计
- **正态性检验**: Shapiro-Wilk检验
- **参数检验**: 独立样本t检验（正态分布）
- **非参数检验**: Mann-Whitney U检验（非正态分布）
- **效应量**: Cohen's d效应量

### 3. 可视化分析
- **箱线图**: 显示数据分布和异常值
- **柱状图**: 显示分类变量的对比
- **散点图**: 显示变量间的相关性

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
成功率 (2cm) & 90.0\% & 95.0\% & 0.234 \\
视角召回率 & 83.3\% & 91.7\% & 0.012* \\
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
- **箱线图**: 显示各指标的数据分布
- **柱状图**: 显示成功率和召回率对比
- **散点图**: 显示角度-误差关系

## 故障排除

### 1. 常见问题

#### 导入错误
```bash
# 确保在正确的目录
cd examples/tutorials

# 检查依赖包
pip install numpy matplotlib seaborn pandas scipy
```

#### 内存不足
```python
# 减少重复次数
self.n_trials_per_condition = 5

# 关闭viewer
show_viewer=False
```

#### 检测失败
```python
# 调整成功阈值
self.success_threshold_strict = 0.05  # 放宽到5cm

# 检查物体位置是否在工作空间内
```

### 2. 调试建议
- 启用详细日志: `logging_level="debug"`
- 单次测试: 先运行单次trial验证
- 检查数据: 查看生成的CSV文件确认数据正确性

## 扩展功能

### 1. 添加新的检测方法
1. 实现检测函数，返回相同格式的结果
2. 在评估器中添加新方法
3. 更新可视化模块

### 2. 添加新的评估指标
1. 在TrialResult中添加新字段
2. 在检测函数中计算新指标
3. 更新分析和可视化模块

### 3. 自定义统计检验
1. 在统计分析器中添加新的检验方法
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
