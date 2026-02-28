# 高级边缘检测方法使用指南

## 概述

本项目现在支持两种先进的边缘检测方法：

1. **HED (Holistically-Nested Edge Detection)** - 基于深度学习的端到端边缘检测
2. **SE (Structure Edge Detection)** - 基于结构化森林的边缘检测

这些方法特别适合处理复杂光照条件和阴影干扰的场景。

## 安装依赖

### 自动安装
```bash
python install_dependencies.py
```

### 手动安装
```bash
pip install torch torchvision scipy opencv-python
```

## 可用的边缘检测方法

### 传统方法
- `canny` - Canny边缘检测
- `hsv` - HSV颜色分割
- `hybrid` - 混合方法
- `adaptive` - 自适应方法

### RGB-D方法
- `rgbd_depth` - 深度边缘检测
- `rgbd_hybrid` - RGB-D混合检测
- `rgbd_3d` - 3D几何特征检测

### 深度学习方法 ⭐
- `hed` - Holistically-Nested Edge Detection
- `se` - Structure Edge Detection

## 使用方法

### 1. 测试模式
运行程序后选择"1"进入测试模式：
```bash
python test.py
```

然后选择"1"测试所有方法，或选择"2"测试单个方法。

### 2. 直接使用特定方法
在代码中直接指定方法：
```python
contours, edges, img, depth = detect_object_boundary(cam, method='hed', debug_vis=True)
```

### 3. 在边界扫描中使用
修改默认方法：
```python
# 在 track_and_scan_boundary_jump 函数中
contours, edges, img, depth = detect_object_boundary(cam, method='hed', debug_vis=True)
```

## 方法对比

| 方法 | 抗阴影能力 | 精度 | 速度 | 适用场景 |
|------|------------|------|------|----------|
| `canny` | 低 | 中等 | 快 | 简单场景 |
| `hsv` | 中等 | 中等 | 快 | 颜色对比明显 |
| `rgbd_depth` | 高 | 高 | 快 | 深度信息丰富 |
| `hed` | **很高** | **很高** | 中等 | **复杂光照** |
| `se` | **很高** | **很高** | 中等 | **结构边缘** |

## 推荐使用场景

### 阴影严重的情况
推荐使用 `hed` 方法：
```python
contours, edges, img, depth = detect_object_boundary(cam, method='hed')
```

### 需要精确结构边缘
推荐使用 `se` 方法：
```python
contours, edges, img, depth = detect_object_boundary(cam, method='se')
```

### 实时性要求高
推荐使用 `rgbd_depth` 方法：
```python
contours, edges, img, depth = detect_object_boundary(cam, method='rgbd_depth')
```

## 性能优化

### GPU加速
如果安装了CUDA版本的PyTorch，HED和SE方法会自动使用GPU加速：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 模型优化
对于生产环境，建议：
1. 使用预训练的HED/SE模型
2. 模型量化以减少内存使用
3. 批处理处理多帧图像

## 故障排除

### PyTorch未安装
如果看到警告"PyTorch未安装"，请运行：
```bash
python install_dependencies.py
```

### 内存不足
如果遇到内存问题，可以：
1. 减小图像分辨率
2. 使用CPU版本的PyTorch
3. 降低批处理大小

### 检测效果不佳
1. 尝试不同的方法
2. 调整光照条件
3. 检查相机参数设置

## 扩展开发

### 添加新的边缘检测方法
1. 在 `detect_object_boundary` 函数中添加新的分支
2. 实现对应的 `_detect_boundary_xxx` 函数
3. 更新方法列表和文档

### 自定义模型
可以替换简化的实现为真正的预训练模型：
```python
# 加载预训练的HED模型
hed_model = load_hed_model('path/to/hed_model.pth')
edges = hed_model(img_tensor)
```

## 参考资料

- [HED论文](https://arxiv.org/abs/1504.06375)
- [SE论文](https://arxiv.org/abs/1412.0773)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [OpenCV边缘检测](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) 