# 目标检测脚本说明

## 原始代码分析

原始的 `demo.py` 文件是一个使用 Caffe 框架进行目标检测的脚本，主要功能包括：

1. **加载预训练模型**：使用 Caffe 框架加载 MobileNet-SSD 模型
2. **图像预处理**：将图像调整为 300x300 大小，并进行归一化处理
3. **目标检测**：使用模型对图像进行前向传播，获取检测结果
4. **结果可视化**：在图像上绘制边界框和类别标签
5. **显示结果**：使用 OpenCV 显示检测结果

## 新脚本说明

由于 Caffe 框架在现代环境中配置较为复杂，我们创建了一个使用 OpenCV 的 DNN 模块的新脚本，实现了相同的功能。

### 依赖项

- Python 3.6+
- OpenCV 4.0+
- NumPy 1.16+

### 如何运行

1. 确保安装了所需的依赖项：
   ```
   pip install opencv-python numpy
   ```

2. 运行脚本：
   ```
   python object_detection.py
   ```

3. 结果将保存在 `D:\LABmodel\python\picresults` 目录中

### 脚本功能

1. **加载预训练模型**：使用 OpenCV 的 DNN 模块加载 MobileNet-SSD 模型
2. **图像预处理**：将图像调整为 300x300 大小，并进行归一化处理
3. **目标检测**：使用模型对图像进行前向传播，获取检测结果
4. **结果可视化**：在图像上绘制边界框和类别标签
5. **保存结果**：将检测结果保存到指定目录

### 解决依赖问题

如果遇到 NumPy 或 OpenCV 的依赖问题，可以尝试以下方法：

1. **使用虚拟环境**：
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install opencv-python numpy
   ```

2. **使用 Anaconda**：
   ```
   conda create -n object-detection python=3.8
   conda activate object-detection
   conda install opencv numpy
   ```

3. **降级 NumPy 版本**：
   ```
   pip install numpy==1.26.4
   ```

## 目录结构

```
d:\LABmodel\python\
├── object_detection.py    # 主脚本
├── simple_object_detection.py  # 简化版本脚本
└── picresults/           # 结果保存目录
```
