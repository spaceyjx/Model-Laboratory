# DeepSeek-V3.1-Teminus 目标检测系统

基于MobileNet-SSD深度学习模型的目标检测实现，使用OpenCV DNN模块进行高效的目标检测。

## 项目结构

```
DeepSeek-V3.1-Teminus python/
├── deepseek_object_detection.py    # 主目标检测程序
├── test_detection.py               # 测试脚本
├── picresults/                     # 检测结果保存目录
└── README.md                       # 项目说明
```

## 功能特性

- ✅ 基于MobileNet-SSD深度学习模型
- ✅ 支持20种常见物体检测
- ✅ 实时目标检测与边界框绘制
- ✅ 批量处理目录中的图像文件
- ✅ 置信度阈值可调节
- ✅ 详细的检测结果统计
- ✅ 多类别颜色区分显示

## 支持的检测类别

1. background (背景)
2. aeroplane (飞机)
3. bicycle (自行车)
4. bird (鸟)
5. boat (船)
6. bottle (瓶子)
7. bus (公交车)
8. car (汽车)
9. cat (猫)
10. chair (椅子)
11. cow (牛)
12. diningtable (餐桌)
13. dog (狗)
14. horse (马)
15. motorbike (摩托车)
16. person (人)
17. pottedplant (盆栽植物)
18. sheep (羊)
19. sofa (沙发)
20. train (火车)
21. tvmonitor (电视显示器)

## 使用方法

### 1. 运行测试脚本
```bash
python test_detection.py
```

### 2. 运行完整目标检测
```bash
python deepseek_object_detection.py
```

### 3. 自定义配置
修改 `deepseek_object_detection.py` 中的路径配置：
- `model_path`: 模型文件路径
- `config_path`: 配置文件路径
- `input_dir`: 输入图像目录
- `output_dir`: 输出结果目录

## 依赖要求

- Python 3.6+
- OpenCV 4.0+
- NumPy

## 模型文件

项目需要以下模型文件：
- `deploy.prototxt`: MobileNet-SSD配置文件
- `mobilenet_iter_73000.caffemodel`: 预训练权重文件

这些文件应放置在：`D:\LABmodel\MobileNet-SSD-Introduction\`

## 输出结果

检测结果将保存在 `picresults/` 目录中，文件名格式为：`原文件名_detected.jpg`

每个结果图像包含：
- 检测到的目标边界框
- 类别标签
- 置信度分数
- 不同类别的颜色区分

## 性能优化

- 使用OpenCV DNN模块进行高效推理
- 支持CPU和GPU加速（如果可用）
- 批量处理优化
- 内存友好的图像处理

## 开发说明

本项目基于DeepSeek-V3.1-Teminus模型开发，专注于计算机视觉目标检测任务。代码结构清晰，易于扩展和维护。