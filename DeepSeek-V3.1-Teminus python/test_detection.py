"""
DeepSeek-V3.1-Teminus 目标检测测试脚本
"""

import cv2
import numpy as np
import os

def create_test_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 绘制一些简单的形状作为测试目标
    # 红色矩形（模拟汽车）
    cv2.rectangle(img, (50, 100), (200, 250), (0, 0, 255), -1)
    
    # 蓝色圆形（模拟瓶子）
    cv2.circle(img, (400, 150), 50, (255, 0, 0), -1)
    
    # 绿色三角形（模拟人）
    pts = np.array([[300, 300], [250, 350], [350, 350]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 0))
    
    return img

def main():
    """主测试函数"""
    print("🧪 DeepSeek-V3.1-Teminus 目标检测测试")
    print("=" * 50)
    
    # 创建测试目录结构
    test_dir = "D:\\LABmodel\\DeepSeek-V3.1-Teminus python\\test_images"
    output_dir = "D:\\LABmodel\\DeepSeek-V3.1-Teminus python\\picresults"
    
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建并保存测试图像
    test_image = create_test_image()
    test_image_path = os.path.join(test_dir, "test_image.jpg")
    cv2.imwrite(test_image_path, test_image)
    
    print(f"✅ 创建测试图像: {test_image_path}")
    print(f"📊 图像尺寸: {test_image.shape}")
    
    # 检查OpenCV版本和功能
    print(f"📋 OpenCV版本: {cv2.__version__}")
    print(f"🔧 DNN模块可用: {'是' if hasattr(cv2, 'dnn') else '否'}")
    
    # 检查模型文件是否存在
    model_dir = "D:\\LABmodel\\MobileNet-SSD-Introduction"
    config_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")
    
    print(f"🔍 检查模型文件...")
    print(f"配置文件: {config_path} - {'✅ 存在' if os.path.exists(config_path) else '❌ 不存在'}")
    print(f"模型文件: {model_path} - {'✅ 存在' if os.path.exists(model_path) else '❌ 不存在'}")
    
    # 简单的图像处理测试
    print("\n🎯 进行简单的图像处理测试...")
    
    # 读取测试图像
    img = cv2.imread(test_image_path)
    if img is not None:
        print("✅ 图像读取成功")
        
        # 简单的图像处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 保存处理结果
        processed_path = os.path.join(output_dir, "processed_test.jpg")
        cv2.imwrite(processed_path, blurred)
        print(f"✅ 图像处理完成: {processed_path}")
    else:
        print("❌ 图像读取失败")
    
    print("\n🎉 测试完成！")
    print("💡 下一步可以运行 deepseek_object_detection.py 进行完整的目标检测")

if __name__ == "__main__":
    main()