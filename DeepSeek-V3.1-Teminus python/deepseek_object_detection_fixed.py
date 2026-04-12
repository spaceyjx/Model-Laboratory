"""
DeepSeek-V3.1-Teminus 目标检测程序（修复版）
基于MobileNet-SSD深度学习模型的目标检测实现
修复了Unicode编码和路径配置问题
"""

import cv2
import numpy as np
import os
import time
import argparse
from datetime import datetime

class DeepSeekObjectDetector:
    """DeepSeek目标检测器"""
    
    def __init__(self, model_path, config_path):
        """初始化检测器"""
        self.model_path = model_path
        self.config_path = config_path
        self.net = None
        self.classes = self._load_classes()
        self.load_model()
    
    def _load_classes(self):
        """加载类别标签"""
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def load_model(self):
        """加载深度学习模型"""
        print("[INFO] DeepSeek-V3.1-Teminus 模型加载中...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
            # 设置计算后端（优先使用CUDA，如果可用）
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[SUCCESS] 模型加载成功！")
            print(f"[INFO] 支持检测的类别: {len(self.classes)}种")
        except cv2.error as e:
            raise Exception(f"OpenCV模型加载失败: {e}")
        except FileNotFoundError as e:
            raise Exception(f"文件不存在: {e}")
        except Exception as e:
            raise Exception(f"模型加载失败: {e}")
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 调整大小为300x300（MobileNet-SSD输入尺寸）
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=0.007843,  # 缩放因子
            size=(300, 300),       # 输入尺寸
            mean=127.5,            # 均值减除
            swapRB=True            # BGR转RGB
        )
        return blob
    
    def detect_objects(self, image, confidence_threshold=0.5):
        """检测图像中的目标"""
        if self.net is None:
            raise Exception("模型未加载，请先调用load_model()")
        
        # 预处理
        blob = self.preprocess_image(image)
        
        # 设置输入并进行前向传播
        self.net.setInput(blob)
        start_time = time.time()
        detections = self.net.forward()
        inference_time = time.time() - start_time
        
        # 解析检测结果
        results = []
        (h, w) = image.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                class_id = int(detections[0, 0, i, 1])
                
                # 计算边界框坐标
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)
                
                # 确保坐标在图像范围内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                results.append({
                    'class_id': class_id,
                    'class_name': self.classes[class_id],
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'area': (x2 - x1) * (y2 - y1)
                })
        
        return results, inference_time
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测结果"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # 绘制边界框
            color = self._get_class_color(detection['class_id'])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # 绘制标签文本
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def _get_class_color(self, class_id):
        """为不同类别生成不同颜色"""
        # 使用HSV色彩空间生成不同色调的颜色
        hue = (class_id * 30) % 180  # 每类间隔30度色调
        hsv_color = np.uint8([[[hue, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr_color))
    
    def validate_image_file(self, image_path):
        """验证图像文件"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            raise ValueError(f"图像文件为空: {image_path}")
        
        return True
    
    def process_image_file(self, image_path, output_dir, confidence_threshold=0.5):
        """处理单个图像文件"""
        print(f"[IMAGE] 处理图像: {os.path.basename(image_path)}")
        
        # 验证图像文件
        try:
            self.validate_image_file(image_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}")
            return False
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] 无法读取图像: {image_path}")
            return False
        
        # 检测目标
        detections, inference_time = self.detect_objects(image, confidence_threshold)
        
        # 绘制结果
        result_image = self.draw_detections(image, detections)
        
        # 保存结果
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        
        success = cv2.imwrite(output_path, result_image)
        
        if success:
            print(f"[SUCCESS] 检测完成: {len(detections)}个目标")
            print(f"[TIME] 推理时间: {inference_time:.3f}秒")
            print(f"[SAVE] 结果保存: {output_path}")
            
            # 打印检测详情
            for detection in detections:
                print(f"   - {detection['class_name']}: {detection['confidence']:.2f}")
        
        return success
    
    def process_directory(self, input_dir, output_dir, confidence_threshold=0.5):
        """处理目录中的所有图像文件"""
        if not os.path.exists(input_dir):
            print(f"[ERROR] 输入目录不存在: {input_dir}")
            return False
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(supported_formats)]
        
        if not image_files:
            print(f"[ERROR] 目录中没有支持的图像文件: {input_dir}")
            return False
        
        print(f"[DIR] 开始处理目录: {input_dir}")
        print(f"[INFO] 找到 {len(image_files)} 个图像文件")
        
        success_count = 0
        total_start_time = time.time()
        
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            if self.process_image_file(image_path, output_dir, confidence_threshold):
                success_count += 1
            print("-" * 50)
        
        total_time = time.time() - total_start_time
        
        print(f"[COMPLETE] 处理完成！")
        print(f"[SUCCESS] 成功处理: {success_count}/{len(image_files)} 个文件")
        print(f"[TIME] 总耗时: {total_time:.2f}秒")
        print(f"[AVG] 平均每张图像: {total_time/len(image_files):.2f}秒")
        
        return success_count > 0


def main():
    """主函数"""
    print("=" * 60)
    print("DeepSeek-V3.1-Teminus 目标检测系统")
    print("=" * 60)
    
    # 配置路径（支持命令行参数）
    parser = argparse.ArgumentParser(description='DeepSeek目标检测程序')
    parser.add_argument('--model-dir', default='D:\\LABmodel\\MobileNet-SSD-Introduction',
                       help='模型文件目录路径')
    parser.add_argument('--output-dir', default='D:\\LABmodel\\DeepSeek-V3.1-Teminus python\\picresults',
                       help='输出结果目录路径')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='检测置信度阈值 (0.0-1.0)')
    
    args = parser.parse_args()
    
    # 构建完整路径
    model_dir = args.model_dir
    config_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")
    input_dir = os.path.join(model_dir, "images")
    output_dir = args.output_dir
    
    # 检查必要文件
    print("[CHECK] 检查文件路径...")
    print(f"配置文件: {config_path} - {'[EXISTS]' if os.path.exists(config_path) else '[MISSING]'}")
    print(f"模型文件: {model_path} - {'[EXISTS]' if os.path.exists(model_path) else '[MISSING]'}")
    print(f"输入目录: {input_dir} - {'[EXISTS]' if os.path.exists(input_dir) else '[MISSING]'}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {args.confidence}")
    
    if not all([os.path.exists(config_path), os.path.exists(model_path), os.path.exists(input_dir)]):
        print("[ERROR] 必要的文件或目录不存在，请检查路径配置")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 初始化检测器
        detector = DeepSeekObjectDetector(model_path, config_path)
        
        # 处理图像
        print("\n[START] 开始目标检测...")
        detector.process_directory(input_dir, output_dir, confidence_threshold=args.confidence)
        
        print("\n[SUCCESS] DeepSeek-V3.1-Teminus 目标检测完成！")
        
    except Exception as e:
        print(f"[ERROR] 程序执行出错: {e}")
        print("[HELP] 请检查：")
        print("  1. OpenCV是否正确安装")
        print("  2. 模型文件路径是否正确")
        print("  3. 图像文件格式是否支持")


if __name__ == "__main__":
    main()