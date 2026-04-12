import numpy as np
import sys
import os
import cv2

# 打印当前工作目录
print(f"Current working directory: {os.getcwd()}")

# 模型和配置文件路径 - 使用绝对路径
net_file = os.path.abspath('d:\\LABmodel\\MobileNet-SSD-Introduction\\deploy.prototxt')
caffe_model = os.path.abspath('d:\\LABmodel\\MobileNet-SSD-Introduction\\mobilenet_iter_73000.caffemodel')
test_dir = os.path.abspath('d:\\LABmodel\\MobileNet-SSD-Introduction\\images')
output_dir = os.path.abspath('d:\\LABmodel\\python\\picresults')

# 检查文件是否存在
print(f"Checking if files exist...")
print(f"Net file: {net_file}, exists: {os.path.exists(net_file)}")
print(f"Caffe model: {caffe_model}, exists: {os.path.exists(caffe_model)}")
print(f"Test directory: {test_dir}, exists: {os.path.exists(test_dir)}")
print(f"Output directory: {output_dir}, exists: {os.path.exists(output_dir)}")

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()
if not os.path.exists(test_dir):
    print(test_dir + " does not exist")
    exit()
if not os.path.exists(output_dir):
    print(output_dir + " does not exist")
    exit()

# 加载模型
print("Loading model...")
try:
    net = cv2.dnn.readNetFromCaffe(net_file, caffe_model)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 类别标签
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

# 查看测试目录中的文件
print(f"Files in test directory: {os.listdir(test_dir)}")

def preprocess(src):
    """预处理图像"""
    img = cv2.resize(src, (300, 300))
    blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
    return blob

def detect(imgfile):
    """检测图像中的目标"""
    print(f"Processing image: {imgfile}")
    origimg = cv2.imread(imgfile)
    
    if origimg is None:
        print(f"Failed to read image: {imgfile}")
        return False
    
    print(f"Image shape: {origimg.shape}")
    
    blob = preprocess(origimg)
    
    # 设置输入并前向传播
    net.setInput(blob)
    out = net.forward()
    
    # 解析输出
    h = origimg.shape[0]
    w = origimg.shape[1]
    
    # 遍历检测结果
    print(f"Number of detections: {out.shape[2]}")
    for i in range(out.shape[2]):
        confidence = out[0, 0, i, 2]
        if confidence > 0.5:  # 置信度阈值
            class_id = int(out[0, 0, i, 1])
            box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(np.int32)
            
            # 绘制边界框和标签
            cv2.rectangle(origimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.putText(origimg, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 保存结果
    filename = os.path.basename(imgfile)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.jpg")
    
    print(f"Saving result to: {output_path}")
    success = cv2.imwrite(output_path, origimg)
    if success:
        print(f"Successfully saved: {output_path}")
    else:
        print(f"Failed to save: {output_path}")
    
    # 检查文件是否真的存在
    if os.path.exists(output_path):
        print(f"File exists after saving: {os.path.getsize(output_path)} bytes")
    else:
        print(f"File does not exist after saving")
    
    return True

# 处理所有测试图像
print("Processing images...")
for f in os.listdir(test_dir):
    if f.endswith(('.jpg', '.jpeg', '.png')):
        detect(os.path.join(test_dir, f))

# 检查输出目录中的文件
print("Checking output directory...")
print(f"Files in output directory: {os.listdir(output_dir)}")

print("Processing complete!")
