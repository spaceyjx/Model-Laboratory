import cv2
import os

# 模型和配置文件路径
net_file = 'd:\\LABmodel\\MobileNet-SSD-Introduction\\deploy.prototxt'
caffe_model = 'd:\\LABmodel\\MobileNet-SSD-Introduction\\mobilenet_iter_73000.caffemodel'
test_dir = 'd:\\LABmodel\\MobileNet-SSD-Introduction\\images'
output_dir = 'd:\\LABmodel\\python\\picresults'

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

# 处理所有测试图像
print("Processing images...")
for f in os.listdir(test_dir):
    if f.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(test_dir, f)
        print(f"Processing image: {img_path}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        
        print(f"Image shape: {img.shape}")
        
        # 预处理图像
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        
        # 设置输入并前向传播
        net.setInput(blob)
        detections = net.forward()
        
        # 解析检测结果
        (h, w) = img.shape[:2]
        
        # 遍历检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # 置信度阈值
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7]
                box[0] *= w
                box[1] *= h
                box[2] *= w
                box[3] *= h
                
                # 绘制边界框和标签
                (x1, y1, x2, y2) = box.astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{CLASSES[class_id]}: {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 保存结果
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.jpg")
        
        print(f"Saving result to: {output_path}")
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"Successfully saved: {output_path}")
        else:
            print(f"Failed to save: {output_path}")

print("Processing complete!")
