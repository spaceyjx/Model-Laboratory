import os
import shutil

# 测试目录
test_dir = 'd:\\LABmodel\\MobileNet-SSD-Introduction\\images'
output_dir = 'd:\\LABmodel\\python\\picresults'

# 检查目录是否存在
print(f"Test directory exists: {os.path.exists(test_dir)}")
print(f"Output directory exists: {os.path.exists(output_dir)}")

# 检查输出目录是否可写
print(f"Output directory is writable: {os.access(output_dir, os.W_OK)}")

# 复制一个图像到输出目录
test_image = os.path.join(test_dir, '000001.jpg')
output_image = os.path.join(output_dir, 'test.jpg')

print(f"Test image exists: {os.path.exists(test_image)}")

if os.path.exists(test_image):
    print(f"Copying {test_image} to {output_image}")
    try:
        shutil.copy(test_image, output_image)
        print(f"Successfully copied image")
        print(f"Output image exists: {os.path.exists(output_image)}")
        print(f"Output image size: {os.path.getsize(output_image)} bytes")
    except Exception as e:
        print(f"Error copying image: {e}")

# 列出输出目录中的文件
print(f"Files in output directory: {os.listdir(output_dir)}")
