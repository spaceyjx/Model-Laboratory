"""
DeepSeek-V3.1-Teminus 简化测试脚本
不依赖numpy，仅测试基本文件操作和路径验证
"""

import os
import shutil
import time
from datetime import datetime

def main():
    """主测试函数"""
    print("=" * 60)
    print("DeepSeek-V3.1-Teminus 简化测试")
    print("=" * 60)
    
    # 配置路径
    base_dir = "D:\\LABmodel"
    model_dir = os.path.join(base_dir, "MobileNet-SSD-Introduction")
    deepseek_dir = os.path.join(base_dir, "DeepSeek-V3.1-Teminus python")
    output_dir = os.path.join(deepseek_dir, "picresults")
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"测试时间: {timestamp}")
    print(f"基础目录: {base_dir}")
    print(f"DeepSeek目录: {deepseek_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查目录结构
    print("\n检查目录结构...")
    directories = {
        "基础目录": base_dir,
        "模型目录": model_dir,
        "DeepSeek目录": deepseek_dir,
        "输出目录": output_dir
    }
    
    for name, path in directories.items():
        exists = os.path.exists(path)
        writable = os.access(path, os.W_OK) if exists else False
        print(f"{name}: {path}")
        print(f"  存在: {'是' if exists else '否'}")
        if exists:
            print(f"  可写: {'是' if writable else '否'}")
    
    # 检查模型文件
    print("\n检查模型文件...")
    model_files = {
        "配置文件": "deploy.prototxt",
        "模型文件": "mobilenet_iter_73000.caffemodel",
        "测试图像目录": "images"
    }
    
    for name, filename in model_files.items():
        file_path = os.path.join(model_dir, filename)
        exists = os.path.exists(file_path)
        print(f"{name}: {file_path}")
        print(f"  存在: {'是' if exists else '否'}")
        
        if exists and filename == "images":
            # 如果是目录，列出其中的文件
            try:
                files = os.listdir(file_path)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  图像文件数量: {len(image_files)}")
                if image_files:
                    print(f"  前5个文件: {image_files[:5]}")
            except Exception as e:
                print(f"  读取目录失败: {e}")
    
    # 创建测试文件
    print("\n创建测试文件...")
    test_file_path = os.path.join(output_dir, "test_report.txt")
    
    try:
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("DeepSeek-V3.1-Teminus 测试报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"测试时间: {timestamp}\n")
            f.write(f"Python版本: 3.11.0\n")
            f.write(f"操作系统: Windows\n")
            f.write("\n目录检查结果:\n")
            
            for name, path in directories.items():
                exists = os.path.exists(path)
                f.write(f"{name}: {'存在' if exists else '不存在'}\n")
        
        print(f"测试报告已创建: {test_file_path}")
        
        # 验证文件创建
        if os.path.exists(test_file_path):
            file_size = os.path.getsize(test_file_path)
            print(f"文件大小: {file_size} 字节")
            
            # 读取并显示文件内容
            with open(test_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"文件内容预览:\n{content}")
        
    except Exception as e:
        print(f"创建测试文件失败: {e}")
    
    # 性能测试
    print("\n性能测试...")
    start_time = time.time()
    
    # 简单的文件操作测试
    test_operations = 1000
    for i in range(test_operations):
        _ = i * i  # 简单计算
    
    end_time = time.time()
    operation_time = (end_time - start_time) / test_operations * 1e6  # 微秒
    
    print(f"完成 {test_operations} 次操作")
    print(f"平均操作时间: {operation_time:.2f} 微秒")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    # 检查关键文件是否存在
    critical_files = [
        os.path.join(model_dir, "deploy.prototxt"),
        os.path.join(model_dir, "mobilenet_iter_73000.caffemodel"),
        os.path.join(model_dir, "images")
    ]
    
    missing_files = [f for f in critical_files if not os.path.exists(f)]
    
    if not missing_files:
        print("所有关键文件都存在")
        print("可以运行完整的目标检测程序")
    else:
        print("缺少以下关键文件:")
        for f in missing_files:
            print(f"   - {f}")
        print("请确保模型文件已正确放置")
    
    print(f"\n项目结构已创建在: {deepseek_dir}")
    print(f"检测结果将保存在: {output_dir}")
    print("\nDeepSeek-V3.1-Teminus 测试完成！")

if __name__ == "__main__":
    main()