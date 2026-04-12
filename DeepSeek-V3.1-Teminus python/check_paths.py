"""
路径配置验证脚本
检查deepseek_object_detection.py中的路径配置是否正确
"""

import os

def check_paths():
    """检查路径配置"""
    print("=" * 60)
    print("路径配置验证")
    print("=" * 60)
    
    # 程序中的路径配置
    base_dir = "D:\\LABmodel"
    model_dir = os.path.join(base_dir, "MobileNet-SSD-Introduction")
    config_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")
    input_dir = os.path.join(model_dir, "images")
    output_dir = os.path.join(base_dir, "DeepSeek-V3.1-Teminus python", "picresults")
    
    paths_to_check = {
        "基础目录": base_dir,
        "模型目录": model_dir,
        "配置文件": config_path,
        "模型文件": model_path,
        "输入目录": input_dir,
        "输出目录": output_dir
    }
    
    all_exist = True
    
    for name, path in paths_to_check.items():
        exists = os.path.exists(path)
        status = "存在" if exists else "不存在"
        
        if not exists:
            all_exist = False
            
        print(f"{name}: {path}")
        print(f"  状态: {status}")
        
        # 如果是目录，检查内容
        if exists and name in ["输入目录", "模型目录"]:
            try:
                files = os.listdir(path)
                if name == "输入目录":
                    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"  图像文件数量: {len(image_files)}")
                    if image_files:
                        print(f"  示例文件: {image_files[:3]}")
                else:
                    print(f"  文件数量: {len(files)}")
                    print(f"  示例文件: {files[:3]}")
            except Exception as e:
                print(f"  读取目录失败: {e}")
        
        print()
    
    return all_exist

def main():
    """主函数"""
    all_paths_exist = check_paths()
    
    print("=" * 60)
    print("验证结果:")
    
    if all_paths_exist:
        print("✅ 所有路径配置正确，程序可以正常运行")
    else:
        print("❌ 部分路径不存在，程序运行时可能出现错误")
        print("💡 建议检查以下内容:")
        print("  1. MobileNet-SSD-Introduction目录是否存在")
        print("  2. 模型文件deploy.prototxt和mobilenet_iter_73000.caffemodel是否已下载")
        print("  3. images目录中是否包含测试图像")
    
    print("=" * 60)

if __name__ == "__main__":
    main()