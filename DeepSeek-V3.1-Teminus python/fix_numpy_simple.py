"""
NumPy DLL加载失败修复工具 - 简化版
专门解决Windows环境下numpy DLL初始化失败问题
"""

import os
import sys
import subprocess
import platform

def main():
    """主修复函数"""
    print("=" * 60)
    print("NumPy DLL加载失败修复工具")
    print("=" * 60)
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查当前numpy状态
    print("\n检查NumPy状态...")
    try:
        import numpy
        print(f"NumPy已安装 - 版本: {numpy.__version__}")
        print("NumPy导入成功！")
        return True
    except Exception as e:
        print(f"NumPy导入失败: {e}")
    
    # 修复步骤
    print("\n开始修复NumPy安装...")
    
    # 步骤1: 卸载当前numpy
    print("步骤1: 卸载当前NumPy版本")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                      check=True, capture_output=True)
        print("卸载成功")
    except:
        print("卸载失败或未安装")
    
    # 步骤2: 安装兼容版本
    print("\n步骤2: 安装兼容的NumPy版本")
    
    # 尝试多个兼容版本
    versions_to_try = ["1.26.4", "1.25.2", "1.24.3", "1.23.5"]
    
    for version in versions_to_try:
        print(f"尝试安装NumPy {version}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                f"numpy=={version}", "--no-cache-dir"
            ], check=True, capture_output=True)
            
            # 验证安装
            try:
                import numpy
                print(f"NumPy {numpy.__version__} 安装并导入成功！")
                return True
            except:
                print(f"安装成功但导入失败，尝试下一个版本...")
                continue
                
        except subprocess.CalledProcessError:
            print(f"安装失败，尝试下一个版本...")
            continue
    
    # 步骤3: 如果所有版本都失败，尝试安装opencv-python-headless
    print("\n步骤3: 尝试安装opencv-python-headless（包含numpy）")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "opencv-python-headless", "--no-cache-dir"
        ], check=True, capture_output=True)
        
        # 验证
        try:
            import numpy
            import cv2
            print(f"opencv-python-headless安装成功")
            print(f"NumPy版本: {numpy.__version__}")
            print(f"OpenCV版本: {cv2.__version__}")
            return True
        except:
            print("安装成功但导入失败")
            
    except subprocess.CalledProcessError:
        print("opencv-python-headless安装失败")
    
    print("\n修复完成，但问题可能仍未解决。")
    print("建议手动下载并安装Visual C++ Redistributable for Visual Studio 2015-2022")
    print("下载地址: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("安装后请重启系统")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 60)
        print("修复成功！现在可以运行目标检测程序了。")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("修复未完全成功，请尝试手动安装Visual C++运行库")
        print("=" * 60)