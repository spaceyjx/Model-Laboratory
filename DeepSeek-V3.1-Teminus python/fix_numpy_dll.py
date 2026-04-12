"""
NumPy DLL加载失败修复工具
专门解决Windows环境下numpy DLL初始化失败问题
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_system_info():
    """检查系统信息"""
    print("=" * 60)
    print("NumPy DLL加载失败诊断工具")
    print("=" * 60)
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"系统架构: {platform.architecture()[0]}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
def check_numpy_installation():
    """检查numpy安装状态"""
    print("\n检查NumPy安装状态...")
    
    try:
        import numpy
        print(f"✅ NumPy已安装 - 版本: {numpy.__version__}")
        print(f"📁 NumPy路径: {numpy.__file__}")
        return True
    except ImportError as e:
        print(f"❌ NumPy未安装或导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ NumPy导入时出现错误: {e}")
        return False

def check_vc_redist():
    """检查Visual C++运行库"""
    print("\n检查Visual C++运行库...")
    
    # 常见的VC++ redist路径
    vc_paths = [
        r"C:\Windows\System32\vcruntime140.dll",
        r"C:\Windows\System32\vcruntime140_1.dll",
        r"C:\Windows\SysWOW64\vcruntime140.dll",
    ]
    
    for path in vc_paths:
        if os.path.exists(path):
            print(f"✅ {os.path.basename(path)} 存在")
        else:
            print(f"❌ {os.path.basename(path)} 缺失")

def fix_numpy_installation():
    """修复numpy安装"""
    print("\n开始修复NumPy安装...")
    
    # 卸载当前numpy
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                      check=True, capture_output=True, text=True)
        print("✅ 已卸载当前NumPy版本")
    except subprocess.CalledProcessError:
        print("⚠️  卸载NumPy时出现问题（可能未安装）")
    
    # 安装兼容版本
    compatible_versions = ["1.26.4", "1.25.2", "1.24.3"]
    
    for version in compatible_versions:
        print(f"\n尝试安装NumPy {version}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                f"numpy=={version}", "--no-cache-dir"
            ], check=True, capture_output=True, text=True)
            
            print(f"✅ NumPy {version} 安装成功")
            
            # 验证安装
            try:
                import numpy
                print(f"✅ NumPy {numpy.__version__} 导入成功")
                return True
            except Exception as e:
                print(f"❌ 导入失败: {e}")
                continue
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 安装NumPy {version} 失败: {e.stderr}")
            continue
    
    return False

def install_opencv():
    """安装OpenCV"""
    print("\n安装OpenCV...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "opencv-python", "--no-cache-dir"
        ], check=True, capture_output=True, text=True)
        
        print("✅ OpenCV安装成功")
        
        # 验证安装
        try:
            import cv2
            print(f"✅ OpenCV {cv2.__version__} 导入成功")
            return True
        except Exception as e:
            print(f"❌ OpenCV导入失败: {e}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ OpenCV安装失败: {e.stderr}")
        return False

def create_test_script():
    """创建测试脚本"""
    print("\n创建测试脚本...")
    
    test_script = """
import numpy as np
import cv2
import sys

print("=" * 50)
print("DeepSeek-V3.1-Teminus 依赖库测试")
print("=" * 50)

print(f"Python版本: {sys.version}")
print(f"NumPy版本: {np.__version__}")
print(f"OpenCV版本: {cv2.__version__}")

# 测试NumPy功能
print("\\n测试NumPy功能...")
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy数组: {arr}")
print(f"数组形状: {arr.shape}")

# 测试OpenCV功能
print("\\n测试OpenCV功能...")
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"创建测试图像: {img.shape}")

print("\\n✅ 所有依赖库测试通过！")
print("🎉 现在可以运行DeepSeek-V3.1-Teminus目标检测程序了！")
"""
    
    test_file = Path("D:/LABmodel/DeepSeek-V3.1-Teminus python/test_dependencies.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"✅ 测试脚本已创建: {test_file}")
    return test_file

def main():
    """主修复函数"""
    check_system_info()
    
    # 检查当前状态
    numpy_ok = check_numpy_installation()
    check_vc_redist()
    
    if not numpy_ok:
        print("\n" + "=" * 60)
        print("开始修复NumPy DLL加载问题...")
        print("=" * 60)
        
        # 修复numpy
        if fix_numpy_installation():
            print("\n✅ NumPy修复成功！")
        else:
            print("\n❌ NumPy修复失败，尝试备用方案...")
            
            # 备用方案：安装opencv-python-headless（包含numpy）
            print("\n尝试安装opencv-python-headless...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "opencv-python-headless", "--no-cache-dir"
                ], check=True)
                print("✅ opencv-python-headless安装成功")
            except:
                print("❌ 备用方案也失败了")
    
    # 安装OpenCV
    install_opencv()
    
    # 创建测试脚本
    test_file = create_test_script()
    
    print("\n" + "=" * 60)
    print("修复完成！")
    print("=" * 60)
    
    print("\n下一步操作：")
    print(f"1. 运行测试脚本: python {test_file}")
    print("2. 如果测试通过，可以运行目标检测程序")
    print("3. 如果仍有问题，请检查Visual C++运行库")
    
    print("\n💡 建议：")
    print("- 确保已安装Visual C++ Redistributable for Visual Studio 2015-2022")
    print("- 可以从 https://aka.ms/vs/17/release/vc_redist.x64.exe 下载")
    print("- 安装后重启系统")

if __name__ == "__main__":
    main()