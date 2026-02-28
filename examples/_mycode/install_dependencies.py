#!/usr/bin/env python3
"""
安装HED和SE边缘检测方法所需的依赖

这个脚本会检查并安装必要的Python包来支持深度学习边缘检测方法
"""

import subprocess
import sys
import os

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=== HED和SE边缘检测方法依赖安装脚本 ===\n")
    
    # 需要安装的包
    required_packages = [
        ("torch", "PyTorch - 深度学习框架"),
        ("torchvision", "TorchVision - 计算机视觉工具"),
        ("scipy", "SciPy - 科学计算库"),
        ("opencv-python", "OpenCV - 计算机视觉库")
    ]
    
    print("检查已安装的包...")
    missing_packages = []
    
    for package, description in required_packages:
        if check_package(package.replace("-", "_")):
            print(f"✓ {package} ({description}) - 已安装")
        else:
            print(f"✗ {package} ({description}) - 未安装")
            missing_packages.append(package)
    
    if not missing_packages:
        print("\n所有依赖包都已安装！")
        print("你现在可以使用HED和SE边缘检测方法了。")
        return
    
    print(f"\n需要安装 {len(missing_packages)} 个包:")
    for package in missing_packages:
        print(f"  - {package}")
    
    choice = input("\n是否自动安装这些包？(y/n): ").lower().strip()
    
    if choice == 'y':
        print("\n开始安装...")
        failed_packages = []
        
        for package in missing_packages:
            print(f"正在安装 {package}...")
            if install_package(package):
                print(f"✓ {package} 安装成功")
            else:
                print(f"✗ {package} 安装失败")
                failed_packages.append(package)
        
        if failed_packages:
            print(f"\n以下包安装失败: {', '.join(failed_packages)}")
            print("请手动安装这些包:")
            for package in failed_packages:
                print(f"  pip install {package}")
        else:
            print("\n所有包安装成功！")
            print("你现在可以使用HED和SE边缘检测方法了。")
    else:
        print("\n请手动安装以下包:")
        for package in missing_packages:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main() 