#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet-100 数据集下载器
使用 Kaggle API 通过 kagglehub 下载 ImageNet-100 数据集
"""

import os
import sys
import shutil
from pathlib import Path
import time

def download_imagenet100():
    """
    使用 kagglehub 下载 ImageNet-100 数据集
    """
    print("🔄 开始下载 ImageNet-100 数据集...")
    print("=" * 50)
    
    try:
        # 导入 kagglehub
        try:
            import kagglehub
            print("✅ kagglehub 已导入")
        except ImportError:
            print("❌ kagglehub 未安装，正在安装...")
            os.system("pip install kagglehub")
            import kagglehub
            print("✅ kagglehub 安装完成")
        
        # 设置下载路径
        download_dir = "data/imagenet100"
        os.makedirs(download_dir, exist_ok=True)
        
        print(f"📁 数据将下载到: {os.path.abspath(download_dir)}")
        print("🚀 开始下载...")
        
        start_time = time.time()
        
        # 下载数据集
        path = kagglehub.dataset_download("ambityga/imagenet100")
        
        end_time = time.time()
        download_time = end_time - start_time
        
        print(f"✅ 下载完成!")
        print(f"📍 数据集路径: {path}")
        print(f"⏱️ 下载用时: {download_time:.2f} 秒")
        
        # 移动或链接到我们的目标目录
        target_path = os.path.abspath(download_dir)
        if path != target_path:
            print(f"📦 复制数据集到目标目录...")
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(path, target_path)
            print(f"✅ 数据集已复制到: {target_path}")
        
        # 检查数据集结构
        print("\n📊 数据集结构:")
        check_dataset_structure(target_path)
        
        return target_path
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        return None

def check_dataset_structure(data_path):
    """
    检查数据集结构
    """
    if not os.path.exists(data_path):
        print("❌ 数据集路径不存在")
        return
    
    print(f"📁 数据集根目录: {data_path}")
    
    # 列出顶级目录
    items = os.listdir(data_path)
    print(f"📋 顶级目录/文件 ({len(items)} 项):")
    
    # 检查是否有压缩文件
    compressed_files = []
    train_path = None
    val_path = None
    
    for item in sorted(items):
        item_path = os.path.join(data_path, item)
        if item.startswith('train.X') or item.startswith('val.X'):
            compressed_files.append(item)
            print(f"  📦 {item} (压缩文件)")
        elif item == 'Labels.json':
            print(f"  📄 {item} (标签文件)")
        elif os.path.isdir(item_path):
            num_files = len(os.listdir(item_path))
            print(f"  📂 {item}/ ({num_files} 项)")
            
            if item == 'train':
                train_path = item_path
            elif item == 'val':
                val_path = item_path
        else:
            file_size = os.path.getsize(item_path) / 1024 / 1024  # MB
            print(f"  📄 {item} ({file_size:.2f} MB)")
    
    # 如果有压缩文件但没有解压目录，提示解压
    if compressed_files and not (train_path and val_path):
        print(f"\n📦 发现 {len(compressed_files)} 个压缩文件")
        print("💡 数据集需要解压，将在首次使用时自动解压")
        return
    
    # 检查训练和验证集
    if train_path:
        train_classes = [d for d in os.listdir(train_path) 
                        if os.path.isdir(os.path.join(train_path, d))]
        print(f"\n🏷️ 训练集类别数: {len(train_classes)}")
        
        # 统计训练样本数
        total_train_samples = 0
        for class_dir in train_classes[:5]:  # 检查前5个类别
            class_path = os.path.join(train_path, class_dir)
            if os.path.isdir(class_path):
                num_samples = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_train_samples += num_samples
                print(f"  📸 {class_dir}: {num_samples} 张图片")
        
        if len(train_classes) > 5:
            print(f"  ... 还有 {len(train_classes) - 5} 个类别")
        
        # 估计总样本数
        if len(train_classes) > 0:
            avg_samples_per_class = total_train_samples / min(5, len(train_classes))
            estimated_total = int(avg_samples_per_class * len(train_classes))
            print(f"📊 估计训练样本总数: ~{estimated_total}")
    
    if val_path:
        val_classes = [d for d in os.listdir(val_path) 
                      if os.path.isdir(os.path.join(val_path, d))]
        print(f"🔍 验证集类别数: {len(val_classes)}")
        
        # 统计验证样本数
        total_val_samples = 0
        for class_dir in val_classes[:5]:
            class_path = os.path.join(val_path, class_dir)
            if os.path.isdir(class_path):
                num_samples = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_val_samples += num_samples
        
        if len(val_classes) > 0:
            avg_samples_per_class = total_val_samples / min(5, len(val_classes))
            estimated_total = int(avg_samples_per_class * len(val_classes))
            print(f"📊 估计验证样本总数: ~{estimated_total}")

def setup_kaggle_credentials():
    """
    设置 Kaggle 凭据
    """
    print("🔑 检查 Kaggle API 凭据...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_json):
        print("✅ Kaggle 凭据已存在")
        return True
    
    print("❌ 未找到 Kaggle 凭据文件")
    print("📋 请按照以下步骤设置 Kaggle API:")
    print("1. 访问 https://www.kaggle.com/account")
    print("2. 滚动到 'API' 部分")
    print("3. 点击 'Create New API Token'")
    print("4. 下载 kaggle.json 文件")
    print(f"5. 将文件保存到: {kaggle_json}")
    print("6. 或者将文件内容粘贴到环境变量中")
    
    return False

def main():
    """
    主函数
    """
    print("🚀 ImageNet-100 数据集下载器")
    print("=" * 50)
    
    # 检查 Kaggle 凭据
    if not setup_kaggle_credentials():
        print("⚠️ 请先设置 Kaggle API 凭据")
        print("💡 或者可以设置环境变量:")
        print("   export KAGGLE_USERNAME=your_username")
        print("   export KAGGLE_KEY=your_api_key")
        
        # 尝试从环境变量获取
        if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
            print("✅ 从环境变量获取到 Kaggle 凭据")
        else:
            print("❌ 未找到环境变量中的凭据")
            return
    
    # 下载数据集
    dataset_path = download_imagenet100()
    
    if dataset_path:
        print("\n" + "=" * 50)
        print("✅ ImageNet-100 下载完成!")
        print(f"📁 数据集位置: {dataset_path}")
        print("🎯 接下来可以:")
        print("1. 检查数据集结构")
        print("2. 更新 dataset.py 文件")
        print("3. 开始训练模型")
    else:
        print("\n❌ 下载失败，请检查网络连接和 Kaggle 凭据")

if __name__ == "__main__":
    main()