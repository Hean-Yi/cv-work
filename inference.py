import os
import torch
import random
import json
import matplotlib.pyplot as plt
import warnings
from torchvision import transforms
from PIL import Image
import numpy as np

from model import OverLoCKModel
from dataset import OverLoCKDataset
from model_visualizer import ModelVisualizer  # 导入新的可视化模块

def load_trained_model(checkpoint_path, class_names, device='cuda'):
    """加载训练好的模型"""
    # 尝试不同的模型导入
    try:
        from scalable_model import ScalableOverLoCKModel
        from rtx4090_configs import RTX4090OptimalConfig
        
        # 使用可扩展模型
        config = RTX4090OptimalConfig()
        model = ScalableOverLoCKModel(
            class_names=class_names,
            config=config
        )
        print("✅ 使用可扩展模型")
    except ImportError:
        # 回退到原始模型
        model = OverLoCKModel(
            class_names=class_names,
            use_clip=True
        )
        print("✅ 使用原始模型")
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同的检查点格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 移动到设备
    model.to(device)
    model.eval()
    
    print(f"📊 模型已加载: {checkpoint_path}")
    return model

def run_comprehensive_evaluation(model, dataloader, device='cuda', save_dir='./visualizations'):
    """
    运行综合评估，包含所有可视化功能
    """
    print("🚀 开始综合模型评估...")
    print("="*60)
    
    # 创建可视化器
    visualizer = ModelVisualizer(model, device=device, save_dir=save_dir)
    
    # 1. 计算评估指标
    print("📊 1/3 计算模型评估指标...")
    metrics = visualizer.calculate_metrics(dataloader, num_samples=500)
    
    # 2. ERF可视化
    print("🔍 2/3 生成有效感受野可视化...")
    erf_maps = visualizer.visualize_effective_receptive_field(dataloader, num_images=100)
    
    # 3. GradCAM可视化
    print("🎯 3/3 生成GradCAM类激活图...")
    
    # 获取一批样本用于GradCAM
    for images, labels in dataloader:
        # 只取前8个样本进行可视化
        sample_images = images[:8].to(device)
        sample_labels = labels[:8].to(device)
        
        gradcams = visualizer.visualize_gradcam(sample_images, sample_labels)
        break  # 只处理第一批
    
    print("✅ 综合评估完成!")
    print(f"📁 结果保存在: {save_dir}")
    
    return metrics, erf_maps, gradcams

def inference_with_visualization():
    """推理并进行可视化的主函数"""
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 检查点路径
    checkpoint_paths = [
        './checkpoints/rtx4090_optimal_best_model.pth',
        './checkpoints/rtx4090_max_best_model.pth', 
        './checkpoints/best_model.pth'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("❌ 未找到模型检查点文件")
        print("💡 请确保训练完成并保存了模型")
        return
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    try:
        # 尝试ImageNet-100
        dataset = OverLoCKDataset(
            data_root="./data/imagenet100",
            mode='val',
            transform=transform,
            dataset_type='imagenet100'
        )
        print("✅ 使用ImageNet-100验证集")
    except:
        try:
            # 回退到CIFAR-10
            dataset = OverLoCKDataset(
                data_root="./data",
                mode='test', 
                transform=transform,
                dataset_type='cifar10'
            )
            print("✅ 使用CIFAR-10测试集")
        except Exception as e:
            print(f"❌ 无法加载数据集: {e}")
            return
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False,
        num_workers=2
    )
    
    print(f"📊 数据集信息:")
    print(f"  样本数量: {len(dataset)}")
    print(f"  类别数量: {dataset.get_num_classes()}")
    
    # 加载模型
    class_names = dataset.get_class_names()
    model = load_trained_model(checkpoint_path, class_names, device)
    
    # 运行综合评估
    save_dir = './evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        metrics, erf_maps, gradcams = run_comprehensive_evaluation(
            model, dataloader, device, save_dir
        )
        
        print("\n🎉 推理和可视化完成!")
        print(f"📈 Top-1 准确率: {metrics.get('top1_accuracy', 'N/A')}")
        print(f"📈 Top-5 准确率: {metrics.get('top5_accuracy', 'N/A')}")
        print(f"🚀 吞吐量: {metrics.get('throughput', 'N/A')} imgs/sec")
        print(f"🔢 参数量: {metrics.get('total_params', 'N/A')/1e6:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行推理和可视化
    success = inference_with_visualization()
    
    if success:
        print("✅ 推理和可视化成功完成!")
    else:
        print("❌ 推理和可视化失败，请检查错误信息")