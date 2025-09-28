#!/usr/bin/env python3
"""
运行置信度校准的主脚本
"""
import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from model import OverLoCKModel
from dataset import OverLoCKDataset
from confidence_calibration import (
    ConfidenceCalibrator, 
    plot_calibration_comparison,
    save_calibration_results,
    calculate_ece,
    calculate_mce
)


def load_model_and_data(checkpoint_path: str = './checkpoints/best_model.pth',
                       data_dir: str = './data',
                       batch_size: int = 32,
                       device: str = 'cuda'):
    """
    加载模型和数据
    Args:
        checkpoint_path: 模型检查点路径
        data_dir: 数据目录
        batch_size: 批次大小
        device: 设备
    Returns:
        model, val_loader, test_loader, class_names
    """
    # 类别名称
    class_names = ['救护车', '棕熊', '海星', '清真寺', '猎豹', 
                   '老虎', '蜜蜂', '野兔', '钢笔', '香蕉']
    
    # 创建模型
    model = OverLoCKModel(class_names=class_names, use_clip=True)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型已加载: {checkpoint_path}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    val_dataset = OverLoCKDataset(
        data_root=data_dir,
        mode='val',
        transform=transform
    )
    
    test_dataset = OverLoCKDataset(
        data_root=data_dir,
        mode='test',
        transform=transform
    )
    
    # 创建数据加载器
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"✅ 数据加载完成 - 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    return model, val_loader, test_loader, class_names


def evaluate_model(model, data_loader, device='cuda', method_name='原始'):
    """
    评估模型性能
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        method_name: 方法名称
    Returns:
        results: 包含所有预测结果的字典
    """
    print(f"🔍 评估 {method_name} 模型性能...")
    
    model.eval()
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            main_logits, aux_logits, clip_logits = model(images, use_aux=False)
            probs = torch.softmax(main_logits, dim=1)
            
            # 获取预测结果
            confidences, predicted = torch.max(probs, 1)
            
            for i in range(len(labels)):
                is_correct = predicted[i].item() == labels[i].item()
                results.append({
                    'confidence': confidences[i].item(),
                    'predicted': predicted[i].item(),
                    'true_label': labels[i].item(),
                    'is_correct': is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
    
    accuracy = 100.0 * correct / total
    print(f"✅ {method_name} 准确率: {accuracy:.2f}%")
    
    return {
        'results': results,
        'accuracy': accuracy,
        'total_samples': total
    }


def evaluate_calibrated_model(model, calibrator, data_loader, method='temperature', device='cuda'):
    """
    评估校准后的模型性能
    Args:
        model: 原始模型
        calibrator: 校准器
        data_loader: 数据加载器
        method: 校准方法 ('temperature', 'platt', 'isotonic')
        device: 设备
    Returns:
        results: 包含所有预测结果的字典
    """
    method_names = {
        'temperature': '温度缩放',
        'platt': 'Platt缩放',
        'isotonic': '等渗回归'
    }
    
    print(f"🔍 评估 {method_names[method]} 校准后模型性能...")
    
    model.eval()
    results = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            main_logits, aux_logits, clip_logits = model(images, use_aux=False)
            
            # 应用校准
            if method == 'temperature':
                probs = calibrator.apply_temperature_scaling(main_logits)
            elif method == 'platt':
                original_probs = torch.softmax(main_logits, dim=1)
                probs = calibrator.apply_platt_scaling(original_probs)
            elif method == 'isotonic':
                original_probs = torch.softmax(main_logits, dim=1)
                probs = calibrator.apply_isotonic_regression(original_probs)
            else:
                raise ValueError(f"未知的校准方法: {method}")
            
            # 获取预测结果
            confidences, predicted = torch.max(probs, 1)
            
            for i in range(len(labels)):
                is_correct = predicted[i].item() == labels[i].item()
                results.append({
                    'confidence': confidences[i].item(),
                    'predicted': predicted[i].item(),
                    'true_label': labels[i].item(),
                    'is_correct': is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
    
    accuracy = 100.0 * correct / total
    print(f"✅ {method_names[method]} 校准后准确率: {accuracy:.2f}%")
    
    return {
        'results': results,
        'accuracy': accuracy,
        'total_samples': total
    }


def print_calibration_metrics(results_dict):
    """
    打印校准指标
    Args:
        results_dict: 结果字典
    """
    print("\n" + "="*60)
    print("📊 置信度校准结果汇总")
    print("="*60)
    
    for method_name, results in results_dict.items():
        if method_name == 'original':
            display_name = '原始模型'
        elif method_name == 'temperature':
            display_name = '温度缩放'
        elif method_name == 'platt':
            display_name = 'Platt缩放'
        elif method_name == 'isotonic':
            display_name = '等渗回归'
        else:
            display_name = method_name
        
        print(f"\n🔹 {display_name}:")
        print(f"   准确率: {results['accuracy']:.2f}%")
        print(f"   ECE: {results['ece']:.4f}")
        print(f"   MCE: {results['mce']:.4f}")
        print(f"   平均置信度: {results['avg_confidence']:.4f}")
        print(f"   置信度标准差: {results['confidence_std']:.4f}")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    print("🚀 开始置信度校准流程...")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 使用设备: {device}")
    
    # 检查必要文件
    checkpoint_path = './checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到模型文件: {checkpoint_path}")
        print("请先训练模型或确保模型文件存在")
        return
    
    # 加载模型和数据
    try:
        model, val_loader, test_loader, class_names = load_model_and_data(
            checkpoint_path=checkpoint_path,
            device=device
        )
    except Exception as e:
        print(f"❌ 加载模型和数据失败: {e}")
        return
    
    # 创建校准器
    calibrator = ConfidenceCalibrator(model, device)
    
    # 1. 评估原始模型
    print("\n" + "="*50)
    print("📊 第一步: 评估原始模型")
    print("="*50)
    
    original_results = evaluate_model(model, test_loader, device, '原始')
    
    # 2. 进行校准训练
    print("\n" + "="*50)
    print("🔧 第二步: 训练校准器")
    print("="*50)
    
    try:
        # 温度缩放校准
        optimal_temp = calibrator.calibrate_temperature_scaling(val_loader)
        
        # Platt缩放校准
        calibrator.calibrate_platt_scaling(val_loader)
        
        # 等渗回归校准
        calibrator.calibrate_isotonic_regression(val_loader)
        
    except Exception as e:
        print(f"❌ 校准训练失败: {e}")
        return
    
    # 3. 评估校准后的模型
    print("\n" + "="*50)
    print("📊 第三步: 评估校准后模型")
    print("="*50)
    
    calibrated_results = {}
    
    # 评估温度缩放
    try:
        temp_results = evaluate_calibrated_model(
            model, calibrator, test_loader, 'temperature', device
        )
        calibrated_results['temperature'] = temp_results
    except Exception as e:
        print(f"⚠️  温度缩放评估失败: {e}")
    
    # 评估Platt缩放
    try:
        platt_results = evaluate_calibrated_model(
            model, calibrator, test_loader, 'platt', device
        )
        calibrated_results['platt'] = platt_results
    except Exception as e:
        print(f"⚠️  Platt缩放评估失败: {e}")
    
    # 评估等渗回归
    try:
        isotonic_results = evaluate_calibrated_model(
            model, calibrator, test_loader, 'isotonic', device
        )
        calibrated_results['isotonic'] = isotonic_results
    except Exception as e:
        print(f"⚠️  等渗回归评估失败: {e}")
    
    # 4. 保存结果和生成可视化
    print("\n" + "="*50)
    print("💾 第四步: 保存结果和生成可视化")
    print("="*50)
    
    try:
        # 保存校准结果
        results_summary = save_calibration_results(
            original_results, 
            calibrated_results,
            './result/calibration_results.json'
        )
        
        # 生成对比图
        plot_calibration_comparison(
            original_results,
            calibrated_results,
            './result/calibration_comparison.png'
        )
        
        # 打印结果汇总
        print_calibration_metrics(results_summary)
        
        print("\n🎉 置信度校准流程完成！")
        print("📁 结果文件:")
        print("   - 校准对比图: ./result/calibration_comparison.png")
        print("   - 校准结果数据: ./result/calibration_results.json")
        
    except Exception as e:
        print(f"❌ 保存结果失败: {e}")
        return


if __name__ == "__main__":
    main()
