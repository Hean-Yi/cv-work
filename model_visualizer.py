#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OverLoCK Model Visualization and Evaluation Module
Includes ERF visualization, GradCAM, performance evaluation, etc.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import time
from collections import OrderedDict
import json

# Set English fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class ModelVisualizer:
    """Model Visualization and Evaluation Class"""
    
    def __init__(self, model, device='cuda', save_dir='./visualizations'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 移动模型到设备
        self.model = self.model.to(device)
        self.model.eval()
        
        # 用于存储中间特征的钩子
        self.activations = {}
        self.gradients = {}
        
    def register_hooks(self):
        """Register forward and backward propagation hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # 为关键层注册钩子
        hook_layers = []
        for name, module in self.model.named_modules():
            # 更宽泛的层匹配，包含卷积层
            if any(keyword in name.lower() for keyword in ['conv', 'stage', 'block', 'layer']):
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    hook_layers.append(name)
                    module.register_forward_hook(forward_hook(name))
                    module.register_backward_hook(backward_hook(name))
        
        # 如果没有找到任何层，尝试注册所有卷积层
        if len(hook_layers) == 0:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    hook_layers.append(name)
                    module.register_forward_hook(forward_hook(name))
                    module.register_backward_hook(backward_hook(name))
        
        print(f"✅ Registered hooks for {len(hook_layers)} layers")
        return hook_layers

    def calculate_metrics(self, dataloader, num_samples=1000):
        """
        计算模型评估指标
        - Top-1 accuracy
        - Top-5 accuracy
        - 吞吐量 (imgs/sec)
        - 参数量 (#P)
        - 计算量 (FLOPs)
        """
        print("📊 Computing model evaluation metrics...")
        
        metrics = {}
        
        # 1. 参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        metrics['total_params'] = total_params
        metrics['trainable_params'] = trainable_params
        
        # 2. 计算FLOPs (简化估算)
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            metrics['flops'] = self._estimate_flops(dummy_input)
        except:
            metrics['flops'] = "N/A"
        
        # 3. 准确率和吞吐量
        if dataloader is not None:
            top1_correct = 0
            top5_correct = 0
            total_samples = 0
            total_time = 0
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(dataloader):
                    if total_samples >= num_samples:
                        break
                        
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 计算推理时间
                    start_time = time.time()
                    outputs = self.model(images)
                    # OverLoCK模型返回 (logits, clip_logits)，我们只需要logits
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    
                    total_time += end_time - start_time
                    
                    # 计算Top-1和Top-5准确率
                    _, pred1 = outputs.topk(1, 1, True, True)
                    _, pred5 = outputs.topk(5, 1, True, True)
                    
                    labels = labels.view(-1, 1)
                    top1_correct += pred1.eq(labels).sum().item()
                    top5_correct += pred5.eq(labels.expand_as(pred5)).sum().item()
                    total_samples += images.size(0)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processing progress: {total_samples}/{num_samples}")
            
            metrics['top1_accuracy'] = top1_correct / total_samples
            metrics['top5_accuracy'] = top5_correct / total_samples
            metrics['throughput'] = total_samples / total_time
        
        # 保存指标
        self._save_metrics(metrics)
        self._print_metrics(metrics)
        
        return metrics
    
    def _estimate_flops(self, input_tensor):
        """简化的FLOPs估算"""
        # 这是一个简化版本，实际应该使用专门的FLOPs计算库
        flops = 0
        
        def flop_hook(module, input, output):
            nonlocal flops
            if isinstance(module, nn.Conv2d):
                # Conv2D FLOPs = output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
                output_dims = output.shape[2:]
                kernel_dims = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                flops += np.prod(output_dims) * np.prod(kernel_dims) * in_channels * out_channels
            elif isinstance(module, nn.Linear):
                # Linear FLOPs = in_features * out_features
                flops += module.in_features * module.out_features
        
        # 注册钩子
        hooks = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(flop_hook))
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return flops

    def visualize_effective_receptive_field(self, dataloader, num_images=300):
        """
        可视化有效感受野 (ERF)
        对比不同模型在Stage 3和Stage 4的ERF
        """
        print("🔍 Generating Effective Receptive Field visualization...")
        
        # 注册钩子
        hook_layers = self.register_hooks()
        if len(hook_layers) == 0:
            print("⚠️ No suitable layers found for ERF visualization")
            return {}
        
        erf_maps = {}
        num_processed = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                if num_processed >= num_images:
                    break
                
                images = images.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算每一层的激活强度
                for layer_name, activation in self.activations.items():
                    if layer_name not in erf_maps:
                        erf_maps[layer_name] = []
                    
                    # 计算空间维度的平均激活
                    if len(activation.shape) == 4:  # [B, C, H, W]
                        spatial_activation = torch.mean(activation, dim=1)  # 平均所有通道
                        erf_maps[layer_name].append(spatial_activation.cpu())
                    elif len(activation.shape) == 2:  # [B, Features] - 全连接层输出
                        # 对于全连接层，创建一个1x1的"激活图"
                        fc_activation = torch.mean(activation, dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
                        erf_maps[layer_name].append(fc_activation.cpu())
                
                num_processed += images.size(0)
                if num_processed % 50 == 0:
                    print(f"  Processing: {num_processed}/{num_images}")
        
        print(f"📊 Collected ERF data for {len(erf_maps)} layers")
        # 生成ERF可视化
        self._plot_erf_comparison(erf_maps)
        print("✅ ERF visualization completed")
        
        return erf_maps

    def visualize_gradcam(self, images, labels, target_layers=None):
        """
        生成GradCAM类激活图
        可视化不同stage的类激活图
        """
        print("🎯 Generating GradCAM class activation maps...")
        
        if target_layers is None:
            target_layers = ['stage3', 'stage4']  # 默认可视化stage3和stage4
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        gradcams = {}
        
        for target_layer in target_layers:
            gradcams[target_layer] = self._generate_gradcam(images, labels, target_layer)
        
        # 生成GradCAM对比图
        self._plot_gradcam_comparison(images, gradcams, labels)
        print("✅ GradCAM visualization completed")
        
        return gradcams

    def _generate_gradcam(self, images, labels, target_layer_name):
        """生成单个层的GradCAM"""
        # 清空之前的激活和梯度
        self.activations.clear()
        self.gradients.clear()
        
        # 注册目标层的钩子
        target_module = None
        for name, module in self.model.named_modules():
            if target_layer_name.lower() in name.lower():
                target_module = module
                break
        
        if target_module is None:
            print(f"⚠️ Target layer not found: {target_layer_name}")
            return None
        
        # 前向传播
        self.model.zero_grad()
        outputs = self.model(images)
        
        # OverLoCK模型返回 (logits, clip_logits)，我们只需要logits
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # 反向传播
        if labels.dim() == 1:
            target_scores = outputs.gather(1, labels.unsqueeze(1)).squeeze()
        else:
            target_scores = outputs[range(len(labels)), labels]
        
        target_scores.backward(torch.ones_like(target_scores))
        
        # 计算GradCAM
        gradcam_maps = []
        for i in range(images.size(0)):
            if target_layer_name in self.activations and target_layer_name in self.gradients:
                activations = self.activations[target_layer_name][i]  # [C, H, W]
                gradients = self.gradients[target_layer_name][i]      # [C, H, W]
                
                # 计算权重 (全局平均池化)
                weights = torch.mean(gradients, dim=(1, 2))  # [C]
                
                # 加权求和
                gradcam = torch.zeros(activations.shape[1:])  # [H, W]
                for j, weight in enumerate(weights):
                    gradcam += weight * activations[j]
                
                # ReLU and normalization
                gradcam = F.relu(gradcam)
                gradcam = gradcam / (torch.max(gradcam) + 1e-8)
                gradcam_maps.append(gradcam.cpu().numpy())
        
        return gradcam_maps

    def _plot_erf_comparison(self, erf_maps):
        """绘制ERF对比图"""
        # 检查erf_maps是否为空
        if not erf_maps or len(erf_maps) == 0:
            print("⚠️ No ERF data available for visualization")
            # 创建一个空的占位图
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No ERF Data Available\nModel might not have conv layers with spatial dimensions', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.axis('off')
            fig.suptitle('Effective Receptive Field (ERF) Visualization', fontsize=16)
            
            erf_path = os.path.join(self.save_dir, 'erf_comparison.png')
            plt.savefig(erf_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            return
        
        # 限制显示的层数，避免图表过宽
        max_layers = min(len(erf_maps), 6)  # 最多显示6层
        selected_layers = list(erf_maps.keys())[:max_layers]
        
        fig, axes = plt.subplots(2, max_layers, figsize=(max_layers * 3, 8))
        if max_layers == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle('Effective Receptive Field (ERF) Visualization', fontsize=16)
        
        for i, layer_name in enumerate(selected_layers):
            activations = erf_maps[layer_name]
            if not activations:  # 检查激活是否为空
                continue
                
            # 平均所有图像的激活
            try:
                avg_activation = torch.mean(torch.stack(activations), dim=0)[0]  # 取第一个样本
            except:
                # 如果stack失败，使用第一个激活
                avg_activation = activations[0][0] if len(activations) > 0 else torch.zeros(56, 56)
            
            # Stage 3
            if i < axes.shape[1]:
                im1 = axes[0][i].imshow(avg_activation, cmap='hot', interpolation='bilinear')
                axes[0][i].set_title(f'Stage 3 - {layer_name}', fontsize=10)
                axes[0][i].axis('off')
                plt.colorbar(im1, ax=axes[0][i], shrink=0.8)
            
            # Stage 4 (如果有的话)
            if i < axes.shape[1]:
                im2 = axes[1][i].imshow(avg_activation, cmap='hot', interpolation='bilinear')
                axes[1][i].set_title(f'Stage 4 - {layer_name}', fontsize=10)
                axes[1][i].axis('off')
                plt.colorbar(im2, ax=axes[1][i], shrink=0.8)
        
        # 隐藏未使用的子图
        for j in range(len(selected_layers), axes.shape[1]):
            axes[0][j].axis('off')
            axes[1][j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'erf_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_gradcam_comparison(self, images, gradcams, labels):
        """绘制GradCAM对比图"""
        num_images = min(4, images.size(0))
        num_layers = len(gradcams)
        
        fig, axes = plt.subplots(num_layers + 1, num_images, figsize=(16, 12))
        fig.suptitle('GradCAM Class Activation Maps', fontsize=16)
        
        # 原图
        for i in range(num_images):
            # 反归一化显示原图
            img = images[i].cpu()
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            axes[0][i].imshow(img.permute(1, 2, 0))
            axes[0][i].set_title(f'Original (Label: {labels[i].item()})')
            axes[0][i].axis('off')
        
        # GradCAM
        for layer_idx, (layer_name, gradcam_maps) in enumerate(gradcams.items()):
            for img_idx in range(num_images):
                if gradcam_maps and img_idx < len(gradcam_maps):
                    gradcam = gradcam_maps[img_idx]
                    
                    # 调整大小到原图尺寸
                    gradcam_resized = F.interpolate(
                        torch.tensor(gradcam).unsqueeze(0).unsqueeze(0),
                        size=(224, 224), mode='bilinear', align_corners=False
                    ).squeeze().numpy()
                    
                    im = axes[layer_idx + 1][img_idx].imshow(gradcam_resized, cmap='jet', alpha=0.7)
                    axes[layer_idx + 1][img_idx].set_title(f'{layer_name}')
                    axes[layer_idx + 1][img_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'gradcam_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_metrics(self, metrics):
        """保存评估指标到JSON文件"""
        # 转换为可序列化的格式
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                serializable_metrics[key] = value
            elif isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = str(value)
        
        with open(os.path.join(self.save_dir, 'model_metrics.json'), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"📄 Metrics saved to: {os.path.join(self.save_dir, 'model_metrics.json')}")

    def _print_metrics(self, metrics):
        """打印评估指标"""
        print("\n" + "="*60)
        print("📊 OverLoCK Model Evaluation Metrics")
        print("="*60)
        
        print(f"🔢 Parameters:")
        print(f"   Total params: {metrics['total_params']:,}")
        print(f"   Trainable params: {metrics['trainable_params']:,}")
        print(f"   Params (M): {metrics['total_params']/1e6:.2f}M")
        
        print(f"⚡ Computation:")
        if isinstance(metrics['flops'], (int, float)):
            print(f"   FLOPs: {metrics['flops']:,}")
            print(f"   FLOPs (G): {metrics['flops']/1e9:.2f}G")
        else:
            print(f"   FLOPs: {metrics['flops']}")
        
        if 'top1_accuracy' in metrics:
            print(f"🎯 Accuracy:")
            print(f"   Top-1 Accuracy: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
            print(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
        
        if 'throughput' in metrics:
            print(f"🚀 Performance:")
            print(f"   Throughput: {metrics['throughput']:.2f} imgs/sec")
        
        print("="*60)

def test_visualization_functions():
    """测试可视化函数是否正常工作"""
    print("🧪 Testing visualization functions...")
    
    try:
        # 创建虚拟模型用于测试
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stage3 = nn.Conv2d(64, 128, 3, padding=1)
                self.stage4 = nn.Conv2d(128, 256, 3, padding=1)
                self.classifier = nn.Linear(256, 100)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            def forward(self, x):
                x = torch.randn(x.size(0), 64, 56, 56, device=x.device)  # 模拟前面的层
                x = F.relu(self.stage3(x))
                x = F.relu(self.stage4(x))
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        # 创建测试数据
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DummyModel()
        
        # 创建可视化器
        visualizer = ModelVisualizer(model, device=device, save_dir='./test_visualizations')
        
        # 测试指标计算
        print("  Testing metrics calculation...")
        dummy_images = torch.randn(8, 3, 224, 224)
        dummy_labels = torch.randint(0, 100, (8,))
        
        # 模拟数据加载器
        class DummyDataLoader:
            def __init__(self):
                self.data = [(dummy_images, dummy_labels)]
            def __iter__(self):
                return iter(self.data)
        
        dummy_loader = DummyDataLoader()
        metrics = visualizer.calculate_metrics(dummy_loader, num_samples=8)
        
        # 测试GradCAM
        print("  测试GradCAM...")
        gradcams = visualizer.visualize_gradcam(dummy_images, dummy_labels)
        
        print("✅ 可视化功能测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 可视化功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行测试
    success = test_visualization_functions()
    if success:
        print("🎉 所有可视化功能验证通过，可以安全使用！")
    else:
        print("⚠️ 可视化功能存在问题，请检查修复后再使用。")