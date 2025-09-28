#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet-100 专用配置文件
针对ImageNet-100数据集优化的OverLoCK模型配置
"""

from dataclasses import dataclass

@dataclass
class ImageNet100Config:
    """ImageNet-100 基础配置"""
    # 数据集配置
    dataset_name: str = "ImageNet-100"
    data_path: str = "data/imagenet100"
    num_classes: int = 100
    image_size: int = 224  # ImageNet标准尺寸
    
    # 模型配置 (基于原有的small配置调整)
    base_channels: int = 64
    overview_channels: int = 128  
    focus_channels: int = 256
    fpn_channels: int = 128
    cbam_channels: int = 64
    
    # 训练配置
    batch_size: int = 64  # ImageNet数据集通常用较大的batch size
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    
    # 数据增强
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True 
    cutmix_alpha: float = 1.0
    
    # 优化器配置
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-6

@dataclass  
class ImageNet100MediumConfig(ImageNet100Config):
    """ImageNet-100 中等配置"""
    # 增大模型容量
    base_channels: int = 96
    overview_channels: int = 192
    focus_channels: int = 384
    fpn_channels: int = 192
    cbam_channels: int = 96
    
    # 调整训练参数
    batch_size: int = 48  # 适当降低batch size以适应更大模型
    learning_rate: float = 8e-4
    epochs: int = 150

@dataclass
class ImageNet100LargeConfig(ImageNet100Config):
    """ImageNet-100 大型配置"""
    # 进一步增大模型
    base_channels: int = 128
    overview_channels: int = 256
    focus_channels: int = 512
    fpn_channels: int = 256
    cbam_channels: int = 128
    
    # 训练配置
    batch_size: int = 32
    learning_rate: float = 6e-4
    epochs: int = 200
    warmup_epochs: int = 10

@dataclass
class ImageNet100RTX4090Config(ImageNet100Config):
    """ImageNet-100 RTX4090优化配置"""
    # 针对RTX4090优化的配置
    base_channels: int = 160
    overview_channels: int = 320
    focus_channels: int = 640
    fpn_channels: int = 320
    cbam_channels: int = 160
    
    # RTX4090优化参数
    batch_size: int = 32  # 最大化显存利用
    learning_rate: float = 5e-4
    epochs: int = 300
    warmup_epochs: int = 15
    
    # 混合精度训练
    use_amp: bool = True
    
    # 更强的正则化
    weight_decay: float = 5e-4
    dropout_rate: float = 0.1
    
    # 学习率调度
    scheduler: str = "cosine_with_restarts"
    restart_epochs: int = 50

# 配置映射字典
IMAGENET100_CONFIGS = {
    "small": ImageNet100Config,
    "medium": ImageNet100MediumConfig, 
    "large": ImageNet100LargeConfig,
    "rtx4090": ImageNet100RTX4090Config
}

def get_imagenet100_config(config_name: str = "small"):
    """
    获取ImageNet-100配置
    
    Args:
        config_name: 配置名称 ["small", "medium", "large", "rtx4090"]
    
    Returns:
        配置对象
    """
    if config_name not in IMAGENET100_CONFIGS:
        available_configs = list(IMAGENET100_CONFIGS.keys())
        raise ValueError(f"未知配置: {config_name}. 可用配置: {available_configs}")
    
    return IMAGENET100_CONFIGS[config_name]()

def print_config_comparison():
    """打印所有配置的对比"""
    print("ImageNet-100 配置对比")
    print("=" * 80)
    
    configs = ["small", "medium", "large", "rtx4090"]
    
    for config_name in configs:
        config = get_imagenet100_config(config_name)
        
        # 计算大致的参数量 (简化估算)
        total_channels = (config.base_channels + config.overview_channels + 
                         config.focus_channels + config.fpn_channels)
        estimated_params = total_channels * total_channels * 9 / 1000000  # 简化估算，单位M
        
        print(f"\n{config_name.upper()} 配置:")
        print(f"  参数量 (估算): ~{estimated_params:.1f}M")
        print(f"  Base通道: {config.base_channels}")
        print(f"  Overview通道: {config.overview_channels}")
        print(f"  Focus通道: {config.focus_channels}")
        print(f"  FPN通道: {config.fpn_channels}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  学习率: {config.learning_rate}")
        print(f"  训练轮数: {config.epochs}")

if __name__ == "__main__":
    print_config_comparison()