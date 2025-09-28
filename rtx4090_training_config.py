#!/usr/bin/env python3
"""
针对RTX 4090优化的训练配置
充分利用24GB显存，提供不同模型规模的最优训练参数
"""

class RTX4090TrainingConfig:
    """RTX 4090优化训练配置"""
    
    # 基础配置
    DEVICE = 'cuda'
    NUM_WORKERS = 8  # 4090配合高性能CPU时的最佳线程数
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True  # 减少worker重启开销
    
    # 小模型配置 (3-6M参数)
    SMALL_MODEL_CONFIG = {
        'batch_size': 128,  # 大batch size提高训练效率
        'learning_rate': 2e-4,  # 大batch size对应更高的学习率
        'weight_decay': 1e-4,
        'gradient_clip_max_norm': 1.0,
        'use_mixed_precision': True,  # 节省显存，加速训练
        'compile_model': True,  # PyTorch 2.0编译加速
        'memory_usage_gb': 2.0,  # 预期显存使用
    }
    
    # 中等模型配置 (8-15M参数)
    MEDIUM_MODEL_CONFIG = {
        'batch_size': 64,
        'learning_rate': 1.5e-4,
        'weight_decay': 1e-4,
        'gradient_clip_max_norm': 1.0,
        'use_mixed_precision': True,
        'compile_model': True,
        'memory_usage_gb': 4.5,
    }
    
    # 大模型配置 (20-30M参数)
    LARGE_MODEL_CONFIG = {
        'batch_size': 48,
        'learning_rate': 1e-4,
        'weight_decay': 2e-4,  # 大模型需要更强的正则化
        'gradient_clip_max_norm': 1.0,
        'use_mixed_precision': True,
        'compile_model': True,
        'gradient_checkpointing': False,  # 4090显存充足，优先速度
        'memory_usage_gb': 8.0,
    }
    
    # 超大模型配置 (40M+参数)
    XLARGE_MODEL_CONFIG = {
        'batch_size': 32,
        'learning_rate': 8e-5,
        'weight_decay': 3e-4,
        'gradient_clip_max_norm': 0.5,  # 大模型梯度更容易爆炸
        'use_mixed_precision': True,
        'compile_model': True,
        'gradient_checkpointing': True,  # 必要时启用以节省显存
        'memory_usage_gb': 16.0,
    }
    
    # 学习率调度配置
    SCHEDULER_CONFIG = {
        'type': 'cosine_with_warmup',
        'warmup_epochs': 5,
        'cosine_restarts': True,
        'restart_period': 50,
        'min_lr_factor': 0.01,
    }
    
    # 数据增强配置 (针对CIFAR-10等小图像)
    DATA_AUGMENTATION_CONFIG = {
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_degrees': 15,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'random_erasing': {
            'probability': 0.25,
            'scale': (0.02, 0.33),
            'ratio': (0.3, 3.3),
        },
        'cutmix': {
            'probability': 0.5,
            'alpha': 1.0,
        },
        'mixup': {
            'probability': 0.5,
            'alpha': 0.8,
        }
    }
    
    # 优化器配置
    OPTIMIZER_CONFIG = {
        'type': 'adamw',  # 可选: 'adamw', 'lion', 'sgd'
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'amsgrad': False,
    }
    
    # 损失函数配置
    LOSS_CONFIG = {
        'primary_loss': 'cross_entropy',
        'label_smoothing': 0.1,  # 提高泛化能力
        'clip_loss_weight': 0.1,  # CLIP辅助损失权重
    }
    
    # 早停和检查点配置
    CHECKPOINT_CONFIG = {
        'save_every_epochs': 10,
        'keep_best_n': 3,
        'early_stopping_patience': 15,
        'early_stopping_min_delta': 0.001,
    }

def get_training_config_for_model_size(model_params_millions):
    """根据模型参数量获取最优训练配置"""
    config = RTX4090TrainingConfig()
    
    if model_params_millions < 8:
        return config.SMALL_MODEL_CONFIG
    elif model_params_millions < 20:
        return config.MEDIUM_MODEL_CONFIG
    elif model_params_millions < 40:
        return config.LARGE_MODEL_CONFIG
    else:
        return config.XLARGE_MODEL_CONFIG

def estimate_training_time(model_params_millions, num_epochs=100, dataset_size=50000):
    """估算训练时间"""
    config = get_training_config_for_model_size(model_params_millions)
    batch_size = config['batch_size']
    
    # 基于经验的时间估算 (RTX 4090)
    batches_per_epoch = dataset_size // batch_size
    
    # 每batch处理时间估算 (毫秒)
    if model_params_millions < 8:
        ms_per_batch = 50  # 小模型很快
    elif model_params_millions < 20:
        ms_per_batch = 80
    elif model_params_millions < 40:
        ms_per_batch = 120
    else:
        ms_per_batch = 200  # 大模型较慢但仍然很快
    
    total_time_hours = (batches_per_epoch * num_epochs * ms_per_batch) / (1000 * 3600)
    
    return {
        'estimated_hours': total_time_hours,
        'batches_per_epoch': batches_per_epoch,
        'ms_per_batch': ms_per_batch,
        'recommended_batch_size': batch_size
    }

def print_training_recommendations():
    """打印RTX 4090训练建议"""
    print("=" * 80)
    print("RTX 4090 OverLoCK模型训练优化建议")
    print("=" * 80)
    
    model_sizes = [
        ("Small (3-6M参数)", 4),
        ("Medium (8-15M参数)", 12),
        ("Large (20-30M参数)", 25),
        ("X-Large (40M+参数)", 45)
    ]
    
    for name, params_m in model_sizes:
        config = get_training_config_for_model_size(params_m)
        time_est = estimate_training_time(params_m)
        
        print(f"\n{name}:")
        print(f"  • 推荐batch_size: {config['batch_size']}")
        print(f"  • 推荐学习率: {config['learning_rate']}")
        print(f"  • 预期显存使用: {config['memory_usage_gb']:.1f} GB")
        print(f"  • 预期训练时间: {time_est['estimated_hours']:.1f} 小时 (100 epochs)")
        print(f"  • 每批处理时间: {time_est['ms_per_batch']} ms")
        
    print("\n" + "=" * 80)
    print("通用优化建议:")
    print("=" * 80)
    print("🚀 性能优化:")
    print("   • 启用混合精度训练 (torch.cuda.amp)")
    print("   • 使用torch.compile()编译模型 (PyTorch 2.0+)")
    print("   • 设置num_workers=8，pin_memory=True")
    print("   • 使用persistent_workers=True减少重启开销")
    
    print("\n💾 显存优化:")
    print("   • 对于超大模型启用gradient_checkpointing")
    print("   • 使用更高效的优化器如Lion (可选)")
    print("   • 合理设置batch_size充分利用显存")
    
    print("\n📊 训练策略:")
    print("   • 使用余弦学习率调度+预热")
    print("   • 添加标签平滑(0.1)提高泛化")
    print("   • 使用数据增强: Cutmix + Mixup + RandomErasing")
    print("   • 设置梯度裁剪防止梯度爆炸")
    
    print("\n⏰ 训练时间估算 (CIFAR-10, 100 epochs):")
    print("   • Small模型: 1-2小时")
    print("   • Medium模型: 3-4小时") 
    print("   • Large模型: 6-8小时")
    print("   • X-Large模型: 12-15小时")
    
    print("\n🎯 最佳实践:")
    print("   • 使用tensorboard监控训练过程")
    print("   • 定期保存检查点，保留最佳模型")
    print("   • 使用早停防止过拟合")
    print("   • 在较小数据集上先验证配置")

if __name__ == "__main__":
    print_training_recommendations()