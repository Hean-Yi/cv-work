#!/usr/bin/env python3
"""
针对RTX 4090的超大模型配置
最大化利用24GB显存
"""

from model_configs import ModelConfig

class RTX4090MaxConfig(ModelConfig):
    """RTX 4090最大化配置 - 充分利用24GB显存"""
    def __init__(self):
        super().__init__()
        self.name = "RTX4090-Max (最大化24GB显存)"
        self.base_channels = [256, 512, 1024]  # 大幅增加通道数
        self.base_blocks = [4, 5, 5]           # 增加深度
        self.fpn_out_channels = 512           # 大幅增加FPN通道数
        self.focus_blocks = 8                 # 增加FocusNet深度
        self.cbam_reduction = 64              # 增加CBAM容量
        self.use_clip = True

class RTX4090OptimalConfig(ModelConfig):
    """RTX 4090最优配置 - 平衡性能和显存使用"""
    def __init__(self):
        super().__init__()
        self.name = "RTX4090-Optimal (性能显存平衡)"
        self.base_channels = [192, 384, 768]  # 适中增加通道数
        self.base_blocks = [3, 4, 4]          # 适当增加深度
        self.fpn_out_channels = 384           # 适中增加FPN通道数
        self.focus_blocks = 6                 # 适中增加FocusNet深度
        self.cbam_reduction = 48              # 适中增加CBAM容量
        self.use_clip = True

def get_rtx4090_training_config(model_size="optimal", multi_gpu=False):
    """获取RTX 4090专用训练配置"""
    if model_size == "max":
        base_config = {
            'batch_size': 24 if not multi_gpu else 24,  # 保持batch_size为24
            'learning_rate': 4e-3,  # 根据论文设置
            'weight_decay': 0.05,   # 根据论文设置
            'gradient_clip_max_norm': 0.3,
            'use_mixed_precision': True,
            'compile_model': True,
            'gradient_checkpointing': True,  # 大模型可能需要
            'warmup_epochs': 8,
            'memory_usage_gb': 22.0 if not multi_gpu else 44.0,  # 双GPU显存估计
        }
    else:  # optimal
        base_config = {
            'batch_size': 32 if not multi_gpu else 32,  # 保持batch_size为32
            'learning_rate': 4e-3,  # 根据论文设置
            'weight_decay': 0.05,   # 根据论文设置
            'gradient_clip_max_norm': 0.5,
            'use_mixed_precision': True,
            'compile_model': True,
            'gradient_checkpointing': False,
            'warmup_epochs': 5,
            'memory_usage_gb': 16.0 if not multi_gpu else 32.0,  # 双GPU显存估计
        }
    
    # 添加多GPU相关配置
    base_config['multi_gpu'] = multi_gpu
    base_config['world_size'] = 2 if multi_gpu else 1
    
    return base_config

if __name__ == "__main__":
    import torch
    from scalable_model import ScalableOverLoCKModel
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("RTX 4090专用配置测试")
    print("=" * 60)
    
    configs = [
        ("Optimal", RTX4090OptimalConfig()),
        ("Max", RTX4090MaxConfig())
    ]
    
    for name, config in configs:
        print(f"\n{name} 配置:")
        model = ScalableOverLoCKModel(class_names, config)
        total_params = sum(p.numel() for p in model.parameters())
        
        train_config = get_rtx4090_training_config("optimal" if name == "Optimal" else "max")
        
        print(f"  • 参数量: {total_params/1e6:.1f}M")
        print(f"  • BaseNet: {config.base_channels}")
        print(f"  • FPN: {config.fpn_out_channels} 通道")
        print(f"  • FocusNet: {config.focus_blocks} 块")
        print(f"  • 推荐batch_size: {train_config['batch_size']}")
        print(f"  • 预计显存: {train_config['memory_usage_gb']:.1f} GB")
        
        # 测试前向传播
        test_input = torch.randn(2, 3, 224, 224)
        try:
            with torch.no_grad():
                logits, clip_logits = model(test_input)
            print(f"  • 前向传播: ✅ 成功")
        except Exception as e:
            print(f"  • 前向传播: ❌ 失败 ({e})")