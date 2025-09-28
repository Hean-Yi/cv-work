#!/usr/bin/env python3
"""
OverLoCK模型的扩展配置
提供不同规模的模型配置，充分利用RTX 4090的24GB显存
"""

class ModelConfig:
    """模型配置基类"""
    def __init__(self):
        # BaseNet配置
        self.base_channels = [64, 128, 256]  # 各阶段通道数
        self.base_blocks = [2, 2, 2]         # 各阶段块数
        
        # FPN配置
        self.fpn_out_channels = 128
        
        # FocusNet配置
        self.focus_blocks = 2
        
        # CBAM配置
        self.cbam_reduction = 16
        
        # 其他配置
        self.use_clip = True
        
    def get_estimated_params(self):
        """估算参数量（简化计算）"""
        # BaseNet参数估算
        base_params = 0
        in_ch = 3
        for i, (out_ch, blocks) in enumerate(zip(self.base_channels, self.base_blocks)):
            if i == 0:
                # stem层
                base_params += 7*7*3*64 + 64  # conv + bn
            # 每个阶段的参数
            stage_params = blocks * (3*3*in_ch*out_ch + out_ch + 3*3*out_ch*out_ch + out_ch)
            if in_ch != out_ch:  # 残差连接
                stage_params += in_ch * out_ch
            base_params += stage_params
            in_ch = out_ch
        
        # FPN参数估算
        fpn_params = sum(ch * self.fpn_out_channels + 3*3*self.fpn_out_channels*self.fpn_out_channels 
                        for ch in self.base_channels)
        
        # FocusNet参数估算
        focus_params = self.focus_blocks * self.base_channels[-1] * 200  # 简化估算
        
        # 其他组件参数估算
        other_params = 100000  # 简化估算
        
        total_params = base_params + fpn_params + focus_params + other_params
        return int(total_params)


class SmallConfig(ModelConfig):
    """小模型配置 - 原始配置"""
    def __init__(self):
        super().__init__()
        self.name = "Small (原始)"
        self.base_channels = [64, 128, 256]
        self.base_blocks = [2, 2, 2]
        self.fpn_out_channels = 128
        self.focus_blocks = 2


class MediumConfig(ModelConfig):
    """中等模型配置 - 适度扩展"""
    def __init__(self):
        super().__init__()
        self.name = "Medium (适度扩展)"
        self.base_channels = [96, 192, 384]  # 增加50%通道数
        self.base_blocks = [2, 2, 3]         # 增加最后阶段的块数
        self.fpn_out_channels = 192          # 增加FPN通道数
        self.focus_blocks = 3                # 增加FocusNet块数


class LargeConfig(ModelConfig):
    """大模型配置 - 显著扩展"""
    def __init__(self):
        super().__init__()
        self.name = "Large (显著扩展)"
        self.base_channels = [128, 256, 512]  # 翻倍通道数
        self.base_blocks = [3, 3, 3]          # 增加所有阶段的块数
        self.fpn_out_channels = 256           # 翻倍FPN通道数
        self.focus_blocks = 4                 # 翻倍FocusNet块数


class XLargeConfig(ModelConfig):
    """超大模型配置 - 激进扩展"""
    def __init__(self):
        super().__init__()
        self.name = "X-Large (激进扩展)"
        self.base_channels = [160, 320, 640]  # 2.5倍通道数
        self.base_blocks = [3, 4, 4]          # 大幅增加块数
        self.fpn_out_channels = 320           # 大幅增加FPN通道数
        self.focus_blocks = 6                 # 大幅增加FocusNet块数
        self.cbam_reduction = 32              # 增加CBAM容量


class XXLargeConfig(ModelConfig):
    """巨大模型配置 - 最大化利用4090显存"""
    def __init__(self):
        super().__init__()
        self.name = "XX-Large (最大化4090)"
        self.base_channels = [192, 384, 768]  # 3倍通道数
        self.base_blocks = [4, 4, 4]          # 大幅增加深度
        self.fpn_out_channels = 384           # 3倍FPN通道数
        self.focus_blocks = 8                 # 4倍FocusNet块数
        self.cbam_reduction = 48              # 进一步增加CBAM容量


def analyze_all_configs():
    """分析所有配置的参数量和显存需求"""
    configs = [
        SmallConfig(),
        MediumConfig(), 
        LargeConfig(),
        XLargeConfig(),
        XXLargeConfig()
    ]
    
    print("=" * 80)
    print("OverLoCK模型配置分析 - RTX 4090优化")
    print("=" * 80)
    
    for config in configs:
        estimated_params = config.get_estimated_params()
        
        # 简化的显存估算（batch_size=32）
        param_memory = estimated_params * 4 / (1024**3)  # GB
        gradient_memory = estimated_params * 4 / (1024**3)  # GB
        optimizer_memory = estimated_params * 8 / (1024**3)  # GB (AdamW)
        activation_memory = 0.3  # GB (固定估算)
        total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        
        # 判断是否适合4090
        if total_memory < 16:
            status = "[推荐]"
        elif total_memory < 20:
            status = "[可用]"
        elif total_memory < 22:
            status = "[紧张]"
        else:
            status = "[超限]"
        
        print(f"\n{config.name}:")
        print(f"  • BaseNet通道数: {config.base_channels}")
        print(f"  • BaseNet块数: {config.base_blocks}")
        print(f"  • FPN通道数: {config.fpn_out_channels}")
        print(f"  • FocusNet块数: {config.focus_blocks}")
        print(f"  • 估算参数量: {estimated_params/1e6:.1f}M ({estimated_params:,})")
        print(f"  • 估算显存: {total_memory:.1f} GB (batch_size=32)")
        print(f"  • 4090适配: {status}")
    
    print("\n" + "=" * 80)
    print("推荐配置选择指南:")
    print("=" * 80)
    print("• Small: 快速实验，资源充足时的基准")
    print("• Medium: 平衡性能和资源，推荐日常使用")
    print("• Large: 追求更好性能，显存使用适中")
    print("• X-Large: 高性能需求，显存使用较多")
    print("• XX-Large: 极致性能，最大化利用4090显存")
    
    print("\n训练优化建议:")
    print("• 使用混合精度训练 (torch.cuda.amp) 可节省约40%显存")
    print("• 启用梯度检查点可进一步节省显存")
    print("• 考虑使用更高效的优化器如Lion代替AdamW")
    print("• 对于XX-Large配置，建议batch_size=16-24")


def get_recommended_config(target="balanced"):
    """获取推荐配置
    
    Args:
        target: "fast" | "balanced" | "performance" | "maximum"
    """
    if target == "fast":
        return SmallConfig()
    elif target == "balanced":
        return MediumConfig()
    elif target == "performance":
        return LargeConfig()
    elif target == "maximum":
        return XLargeConfig()
    else:
        return MediumConfig()


if __name__ == "__main__":
    analyze_all_configs()