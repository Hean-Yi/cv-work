#!/usr/bin/env python3
"""
可扩展的OverLoCK模型实现
支持不同规模的配置，充分利用RTX 4090显存
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from model_configs import ModelConfig, get_recommended_config


class ContMixOptimized(nn.Module):
    """
    ContMix - 优化版本
    
    优化点：
    1. 减少内存使用 - 避免存储完整的亲和力矩阵
    2. 提高计算效率 - 使用深度可分离卷积近似
    3. 保持核心思想 - 依然是动态上下文引导
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 num_groups: int = 4, region_size: int = 7, 
                 use_efficient_impl: bool = True):
        super(ContMixOptimized, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.region_size = region_size
        self.use_efficient_impl = use_efficient_impl
        self.group_ch = in_ch // num_groups
        
        # 轻量级的Q/K生成
        self.qk_dim = min(in_ch // 2, 128)  # 降维以提高效率
        self.query_conv = nn.Conv2d(in_ch, self.qk_dim, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_ch, self.qk_dim, kernel_size=1, bias=False)
        
        # 区域池化
        self.region_pool = nn.AdaptiveAvgPool2d(region_size)
        
        if use_efficient_impl:
            # 优化实现：使用注意力机制 + 深度可分离卷积
            self.attention_proj = nn.Linear(region_size * region_size, kernel_size * kernel_size)
            self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                   padding=kernel_size//2, groups=in_ch, bias=False)
            self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            # 标准实现：简化版的动态卷积
            self.dynamic_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                                        padding=kernel_size//2, bias=False)
            self.context_modulation = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, in_ch // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch // 4, in_ch, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """优化版本的前向传播"""
        B, C, H, W = x.shape
        
        if self.use_efficient_impl:
            return self._efficient_forward(x, context)
        else:
            return self._standard_forward(x, context)
    
    def _efficient_forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """高效实现版本"""
        B, C, H, W = x.shape
        
        # 生成轻量级的Q和K
        Q = self.query_conv(x)        # [B, qk_dim, H, W]
        K = self.key_conv(context)    # [B, qk_dim, H, W]
        K_pooled = self.region_pool(K)  # [B, qk_dim, S, S]
        
        # 计算全局上下文注意力
        Q_global = F.adaptive_avg_pool2d(Q, 1)  # [B, qk_dim, 1, 1]
        K_global = K_pooled.view(B, self.qk_dim, -1)  # [B, qk_dim, S²]
        
        # 注意力权重计算
        attention = torch.bmm(
            Q_global.view(B, 1, self.qk_dim), 
            K_global
        )  # [B, 1, S²]
        attention = F.softmax(attention, dim=-1)
        
        # 生成动态权重
        dynamic_weights = self.attention_proj(attention)  # [B, 1, K²]
        dynamic_weights = dynamic_weights.view(B, 1, self.kernel_size, self.kernel_size)
        
        # 应用深度可分离卷积 + 动态调制
        x_dw = self.dw_conv(x)  # 深度卷积
        
        # 动态调制 (简化版)
        scale = F.adaptive_avg_pool2d(dynamic_weights, 1)  # [B, 1, 1, 1]
        x_modulated = x_dw * scale.expand_as(x_dw)
        
        # 点卷积输出
        output = self.pw_conv(x_modulated)
        
        return output
    
    def _standard_forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """标准实现版本"""
        # 上下文调制
        context_weight = self.context_modulation(context)
        x_modulated = x * context_weight
        
        # 标准卷积
        output = self.dynamic_conv(x_modulated)
        
        return output


class ScalableDynamicBlock(nn.Module):
    """可扩展的Dynamic Block - Focus-Net的核心构建块"""
    def __init__(self, in_ch: int, context_ch: int):
        super(ScalableDynamicBlock, self).__init__()
        # 残差深度卷积
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, 
                                groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        
        # GDSA (Gated Dynamic Spatial Aggregator)
        self.contmix = ContMixOptimized(in_ch, in_ch, use_efficient_impl=True)
        
        # 门控机制
        self.gate_conv = nn.Conv2d(in_ch + context_ch, in_ch, kernel_size=1, bias=False)
        self.gate_act = nn.Sigmoid()
        
        # ConvFFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch)
        )
        
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity = x
        
        # 残差深度卷积
        x = self.relu(self.bn1(self.dw_conv(x)))
        
        # 上下文融合
        fused = torch.cat([x, context], dim=1)
        gate = self.gate_act(self.gate_conv(fused))
        
        # GDSA with ContMix
        x_contmix = self.contmix(x, context)
        x = x_contmix * gate + x * (1 - gate)
        
        # 残差连接
        x = x + identity
        
        # ConvFFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        
        return x, context


class ScalableBaseBlock(nn.Module):
    """可扩展的基础残差卷积块"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super(ScalableBaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # 避免inplace操作
        
        # 残差连接
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return self.relu(out)


class ScalableBaseNet(nn.Module):
    """可扩展的基础特征提取网络 - 统一为3阶段架构"""
    def __init__(self, config: ModelConfig):
        super(ScalableBaseNet, self).__init__()
        self.config = config
        
        # Stage 1: 初始特征提取 - 输入→H/4×W/4
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 2: H/4×W/4 → H/8×W/8
        # 使用config中的第一个通道配置
        stage2_ch = config.base_channels[0] if config.base_channels else 128
        stage2_blocks = config.base_blocks[0] if config.base_blocks else 2
        
        self.stage2 = nn.ModuleList()
        self.stage2.append(ScalableBaseBlock(64, stage2_ch, stride=2))  # 降采样
        for _ in range(1, stage2_blocks):
            self.stage2.append(ScalableBaseBlock(stage2_ch, stage2_ch))
        
        # Stage 3: H/8×W/8 → H/16×W/16 (中层特征)
        # 使用config中的第二个通道配置
        stage3_ch = config.base_channels[1] if len(config.base_channels) > 1 else 256
        stage3_blocks = config.base_blocks[1] if len(config.base_blocks) > 1 else 2
        
        self.stage3 = nn.ModuleList()
        self.stage3.append(ScalableBaseBlock(stage2_ch, stage3_ch, stride=2))  # 降采样
        for _ in range(1, stage3_blocks):
            self.stage3.append(ScalableBaseBlock(stage3_ch, stage3_ch))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回3阶段特征，与model.py保持一致"""
        # Stage 1: H/4×W/4
        x1 = self.stem(x)
        
        # Stage 2: H/8×W/8  
        x2 = x1
        for block in self.stage2:
            x2 = block(x2)
        
        # Stage 3: H/16×W/16 (中层特征)
        x3 = x2
        for block in self.stage3:
            x3 = block(x3)
        
        return x1, x2, x3


class ScalableOverviewNet(nn.Module):
    """可扩展的轻量级上下文先验生成网络 - 统一与model.py架构"""
    def __init__(self, in_ch: int, out_ch: int = None):
        super(ScalableOverviewNet, self).__init__()
        if out_ch is None:
            out_ch = in_ch
            
        # 快速下采样到 H/32 × W/32
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        
        # 轻量级处理块
        self.process_block = nn.Sequential(
            ScalableBaseBlock(out_ch, out_ch),
            ScalableBaseBlock(out_ch, out_ch)
        )
        
        # 上采样回中层特征分辨率用于引导
        self.context_proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, mid_level_feat: torch.Tensor) -> torch.Tensor:
        """生成粗糙但语义丰富的上下文先验"""
        # 生成粗糙但语义丰富的上下文先验
        context = self.downsample(mid_level_feat)  # H/32 × W/32
        context = self.process_block(context)
        
        # 上采样用于后续引导
        context = F.interpolate(context, size=mid_level_feat.shape[2:], 
                               mode='bilinear', align_corners=False)
        context = self.context_proj(context)
        
        return context


class ScalableFocusNet(nn.Module):
    """可扩展的高分辨率细节感知网络 - 统一使用DynamicBlock"""
    def __init__(self, config: ModelConfig):
        super(ScalableFocusNet, self).__init__()
        # 使用最后一层的通道数作为输入
        in_ch = config.base_channels[1] if len(config.base_channels) > 1 else 256
        
        # 构建多个动态块
        self.blocks = nn.ModuleList()
        num_blocks = config.focus_blocks if hasattr(config, 'focus_blocks') else 6
        
        for _ in range(num_blocks):
            self.blocks.append(ScalableDynamicBlock(in_ch, in_ch))

    def forward(self, base_feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Focus-Net前向传播，使用上下文流引导"""
        x = base_feat
        
        # 通过所有动态块传播上下文
        for block in self.blocks:
            x, context = block(x, context)
            
        return x


class ScalableFPN(nn.Module):
    """可扩展的FPN多尺度特征融合模块"""
    def __init__(self, config: ModelConfig):
        super(ScalableFPN, self).__init__()
        in_channels_list = config.base_channels
        out_ch = config.fpn_out_channels
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        # 从最高层开始
        last_feat = self.lateral_convs[-1](feats[-1])
        out_feats = []
        out = self.output_convs[-1](last_feat)
        out_feats.append(out)
        
        # 自顶向下融合
        for i in range(len(feats) - 2, -1, -1):
            lateral = self.lateral_convs[i](feats[i])
            # 上采样
            top_down = F.interpolate(last_feat, size=lateral.shape[-2:], 
                                   mode='bilinear', align_corners=False)
            last_feat = lateral + top_down
            out = self.output_convs[i](last_feat)
            out_feats.insert(0, out)
        
        return out_feats


class ScalableCBAM(nn.Module):
    """可扩展的CBAM注意力模块"""
    def __init__(self, config: ModelConfig):
        super(ScalableCBAM, self).__init__()
        in_ch = config.fpn_out_channels
        reduction = config.cbam_reduction
        
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        avg_pool = torch.mean(x, dim=[2, 3], keepdim=True)
        max_pool = torch.max(x, dim=2, keepdim=True)[0]
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]
        
        channel_att = self.channel_att(avg_pool) + self.channel_att(max_pool)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_att(spatial_input)
        x = x * spatial_att
        
        return x


class ScalableOverLoCKModel(nn.Module):
    """可扩展的OverLoCK主模型 - 统一论文架构"""
    def __init__(self, class_names: List[str], config: Optional[ModelConfig] = None):
        super(ScalableOverLoCKModel, self).__init__()
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # 使用配置或默认配置
        if config is None:
            config = get_recommended_config("balanced")
        self.config = config
        
        # OverLoCK三阶段架构
        self.base_net = ScalableBaseNet(config)  # BaseNet: 3阶段特征提取
        
        # 获取中层特征通道数
        mid_ch = config.base_channels[1] if len(config.base_channels) > 1 else 256
        
        self.overview_net = ScalableOverviewNet(mid_ch, mid_ch)  # OverviewNet: 上下文先验
        self.focus_net = ScalableFocusNet(config)  # FocusNet: 细节感知
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(mid_ch, self.num_classes)
        
        # 辅助分类器
        self.aux_classifier = nn.Linear(mid_ch, self.num_classes)
        
        # 可选的CLIP头
        if hasattr(config, 'use_clip') and config.use_clip:
            self.clip_head = nn.Parameter(
                torch.randn(self.num_classes, mid_ch), 
                requires_grad=False
            )
            nn.init.normal_(self.clip_head, std=0.02)
            self.clip_head.data = F.normalize(self.clip_head.data, dim=1)
        else:
            self.clip_head = None
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, use_aux: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """OverLoCK三阶段前向传播"""
        # BaseNet: 三阶段特征提取
        stage1_feat, stage2_feat, mid_level_feat = self.base_net(x)  # H/4, H/8, H/16
        
        # OverviewNet: 生成轻量级上下文先验
        context = self.overview_net(mid_level_feat)
        
        # FocusNet: 上下文引导的细节感知
        focus_out = self.focus_net(mid_level_feat, context)
        
        # 全局池化
        pooled = self.global_pool(focus_out).view(focus_out.size(0), -1)
        
        # 主分类预测
        main_logits = self.classifier(pooled)
        
        # 辅助分类器输出
        aux_logits = None
        if use_aux:
            aux_pooled = self.global_pool(context).view(context.size(0), -1)
            aux_logits = self.aux_classifier(aux_pooled)
        
        # CLIP输出
        clip_logits = None
        if self.clip_head is not None:
            visual_feats = F.normalize(pooled, dim=1)
            clip_logits = visual_feats @ self.clip_head.t()
        
        return main_logits, aux_logits, clip_logits

    def count_parameters(self):
        """统计模型参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def print_model_info(self):
        """打印模型信息"""
        total_params, trainable_params = self.count_parameters()
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        print(f"模型配置: {self.config.name}")
        print(f"总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: {model_size_mb:.1f} MB")
        print(f"BaseNet通道数: {self.config.base_channels}")
        print(f"BaseNet块数: {self.config.base_blocks}")
        
        # 获取中层特征通道数
        mid_ch = self.config.base_channels[1] if len(self.config.base_channels) > 1 else 256
        print(f"中层特征通道数: {mid_ch}")
        
        focus_blocks = getattr(self.config, 'focus_blocks', 6)
        print(f"FocusNet块数: {focus_blocks}")


# 便捷函数
def create_overlock_model(class_names: List[str], model_size: str = "balanced"):
    """创建OverLoCK模型的便捷函数
    
    Args:
        class_names: 类别名称列表
        model_size: "small", "balanced", "large", "xlarge", "xxlarge"
    
    Returns:
        ScalableOverLoCKModel实例
    """
    size_map = {
        "small": "fast",
        "balanced": "balanced", 
        "medium": "balanced",
        "large": "performance",
        "xlarge": "maximum",
        "xxlarge": "maximum"
    }
    
    config = get_recommended_config(size_map.get(model_size, "balanced"))
    model = ScalableOverLoCKModel(class_names, config)
    return model


if __name__ == "__main__":
    # 测试不同规模的模型
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("测试不同规模的OverLoCK模型:")
    print("=" * 60)
    
    for size in ["small", "balanced", "large"]:
        print(f"\n{size.upper()} 模型:")
        model = create_overlock_model(class_names, size)
        model.print_model_info()
        
        # 测试前向传播
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            main_logits, aux_logits, clip_logits = model(x, use_aux=True)
            print(f"主输出形状: {main_logits.shape}")
            if aux_logits is not None:
                print(f"辅助输出形状: {aux_logits.shape}")
            if clip_logits is not None:
                print(f"CLIP输出形状: {clip_logits.shape}")