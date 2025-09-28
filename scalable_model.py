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
    """可扩展的基础特征提取网络"""
    def __init__(self, config: ModelConfig):
        super(ScalableBaseNet, self).__init__()
        self.config = config
        
        # Stem层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 构建各阶段
        self.stages = nn.ModuleList()
        in_ch = 64
        
        for i, (out_ch, num_blocks) in enumerate(zip(config.base_channels, config.base_blocks)):
            stage = nn.ModuleList()
            
            # 第一个块可能需要降采样
            stride = 2 if i > 0 else 1
            stage.append(ScalableBaseBlock(in_ch, out_ch, stride))
            
            # 后续块
            for _ in range(1, num_blocks):
                stage.append(ScalableBaseBlock(out_ch, out_ch))
            
            self.stages.append(stage)
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        features = []
        
        for stage in self.stages:
            for block in stage:
                x = block(x)
            features.append(x)
        
        return features


class ScalableOverviewNet(nn.Module):
    """可扩展的轻量级全局上下文注意力网络"""
    def __init__(self, in_ch: int):
        super(ScalableOverviewNet, self).__init__()
        # 使用更高效的深度可分离卷积
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.conv(y)
        y = self.upsample(y)
        return torch.sigmoid(y)


class ScalableFocusBlock(nn.Module):
    """可扩展的FocusNet块"""
    def __init__(self, in_ch: int, out_ch: int):
        super(ScalableFocusBlock, self).__init__()
        # 使用深度可分离卷积提高效率
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)
        
        # 上下文融合
        self.context_fc = nn.Linear(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        
        # 上下文注意力
        ctx = context.view(B, C, -1).mean(dim=2)  # Global average pooling
        attn = self.sigmoid(self.context_fc(ctx)).view(B, C, 1, 1)
        
        # 应用注意力
        x_attended = x * attn
        
        # 深度可分离卷积
        out = self.relu(self.bn1(self.dwconv(x_attended)))
        out = self.relu(self.bn2(self.pwconv(out)))
        
        return out


class ScalableFocusNet(nn.Module):
    """可扩展的高分辨率细节感知网络"""
    def __init__(self, config: ModelConfig):
        super(ScalableFocusNet, self).__init__()
        in_ch = config.base_channels[-1]  # 使用最后一层的通道数
        
        self.blocks = nn.ModuleList()
        for _ in range(config.focus_blocks):
            self.blocks.append(ScalableFocusBlock(in_ch, in_ch))

    def forward(self, base_feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = base_feat * context
        
        for block in self.blocks:
            x = block(x, context)
            
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
    """可扩展的OverLoCK主模型"""
    def __init__(self, class_names: List[str], config: Optional[ModelConfig] = None):
        super(ScalableOverLoCKModel, self).__init__()
        
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # 使用配置或默认配置
        if config is None:
            config = get_recommended_config("balanced")
        self.config = config
        
        # 初始化子网络
        self.base_net = ScalableBaseNet(config)
        self.overview_net = ScalableOverviewNet(config.base_channels[-1])
        self.focus_net = ScalableFocusNet(config)
        self.fpn = ScalableFPN(config)
        self.cbam = ScalableCBAM(config)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(config.fpn_out_channels, self.num_classes)
        
        # 可选的CLIP头
        if config.use_clip:
            self.clip_head = nn.Parameter(
                torch.randn(self.num_classes, config.fpn_out_channels), 
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
        # BaseNet特征提取
        feats = self.base_net(x)
        
        # Overview网络生成上下文注意力
        context = self.overview_net(feats[-1])
        
        # Focus网络细节感知
        focus_out = self.focus_net(feats[-1], context)
        
        # FPN多尺度融合
        fused_feats = self.fpn(feats[:-1] + [focus_out])
        fused_feature = fused_feats[0]  # 使用最高分辨率特征
        
        # CBAM注意力增强
        attended = self.cbam(fused_feature)
        
        # 全局池化
        pooled = self.global_pool(attended).view(attended.size(0), -1)
        
        # 分类预测
        logits = self.classifier(pooled)
        
        # 辅助分类器输出 (可选)
        aux_logits = None
        if use_aux:
            # 使用context特征进行辅助分类
            aux_pooled = self.global_pool(context).view(context.size(0), -1)
            if not hasattr(self, 'aux_classifier'):
                # 动态创建辅助分类器
                self.aux_classifier = nn.Linear(aux_pooled.size(1), self.num_classes)
                if aux_pooled.is_cuda:
                    self.aux_classifier = self.aux_classifier.cuda()
            aux_logits = self.aux_classifier(aux_pooled)
        
        # CLIP辅助分类
        clip_logits = None
        if self.clip_head is not None:
            visual_feats = F.normalize(pooled, dim=1)
            clip_logits = visual_feats @ self.clip_head.t()
        
        return logits, aux_logits, clip_logits

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
        print(f"FPN通道数: {self.config.fpn_out_channels}")
        print(f"FocusNet块数: {self.config.focus_blocks}")


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
            main_logits, aux_logits, clip_logits = model(x, use_aux=False)
            print(f"输出形状: {main_logits.shape}")
            if clip_logits is not None:
                print(f"CLIP输出形状: {clip_logits.shape}")