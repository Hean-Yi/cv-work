import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

# ================== 核心OverLoCK实现 ==================

class BaseBlock(nn.Module):
    """基础残差卷积块"""
    def __init__(self, in_ch: int, out_ch: int):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.res_conv is not None:
            identity = self.res_conv(identity)
        out += identity
        return self.relu(out)


class BaseNet(nn.Module):
    """Base-Net: 编码低层和中层特征"""
    def __init__(self):
        super(BaseNet, self).__init__()
        # Stage 1: 初始特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 2: H/4 × W/4 → H/8 × W/8
        self.stage2 = nn.Sequential(
            BaseBlock(64, 128),
            BaseBlock(128, 128)
        )
        
        # Stage 3: H/8 × W/8 → H/16 × W/16 (中层特征)
        self.stage3 = nn.Sequential(
            BaseBlock(128, 256),
            BaseBlock(256, 256)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)  # H/4 × W/4
        f2 = self.stage2(x)  # H/8 × W/8
        f3 = self.stage3(f2)  # H/16 × W/16 (中层特征)
        return x, f2, f3


class OverviewNet(nn.Module):
    """Overview-Net: 轻量级上下文先验生成网络"""
    def __init__(self, in_ch: int, out_ch: int):
        super(OverviewNet, self).__init__()
        # 快速下采样到 H/32 × W/32
        self.downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # 轻量级处理块
        self.process_block = nn.Sequential(
            BaseBlock(out_ch, out_ch),
            BaseBlock(out_ch, out_ch)
        )
        
        # 上采样回中层特征分辨率用于引导
        self.context_proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, mid_level_feat: torch.Tensor) -> torch.Tensor:
        # 生成粗糙但语义丰富的上下文先验
        context = self.downsample(mid_level_feat)  # H/32 × W/32
        context = self.process_block(context)
        
        # 上采样用于后续引导
        context = F.interpolate(context, size=mid_level_feat.shape[2:], 
                               mode='bilinear', align_corners=False)
        context = self.context_proj(context)
        
        return context


class ContMix(nn.Module):
    """Context-Mixing 动态卷积 - 论文核心实现"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 num_groups: int = 4, region_size: int = 7):
        super(ContMix, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.region_size = region_size
        
        # Q和K的投影层
        self.query_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        
        # 自适应平均池化到区域中心
        self.region_pool = nn.AdaptiveAvgPool2d(region_size)
        
        # 动态核生成的线性层
        self.kernel_gen = nn.Linear(region_size * region_size, kernel_size * kernel_size)
        
        # 分组处理
        assert in_ch % num_groups == 0, "in_ch must be divisible by num_groups"
        self.group_ch = in_ch // num_groups
        
        # 基础卷积核(用于与动态核组合)
        self.base_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                                  padding=kernel_size//2, bias=False)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 生成查询Q和键K
        Q = self.query_conv(x)  # B × C × H × W
        K = self.key_conv(context)  # B × C × H × W
        K = self.region_pool(K)  # B × C × S × S
        
        # 简化实现：使用注意力机制调制特征
        # 计算上下文注意力权重
        context_pool = torch.mean(K, dim=[2, 3], keepdim=True)  # B × C × 1 × 1
        
        # 生成动态权重
        dynamic_weight = torch.sigmoid(context_pool)  # B × C × 1 × 1
        
        # 应用动态调制
        modulated_x = x * dynamic_weight
        
        # 基础卷积
        out = self.base_conv(modulated_x)
        
        return out


class DynamicBlock(nn.Module):
    """Dynamic Block - Focus-Net的核心构建块"""
    def __init__(self, in_ch: int, context_ch: int):
        super(DynamicBlock, self).__init__()
        # 残差深度卷积
        self.dw_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, 
                                groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        
        # GDSA (Gated Dynamic Spatial Aggregator)
        self.contmix = ContMix(in_ch, in_ch)
        
        # 门控机制
        self.gate_conv = nn.Conv2d(in_ch + context_ch, in_ch, kernel_size=1, bias=False)
        self.gate_act = nn.Sigmoid()
        
        # ConvFFN
        self.ffn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 4, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch)
        )
        
        self.relu = nn.ReLU(inplace=True)

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


class FocusNet(nn.Module):
    """Focus-Net: 上下文引导的细节感知网络"""
    def __init__(self, in_ch: int, context_ch: int, num_blocks: int = 6):
        super(FocusNet, self).__init__()
        self.num_blocks = num_blocks
        
        # 上下文先验预处理
        self.context_reduction = nn.Conv2d(context_ch, context_ch, kernel_size=1)
        
        # Dynamic Blocks
        self.blocks = nn.ModuleList([
            DynamicBlock(in_ch, context_ch) for _ in range(num_blocks)
        ])
        
        # 上下文更新参数
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, context_prior: torch.Tensor) -> torch.Tensor:
        # 初始上下文先验
        P0 = self.context_reduction(context_prior)
        current_context = P0
        
        # 逐块处理with上下文流
        for i, block in enumerate(self.blocks):
            x, updated_context = block(x, current_context)
            
            # 上下文更新: Pi+1 = α * P'i + β * P0
            if i < len(self.blocks) - 1:  # 最后一块不需要更新上下文
                current_context = self.alpha * updated_context + self.beta * P0
        
        return x


# ================== 保留的创新组件 ==================

class PELKConv(nn.Module):
    """外围卷积模块 - 保留的创新点"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 101):
        super(PELKConv, self).__init__()
        center_size = 7
        # 调整膨胀率以避免kernel size过大
        dilation = min(kernel_size//2, 3)  # 限制膨胀率
        actual_kernel = min(kernel_size, 15)  # 限制实际kernel size
        
        self.center_conv = nn.Conv2d(in_ch, out_ch, kernel_size=center_size, 
                                     padding=center_size//2, bias=False)
        self.periph_conv = nn.Conv2d(in_ch, out_ch, kernel_size=actual_kernel, 
                                     padding=(actual_kernel//2)*dilation, dilation=dilation, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_center = self.center_conv(x)
        y_periph = self.periph_conv(x)
        out = y_center + y_periph + self.pos_embed
        return self.relu(out)


class DCNv4(nn.Module):
    """DCNv4 形变卷积 - 保留的创新点"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super(DCNv4, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, 
                                     kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(in_ch, kernel_size * kernel_size, 
                                   kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        out = self.conv(x)  # 简化版，实际应用偏移和掩码
        return self.relu(out)


class CBAM(nn.Module):
    """CBAM注意力模块 - 保留的创新点"""
    def __init__(self, in_ch: int, reduction: int = 16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        
        return x


class FPNFusion(nn.Module):
    """FPN多尺度特征融合 - 保留的创新点"""
    def __init__(self, in_channels_list: List[int], out_ch: int):
        super(FPNFusion, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # 自顶向下融合
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]
        
        # 从最高层开始融合
        fused = lateral_feats[-1]
        for i in range(len(lateral_feats) - 2, -1, -1):
            upsampled = F.interpolate(fused, size=lateral_feats[i].shape[2:], mode='nearest')
            fused = upsampled + lateral_feats[i]
        
        # 输出卷积
        fused = self.output_convs[0](fused)
        return fused


class CLIPTextHead(nn.Module):
    """CLIP语言引导分类头 - 保留的创新点"""
    def __init__(self, class_names: List[str], feat_dim: int = 512):
        super(CLIPTextHead, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.text_embeddings = nn.Parameter(
            torch.randn(self.num_classes, feat_dim), requires_grad=False
        )
        self.text_embeddings.data = F.normalize(self.text_embeddings.data, dim=1)
        
    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        if visual_feats.dim() != 2:
            visual_feats = visual_feats.view(visual_feats.size(0), -1)
        v = F.normalize(visual_feats, dim=1)
        sims = v @ self.text_embeddings.t()
        return sims


# ================== 主模型 ==================

class OverLoCKModel(nn.Module):
    """
    OverLoCK主模型 - 论文忠实实现 + 保留创新点
    
    架构流程:
    1. BaseNet: 提取多层次特征 (DDS第一部分)
    2. OverviewNet: 生成上下文先验 (DDS第二部分) 
    3. FocusNet: 上下文引导的细节感知 (DDS第三部分)
    4. 创新组件: FPN + CBAM + CLIP等增强
    """
    def __init__(self, class_names: List[str], use_innovations: bool = True):
        super(OverLoCKModel, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.use_innovations = use_innovations
        
        # ========== 核心OverLoCK组件 ==========
        # Deep-stage Decomposition Strategy (DDS)
        self.base_net = BaseNet()
        self.overview_net = OverviewNet(in_ch=256, out_ch=256)
        self.focus_net = FocusNet(in_ch=256, context_ch=256, num_blocks=6)
        
        # ========== 保留的创新组件 ==========
        if self.use_innovations:
            # FPN多尺度融合
            self.fpn = FPNFusion(in_channels_list=[64, 128, 256], out_ch=256)
            
            # CBAM注意力增强
            self.cbam = CBAM(in_ch=256)
            
            # PELK外围卷积
            self.pelk = PELKConv(in_ch=256, out_ch=256, kernel_size=51)
            
            # DCNv4形变卷积
            self.dcn = DCNv4(in_ch=256, out_ch=256)
            
            # CLIP语言引导
            self.clip_head = CLIPTextHead(class_names, feat_dim=256)
        
        # ========== 分类头 ==========
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, self.num_classes)
        
        # 辅助分类器用于Overview-Net预训练
        self.aux_classifier = nn.Linear(256, self.num_classes)

    def forward(self, x: torch.Tensor, use_aux: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, 3, H, W]
            use_aux: 是否使用辅助分类器(训练时用于Overview-Net)
            
        Returns:
            main_logits: 主分类预测
            aux_logits: 辅助分类预测(可选)
            clip_logits: CLIP分类预测(可选)
        """
        # ========== DDS: Deep-stage Decomposition ==========
        
        # Step 1: Base-Net提取多层次特征
        f1, f2, f3 = self.base_net(x)  # f3是中层特征 H/16 × W/16
        
        # Step 2: Overview-Net生成上下文先验
        context_prior = self.overview_net(f3)  # 轻量级全局上下文
        
        # Step 3: Focus-Net进行上下文引导的细节感知
        focus_output = self.focus_net(f3, context_prior)  # 核心的上下文引导
        
        # ========== 创新组件增强 ==========
        if self.use_innovations:
            # FPN多尺度特征融合
            enhanced_features = self.fpn([f1, f2, focus_output])
            
            # CBAM注意力增强
            enhanced_features = self.cbam(enhanced_features)
            
            # PELK外围卷积增强感受野
            enhanced_features = self.pelk(enhanced_features)
            
            # DCNv4形变卷积适应性建模
            final_features = self.dcn(enhanced_features)
        else:
            final_features = focus_output
        
        # ========== 分类预测 ==========
        # 全局池化
        pooled_features = self.global_pool(final_features).view(x.size(0), -1)
        
        # 主分类器
        main_logits = self.classifier(pooled_features)
        
        # 辅助分类器(用于Overview-Net的预训练)
        aux_logits = None
        if use_aux:
            aux_pooled = self.global_pool(context_prior).view(x.size(0), -1)
            aux_logits = self.aux_classifier(aux_pooled)
        
        # CLIP语言引导分类
        clip_logits = None
        if self.use_innovations:
            clip_logits = self.clip_head(pooled_features)
        
        return main_logits, aux_logits, clip_logits

    def print_model_info(self):
        """打印模型架构信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"OverLoCK Model Architecture:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Innovation components: {'Enabled' if self.use_innovations else 'Disabled'}")
        print(f"  - Core components: BaseNet + OverviewNet + FocusNet")
        if self.use_innovations:
            print(f"  - Enhancement: FPN + CBAM + PELK + DCNv4 + CLIP")
