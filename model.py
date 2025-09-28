import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

# ================== 模块1: MODEL 模块 ==================



"""
1. 输入图片(224*224像素)
      ↓
2. BaseNet提取三层特征(轮廓→纹理→部件)
      ↓
3. OverviewNet看低级特征, 生成"注意力地图"
      ↓
4. FocusNet用注意力地图处理高级特征, 深入分析细节
      ↓
5. FPN融合所有层次的特征
      ↓
6. CBAM增强重要特征, 抑制无关信息
      ↓
7. 全局池化(压缩成一个特征向量)
      ↓
8. 分类器判断类别 + CLIP辅助验证
      ↓
9. 输出：这是猫(95%概率)
"""


# -------- 基础模块 --------
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

"""
这个基础特征提取网络就像人的眼睛一样, 先是通过一个大卷积和池化层快速捕捉整体信息, 然后通过多个残差块逐步提取更细致的特征。
它把图片分为三个层次来理解, 
f2 = self.stage2(x)  # 低级特征 -- 边缘和纹理
f3 = self.stage3(f2)  # 中级特征 -- 形状和部分物体
f4 = self.stage4(f3)  # 高级特征 -- 复杂物体和语义信息
"""
class BaseNet(nn.Module):
    """基础特征提取网络"""
    def __init__(self):
        super(BaseNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            BaseBlock(64, 64),
            BaseBlock(64, 64)
        )
        self.stage3 = nn.Sequential(
            BaseBlock(64, 128),
            BaseBlock(128, 128)
        )
        self.stage4 = nn.Sequential(
            BaseBlock(128, 256),
            BaseBlock(256, 256)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f2 = self.stage2(x)  # 低级特征
        f3 = self.stage3(f2)  # 中级特征
        f4 = self.stage4(f3)  # 高级特征
        return [f2, f3, f4]

"""
快速扫描整张图片, 找出"哪里重要"
把图片缩小看整体, 生成一个注意力图, 让后续网络重点关注这些区域
再放大回去, 让注意力图和原图对齐
"""

# -------- 概览网络模块 --------
class OverviewNet(nn.Module):
    """轻量级全局上下文注意力网络"""
    def __init__(self, in_ch: int):
        super(OverviewNet, self).__init__()
        self.down = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.down(x))
        y = self.relu(self.conv(y))
        y = self.upsample(y)
        return torch.sigmoid(y)


"""
结合局部细节和全局上下文
动态调整卷积核, 适应不同区域的特征
"""
# -------- 动态卷积模块 --------
class ContMix(nn.Module):
    """Context-Mixing 动态卷积模块"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super(ContMix, self).__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, 
                                padding=kernel_size//2, groups=in_ch)
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fc = nn.Linear(in_ch, in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = context.size()
        ctx = context.view(B, C, -1).mean(dim=2)
        attn = self.sigmoid(self.fc(ctx)).view(B, C, 1, 1)
        x_mod = x * attn
        out = self.dwconv(x_mod)
        out = self.pwconv(out)
        return out

"""
中间有一个小卷积核, 用于捕捉局部细节
外围是一个大卷积核, 用于捕捉全局信息
这个大卷积核通过膨胀卷积实现, 可以有非常大的感受野
可以理解为鱼眼镜头, 中间看得清楚, 周围看得模糊

为什么需要这个?
因为有些细节需要结合全局信息来理解, 例如一个小物体可能只有在特定背景下才有意义
有一个圆形的球 -- > 可能是足球, 乒乓球, 或者只是一个白色的球
通过PELKConv, 网络可以同时看到局部和全局信息, 更好地理解细节
有一个圆形的球, 还有草地 --> 可能是足球
"""
# -------- 外围卷积模块 --------
class PELKConv(nn.Module):
    """外围卷积模块, 支持极大感受野"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 101):
        super(PELKConv, self).__init__()
        # 中心卷积核大小的作用是捕捉局部细节
        center_size = 7 # 中心卷积核大小
        # 设计两个卷积核: 一个小的捕捉局部细节, 一个大的捕捉全局信息
        self.center_conv = nn.Conv2d(in_ch, out_ch, kernel_size=center_size, 
                                     padding=center_size//2, bias=False)
        self.periph_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                                     padding=kernel_size//2, dilation=kernel_size//2, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, out_ch, 1, 1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_center = self.center_conv(x)
        y_periph = self.periph_conv(x)
        out = y_center + y_periph + self.pos_embed
        return self.relu(out)

"""
这个模块与普通的卷积不同的一点是这个卷积核不是固定的, 而是动态生成的
它根据输入特征图的内容, 生成适合当前区域的卷积核
普通的卷积核是固定的, 对所有位置都一样
动态卷积核可以根据不同区域的特征, 生成不同的卷积核
"""
# -------- 形变卷积模块 --------
class DCNv4(nn.Module):
    """DCNv4 形变卷积模块(简化实现)"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, modulation: bool = False):
        super(DCNv4, self).__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, 
                                     kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(in_ch, kernel_size * kernel_size, 
                                   kernel_size=3, padding=1) if modulation else None
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        if self.mask_conv is not None:
            mask = torch.sigmoid(self.mask_conv(x))
        out = self.conv(x)
        return self.relu(out)



"""
这个就是接受注意力图和高级特征, 进行细节感知的模块
它先用动态卷积根据注意力图调整特征, 然后用形变卷积捕捉更复杂的细节
再重要的区域仔细观察, 不重要的区域快速略过
"""
# -------- FocusNet 模块 --------
class FocusBlock(nn.Module):
    """包含动态卷积和形变卷积的高分辨率特征块"""
    def __init__(self, in_ch: int, out_ch: int, use_contmix: bool = True, use_dcn: bool = True):
        super(FocusBlock, self).__init__()
        self.use_contmix = use_contmix
        if use_contmix:
            self.contmix = ContMix(in_ch, out_ch)
        else:
            self.conv = BaseBlock(in_ch, out_ch)
        
        self.use_dcn = use_dcn
        self.dcn = DCNv4(out_ch, out_ch) if use_dcn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.use_contmix:
            out = self.contmix(x, context)
        else:
            out = self.conv(x)
        if self.use_dcn:
            out = self.dcn(out)
        return self.relu(out)
    
class FocusNet(nn.Module):
    """高分辨率细节感知网络"""
    def __init__(self, in_ch: int):
        super(FocusNet, self).__init__()
        self.block1 = FocusBlock(in_ch, in_ch, use_contmix=True, use_dcn=True)
        self.block2 = FocusBlock(in_ch, in_ch, use_contmix=True, use_dcn=True)

    def forward(self, base_feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = base_feat * context
        x = self.block1(x, context)
        x = self.block2(x, context)
        return x


"""
注意力模块分为两个部分, 一个部分是通道注意力, 另一个部分是空间注意力
通道注意力是决定关注哪些特征(颜色, 纹理, 形状等)
空间注意力是决定关注图像的哪些位置(左上, 右下, 中间等)

通道注意力是决定"看什么", 空间注意力是决定"看哪里"
"""
# -------- 注意力模块 --------
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_ch: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(scale))
        return x * scale


class CBAM(nn.Module):
    """CBAM: 通道和空间注意力组合模块"""
    def __init__(self, in_ch: int, reduction: int = 16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_ch, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

"""
把先前的多层特征融合起来, 让模型同时利用低级和高级信息
低级特征提供细节, 高级特征提供语义
"""
# -------- 特征融合模块 --------
class FPNFusion(nn.Module):
    """FPN 多尺度特征融合模块"""
    def __init__(self, in_channels_list: List[int], out_ch: int):
        super(FPNFusion, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=1) for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        last_feat = self.lateral_convs[-1](feats[-1])
        out_feats = []
        out = self.output_convs[-1](last_feat)
        out_feats.append(out)
        
        for i in range(len(feats) - 2, -1, -1):
            upsampled = F.interpolate(last_feat, size=feats[i].shape[2:], mode='nearest')
            last_feat = upsampled + self.lateral_convs[i](feats[i])
            out = self.output_convs[i](last_feat)
            out_feats.insert(0, out)
        return out_feats

"""
用文字帮助模型理解类别
把"猫", "狗"等类别名称转换成向量, 然后和图像特征对比
如果图像特征和"猫"的向量很接近, 那么就很可能是猫
"""
# -------- CLIP 语言引导模块 --------
class CLIPTextHead(nn.Module):
    """CLIP 语言引导分类头(简化版本)"""
    def __init__(self, class_names: List[str], feat_dim: int = 512):
        super(CLIPTextHead, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        # 简化实现：使用随机初始化的文本嵌入
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


# -------- 主模型 --------
class OverLoCKModel(nn.Module):
    """OverLoCK 主模型：集成所有组件"""
    def __init__(self, class_names: List[str], use_clip: bool = True):
        super(OverLoCKModel, self).__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.use_clip = use_clip
        
        # 初始化子网络
        self.base_net = BaseNet()
        self.overview_net = OverviewNet(in_ch=256)
        self.focus_net = FocusNet(in_ch=256)
        self.fpn = FPNFusion(in_channels_list=[64, 128, 256], out_ch=128)
        self.cbam = CBAM(in_ch=128)
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, self.num_classes)
        
        # 可选的CLIP头
        if self.use_clip:
            self.clip_head = CLIPTextHead(class_names, feat_dim=128)
        else:
            self.clip_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # BaseNet 特征提取
        feats = self.base_net(x)
        f2, f3, f4 = feats
        
        # Overview 网络生成上下文注意力
        context = self.overview_net(f4)
        
        # Focus 网络细节感知
        focus_out = self.focus_net(f4, context)
        
        # FPN 多尺度融合
        fused_feats = self.fpn([f2, f3, focus_out])
        fused_feature = fused_feats[0]
        
        # CBAM 注意力增强
        attended = self.cbam(fused_feature)
        
        # 全局池化
        pooled = self.global_pool(attended).view(attended.size(0), -1)
        
        # 分类预测
        logits = self.classifier(pooled)
        
        # CLIP 辅助分类
        clip_logits = self.clip_head(pooled) if self.clip_head else None
        
        return logits, clip_logits
