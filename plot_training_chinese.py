#!/usr/bin/env python3
"""
训练过程可视化脚本 - 中文版本
"""
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置中文字体 - 优先使用系统中文字体"""
    # 尝试本地字体文件路径
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        './fonts/NotoSansCJK-Regular.ttc',
        './fonts/NotoSansSC-Regular.ttf',
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
        'C:/Windows/Fonts/simhei.ttf'  # Windows 黑体
    ]
    
    # 首先尝试本地字体文件
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                fm.fontManager.addfont(font_path)
                font_name = font_prop.get_name()
                
                # 强制设置字体
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                
                print(f"✅ 使用本地中文字体: {font_name} ({font_path})")
                return True
            except Exception as e:
                print(f"⚠️  字体文件 {font_path} 加载失败: {e}")
                continue
    
    # 系统中文字体列表（按优先级排序）
    chinese_fonts = [
        'Noto Sans CJK SC',
        'WenQuanYi Micro Hei',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB',
        'Source Han Sans CN',
        'Noto Sans SC'
    ]
    
    # 尝试系统字体名称
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            
            # 测试字体是否可用
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', fontsize=12, fontfamily=font_name)
            plt.close(fig)
            
            print(f"✅ 使用系统中文字体: {font_name}")
            return True
        except Exception as e:
            print(f"⚠️  字体 {font_name} 不可用: {e}")
            continue
    
    # 最后的备选方案
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("⚠️  使用默认字体，中文可能显示为方框")
    return False

def plot_training_history():
    """绘制训练历史图表"""
    # 设置中文字体
    font_available = setup_chinese_font()
    
    # 加载训练历史
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')
    history = checkpoint.get('history', {})
    
    if not history:
        print("未找到训练历史数据")
        return
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    if font_available:
        # 使用中文标签
        # 1. 训练和验证损失
        ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
        ax1.set_title('损失变化')
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 训练和验证准确率
        ax2.plot(epochs, history['train_acc'], 'g-', label='训练准确率')
        ax2.plot(epochs, history['val_acc'], 'orange', label='验证准确率')
        ax2.set_title('准确率变化')
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 过拟合分析
        gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        ax3.plot(epochs, gap, 'purple', label='过拟合程度')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('过拟合分析')
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('训练准确率 - 验证准确率 (%)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 统计信息
        ax4.axis('off')
        stats = f"""训练统计信息:
        
最佳训练轮次: {checkpoint.get('epoch', 0) + 1}
最佳验证准确率: {checkpoint.get('val_acc', 0):.2f}%
最终训练准确率: {history['train_acc'][-1]:.2f}%
最终验证准确率: {history['val_acc'][-1]:.2f}%
最终训练损失: {history['train_loss'][-1]:.4f}
最终验证损失: {history['val_loss'][-1]:.4f}
最终过拟合程度: {gap[-1]:.2f}%
        """
    else:
        # 使用英文标签作为备选
        # 1. 训练和验证损失
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 训练和验证准确率
        ax2.plot(epochs, history['train_acc'], 'g-', label='Training Accuracy')
        ax2.plot(epochs, history['val_acc'], 'orange', label='Validation Accuracy')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 过拟合分析
        gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
        ax3.plot(epochs, gap, 'purple', label='Overfitting Gap')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Overfitting Analysis')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train Acc - Val Acc (%)')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 统计信息
        ax4.axis('off')
        stats = f"""Training Statistics:
        
Best Epoch: {checkpoint.get('epoch', 0) + 1}
Best Val Accuracy: {checkpoint.get('val_acc', 0):.2f}%
Final Train Accuracy: {history['train_acc'][-1]:.2f}%
Final Val Accuracy: {history['val_acc'][-1]:.2f}%
Final Train Loss: {history['train_loss'][-1]:.4f}
Final Val Loss: {history['val_loss'][-1]:.4f}
Final Overfitting Gap: {gap[-1]:.2f}%
        """
    
    ax4.text(0.1, 0.9, stats, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('./result/training_history_chinese.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 中文训练历史图表已保存: ./result/training_history_chinese.png")

if __name__ == "__main__":
    plot_training_history()
