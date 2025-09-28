#!/usr/bin/env python3
"""
置信度分析和可视化脚本 - 中文版本
"""
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

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

def plot_confidence_analysis():
    """分析和可视化置信度"""
    # 设置中文字体
    font_available = setup_chinese_font()
    
    # 加载推理结果
    with open('./result/inference_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    confidences = [r['confidence'] for r in results]
    correct_confidences = [r['confidence'] for r in results if r['is_correct']]
    incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    if font_available:
        # 使用中文标签
        # 1. 整体置信度分布
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'平均值: {np.mean(confidences):.3f}')
        ax1.set_title('整体置信度分布')
        ax1.set_xlabel('置信度')
        ax1.set_ylabel('数量')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 正确vs错误预测的置信度对比
        if correct_confidences and incorrect_confidences:
            ax2.hist(correct_confidences, bins=15, alpha=0.7, color='green', 
                    label=f'正确预测 ({len(correct_confidences)})', edgecolor='black')
            ax2.hist(incorrect_confidences, bins=15, alpha=0.7, color='red', 
                    label=f'错误预测 ({len(incorrect_confidences)})', edgecolor='black')
            ax2.set_title('置信度对比: 正确 vs 错误预测')
            ax2.set_xlabel('置信度')
            ax2.set_ylabel('数量')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 置信度vs准确率关系
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_results = [r for r in results 
                          if confidence_bins[i] <= r['confidence'] < confidence_bins[i+1]]
            if bin_results:
                accuracy = sum(r['is_correct'] for r in bin_results) / len(bin_results)
                bin_accuracies.append(accuracy * 100)
                bin_counts.append(len(bin_results))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        ax3.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='orange', edgecolor='black')
        ax3.plot([0, 1], [0, 100], 'r--', label='完美校准')
        ax3.set_title('置信度校准分析')
        ax3.set_xlabel('置信度区间')
        ax3.set_ylabel('准确率 (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. 统计摘要
        ax4.axis('off')
        
        # 计算错误预测的平均置信度
        incorrect_mean_conf = np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else None
        incorrect_mean_str = f"{incorrect_mean_conf:.3f}" if incorrect_mean_conf is not None else "无"
        
        stats_text = f"""置信度统计信息:
        
总样本数: {len(results)}
整体准确率: {data['accuracy']:.1f}%

平均置信度: {np.mean(confidences):.3f}
置信度标准差: {np.std(confidences):.3f}
最高置信度: {max(confidences):.3f}
最低置信度: {min(confidences):.3f}

正确预测:
  数量: {len(correct_confidences)}
  平均置信度: {np.mean(correct_confidences):.3f}

错误预测:
  数量: {len(incorrect_confidences)}
  平均置信度: {incorrect_mean_str}
        """
    else:
        # 使用英文标签作为备选
        # 1. 整体置信度分布
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.set_title('Overall Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 正确vs错误预测的置信度对比
        if correct_confidences and incorrect_confidences:
            ax2.hist(correct_confidences, bins=15, alpha=0.7, color='green', 
                    label=f'Correct ({len(correct_confidences)})', edgecolor='black')
            ax2.hist(incorrect_confidences, bins=15, alpha=0.7, color='red', 
                    label=f'Incorrect ({len(incorrect_confidences)})', edgecolor='black')
            ax2.set_title('Confidence: Correct vs Incorrect')
            ax2.set_xlabel('Confidence')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 置信度vs准确率关系
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_results = [r for r in results 
                          if confidence_bins[i] <= r['confidence'] < confidence_bins[i+1]]
            if bin_results:
                accuracy = sum(r['is_correct'] for r in bin_results) / len(bin_results)
                bin_accuracies.append(accuracy * 100)
                bin_counts.append(len(bin_results))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        ax3.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='orange', edgecolor='black')
        ax3.plot([0, 1], [0, 100], 'r--', label='Perfect Calibration')
        ax3.set_title('Confidence Calibration')
        ax3.set_xlabel('Confidence Bin')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. 统计摘要
        ax4.axis('off')
        
        # 计算错误预测的平均置信度
        incorrect_mean_conf = np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else None
        incorrect_mean_str = f"{incorrect_mean_conf:.3f}" if incorrect_mean_conf is not None else "N/A"
        
        stats_text = f"""Confidence Statistics:
        
Total Samples: {len(results)}
Overall Accuracy: {data['accuracy']:.1f}%

Mean Confidence: {np.mean(confidences):.3f}
Std Confidence: {np.std(confidences):.3f}
Max Confidence: {max(confidences):.3f}
Min Confidence: {min(confidences):.3f}

Correct Predictions:
  Count: {len(correct_confidences)}
  Mean Conf: {np.mean(correct_confidences):.3f}

Incorrect Predictions:
  Count: {len(incorrect_confidences)}
  Mean Conf: {incorrect_mean_str}
        """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('./result/confidence_analysis_chinese.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 中文置信度分析图表已保存: ./result/confidence_analysis_chinese.png")

if __name__ == "__main__":
    plot_confidence_analysis()
