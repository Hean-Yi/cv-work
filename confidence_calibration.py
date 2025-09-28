#!/usr/bin/env python3
"""
置信度校准模块
实现温度缩放、Platt缩放等校准方法
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import json
from typing import Tuple, List, Dict, Optional
from torch.utils.data import DataLoader


class TemperatureScaling(nn.Module):
    """
    温度缩放校准方法
    通过学习一个温度参数来校准置信度
    """
    def __init__(self, temperature: float = 1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放
        Args:
            logits: 模型原始输出 [batch_size, num_classes]
        Returns:
            校准后的概率分布 [batch_size, num_classes]
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, 
            lr: float = 0.01, max_iter: int = 50) -> float:
        """
        训练温度参数
        Args:
            logits: 验证集的模型输出
            labels: 验证集的真实标签
            lr: 学习率
            max_iter: 最大迭代次数
        Returns:
            最优温度值
        """
        # 确保所有张量在同一设备上
        device = logits.device
        self.temperature = self.temperature.to(device)
        labels = labels.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        return self.temperature.item()


class PlattScaling:
    """
    Platt缩放校准方法
    使用逻辑回归来校准置信度
    """
    def __init__(self):
        self.calibrator = LogisticRegression()
    
    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        """
        训练Platt缩放参数
        Args:
            confidences: 原始置信度 [n_samples]
            labels: 二元标签 (0或1) [n_samples]
        """
        confidences = confidences.reshape(-1, 1)
        self.calibrator.fit(confidences, labels)
    
    def predict_proba(self, confidences: np.ndarray) -> np.ndarray:
        """
        预测校准后的概率
        Args:
            confidences: 原始置信度 [n_samples]
        Returns:
            校准后的概率 [n_samples]
        """
        confidences = confidences.reshape(-1, 1)
        return self.calibrator.predict_proba(confidences)[:, 1]


class IsotonicCalibration:
    """
    等渗回归校准方法
    使用等渗回归来校准置信度
    """
    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        """
        训练等渗回归参数
        Args:
            confidences: 原始置信度 [n_samples]
            labels: 二元标签 (0或1) [n_samples]
        """
        self.calibrator.fit(confidences, labels)
    
    def predict(self, confidences: np.ndarray) -> np.ndarray:
        """
        预测校准后的概率
        Args:
            confidences: 原始置信度 [n_samples]
        Returns:
            校准后的概率 [n_samples]
        """
        return self.calibrator.predict(confidences)


class ConfidenceCalibrator:
    """
    置信度校准器主类
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.temperature_scaler = None
        self.platt_scalers = {}  # 每个类别一个Platt缩放器
        self.isotonic_scalers = {}  # 每个类别一个等渗回归器
        self.class_names = getattr(model, 'class_names', [])
        
    def calibrate_temperature_scaling(self, val_loader: DataLoader) -> float:
        """
        使用温度缩放进行校准
        Args:
            val_loader: 验证数据加载器
        Returns:
            最优温度值
        """
        print("🌡️  开始温度缩放校准...")
        
        # 收集验证集的logits和标签
        all_logits = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                main_logits, aux_logits, clip_logits = self.model(images, use_aux=False)
                all_logits.append(main_logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 训练温度缩放，保持在GPU上
        self.temperature_scaler = TemperatureScaling().to(self.device)
        optimal_temp = self.temperature_scaler.fit(all_logits, all_labels)
        
        print(f"✅ 温度缩放校准完成，最优温度: {optimal_temp:.4f}")
        return optimal_temp
    
    def calibrate_platt_scaling(self, val_loader: DataLoader):
        """
        使用Platt缩放进行校准
        Args:
            val_loader: 验证数据加载器
        """
        print("📊 开始Platt缩放校准...")
        
        # 收集每个类别的置信度和标签
        class_confidences = {i: [] for i in range(len(self.class_names))}
        class_labels = {i: [] for i in range(len(self.class_names))}
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=1)
                
                for i, (prob_vec, true_label) in enumerate(zip(probs, labels)):
                    for class_idx in range(len(self.class_names)):
                        confidence = prob_vec[class_idx].cpu().numpy()
                        is_correct = 1 if true_label.item() == class_idx else 0
                        
                        class_confidences[class_idx].append(confidence)
                        class_labels[class_idx].append(is_correct)
        
        # 为每个类别训练Platt缩放器
        for class_idx in range(len(self.class_names)):
            if len(set(class_labels[class_idx])) > 1:  # 确保有正负样本
                scaler = PlattScaling()
                scaler.fit(np.array(class_confidences[class_idx]), 
                          np.array(class_labels[class_idx]))
                self.platt_scalers[class_idx] = scaler
                print(f"✅ 类别 {self.class_names[class_idx]} 的Platt缩放器训练完成")
            else:
                print(f"⚠️  类别 {self.class_names[class_idx]} 缺少正负样本，跳过Platt缩放")
    
    def calibrate_isotonic_regression(self, val_loader: DataLoader):
        """
        使用等渗回归进行校准
        Args:
            val_loader: 验证数据加载器
        """
        print("📈 开始等渗回归校准...")
        
        # 收集每个类别的置信度和标签
        class_confidences = {i: [] for i in range(len(self.class_names))}
        class_labels = {i: [] for i in range(len(self.class_names))}
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=1)
                
                for i, (prob_vec, true_label) in enumerate(zip(probs, labels)):
                    for class_idx in range(len(self.class_names)):
                        confidence = prob_vec[class_idx].cpu().numpy()
                        is_correct = 1 if true_label.item() == class_idx else 0
                        
                        class_confidences[class_idx].append(confidence)
                        class_labels[class_idx].append(is_correct)
        
        # 为每个类别训练等渗回归器
        for class_idx in range(len(self.class_names)):
            if len(set(class_labels[class_idx])) > 1:  # 确保有正负样本
                scaler = IsotonicCalibration()
                scaler.fit(np.array(class_confidences[class_idx]), 
                          np.array(class_labels[class_idx]))
                self.isotonic_scalers[class_idx] = scaler
                print(f"✅ 类别 {self.class_names[class_idx]} 的等渗回归器训练完成")
            else:
                print(f"⚠️  类别 {self.class_names[class_idx]} 缺少正负样本，跳过等渗回归")
    
    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放
        Args:
            logits: 原始logits
        Returns:
            校准后的概率
        """
        if self.temperature_scaler is None:
            raise ValueError("温度缩放器未训练，请先调用calibrate_temperature_scaling")
        
        # 确保温度缩放器在正确的设备上
        self.temperature_scaler = self.temperature_scaler.to(logits.device)
        return self.temperature_scaler(logits)
    
    def apply_platt_scaling(self, probs: torch.Tensor) -> torch.Tensor:
        """
        应用Platt缩放
        Args:
            probs: 原始概率分布
        Returns:
            校准后的概率分布
        """
        if not self.platt_scalers:
            raise ValueError("Platt缩放器未训练，请先调用calibrate_platt_scaling")
        
        device = probs.device
        calibrated_probs = torch.zeros_like(probs)
        
        for class_idx in range(probs.shape[1]):
            if class_idx in self.platt_scalers:
                original_probs = probs[:, class_idx].cpu().numpy()
                calibrated = self.platt_scalers[class_idx].predict_proba(original_probs)
                calibrated_probs[:, class_idx] = torch.from_numpy(calibrated).to(device)
            else:
                calibrated_probs[:, class_idx] = probs[:, class_idx]
        
        # 重新归一化
        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)
        return calibrated_probs
    
    def apply_isotonic_regression(self, probs: torch.Tensor) -> torch.Tensor:
        """
        应用等渗回归
        Args:
            probs: 原始概率分布
        Returns:
            校准后的概率分布
        """
        if not self.isotonic_scalers:
            raise ValueError("等渗回归器未训练，请先调用calibrate_isotonic_regression")
        
        device = probs.device
        calibrated_probs = torch.zeros_like(probs)
        
        for class_idx in range(probs.shape[1]):
            if class_idx in self.isotonic_scalers:
                original_probs = probs[:, class_idx].cpu().numpy()
                calibrated = self.isotonic_scalers[class_idx].predict(original_probs)
                calibrated_probs[:, class_idx] = torch.from_numpy(calibrated).to(device)
            else:
                calibrated_probs[:, class_idx] = probs[:, class_idx]
        
        # 重新归一化
        calibrated_probs = calibrated_probs / calibrated_probs.sum(dim=1, keepdim=True)
        return calibrated_probs


def calculate_ece(confidences: np.ndarray, accuracies: np.ndarray, 
                  n_bins: int = 10) -> float:
    """
    计算期望校准误差 (Expected Calibration Error)
    Args:
        confidences: 置信度数组
        accuracies: 准确率数组 (0或1)
        n_bins: 分箱数量
    Returns:
        ECE值
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前bin中的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def calculate_mce(confidences: np.ndarray, accuracies: np.ndarray, 
                  n_bins: int = 10) -> float:
    """
    计算最大校准误差 (Maximum Calibration Error)
    Args:
        confidences: 置信度数组
        accuracies: 准确率数组 (0或1)
        n_bins: 分箱数量
    Returns:
        MCE值
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    mce = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前bin中的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    
    return mce


def setup_chinese_font():
    """设置中文字体"""
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        './fonts/NotoSansCJK-Regular.ttc',
        './fonts/NotoSansSC-Regular.ttf',
        '/System/Library/Fonts/PingFang.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf'
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                fm.fontManager.addfont(font_path)
                font_name = font_prop.get_name()
                plt.rcParams['font.family'] = font_name
                plt.rcParams['axes.unicode_minus'] = False
                return True
            except Exception:
                continue
    
    # 备选方案
    chinese_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei', 
                     'Microsoft YaHei', 'PingFang SC']
    for font_name in chinese_fonts:
        try:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except Exception:
            continue
    
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return False


def plot_calibration_comparison(original_results: Dict, 
                              calibrated_results: Dict,
                              save_path: str = './result/calibration_comparison.png'):
    """
    绘制校准前后的对比图
    Args:
        original_results: 原始结果字典
        calibrated_results: 校准后结果字典
        save_path: 保存路径
    """
    font_available = setup_chinese_font()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    methods = ['temperature', 'platt', 'isotonic']
    method_names = ['温度缩放', 'Platt缩放', '等渗回归'] if font_available else \
                   ['Temperature', 'Platt', 'Isotonic']
    
    # 原始结果
    orig_conf = np.array([r['confidence'] for r in original_results['results']])
    orig_acc = np.array([r['is_correct'] for r in original_results['results']], dtype=int)
    
    for i, (method, method_name) in enumerate(zip(methods, method_names)):
        if method not in calibrated_results:
            continue
            
        calib_conf = np.array([r['confidence'] for r in calibrated_results[method]['results']])
        calib_acc = np.array([r['is_correct'] for r in calibrated_results[method]['results']], dtype=int)
        
        # 校准图 (上排)
        ax1 = axes[0, i]
        
        # 绘制原始和校准后的校准曲线
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        orig_bin_acc = []
        orig_bin_conf = []
        calib_bin_acc = []
        calib_bin_conf = []
        bin_centers = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_center = (bin_lower + bin_upper) / 2
            bin_centers.append(bin_center)
            
            # 原始结果
            in_bin_orig = (orig_conf > bin_lower) & (orig_conf <= bin_upper)
            if in_bin_orig.sum() > 0:
                orig_bin_acc.append(orig_acc[in_bin_orig].mean())
                orig_bin_conf.append(orig_conf[in_bin_orig].mean())
            else:
                orig_bin_acc.append(0)
                orig_bin_conf.append(bin_center)
            
            # 校准后结果
            in_bin_calib = (calib_conf > bin_lower) & (calib_conf <= bin_upper)
            if in_bin_calib.sum() > 0:
                calib_bin_acc.append(calib_acc[in_bin_calib].mean())
                calib_bin_conf.append(calib_conf[in_bin_calib].mean())
            else:
                calib_bin_acc.append(0)
                calib_bin_conf.append(bin_center)
        
        # 绘制校准曲线
        ax1.plot([0, 1], [0, 1], 'k--', label='完美校准' if font_available else 'Perfect Calibration')
        ax1.plot(orig_bin_conf, orig_bin_acc, 'ro-', label='原始' if font_available else 'Original')
        ax1.plot(calib_bin_conf, calib_bin_acc, 'bo-', label='校准后' if font_available else 'Calibrated')
        
        ax1.set_xlabel('置信度' if font_available else 'Confidence')
        ax1.set_ylabel('准确率' if font_available else 'Accuracy')
        ax1.set_title(f'{method_name} 校准曲线' if font_available else f'{method_name} Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 置信度分布对比 (下排)
        ax2 = axes[1, i]
        
        ax2.hist(orig_conf, bins=20, alpha=0.5, color='red', 
                label=f'原始 (ECE: {calculate_ece(orig_conf, orig_acc):.3f})' if font_available else 
                      f'Original (ECE: {calculate_ece(orig_conf, orig_acc):.3f})')
        ax2.hist(calib_conf, bins=20, alpha=0.5, color='blue',
                label=f'校准后 (ECE: {calculate_ece(calib_conf, calib_acc):.3f})' if font_available else
                      f'Calibrated (ECE: {calculate_ece(calib_conf, calib_acc):.3f})')
        
        ax2.set_xlabel('置信度' if font_available else 'Confidence')
        ax2.set_ylabel('数量' if font_available else 'Count')
        ax2.set_title(f'{method_name} 置信度分布' if font_available else f'{method_name} Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 校准对比图已保存: {save_path}")


def save_calibration_results(original_results: Dict, 
                           calibrated_results: Dict,
                           save_path: str = './result/calibration_results.json'):
    """
    保存校准结果
    Args:
        original_results: 原始结果
        calibrated_results: 校准后结果
        save_path: 保存路径
    """
    # 计算各种指标
    orig_conf = np.array([r['confidence'] for r in original_results['results']])
    orig_acc = np.array([r['is_correct'] for r in original_results['results']], dtype=int)
    
    results_summary = {
        'original': {
            'accuracy': original_results['accuracy'],
            'ece': calculate_ece(orig_conf, orig_acc),
            'mce': calculate_mce(orig_conf, orig_acc),
            'avg_confidence': float(orig_conf.mean()),
            'confidence_std': float(orig_conf.std())
        }
    }
    
    for method in ['temperature', 'platt', 'isotonic']:
        if method in calibrated_results:
            calib_conf = np.array([r['confidence'] for r in calibrated_results[method]['results']])
            calib_acc = np.array([r['is_correct'] for r in calibrated_results[method]['results']], dtype=int)
            
            results_summary[method] = {
                'accuracy': calibrated_results[method]['accuracy'],
                'ece': calculate_ece(calib_conf, calib_acc),
                'mce': calculate_mce(calib_conf, calib_acc),
                'avg_confidence': float(calib_conf.mean()),
                'confidence_std': float(calib_conf.std())
            }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 校准结果已保存: {save_path}")
    return results_summary
