import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
from tqdm import tqdm
from confidence_calibration import ConfidenceCalibrator


class OverLoCKTrainer:
    """OverLoCK 训练器类"""
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 100,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints',
                 clip_loss_weight: float = 0.1,
                 multi_gpu: bool = False):
        """
        Args:
            model: OverLoCK模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            batch_size: 批次大小
            learning_rate: 学习率
            num_epochs: 训练轮数
            device: 设备
            save_dir: 模型保存目录
            clip_loss_weight: CLIP损失权重
            multi_gpu: 是否使用多GPU训练
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.clip_loss_weight = clip_loss_weight
        self.multi_gpu = multi_gpu

        # 多GPU设置
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 Detected {torch.cuda.device_count()} GPUs, enabling multi-GPU parallel training")
            model = nn.DataParallel(model)
            # 双GPU时有效batch_size翻倍
            effective_batch_size = batch_size * torch.cuda.device_count()
            print(f"📊 Multi-GPU training: effective_batch_size={effective_batch_size} (per GPU: {batch_size})")
        elif multi_gpu:
            print("⚠️ Requested multi-GPU training but only 1 GPU detected, using single GPU training")
            effective_batch_size = batch_size
        else:
            effective_batch_size = batch_size
            
        self.effective_batch_size = effective_batch_size
        self.model = model
        
        # 移动模型到设备
        self.model.to(self.device)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 数据加载器 - 针对多GPU优化
        num_workers = 8 if multi_gpu else 4  # 双GPU时增加workers
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,  # 保持worker进程存活
            prefetch_factor=2         # 预取数据
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
        else:
            self.val_loader = None

        # 优化器和损失函数 - 使用论文中的设置
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,  # 使用传入的学习率（应该是4e-3）
            weight_decay=0.05,  # 论文中的权重衰减
            betas=(0.9, 0.999)
        )
        
        # 使用带预热的余弦退火调度器
        warmup_epochs = max(1, num_epochs // 10)  # 预热轮数为总轮数的10%
        # 确保T_0至少为1，避免学习率调度器错误
        T_0 = max(1, num_epochs - warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=T_0,
            T_mult=1,
            eta_min=learning_rate * 0.01  # 最小学习率为初始学习率的1%
        )
        
        # 添加标签平滑
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 移动模型到设备
        self.model.to(self.device)

        # 修复inplace操作
        self._disable_inplace_operations()

        # 训练历史记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def _disable_inplace_operations(self):
        """禁用模型中的所有inplace操作"""
        inplace_found = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'inplace') and module.inplace:
                module.inplace = False
                inplace_found.append(name)
        
        if inplace_found:
            print(f"Disabled inplace operations for modules: {inplace_found}")
        else:
            print("No inplace operations found, model is normal")

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        import time
        
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 性能监控
        epoch_start_time = time.time()
        batch_times = []

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            batch_start_time = time.time()
            
            try:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向传播
                main_logits, aux_logits, clip_logits = self.model(images, use_aux=True)

                # 计算损失
                loss_cls = self.criterion(main_logits, labels)
                loss = loss_cls

                # 添加辅助损失(如果有)
                if aux_logits is not None:
                    loss_aux = self.criterion(aux_logits, labels)
                    loss = loss + 0.3 * loss_aux  # 辅助损失权重

                # 添加CLIP损失(如果有)
                if clip_logits is not None:
                    loss_clip = self.criterion(clip_logits, labels)
                    loss = loss + self.clip_loss_weight * loss_clip

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                # 统计
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                # 计算吞吐量（每秒处理的样本数）
                throughput = self.effective_batch_size / batch_time
                total_loss += loss.item()
                _, predicted = torch.max(main_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条 - 显示性能信息
                if batch_idx % 10 == 0:  # 每10个batch显示一次性能
                    avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                    avg_throughput = self.effective_batch_size / avg_batch_time
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*correct/total:.2f}%',
                        'Throughput': f'{avg_throughput:.1f} imgs/s',
                        'Time/batch': f'{avg_batch_time:.3f}s'
                    })
                
            except Exception as e:
                print(f"Error during training: {e}")
                print("Suggest checking model structure")
                raise e

        # 关闭异常检测
        torch.autograd.set_detect_anomaly(False)
        
        # Epoch性能统计
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_samples = total
        epoch_throughput = total_samples / epoch_duration
        
        print(f"\n📊 Epoch Performance:")
        print(f"   Total time: {epoch_duration:.2f}s")
        print(f"   Avg batch time: {avg_batch_time:.3f}s") 
        print(f"   Epoch throughput: {epoch_throughput:.1f} imgs/s")
        print(f"   Total samples processed: {total_samples}")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                main_logits, aux_logits, clip_logits = self.model(images, use_aux=False)

                # 计算损失
                loss = self.criterion(main_logits, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(main_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def train(self):
        """完整训练流程"""
        print(f"Starting training on device: {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")

        # 检查是否存在最佳模型，如果存在则加载其验证准确率
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        best_val_acc = 0.0
        
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                best_val_acc = checkpoint.get('val_acc', 0.0)
                print(f"🔍 Found existing best model, validation accuracy: {best_val_acc:.2f}%")
                print(f"📊 New training will only save as best model when validation accuracy exceeds {best_val_acc:.2f}%")
            except Exception as e:
                print(f"⚠️ Unable to load existing best model: {e}")
                print("🔄 Starting from validation accuracy 0.0%")
                best_val_acc = 0.0
        else:
            print("🆕 No existing best model found, starting from validation accuracy 0.0%")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")

            # 训练
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # 验证
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 调整学习率
            self.scheduler.step()

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)

        # 训练完成后运行可视化
        print("\n" + "="*60)
        print("🎨 Training completed! Starting visualization generation...")
        print("="*60)
        
        try:
            self.run_post_training_visualization()
        except Exception as e:
            print(f"⚠️ Visualization generation failed: {e}")
            print("💡 You can manually run later: python inference.py")

    def run_post_training_visualization(self):
        """训练完成后运行可视化"""
        try:
            from model_visualizer import ModelVisualizer
            
            # 创建可视化器
            visualizer = ModelVisualizer(self.model, device=self.device, save_dir='./visualizations')
            
            print("📊 1/3 Computing model evaluation metrics...")
            # 使用验证集进行评估
            if self.val_loader is not None:
                metrics = visualizer.calculate_metrics(self.val_loader, num_samples=200)
            else:
                metrics = visualizer.calculate_metrics(self.train_loader, num_samples=200)
            
            print("🔍 2/3 Generating Effective Receptive Field visualization...")
            # ERF可视化
            if self.val_loader is not None:
                erf_maps = visualizer.visualize_effective_receptive_field(self.val_loader, num_images=50)
            else:
                erf_maps = visualizer.visualize_effective_receptive_field(self.train_loader, num_images=50)
            
            print("🎯 3/3 Generating GradCAM class activation maps...")
            # GradCAM可视化
            dataloader = self.val_loader if self.val_loader is not None else self.train_loader
            for images, labels in dataloader:
                sample_images = images[:4].to(self.device)  # 只取4个样本
                sample_labels = labels[:4].to(self.device)
                gradcams = visualizer.visualize_gradcam(sample_images, sample_labels)
                break
            
            print("✅ Visualization completed! Results saved in ./visualizations/ directory")
            print(f"📈 Final validation accuracy: {metrics.get('top1_accuracy', 'N/A')}")
            
        except ImportError:
            print("⚠️ Visualization module not found, please ensure model_visualizer.py is uploaded")
        except Exception as e:
            print(f"❌ 可视化过程出错: {e}")
            import traceback
            traceback.print_exc()

    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        # 重新修复inplace操作
        self._disable_inplace_operations()
        
        print(f"Model loaded from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['val_acc']
    
    def calibrate_confidence(self, test_dataset: Optional[Dataset] = None, 
                           enable_calibration: bool = True):
        """
        训练完成后进行置信度校准
        Args:
            test_dataset: 测试数据集，如果为None则使用验证集
            enable_calibration: 是否启用校准功能
        """
        if not enable_calibration:
            print("⚠️  置信度校准功能已禁用")
            return
            
        if self.val_loader is None:
            print("⚠️  没有验证集，无法进行置信度校准")
            return
        
        print("\n" + "="*60)
        print("🔧 开始置信度校准...")
        print("="*60)
        
        # 创建校准器
        calibrator = ConfidenceCalibrator(self.model, str(self.device))
        
        try:
            # 使用验证集进行校准训练
            print("📊 使用验证集训练校准器...")
            
            # 温度缩放校准
            optimal_temp = calibrator.calibrate_temperature_scaling(self.val_loader)
            
            # Platt缩放校准
            calibrator.calibrate_platt_scaling(self.val_loader)
            
            # 等渗回归校准
            calibrator.calibrate_isotonic_regression(self.val_loader)
            
            # 保存校准器
            calibrator_path = os.path.join(self.save_dir, 'confidence_calibrator.pth')
            torch.save({
                'temperature_scaler': calibrator.temperature_scaler.state_dict() if calibrator.temperature_scaler else None,
                'platt_scalers': calibrator.platt_scalers,
                'isotonic_scalers': calibrator.isotonic_scalers,
                'class_names': calibrator.class_names,
                'optimal_temperature': optimal_temp
            }, calibrator_path)
            
            print(f"✅ 置信度校准器已保存: {calibrator_path}")
            
            # 如果有测试集，进行校准效果评估
            if test_dataset is not None:
                print("📈 评估校准效果...")
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
                
                # 评估原始模型
                original_results = self._evaluate_model_confidence(test_loader, "原始模型")
                
                # 评估校准后的模型
                calibrated_results = {}
                
                # 温度缩放
                temp_results = self._evaluate_calibrated_model_confidence(
                    test_loader, calibrator, 'temperature', "温度缩放"
                )
                if temp_results:
                    calibrated_results['temperature'] = temp_results
                
                # Platt缩放
                platt_results = self._evaluate_calibrated_model_confidence(
                    test_loader, calibrator, 'platt', "Platt缩放"
                )
                if platt_results:
                    calibrated_results['platt'] = platt_results
                
                # 等渗回归
                isotonic_results = self._evaluate_calibrated_model_confidence(
                    test_loader, calibrator, 'isotonic', "等渗回归"
                )
                if isotonic_results:
                    calibrated_results['isotonic'] = isotonic_results
                
                # 保存和可视化结果
                if calibrated_results:
                    from confidence_calibration import (
                        save_calibration_results, 
                        plot_calibration_comparison
                    )
                    
                    # 保存结果
                    results_summary = save_calibration_results(
                        original_results,
                        calibrated_results,
                        './result/training_calibration_results.json'
                    )
                    
                    # 生成可视化
                    plot_calibration_comparison(
                        original_results,
                        calibrated_results,
                        './result/training_calibration_comparison.png'
                    )
                    
                    print("✅ 校准结果已保存:")
                    print("   - 数据: ./result/training_calibration_results.json")
                    print("   - 图表: ./result/training_calibration_comparison.png")
            
            print("🎉 置信度校准完成！")
            
        except Exception as e:
            print(f"❌ 置信度校准失败: {e}")
            print("继续正常训练流程...")
    
    def _evaluate_model_confidence(self, data_loader, method_name="模型"):
        """评估模型的置信度性能"""
        self.model.eval()
        results = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                main_logits, _, _ = self.model(images)
                probs = torch.softmax(main_logits, dim=1)
                confidences, predicted = torch.max(probs, 1)
                
                for i in range(len(labels)):
                    is_correct = predicted[i].item() == labels[i].item()
                    results.append({
                        'confidence': confidences[i].item(),
                        'predicted': predicted[i].item(),
                        'true_label': labels[i].item(),
                        'is_correct': is_correct
                    })
                    
                    if is_correct:
                        correct += 1
                    total += 1
        
        accuracy = 100.0 * correct / total
        print(f"✅ {method_name} 准确率: {accuracy:.2f}%")
        
        return {
            'results': results,
            'accuracy': accuracy,
            'total_samples': total
        }
    
    def _evaluate_calibrated_model_confidence(self, data_loader, calibrator, method, method_name):
        """评估校准后模型的置信度性能"""
        try:
            self.model.eval()
            results = []
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in data_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    main_logits, _, _ = self.model(images)
                    
                    # 应用校准
                    if method == 'temperature':
                        probs = calibrator.apply_temperature_scaling(main_logits)
                    elif method == 'platt':
                        original_probs = torch.softmax(main_logits, dim=1)
                        probs = calibrator.apply_platt_scaling(original_probs)
                    elif method == 'isotonic':
                        original_probs = torch.softmax(main_logits, dim=1)
                        probs = calibrator.apply_isotonic_regression(original_probs)
                    else:
                        continue
                    
                    confidences, predicted = torch.max(probs, 1)
                    
                    for i in range(len(labels)):
                        is_correct = predicted[i].item() == labels[i].item()
                        results.append({
                            'confidence': confidences[i].item(),
                            'predicted': predicted[i].item(),
                            'true_label': labels[i].item(),
                            'is_correct': is_correct
                        })
                        
                        if is_correct:
                            correct += 1
                        total += 1
            
            accuracy = 100.0 * correct / total
            print(f"✅ {method_name} 校准后准确率: {accuracy:.2f}%")
            
            return {
                'results': results,
                'accuracy': accuracy,
                'total_samples': total
            }
            
        except Exception as e:
            print(f"⚠️  {method_name} 评估失败: {e}")
            return None
