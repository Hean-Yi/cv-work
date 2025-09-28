import os
import torch
from scalable_model import create_overlock_model, ScalableOverLoCKModel
from rtx4090_configs import RTX4090OptimalConfig, RTX4090MaxConfig, get_rtx4090_training_config
from dataset import OverLoCKDataset
from trainer import OverLoCKTrainer


def main():
    """主函数：演示如何使用框架"""

    # 数据集选择: "cifar10" 或 "imagenet100"
    dataset_type = "imagenet100"  # 修改这里来选择数据集
    
    if dataset_type == "imagenet100":
        print("🎯 使用ImageNet-100数据集 (100类, 224x224)")
        data_root = "./data/imagenet100"
        num_classes = 100
    else:
        print("🎯 使用CIFAR-10数据集 (10类, 224x224)")
        data_root = "./data"
        num_classes = 10

    # 1. 设置参数 - RTX 4090最优化配置
    print("🚀 使用RTX 4090最优化OverLoCK模型配置")
    
    # 检测GPU数量
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    multi_gpu = gpu_count > 1
    
    if multi_gpu:
        print(f"🔥 检测到 {gpu_count} 个GPU，将启用多GPU并行训练！")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
    else:
        print(f"📱 使用单GPU训练 (检测到 {gpu_count} 个GPU)")
    
    # 选择模型规模: "optimal" 或 "max"
    model_size = "optimal"  # 推荐使用optimal，平衡性能和显存
    # model_size = "max"    # 最大模型，144M参数，需要22GB显存
    
    if model_size == "max":
        print("📊 使用MAX配置: ~144M参数, 预期显存使用: ~22GB")
        config = RTX4090MaxConfig()
    else:
        print("📊 使用OPTIMAL配置: ~65M参数, 预期显存使用: ~16GB")
        config = RTX4090OptimalConfig()
    
    # 获取对应的训练配置（包含多GPU支持）
    train_config = get_rtx4090_training_config(model_size, multi_gpu=multi_gpu)
    
    batch_size = train_config['batch_size']
    learning_rate = train_config['learning_rate']
    num_epochs = 3  # 改为3个epoch，更容易看出性能差异
    weight_decay = train_config['weight_decay']
    use_mixed_precision = train_config['use_mixed_precision']
    gradient_clip_max_norm = train_config['gradient_clip_max_norm']
    
    print(f"⚙️ 训练配置: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
    if multi_gpu:
        print(f"🚀 双GPU有效batch_size: {batch_size * 2} (每GPU: {batch_size})")
    print(f"💾 启用混合精度训练: {use_mixed_precision}")
    print(f"⚡ 梯度裁剪: {gradient_clip_max_norm}")
    
    # 设置训练模式
    training_mode = "new"  # "new": 重新训练, "load": 只加载模型, "continue": 继续训练
    checkpoint_path = f"./checkpoints/rtx4090_{model_size}_best_model.pth"  # 专用路径

    # 定义数据变换 - 增加数据增强
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 先放大
        transforms.RandomCrop((224, 224)),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 创建数据集
    train_dataset = OverLoCKDataset(
        data_root=data_root,
        mode='train',
        transform=train_transform,
        dataset_type=dataset_type
    )

    val_dataset = OverLoCKDataset(
        data_root=data_root,
        mode='val',
        transform=val_transform,
        dataset_type=dataset_type
    )

    test_dataset = OverLoCKDataset(
        data_root=data_root,
        mode='test',
        transform=val_transform,
        dataset_type=dataset_type
    )

    # 加入测试集数量检查
    print("=== 数据集统计 ===")
    print(f"训练集样本数量: {len(train_dataset)}")
    print(f"验证集样本数量: {len(val_dataset)}")
    print("=" * 20)

    # 获取实际的类别名称
    class_names = train_dataset.get_class_names()
    print(f"实际类别名称: {class_names}")
    print(f"类别数量: {len(class_names)}")

    # 3. 创建RTX 4090优化模型
    print(f"\n🏗️ 创建{model_size.upper()}规模的OverLoCK模型...")
    model = ScalableOverLoCKModel(
        class_names=class_names,
        config=config
    )

    # 4. 打印模型信息
    print(f"\n📊 {config.name} Architecture:")
    model.print_model_info()
    
    # 估算显存使用
    total_params = sum(p.numel() for p in model.parameters())
    estimated_memory = train_config['memory_usage_gb']
    print(f"💾 预估显存使用: {estimated_memory:.1f} GB (RTX 4090: 24GB)")
    print(f"🎯 显存利用率: {estimated_memory/24*100:.1f}%")
    
    if model_size == "max":
        print("⚠️ MAX配置需要大量显存，确保RTX 4090有足够的显存空间")
    else:
        print("✅ OPTIMAL配置显存使用适中，推荐用于生产环境")

    # 5. 根据训练模式选择操作
    if training_mode == "load" and os.path.exists(checkpoint_path):
        print(f"\n🔄 加载预训练模型: {checkpoint_path}")
        # 只加载模型权重，不训练
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移动模型到设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"✅ 模型已加载，验证准确率: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        print(f"📊 已完成训练轮数: {checkpoint.get('epoch', 'N/A') + 1}")
        
    elif training_mode == "continue" and os.path.exists(checkpoint_path):
        print(f"\n🔄 继续训练模式: 从检查点恢复训练")
        print(f"📂 检查点路径: {checkpoint_path}")
        
        # 创建训练器 - 多GPU支持
        trainer = OverLoCKTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device='cuda',
            save_dir='./checkpoints',
            clip_loss_weight=0.1,  # X-Large模型用较小的CLIP权重
            multi_gpu=multi_gpu
        )
        
        # 应用RTX 4090优化设置
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 加载检查点并继续训练
        start_epoch, current_val_acc = trainer.load_checkpoint(checkpoint_path)
        print(f"✅ 检查点已加载")
        print(f"📊 当前验证准确率: {current_val_acc:.2f}%")
        print(f"🔢 已完成轮数: {start_epoch + 1}")
        print(f"📈 训练历史已恢复，包含 {len(trainer.history['train_loss'])} 个训练记录")
        
        print("\n" + "="*60)
        print(f"继续训练 - 从第 {start_epoch + 2} 轮开始")
        print("="*60)
        # 继续训练
        trainer.train()
        
    elif training_mode == "new":
        print("\n🆕 开始X-Large模型全新训练...")
        # 创建训练器 - 多GPU支持
        trainer = OverLoCKTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device='cuda',
            save_dir='./checkpoints',
            clip_loss_weight=0.1,
            multi_gpu=multi_gpu
        )
        
        print(f"🚀 多GPU模型训练器已创建")
        if multi_gpu:
            print(f"⚡ 双GPU训练：batch_size={batch_size} (每GPU约{batch_size//2})")
            print(f"💾 预期显存使用: {estimated_memory:.1f} GB × 2 = {estimated_memory*2:.1f} GB")
        print(f"� 学习率: {learning_rate} (论文推荐)")
        print(f"⚖️ 权重衰减: {weight_decay} (论文推荐)")
        print(f"⏱️ 预期训练时间: 2-5分钟 (1轮测试)")
        
        print("\n" + "="*60)
        print("开始X-Large OverLoCK模型训练 (RTX 4090优化) - 测试模式")
        print("="*60)
        print(f"🚀 模型规模: {total_params/1e6:.1f}M 参数")
        print(f"💾 显存使用: ~{estimated_memory:.1f} GB")
        print(f"⏱️ 预期时间: 5-10分钟 (1轮测试)")
        # 开始训练
        trainer.train()
        
    else:
        if training_mode == "continue":
            print(f"\n⚠️  继续训练模式但未找到检查点: {checkpoint_path}")
            print("🔄 自动切换到全新训练模式")
        elif training_mode == "load":
            print(f"\n⚠️  加载模式但未找到检查点: {checkpoint_path}")
            print("🔄 自动切换到全新训练模式")
            
        print("\n🆕 开始X-Large模型全新训练...")
        # 创建训练器 - 多GPU支持
        trainer = OverLoCKTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            device='cuda',
            save_dir='./checkpoints',
            clip_loss_weight=0.1,
            multi_gpu=multi_gpu
        )
        
        print(f"🚀 多GPU模型训练器已创建")
        if multi_gpu:
            print(f"⚡ 双GPU训练：batch_size={batch_size} (每GPU约{batch_size//2})")
            print(f"💾 预期显存使用: {estimated_memory:.1f} GB × 2 = {estimated_memory*2:.1f} GB")
        print(f"📈 学习率: {learning_rate} (论文推荐)")
        print(f"⚖️ 权重衰减: {weight_decay} (论文推荐)")
        print(f"⏱️ 预期训练时间: 2-5分钟 (1轮测试)")
        
        print("\n" + "="*60)
        if multi_gpu:
            print("开始OverLoCK模型双GPU并行训练 - 测试模式")
        else:
            print("开始OverLoCK模型训练 - 测试模式")
        print("="*60)
        print(f"🚀 模型规模: {total_params/1e6:.1f}M 参数")
        print(f"💾 显存使用: ~{estimated_memory:.1f} GB" + (f" × 2" if multi_gpu else ""))
        print(f"⏱️ 预期时间: 2-5分钟 (1轮测试)")
        # 开始训练
        trainer.train()

    # 6. 测试推理(批量处理示例) - 使用test_dataset的随机数据
    model.eval()
    with torch.no_grad():
        # 从验证集中随机选择8个样本
        import random
        import json
        import matplotlib.pyplot as plt
        import warnings
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # 抑制matplotlib字体警告
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        
        # 设置中文字体函数
        def setup_chinese_font():
            """设置中文字体 - 优先使用系统中文字体"""
            import matplotlib.font_manager as fm
            
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
        
        # 设置中文字体
        setup_chinese_font()
        
        # 创建结果保存目录
        result_dir = "./result"
        os.makedirs(result_dir, exist_ok=True)
        
        # 定义图像预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 随机选择样本索引
        batch_size = 100
        random_indices = random.sample(range(len(val_dataset)), min(batch_size, len(val_dataset)))
        
        test_images = []
        test_labels = []
        test_paths = []
        original_images = []  # 保存原始图像用于可视化
        
        # 创建一个没有transform的数据集来获取原始图像
        val_dataset_no_transform = OverLoCKDataset(
            data_root=data_root,  # 使用正确的data_root
            mode='val',
            transform=None,
            dataset_type=dataset_type  # 使用正确的dataset_type
        )
        
        # 确保两个数据集长度一致
        max_samples = min(len(val_dataset), len(val_dataset_no_transform))
        random_indices = random.sample(range(max_samples), min(batch_size, max_samples))
        
        for idx in random_indices:
            # 检查索引范围安全性
            if idx < len(val_dataset) and idx < len(val_dataset_no_transform):
                # 获取已经变换的图像和标签
                image, label = val_dataset[idx]
                test_images.append(image)
                test_labels.append(label)
                test_paths.append(val_dataset.samples[idx][0])  # 获取图像路径
                
                # 获取原始图像用于可视化
                original_image, _ = val_dataset_no_transform[idx]
                original_images.append(original_image)
        
        # 创建批量数据
        test_batch = torch.stack(test_images)
        test_labels = torch.tensor(test_labels)
        
        if torch.cuda.is_available():
            test_batch = test_batch.cuda()
            test_labels = test_labels.cuda()
            model = model.cuda()

        # 推理
        logits, clip_logits = model(test_batch)
        predictions = torch.argmax(logits, dim=1)
        
        # 计算正确率
        correct = (predictions.cpu() == test_labels.cpu()).sum().item()
        accuracy = correct / len(test_labels) * 100

        print("\nInference Test (using val_dataset random samples):")
        print(f"Input shape: {test_batch.shape}")
        print(f"Output logits shape: {logits.shape}")
        print(f"Predictions: {predictions.cpu().numpy()}")
        print(f"Ground truth: {test_labels.cpu().numpy()}")
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(test_labels)})")
        
        # 保存预测结果到JSON文件
        results = []
        for i in range(len(predictions)):
            pred_class = class_names[predictions[i].cpu().item()]
            true_class = class_names[test_labels[i].cpu().item()]
            image_path = os.path.basename(test_paths[i])
            is_correct = predictions[i].cpu().item() == test_labels[i].cpu().item()
            
            result = {
                "sample_id": i + 1,
                "image_name": image_path,
                "predicted_class": pred_class,
                "true_class": true_class,
                "is_correct": is_correct,
                "confidence": torch.softmax(logits[i], dim=0).max().cpu().item()
            }
            results.append(result)
        
        # 保存结果到JSON文件
        with open(os.path.join(result_dir, "inference_results.json"), "w", encoding="utf-8") as f:
            json.dump({
                "accuracy": accuracy,
                "total_samples": len(test_labels),
                "correct_predictions": correct,
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        # 创建可视化图像 - 显示前16个样本作为示例
        display_count = min(16, len(predictions))
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle(f'Inference Results Sample (showing {display_count}/{len(predictions)}) - Accuracy: {accuracy:.2f}%', fontsize=16)
        
        for i in range(display_count):
            row = i // 4
            col = i % 4
            
            # 显示原始图像
            axes[row, col].imshow(original_images[i])
            axes[row, col].axis('off')
            
            # 设置标题
            pred_class = class_names[predictions[i].cpu().item()]
            true_class = class_names[test_labels[i].cpu().item()]
            is_correct = predictions[i].cpu().item() == test_labels[i].cpu().item()
            confidence = torch.softmax(logits[i], dim=0).max().cpu().item()
            
            title_color = 'green' if is_correct else 'red'
            title = f'#{i+1} Pred: {pred_class}\nTrue: {true_class}\nConf: {confidence:.3f}'
            axes[row, col].set_title(title, color=title_color, fontsize=10)
        
        # 保存可视化结果
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "inference_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 显示预测结果详情
        print("\nDetailed predictions:")
        for result in results:
            status = "✓" if result["is_correct"] else "✗"
            print(f"Sample {result['sample_id']}: {result['image_name']} -> "
                  f"Predicted: {result['predicted_class']}, True: {result['true_class']} "
                  f"[Conf: {result['confidence']:.3f}] {status}")
        
        print(f"\nResults saved to: {result_dir}/")
        print(f"- inference_results.json: Detailed results in JSON format")
        print(f"- inference_visualization.png: Visual results with sample images (showing first 16)")


if __name__ == "__main__":
    main()
