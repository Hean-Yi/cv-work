import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

class OverLoCKDataset(Dataset):
    """通用数据集读取类（支持 CIFAR-10 和 ImageNet-100）"""
    def __init__(self, data_root: str, mode: str = 'train', transform=None, dataset_type: str = 'auto'):
        """
        Args:
            data_root: 数据集根目录，例如 "cifar-10-images" 或 "data/imagenet100"
            mode: 'train' 或 'test' 或 'val'
            transform: torchvision.transforms 图像变换
            dataset_type: 'cifar10', 'imagenet100', 'auto' (自动检测)
        """
        super(OverLoCKDataset, self).__init__()
        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.dataset_type = dataset_type

        self.samples = []       # List of (image_path, label)
        self.class_names = []   # 类别名称列表

        # 自动检测数据集类型
        if self.dataset_type == 'auto':
            self.dataset_type = self._detect_dataset_type()
        
        # 加载所有样本信息
        self._load_annotations()

    def _detect_dataset_type(self) -> str:
        """自动检测数据集类型"""
        # 检查是否存在典型的ImageNet-100结构
        possible_modes = ['train', 'val', 'test']
        for mode in possible_modes:
            mode_dir = os.path.join(self.data_root, mode)
            if os.path.exists(mode_dir):
                # 检查类别数量来判断数据集类型
                classes = [d for d in os.listdir(mode_dir) 
                          if os.path.isdir(os.path.join(mode_dir, d))]
                if len(classes) >= 90:  # ImageNet-100 有100个类别
                    return 'imagenet100'
                elif len(classes) == 10:  # CIFAR-10 有10个类别
                    return 'cifar10'
        
        # 默认返回 cifar10
        return 'cifar10'

    def _check_and_extract_imagenet100(self):
        """检查并解压ImageNet-100数据集"""
        import zipfile
        import tarfile
        
        # 检查是否已经解压
        if os.path.exists(os.path.join(self.data_root, 'train')) and os.path.exists(os.path.join(self.data_root, 'val')):
            return  # 已经解压，无需操作
        
        print("🔄 检测到ImageNet-100压缩文件，开始解压...")
        
        # 查找压缩文件
        compressed_files = []
        for file in os.listdir(self.data_root):
            if file.startswith('train.X') or file.startswith('val.X'):
                compressed_files.append(os.path.join(self.data_root, file))
        
        if not compressed_files:
            print("⚠️ 未找到ImageNet-100压缩文件")
            return
        
        # 合并并解压文件 (假设是分卷压缩)
        try:
            # 尝试解压train文件
            train_files = [f for f in compressed_files if 'train' in f]
            if train_files:
                print("📦 解压训练集...")
                # 如果是zip分卷，需要先合并
                train_files.sort()  # 确保顺序正确
                
                # 尝试直接解压第一个文件（通常包含所有数据）
                try:
                    with zipfile.ZipFile(train_files[0], 'r') as zip_ref:
                        zip_ref.extractall(self.data_root)
                        print("✅ 训练集解压完成")
                except:
                    # 如果失败，尝试tar格式
                    try:
                        with tarfile.open(train_files[0], 'r') as tar_ref:
                            tar_ref.extractall(self.data_root)
                            print("✅ 训练集解压完成")
                    except:
                        print("❌ 训练集解压失败，请手动解压")
            
            # 尝试解压val文件
            val_files = [f for f in compressed_files if 'val' in f]
            if val_files:
                print("📦 解压验证集...")
                try:
                    with zipfile.ZipFile(val_files[0], 'r') as zip_ref:
                        zip_ref.extractall(self.data_root)
                        print("✅ 验证集解压完成")
                except:
                    try:
                        with tarfile.open(val_files[0], 'r') as tar_ref:
                            tar_ref.extractall(self.data_root)
                            print("✅ 验证集解压完成")
                    except:
                        print("❌ 验证集解压失败，请手动解压")
                        
        except Exception as e:
            print(f"⚠️ 解压过程中出现错误: {e}")
            print("💡 请手动解压ImageNet-100数据集文件")

    def _handle_imagenet100_structure(self):
        """处理ImageNet-100的特殊目录结构 (train.X1, train.X2, ..., val.X)"""
        import shutil
        
        # 检查是否已经合并
        train_dir = os.path.join(self.data_root, 'train')
        val_dir = os.path.join(self.data_root, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            return  # 已经合并，无需操作
        
        print("🔄 检测到ImageNet-100分块目录结构，正在合并...")
        
        # 合并训练集 (train.X1, train.X2, train.X3, train.X4)
        train_parts = ['train.X1', 'train.X2', 'train.X3', 'train.X4']
        train_parts = [os.path.join(self.data_root, part) for part in train_parts if os.path.exists(os.path.join(self.data_root, part))]
        
        if train_parts and not os.path.exists(train_dir):
            print(f"📦 合并训练集 ({len(train_parts)} 个分块)...")
            os.makedirs(train_dir, exist_ok=True)
            
            # 收集所有类别
            all_classes = set()
            for part_dir in train_parts:
                if os.path.isdir(part_dir):
                    classes = [d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))]
                    all_classes.update(classes)
            
            print(f"📂 发现 {len(all_classes)} 个类别")
            
            # 为每个类别创建目录并合并图片
            for class_name in sorted(all_classes):
                class_dir = os.path.join(train_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # 从各个分块复制该类别的图片
                total_images = 0
                for part_dir in train_parts:
                    part_class_dir = os.path.join(part_dir, class_name)
                    if os.path.exists(part_class_dir):
                        images = [f for f in os.listdir(part_class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        for img in images:
                            src = os.path.join(part_class_dir, img)
                            dst = os.path.join(class_dir, img)
                            if not os.path.exists(dst):  # 避免重复复制
                                shutil.copy2(src, dst)
                                total_images += 1
                
                if total_images > 0:
                    print(f"  ✅ {class_name}: {total_images} 张图片")
            
            print("✅ 训练集合并完成")
        
        # 处理验证集 (val.X)
        val_part = os.path.join(self.data_root, 'val.X')
        if os.path.exists(val_part) and os.path.isdir(val_part) and not os.path.exists(val_dir):
            print("📦 处理验证集...")
            
            # 直接重命名或复制val.X到val
            try:
                shutil.move(val_part, val_dir)
                print("✅ 验证集处理完成 (重命名)")
            except:
                # 如果重命名失败，尝试复制
                shutil.copytree(val_part, val_dir)
                print("✅ 验证集处理完成 (复制)")
        
        # 验证结果
        if os.path.exists(train_dir):
            train_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            print(f"📊 训练集: {train_classes} 个类别")
        
        if os.path.exists(val_dir):
            val_classes = len([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
            print(f"📊 验证集: {val_classes} 个类别")

    def _load_annotations(self):
        """扫描文件夹，将每个图片路径和标签存入 samples"""
        # 检查ImageNet-100的特殊目录结构
        if self.dataset_type == 'imagenet100':
            self._handle_imagenet100_structure()
        
        # 根据数据集类型选择合适的模式目录
        if self.dataset_type == 'imagenet100':
            # ImageNet-100 使用合并后的目录结构
            if self.mode == 'test' or self.mode == 'val':
                mode_name = 'val'
            else:
                mode_name = 'train'
        else:
            # CIFAR-10 支持 train/test
            mode_name = self.mode
            
        mode_dir = os.path.join(self.data_root, mode_name)
        
        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"数据集目录不存在: {mode_dir}")
        
        self.class_names = sorted([d for d in os.listdir(mode_dir) 
                                  if os.path.isdir(os.path.join(mode_dir, d))])  # 只获取目录

        # 根据数据集类型选择图像文件扩展名
        if self.dataset_type == 'imagenet100':
            valid_extensions = [".JPEG", ".jpg", ".jpeg", ".png", ".PNG"]
        else:
            valid_extensions = [".JPEG", ".jpg", ".jpeg", ".png", ".PNG"]

        # 遍历每个类别文件夹
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(mode_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for fname in os.listdir(class_dir):
                # 检查文件扩展名
                if any(fname.endswith(ext) for ext in valid_extensions):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self) -> int:
        """返回数据集样本数量"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        返回一个样本
        Returns:
            image: [C, H, W] 的 torch.Tensor
            label: 类别索引 int
        """
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")  # 打开图像并转换成 RGB
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # 返回一个默认的黑色图像
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # 应用变换（如 ToTensor、归一化）
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self) -> List[str]:
        """返回类别名称列表"""
        return self.class_names

    def get_num_classes(self) -> int:
        """返回类别数量"""
        return len(self.class_names)
    
    def get_dataset_info(self) -> dict:
        """返回数据集信息"""
        return {
            'dataset_type': self.dataset_type,
            'mode': self.mode,
            'num_classes': self.get_num_classes(),
            'num_samples': len(self.samples),
            'class_names': self.class_names[:10],  # 只显示前10个类别名
            'data_root': self.data_root
        }
