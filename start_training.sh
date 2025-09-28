#!/bin/bash
# OverLoCK ImageNet-100训练脚本
# 支持ImageNet-100数据集的后台训练

echo "🚀 OverLoCK ImageNet-100模型训练脚本"
echo "=========================================="

# 1. 检查GPU是否可用
echo "📊 检查GPU状态..."
nvidia-smi
echo ""

# 2. 检查Python环境和依赖
echo "🐍 检查Python环境..."
python --version
echo ""

echo "📦 检查必要的Python包..."
python -c "
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA version: {torch.version.cuda}')
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('❌ PyTorch not found')

try:
    import torchvision
    print(f'✅ TorchVision: {torchvision.__version__}')
except ImportError:
    print('❌ TorchVision not found')

try:
    import kagglehub
    print('✅ KaggleHub available')
except ImportError:
    print('❌ KaggleHub not found - pip install kagglehub')

try:
    import tqdm
    print('✅ tqdm available')
except ImportError:
    print('❌ tqdm not found')

try:
    import PIL
    print('✅ PIL available')
except ImportError:
    print('❌ PIL not found')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError:
    print('❌ NumPy not found')
"
echo ""

# 3. 检查必要的文件
echo "📁 检查必要文件..."
files=(
    "main.py"
    "scalable_model.py" 
    "rtx4090_configs.py"
    "trainer.py"
    "dataset.py"
    "download_imagenet100.py"
    "imagenet100_configs.py"
    "model_configs.py"
)

missing_files=()
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ $file"
    else
        echo "❌ $file (缺失)"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "⚠️ 缺失文件，请先上传："
    for file in "${missing_files[@]}"; do
        echo "   scp -P 40908 c:\\OverLoCK\\$file lianghangchun@59.72.109.235:~/OverLoCK9/"
    done
    echo ""
    exit 1
fi

# 4. 检查Kaggle凭据
echo ""
echo "🔑 检查Kaggle凭据..."
if [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "✅ Kaggle凭据文件存在"
elif [[ -n "$KAGGLE_USERNAME" && -n "$KAGGLE_KEY" ]]; then
    echo "✅ Kaggle环境变量已设置"
else
    echo "❌ 未找到Kaggle凭据"
    echo "💡 请设置环境变量或上传kaggle.json文件:"
    echo "   export KAGGLE_USERNAME=isohean"
    echo "   export KAGGLE_KEY=d8a22c78fd752024036354877684d5b9"
    echo "   或者将kaggle.json复制到 ~/.kaggle/"
fi

# 5. 检查并下载ImageNet-100数据集
echo ""
echo "📂 检查ImageNet-100数据集..."
if [[ -d "./data/imagenet100" ]]; then
    echo "✅ ImageNet-100数据集已存在"
    echo "📊 数据集结构:"
    find ./data/imagenet100 -maxdepth 2 -type d | head -10
else
    echo "❌ ImageNet-100数据集不存在"
    echo "� 开始下载ImageNet-100数据集..."
    
    # 设置Kaggle凭据（如果未设置）
    if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
        export KAGGLE_USERNAME=isohean
        export KAGGLE_KEY=d8a22c78fd752024036354877684d5b9
        echo "🔑 已设置Kaggle凭据"
    fi
    
    # 安装kagglehub（如果未安装）
    python -c "import kagglehub" 2>/dev/null || pip install kagglehub
    
    # 下载数据集
    echo "⏳ 正在下载ImageNet-100数据集..."
    python download_imagenet100.py
    
    if [[ -d "./data/imagenet100" ]]; then
        echo "✅ ImageNet-100数据集下载完成"
    else
        echo "❌ 数据集下载失败"
        echo "💡 请手动运行: python download_imagenet100.py"
        exit 1
    fi
fi

# 6. 创建必要的目录
echo ""
echo "📁 创建必要目录..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
echo "✅ 目录创建完成"

# 7. 检查main.py中的数据集配置
echo ""
echo "🔍 检查main.py数据集配置..."
dataset_config=$(grep -n "dataset_type.*=" main.py | head -1)
if [[ $dataset_config == *"imagenet100"* ]]; then
    echo "✅ main.py已配置为使用ImageNet-100"
    echo "   $dataset_config"
else
    echo "⚠️ main.py未配置为ImageNet-100"
    echo "   当前配置: $dataset_config"
    echo "💡 请确保main.py中设置: dataset_type = \"imagenet100\""
fi

# 8. 开始训练
echo ""
echo "🚀 开始训练ImageNet-100 OverLoCK模型..."
echo "⏰ 开始时间: $(date)"
echo "📊 数据集: ImageNet-100 (100类, 224x224)"
echo "🎯 模型配置: RTX4090 Optimal (~65M参数)"
echo "=========================================="

# 使用nohup在后台运行，并重定向输出
nohup python main.py > logs/imagenet100_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
training_pid=$!

echo "✅ ImageNet-100训练已在后台启动"
echo "📊 进程ID: $training_pid"
echo "📋 日志文件: logs/imagenet100_training_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "📖 常用监控命令:"
echo "   查看训练日志: tail -f logs/imagenet100_training_*.log"
echo "   查看GPU使用: watch -n 1 nvidia-smi"
echo "   查看进程状态: ps aux | grep python"
echo "   停止训练: kill $training_pid"
echo ""
echo "🎯 ImageNet-100预期训练时间:"
echo "   OPTIMAL配置: 4-6小时 (100 epochs)"
echo "   MAX配置: 8-12小时 (100 epochs)"
echo ""
echo "📈 预期性能指标:"
echo "   Top-1 准确率: 70-80%"
echo "   训练样本: ~50,000张"
echo "   验证样本: ~5,000张"