#!/bin/bash
# ImageNet-100数据集手动解压脚本

echo "📦 ImageNet-100数据集手动解压脚本"
echo "=================================="

# 设置数据集路径
DATA_DIR="./data/imagenet100"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ 数据集目录不存在: $DATA_DIR"
    echo "💡 请确保已下载ImageNet-100数据集"
    exit 1
fi

cd "$DATA_DIR"
echo "📁 当前目录: $(pwd)"

# 列出现有文件
echo ""
echo "📋 当前文件列表:"
ls -la

# 检查压缩文件
echo ""
echo "🔍 检查压缩文件..."

TRAIN_FILES=(train.X*)
VAL_FILES=(val.X*)

echo "训练集文件: ${TRAIN_FILES[@]}"
echo "验证集文件: ${VAL_FILES[@]}"

# 解压训练集
if [[ -f "train.X1" ]]; then
    echo ""
    echo "📦 开始解压训练集..."
    
    # 尝试不同的解压方法
    if command -v 7z &> /dev/null; then
        echo "使用7z解压..."
        7z x train.X1
    elif command -v unzip &> /dev/null; then
        echo "使用unzip解压..."
        unzip train.X1
    elif command -v tar &> /dev/null; then
        echo "尝试tar解压..."
        tar -xf train.X1
    else
        echo "❌ 未找到解压工具"
        echo "💡 请安装 7zip, unzip 或 tar"
    fi
    
    if [[ -d "train" ]]; then
        echo "✅ 训练集解压成功"
    else
        echo "❌ 训练集解压失败"
    fi
else
    echo "⚠️ 未找到train.X1文件"
fi

# 解压验证集
if [[ -f "val.X" ]]; then
    echo ""
    echo "📦 开始解压验证集..."
    
    if command -v 7z &> /dev/null; then
        echo "使用7z解压..."
        7z x val.X
    elif command -v unzip &> /dev/null; then
        echo "使用unzip解压..."
        unzip val.X
    elif command -v tar &> /dev/null; then
        echo "尝试tar解压..."
        tar -xf val.X
    else
        echo "❌ 未找到解压工具"
    fi
    
    if [[ -d "val" ]]; then
        echo "✅ 验证集解压成功"
    else
        echo "❌ 验证集解压失败"
    fi
else
    echo "⚠️ 未找到val.X文件"
fi

# 检查解压结果
echo ""
echo "📊 解压结果检查:"
echo "=================================="

if [[ -d "train" ]]; then
    train_classes=$(find train -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✅ 训练集目录存在，类别数: $train_classes"
    
    # 显示前5个类别
    echo "📂 前5个训练类别:"
    find train -mindepth 1 -maxdepth 1 -type d | head -5 | while read dir; do
        class_name=$(basename "$dir")
        img_count=$(find "$dir" -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" | wc -l)
        echo "   $class_name: $img_count 张图片"
    done
else
    echo "❌ 训练集目录不存在"
fi

if [[ -d "val" ]]; then
    val_classes=$(find val -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✅ 验证集目录存在，类别数: $val_classes"
    
    # 显示前5个类别
    echo "📂 前5个验证类别:"
    find val -mindepth 1 -maxdepth 1 -type d | head -5 | while read dir; do
        class_name=$(basename "$dir")
        img_count=$(find "$dir" -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" | wc -l)
        echo "   $class_name: $img_count 张图片"
    done
else
    echo "❌ 验证集目录不存在"
fi

# 检查Labels.json
if [[ -f "Labels.json" ]]; then
    echo "✅ Labels.json 文件存在"
else
    echo "⚠️ Labels.json 文件不存在"
fi

echo ""
echo "📁 最终目录结构:"
ls -la

echo ""
if [[ -d "train" && -d "val" ]]; then
    echo "🎉 ImageNet-100数据集解压完成！"
    echo "💡 现在可以运行训练程序:"
    echo "   cd .."
    echo "   python main.py"
else
    echo "⚠️ 数据集解压不完整"
    echo "💡 可能需要手动处理压缩文件"
fi