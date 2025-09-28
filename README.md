# OverLoCK - Computer Vision Work

## Overview
This repository contains the implementation of OverLoCK (Overview-first-Look-Closely-next) ConvNet with Context-Mixing Dynamic Kernels for computer vision tasks.

## Features
- OverLoCK architecture with BaseNet, OverviewNet, and FocusNet components
- Multi-GPU training support (RTX 4090 optimized)
- ImageNet-100 dataset support
- Comprehensive visualization tools (ERF, GradCAM, performance metrics)
- Mixed precision training
- Advanced data augmentation

## Model Architecture
- **BaseNet**: Feature extraction backbone
- **OverviewNet**: Global context understanding
- **FocusNet**: Fine-grained detail analysis
- **FPN**: Feature Pyramid Network for multi-scale features
- **CBAM**: Convolutional Block Attention Module

## Training Configuration
- Learning Rate: 4e-3 (paper recommended)
- Weight Decay: 0.05
- Batch Size: 32 (single GPU) / 64 effective (dual GPU)
- Mixed Precision: Enabled
- Gradient Clipping: 0.5

## Files Structure

### Core Model Files
- `main.py`: Main training script
- `model.py`: OverLoCK model implementation  
- `scalable_model.py`: Scalable model wrapper
- `trainer.py`: Training and validation logic

### Dataset & Configuration
- `dataset.py`: Dataset handling (CIFAR-10, ImageNet-100)
- `model_configs.py`: Base model architecture configurations
- `rtx4090_configs.py`: RTX 4090 optimized configurations
- `imagenet100_configs.py`: ImageNet-100 specific configurations

### Utilities & Tools
- `model_visualizer.py`: Visualization and evaluation tools (ERF, GradCAM)
- `confidence_calibration.py`: Model calibration utilities
- `inference.py`: Model inference and evaluation
- `run_calibration.py`: Calibration execution script

### Setup & Data
- `download_imagenet100.py`: ImageNet-100 dataset downloader
- `extract_imagenet100.sh`: Dataset extraction script
- `start_training.sh`: Training startup script

### Documentation
- `README.md`: Project documentation
- `参考论文/`: Reference papers

## Usage
```bash
# Start training
./start_training.sh

# Or run directly
python main.py
```

## Requirements
- PyTorch >= 1.12
- CUDA >= 11.6
- torchvision
- matplotlib
- tqdm
- PIL

## Performance
- Model Parameters: ~65M (optimal) / ~144M (max)
- RTX 4090 Memory Usage: ~16GB (optimal) / ~22GB (max)
- Multi-GPU Support: 2x RTX 4090 D
- Training Speed: ~2x improvement with dual GPU

## Visualization Features
- Effective Receptive Field (ERF) analysis
- GradCAM class activation maps
- Performance metrics (Top-1/Top-5 accuracy, throughput)
- Training curves and loss visualization

## Paper Reference
Based on "OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels" (CVPR 2025)