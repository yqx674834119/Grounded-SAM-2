#!/bin/bash

# 脚本出错时立即退出
set -e

# Conda 环境名称
ENV_NAME="sam2"
PYTHON_VERSION="3.10"

echo ">>> 创建 Conda 环境: $ENV_NAME"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo ">>> 激活 Conda 环境"
# 激活 Conda 环境（注意: 在脚本中使用 conda activate 需要初始化 shell）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ">>> 安装 PyTorch + CUDA 12.1"
conda install -y pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo ">>> 安装 Segment Anything v2 (当前目录)"
pip install -e ".[notebook,dev,interactive-demo]"

echo ">>> 安装 Grounding DINO (位于 ./grounding_dino)"
pip install --no-build-isolation -e grounding_dino

echo "✅ 环境 $ENV_NAME 安装完成！"
