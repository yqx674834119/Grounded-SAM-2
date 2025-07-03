## Set up environment
if you don't have install cuda, you can install cuda 12.1, and set CUDA_HOME in .bashrc,  [cuda-12-1-0-download-archive](https://developer.nvidia.com/cuda-12-1-0-download-archive)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Set CUDA_HOME in .bashrc, 
```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
source ~/.bashrc
echo $CUDA_HOME
```

## Download checkpoint
### Download the pretrained SAM 2 checkpoints:
```bash
cd checkpoints
bash download_ckpts.sh
```
### Download the pretrained Grounding DINO checkpoints:
```bash
cd gdino_checkpoints
bash download_ckpts.sh
```
### setup conda environment
please save as a setup.sh, and run it.
```

```
```bash
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

```

### Configure Huggingface Endpoint
```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```