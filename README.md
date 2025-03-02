# LLM Chat Application with Docker GPU Support

基于Docker的LLM聊天应用，支持GPU加速和联网搜索功能。

A Docker-based LLM chat application with GPU acceleration and internet search capabilities.

## Features | 功能特点

- Local large language model support | 支持本地加载大型语言模型
- Integrated DuckDuckGo search functionality | 集成DuckDuckGo搜索功能
- Chunk generation strategy support | 支持分块生成策略
- Performance statistics | 提供性能统计信息
- Complete Docker GPU environment setup | 完整的Docker GPU环境配置

## Requirements | 系统要求

- NVIDIA GPU | NVIDIA GPU 显卡
- Docker and Docker Compose | 已安装 Docker 和 Docker Compose
- NVIDIA Container Toolkit | 已安装 NVIDIA Container Toolkit

## Installation | 安装步骤

### 1. Install NVIDIA Container Toolkit | 安装 NVIDIA Container Toolkit

If you haven't installed NVIDIA Container Toolkit, follow these steps:

如果您尚未安装NVIDIA Container Toolkit，请按照以下步骤安装：

```bash
# Add NVIDIA package repository | 添加NVIDIA软件包仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Update package list and install | 更新软件包列表并安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker service | 重启Docker服务
sudo systemctl restart docker
```

### 2. Prepare Model Files | 准备模型文件

Place your model files in the `models` directory with the following structure:

将您的模型文件放置在项目根目录下的`models`文件夹中。模型目录结构应如下：

```
models/
  └── your-model-name/          # 您的模型名称
      └── 1/
          ├── config.json       # 模型配置文件
          ├── tokenizer.json    # 分词器配置文件
          ├── model-00001-of-00000N.safetensors  # 模型文件
          ├── model-00002-of-00000N.safetensors
          └── ...
```

### 3. Start the Application | 启动应用

Use the provided start script | 使用提供的启动脚本：

```bash
# Add execution permission | 添加执行权限
chmod +x start.sh

# Start the application | 启动应用
./start.sh
```

Or use Docker Compose directly | 或者直接使用Docker Compose命令：

```bash
docker-compose up --build
```

The application will be available at http://localhost:7860

应用将在 http://localhost:7860 上运行。

## Usage | 使用说明

1. Open http://localhost:7860 in your browser | 在浏览器中打开 http://localhost:7860
2. Select a model from the dropdown menu | 从下拉菜单中选择要使用的模型
3. Enter your question | 输入您的问题
4. Toggle internet search and chunk generation options as needed | 选择是否启用联网搜索和分块生成策略
5. Click submit to get the response | 点击提交按钮获取回答

## Configuration | 配置说明

### Model Settings | 模型配置

Adjust model parameters in `app.py` | 您可以通过修改`app.py`文件中的以下参数来调整模型配置：

```python
# Modify generation parameters | 修改生成参数
max_new_tokens=800  # Maximum number of tokens to generate | 调整生成的最大token数
temperature=0.7     # Sampling temperature | 调整随机性
top_k=50            # Top-k sampling | 调整采样范围
top_p=0.95          # Nucleus sampling | 调整核采样参数
```

### Docker Settings | Docker配置

Adjust Docker configuration in `docker-compose.yml` | 您可以通过修改`docker-compose.yml`文件来调整Docker配置：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specify GPU device | 指定使用的GPU设备
```

## License | 许可证

MIT