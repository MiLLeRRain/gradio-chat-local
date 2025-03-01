# LLM Chat Application with Docker GPU Support
# 基于Docker的LLM聊天应用（支持GPU）

This project provides a Docker-based LLM chat application with GPU acceleration and internet search capabilities.

这个项目提供了一个基于Docker的LLM聊天应用，支持GPU加速和联网搜索功能。

## Features | 功能特点

- Support for local loading of large language models | 支持本地加载大型语言模型
- Integrated DuckDuckGo search functionality | 集成DuckDuckGo搜索功能
- Support for chunk generation strategy | 支持分块生成策略
- Performance statistics | 提供性能统计信息
- Complete Docker GPU environment configuration | 完整的Docker GPU环境配置

## System Requirements | 系统要求

- NVIDIA GPU | NVIDIA GPU 显卡
- Docker and Docker Compose installed | 已安装 Docker 和 Docker Compose
- NVIDIA Container Toolkit installed | 已安装 NVIDIA Container Toolkit

## Installation Steps | 安装步骤

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

Place your model files in the `models` folder in the project root directory. The model directory structure should be as follows:

将您的模型文件放置在项目根目录下的`models`文件夹中。模型目录结构应如下：

```
models/
  └── deepseek-r1-distill-qwen-7b/
      └── 1/
          ├── config.json
          ├── tokenizer.json
          ├── model-00001-of-000002.safetensors
          ├── model-00002-of-000002.safetensors
          └── ...
```

### 3. Launch Application | 启动应用

Use the provided startup script to run the application:

使用提供的启动脚本运行应用：

```bash
# Add execution permissions | 添加执行权限
chmod +x start.sh

# Start the application | 启动应用
./start.sh
```

Or directly use Docker Compose command:

或者直接使用Docker Compose命令：

```bash
docker-compose up --build
```

The application will run at http://localhost:7860

应用将在 http://localhost:7860 上运行。

## Usage Guide | 使用说明

1. Open http://localhost:7860 in your browser | 在浏览器中打开 http://localhost:7860
2. Select the model you want to use from the dropdown menu | 从下拉菜单中选择要使用的模型
3. Enter your question | 输入您的问题
4. Choose whether to enable internet search and chunk generation strategy | 选择是否启用联网搜索和分块生成策略
5. Click the submit button to get an answer | 点击提交按钮获取回答

## Configuration | 配置说明

### Model Configuration | 模型配置

You can adjust model parameters by modifying the following parameters in the `app.py` file:

您可以通过修改`app.py`文件中的以下参数来调整模型配置：

```python
# Modify generation parameters | 修改生成参数
max_new_tokens=800  # Adjust maximum token count | 调整生成的最大token数
temperature=0.7     # Adjust randomness | 调整随机性
top_k=50            # Adjust sampling range | 调整采样范围
top_p=0.95          # Adjust nucleus sampling parameter | 调整核采样参数
```

### Docker Configuration | Docker配置

You can adjust Docker settings by modifying the `docker-compose.yml` file:

您可以通过修改`docker-compose.yml`文件来调整Docker配置：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specify GPU device to use | 指定使用的GPU设备
```

## Troubleshooting | 故障排除

### GPU Not Available | GPU不可用

If you encounter GPU unavailability issues, please check:

如果遇到GPU不可用的问题，请检查：

1. Confirm NVIDIA drivers are correctly installed | 确认NVIDIA驱动已正确安装
2. Confirm NVIDIA Container Toolkit is correctly installed | 确认NVIDIA Container Toolkit已正确安装
3. Confirm Docker service has been restarted | 确认Docker服务已重启
4. Use `nvidia-smi` command to check GPU status | 使用`nvidia-smi`命令检查GPU状态

### Model Loading Failed | 模型加载失败

If model loading fails, please check:

如果模型加载失败，请检查：

1. Confirm model file structure is correct | 确认模型文件结构正确
2. Confirm model files are complete | 确认模型文件完整
3. Check specific error messages in the logs | 检查日志中的具体错误信息

## License | 许可证

MIT