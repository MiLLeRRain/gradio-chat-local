# 混合对话生成系统 (Hybrid Chat Generation System)

A sophisticated chat application that combines local large language models with API services, featuring internet search capabilities, customizable UI, and detailed performance monitoring.

## ✨ Features

- **Local Model Support**: Run AI models locally on your own hardware
- **API Integration**: Connect to DeepSeek API for cloud-based inference
- **Internet Search**: Incorporate DuckDuckGo search results into responses
- **Chunk Generation**: Break large responses into manageable chunks
- **Performance Monitoring**: Real-time GPU statistics and token generation metrics
- **Resizable UI**: Customizable interface with dopamine-inducing color scheme
- **Authentication**: Secure access with username/password protection

## 🖥️ System Requirements

- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **Docker**: Docker and Docker Compose for containerized deployment
- **NVIDIA Tools**: NVIDIA Container Toolkit (nvidia-docker)
- **Storage**: At least 10GB for model storage (varies by model size)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gradio-chat-local.git
cd gradio-chat-local
```

### 2. Prepare Model Files

Place your model files in the `models` directory with the following structure:

```
models/
  └── your-model-name/          # Main model folder
      └── transformers/         # For transformers format models
          ├── config.json
          ├── tokenizer.json
          └── model files...
```

Alternatively, use numbered version subdirectories:

```
models/
  └── your-model-name/
      └── 1/                    # Version number
          ├── config.json
          ├── tokenizer.json
          └── model files...
```

### 3. Configure API Access (Optional)

Create an `api_config.json` file in the root directory:

```json
{
  "api_models": ["deepseek-chat", "deepseek-reasoner"],
  "api_keys": {
    "deepseek": "your_deepseek_api_key"
  },
  "auth": {
    "username": "admin",
    "password": "your_secure_password"
  }
}
```

### 4. Start the Application

Using Docker:

```bash
docker-compose up --build
```

Or run directly with Python:

```bash
pip install -r requirements.txt
python app.py
```

The application will be available at http://localhost:7860

## 🎮 Usage Guide

### Accessing the Interface

1. Navigate to http://localhost:7860 in your browser
2. Log in with your username and password (default: admin/password)

### Basic Operation

1. Select the model type (Local or API)
2. Choose a specific model from the dropdown menu
3. Type your question in the input field
4. Toggle options as needed:
   - Enable internet search for up-to-date information
   - Enable chunk generation for longer responses
5. Click "Submit" or press Enter to get your response

### Advanced Features

- **System Prompt**: When using API models, you can provide a system prompt to guide the model
- **GPU Monitoring**: View real-time GPU statistics in the interface
- **Response Stats**: See token count, generation time, and speed metrics
- **Resizable Chat History**: Drag the bottom of the chat window to resize

## ⚙️ Configuration Options

### Docker Settings

Adjust Docker configuration in `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Specify GPU devices
  - NVIDIA_VISIBLE_DEVICES=all
```

### Application Settings

Key configuration files:

- `api_config.json`: API keys and authentication settings
- `style.css`: UI customization and theming
- `custom.js`: Additional JavaScript functionality

## 🔧 Troubleshooting

### Common Issues

- **GPU Not Detected**: Ensure CUDA drivers are installed and working
- **Model Loading Errors**: Check model path structure and permissions
- **API Connection Failures**: Verify your API key and internet connection

### Support

For issues or feature requests, please open an issue on the GitHub repository.

## 📜 License

MIT License

## 🙏 Acknowledgements

- [Gradio](https://www.gradio.app/) for the web interface framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model handling
- [DuckDuckGo](https://duckduckgo.com/) for search functionality