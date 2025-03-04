# LLM Chat Application with Docker GPU Support

A Docker-based LLM chat application with GPU acceleration and internet search capabilities.

## Features

- Local large language model support
- Integrated DuckDuckGo search functionality
- Chunk generation strategy support
- Performance statistics
- Complete Docker GPU environment setup

## Requirements

- NVIDIA GPU
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd llm-chat-local
```

### 2. Prepare Model Files

Place your model files in the `models` directory with the following structure:

```
models/
  └── your-model-name/
      └── 1/
          ├── config.json
          ├── tokenizer.json
          └── model files
```

### 3. Start the Application

Use the provided start script:

```bash
chmod +x start.sh
./start.sh
```

Or use Docker Compose directly:

```bash
docker-compose up --build
```

The application will be available at http://localhost:7860

## Usage

1. Open http://localhost:7860 in your browser
2. Select a model from the dropdown menu
3. Enter your question
4. Toggle internet search and chunk generation options as needed
5. Click submit to get the response

## Configuration

### Docker Settings

Adjust Docker configuration in `docker-compose.yml`:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # Specify GPU device
```

## License

MIT