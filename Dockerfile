# Use NVIDIA CUDA base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for GitHub Copilot integration
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && node --version \
    && npm --version

# Set Python aliases
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code (excluding models directory)
COPY app.py .
COPY copilot_proxy.py .
COPY copilot_config.json .
COPY requirements.txt .
COPY start.sh .
# Copy any other necessary files, but exclude models
COPY .gitignore .
COPY docker-compose.yml .

# Expose Gradio port
EXPOSE 7860

# Set default command
CMD ["python3", "app.py"]