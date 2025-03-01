#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -i nvidia > /dev/null; then
    echo "Warning: NVIDIA Docker runtime not detected. GPU acceleration may not work."
    echo "Please ensure nvidia-container-toolkit is installed and configured."
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create models directory if it doesn't exist
mkdir -p ./models
mkdir -p ./data

echo "Starting LLM Chat Application with Docker Compose..."

# Function to get file modification time
get_mod_time() {
    stat -c %Y "$1" 2>/dev/null || echo "0"
}

# Store the last build time if image exists
LAST_BUILD_FILE=".last_build_time"

# Check if the image already exists
IMAGE_NAME=$(grep -m 1 "image:" docker-compose.yml | awk '{print $2}' || echo "")
SERVICE_NAME=$(grep -m 1 "services:" -A 1 docker-compose.yml | tail -n 1 | awk '{print $1}' | tr -d ':')

# Get current modification times
APP_MOD_TIME=$(get_mod_time "app.py")
DOCKER_MOD_TIME=$(get_mod_time "Dockerfile")
REQ_MOD_TIME=$(get_mod_time "requirements.txt")

# Read last build time
LAST_BUILD_TIME=0
if [ -f "$LAST_BUILD_FILE" ]; then
    LAST_BUILD_TIME=$(cat "$LAST_BUILD_FILE")
fi

# Check if any key files were modified since last build
NEEDS_REBUILD=false
if [ $APP_MOD_TIME -gt $LAST_BUILD_TIME ] || 
   [ $DOCKER_MOD_TIME -gt $LAST_BUILD_TIME ] || 
   [ $REQ_MOD_TIME -gt $LAST_BUILD_TIME ]; then
    NEEDS_REBUILD=true
fi

# Determine if we need to build
BUILD_NEEDED=false
if [ -z "$IMAGE_NAME" ]; then
    if [ -n "$SERVICE_NAME" ]; then
        IMAGE_EXISTS=$(docker images | grep "gradio-chat-local_$SERVICE_NAME" | wc -l)
        if [ "$IMAGE_EXISTS" -eq "0" ] || [ "$NEEDS_REBUILD" = true ]; then
            BUILD_NEEDED=true
        fi
    else
        BUILD_NEEDED=true
    fi
else
    IMAGE_EXISTS=$(docker images | grep "$IMAGE_NAME" | wc -l)
    if [ "$IMAGE_EXISTS" -eq "0" ] || [ "$NEEDS_REBUILD" = true ]; then
        BUILD_NEEDED=true
    fi
fi

# Build and start the container
if [ "$BUILD_NEEDED" = true ]; then
    echo "Building image (detected changes in application files)..."
    docker-compose up --build
    # Store current time as last build time
    date +%s > "$LAST_BUILD_FILE"
else
    echo "No changes detected in application files. Starting without rebuild..."
    docker-compose up
fi