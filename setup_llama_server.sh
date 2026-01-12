#!/bin/bash

#Script to setup Llama C++ server with Qwen 2.5 7B + 0.5B draft model
# Requires: llama.cpp installed with server binary

# Configuration
PORT=8081
HOST="127.0.0.1"
MAIN_MODEL="models/qwen2.5-7b-instruct-q4_k_m.gguf"
DRAFT_MODEL="models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
N_GPU_LAYERS=0  # CPU only (no GPU offloading)

# Download models if they don't exist
if [ ! -f "$MAIN_MODEL" ]; then
    echo "Downloading Qwen 2.5 7B model..."
    mkdir -p models
    # Example: download from HuggingFace
    # wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf -O "$MAIN_MODEL"
    echo "Please download the model manually and place it at: $MAIN_MODEL"
    exit 1
fi

if [ ! -f "$DRAFT_MODEL" ]; then
    echo "Downloading Qwen 2.5 0.5B draft model..."
    mkdir -p models
    # Example: download from HuggingFace
    # wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf -O "$DRAFT_MODEL"
    echo "Please download the draft model manually and place it at: $DRAFT_MODEL"
    exit 1
fi

# Start llama.cpp server with speculative decoding
echo "Starting Llama C++ server on $HOST:$PORT"
echo "Main model: $MAIN_MODEL"
echo "Draft model: $DRAFT_MODEL"

./llama-server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MAIN_MODEL" \
    --model-draft "$DRAFT_MODEL" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --threads 8 \
    --log-format text

# Notes:
# - N_GPU_LAYERS=0 means CPU-only (set to >0 to offload layers to GPU if available)
# - Speculative decoding parameters (n_min, n_max, p_min) are set per-request in Python code
# - This allows dynamic experimentation with different draft step sizes
# - Download models from: https://huggingface.co/Qwen
