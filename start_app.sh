#!/bin/bash

# Set environment variables for Apple Silicon optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ONNXRUNTIME_PROVIDER_NAMES=CoreMLExecutionProvider,CPUExecutionProvider

# Force ONNX Runtime to use CoreML
export ONNXRUNTIME_PROVIDER_NAMES=CoreMLExecutionProvider,CPUExecutionProvider

# Start the Flask application
python app.py 