#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p code

MODEL_PATH="code/emotion_model.onnx"
MODEL_URL="https://huggingface.co/onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX/resolve/main/onnx/model.onnx"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading emotion model..."
  curl -L "$MODEL_URL" -o "$MODEL_PATH"
  echo "Saved model to $MODEL_PATH"
else
  echo "Emotion model already exists. Skipping download."
fi

echo "Downloading required models/files..."
python code/main.py download-files

echo "Starting voice agent console..."
python code/main.py console
