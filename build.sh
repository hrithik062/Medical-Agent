#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Downloading required models/files..."
python code/main.py download-files

echo "Starting voice agent console..."
python code/main.py console
