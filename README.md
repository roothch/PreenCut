# AI Video Editing Tool with Automatic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio Interface](https://img.shields.io/badge/Web%20UI-Gradio-FF4B4B.svg)](https://gradio.app/)

An intelligent video editing tool that automatically transcribes audio, generates subtitles, segments content using AI, and enables smart clipping based on semantic analysis.

![Demo Screenshot](docs/screenshot.png) *(Example screenshot - replace with your actual screenshot)*

## Key Features

- 🎙️ **Multi-model Speech Recognition**
  - Supports Faster-Whisper (with GPU acceleration)
  - Multiple model sizes (large-v2, large-v3, etc.)
  - Fallback to CPU when GPU unavailable

- ✂️ **AI-Powered Content Segmentation**
  - LLM-based semantic analysis of transcripts
  - Automatic video segmentation by topics
  - Customizable segmentation prompts

- 🖥️ **User-Friendly Interface**
  - Gradio web UI with progress tracking
  - Preview segments before export
  - Support for multiple file formats

- ⚙️ **Flexible Configuration**
  - Adjustable GPU/CPU utilization
  - Multiple output quality options
  - Extensible plugin architecture

## Installation

### Prerequisites
- Python 3.8+
- FFmpeg (`sudo apt install ffmpeg` on Linux)
- NVIDIA GPU with drivers (optional but recommended)

### Quick Start
```bash
git clone https://github.com/yourusername/ai-video-editor.git
cd ai-video-editor

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'

# Launch the application
python main.py
