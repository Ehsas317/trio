#!/bin/bash
# Trio Project Setup Script for Mac Mini M4
set -e

echo "========================================="
echo "  Trio AI Ecosystem - Setup Script"
echo "  Optimized for Mac Mini M4"
echo "========================================="
echo ""

PROJECT_DIR="$HOME/trio_project_m4"
REPO_URL="https://github.com/Ehsas317/trio.git"

# Check macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Warning: This script is designed for macOS."
fi

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "Installing system dependencies..."
brew install python@3.11 ffmpeg git whisper.cpp ollama

# Create project directory
echo "Setting up project at $PROJECT_DIR..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory exists. Updating..."
    cd "$PROJECT_DIR"
    git pull || true
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Create directory structure
mkdir -p checkpoints/{nami,rush,vex}
mkdir -p logs/{nami,rush,vex,dashboard,controller}
mkdir -p models/whisper models/llm
mkdir -p shared/state
mkdir -p vector_store
mkdir -p assistants/rush/{audio_input,transcripts,summaries}
mkdir -p assistants/vex/{video_input,clips_output,metadata_output}
mkdir -p assistants/nami/{cache,downloads}

# Download whisper model if not present
WHISPER_MODEL="$PROJECT_DIR/models/whisper/ggml-medium.en.bin"
if [ ! -f "$WHISPER_MODEL" ]; then
    echo "Downloading whisper model..."
    mkdir -p models/whisper
    curl -L --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin" \
        -o "$WHISPER_MODEL" || echo "Download failed. Please download manually."
fi

# Pull Ollama model
echo "Pulling Ollama model..."
ollama pull llama3:8b || echo "Ollama pull failed. Please run 'ollama pull llama3:8b' manually."

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit config files:"
echo "   nano $PROJECT_DIR/assistants/nami/nami_config.json"
echo ""
echo "2. Activate virtual environment:"
echo "   source $PROJECT_DIR/.venv/bin/activate"
echo ""
echo "3. Run an assistant:"
echo "   python $PROJECT_DIR/assistants/nami/nami_main.py"
echo ""
echo "4. Or setup launchd auto-start:"
echo "   cp $PROJECT_DIR/launchd/*.plist ~/Library/LaunchAgents/"
echo "   # Remember to update USERNAME in the plist files!"
echo ""
