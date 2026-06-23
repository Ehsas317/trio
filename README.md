# Trio Project v2.0

Your Personal AI Ecosystem on Mac Mini M4. Three specialized AI assistants working together with a unified core framework, vector database knowledge sharing, and a modern web dashboard.

## What's New in v2.0

- **Modern Python packaging** with `pyproject.toml` and proper package structure
- **Upgraded dependencies** to latest stable versions (2025)
- **Official Ollama Python client** instead of raw HTTP requests
- **Modern ChromaDB v0.6.x** API compatibility
- **PySceneDetect v0.6.5+** API support
- **python-telegram-bot v21+** with modern async patterns
- **Flask 3.x** with dark-themed dashboard
- **Type hints** throughout with `from __future__ import annotations`
- **Pathlib** usage instead of string paths
- **Vector store query** command in NAMI Telegram bot
- **Improved error handling** and logging

## The Three Assistants

| Assistant | Name | Role |
|-----------|------|------|
| **NAMI** | Network and Machine Intelligence | Telegram bot for system control and monitoring |
| **RUSH** | Recording and Understanding Speech Helper | Audio transcription via whisper.cpp + Ollama summarization |
| **VEX** | Video Exploration Helper | Scene detection via PySceneDetect + ffmpeg clip extraction |

## Architecture

```
trio/
├── core/                    # Shared framework
│   ├── state_manager.py     # Assistant status tracking
│   ├── memory_manager.py    # System memory monitoring
│   ├── checkpoint_manager.py # Task checkpoint save/load
│   ├── controller.py        # Process lifecycle orchestration
│   └── vector_store.py      # ChromaDB + embeddings knowledge base
├── assistants/
│   ├── nami/                # Telegram control bot
│   ├── rush/                # Audio transcription & summarization
│   └── vex/                 # Video scene detection
├── dashboard/               # Flask web monitoring UI
└── launchd/                 # macOS scheduling configs
```

## Quick Start

### 1. Install Homebrew Dependencies

```bash
brew install python@3.11 ffmpeg git whisper.cpp ollama
```

### 2. Clone and Setup

```bash
git clone https://github.com/Ehsas317/trio.git ~/trio_project_m4
cd ~/trio_project_m4
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Assistants

Edit each assistant's config JSON (replace placeholders):

```bash
# NAMI - Add your Telegram bot token
nano assistants/nami/nami_config.json

# RUSH - Set whisper model path
nano assistants/rush/rush_config.json

# VEX - Adjust scene detection threshold if needed
nano assistants/vex/vex_config.json
```

### 4. Pull Ollama Model

```bash
ollama pull llama3:8b
```

### 5. Download Whisper Model

```bash
mkdir -p models/whisper
cd models/whisper
# Download a model (tiny.en, base.en, small.en, medium.en)
curl -L -o ggml-medium.en.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin"
```

### 6. Run

```bash
# Start each assistant individually
python assistants/nami/nami_main.py
python assistants/rush/rush_main.py
python assistants/vex/vex_main.py

# Or start the dashboard
python dashboard/app.py
```

### 7. Launchd Auto-Start (macOS)

```bash
# Update USERNAME in plist files first, then:
cp launchd/*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.user.trio.nami.plist
launchctl load ~/Library/LaunchAgents/com.user.trio.rush.plist
launchctl load ~/Library/LaunchAgents/com.user.trio.vex.plist
launchctl load ~/Library/LaunchAgents/com.user.trio.dashboard.plist
```

## NAMI Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/help` | List commands |
| `/status` | System & assistant status |
| `/run <cmd>` | Execute shell command (confirmed) |
| `/ls [path]` | List directory |
| `/cd <path>` | Change directory |
| `/read <file>` | Read file contents |
| `/control <assistant> <action>` | Start/stop/pause/resume assistants |
| `/query <text>` | Search vector knowledge base |

## Tech Stack

- **Python 3.11+** with modern type hints
- **whisper.cpp** for Metal-accelerated speech recognition
- **Ollama** for local LLM inference
- **ChromaDB** for vector embeddings knowledge sharing
- **python-telegram-bot v21+** for Telegram integration
- **Flask 3.x** for web dashboard
- **PySceneDetect v0.6.5+** for video scene detection
- **ffmpeg** for video clip extraction
- **launchd** for macOS process scheduling

## License

MIT License - see [LICENSE](LICENSE)
