"""RUSH - Recording and Understanding Speech Helper."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ollama
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

CORE_DIR = Path(__file__).resolve().parents[2] / "core"
sys.path.insert(0, str(CORE_DIR.parent))

from core.checkpoint_manager import CheckpointManager
from core.state_manager import StateManager
from core.vector_store import VectorStoreClient

CONFIG_FILE = Path(__file__).with_suffix(".json")
config: Dict[str, Any] = {}

paused = threading.Event()
shutdown_requested = threading.Event()
current_process: Optional[subprocess.Popen] = None
current_task_info: Dict[str, Any] = {}
state_manager: Optional[StateManager] = None
checkpoint_manager: Optional[CheckpointManager] = None
vector_store: Optional[VectorStoreClient] = None


def load_config() -> bool:
    global config
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        required = ["whisper_model_path", "ollama_model_name", "audio_input_dir",
                    "transcript_output_dir", "summary_output_dir"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing config key: {key}")
            if "dir" in key or "path" in key:
                config[key] = str(Path(config[key]).expanduser())
        Path(config["transcript_output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["summary_output_dir"]).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Config error: {e}")
        return False


def setup_logging() -> None:
    level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    log_dir = Path(config.get("audio_input_dir", ".")).parents[1] / "logs" / "rush"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "rush_assistant.log"),
            logging.StreamHandler(),
        ],
    )


def handle_signal(sig: int, _frame) -> None:
    global current_process
    if sig == signal.SIGUSR1:
        logging.info("Pause signal received.")
        paused.set()
        if current_process and current_process.poll() is None:
            try:
                current_process.send_signal(signal.SIGSTOP)
            except Exception as e:
                logging.error(f"SIGSTOP error: {e}")
    elif sig == signal.SIGUSR2:
        logging.info("Resume signal received.")
        if current_process and current_process.poll() is None:
            try:
                current_process.send_signal(signal.SIGCONT)
            except Exception as e:
                logging.error(f"SIGCONT error: {e}")
        paused.clear()
    elif sig in (signal.SIGINT, signal.SIGTERM):
        logging.info(f"Shutdown signal {sig} received.")
        shutdown_requested.set()
        paused.clear()
        if current_process and current_process.poll() is None:
            current_process.terminate()
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current_process.kill()


def transcribe_audio(audio_path: str) -> Optional[str]:
    global current_process
    logging.info(f"Transcribing: {audio_path}")
    base_name = Path(audio_path).stem
    out_dir = Path(config["transcript_output_dir"])
    out_base = out_dir / base_name
    transcript_path = out_base.with_suffix(".txt")

    whisper_exec = config.get("whisper_executable", "whisper")
    cmd = [
        whisper_exec,
        "-m", config["whisper_model_path"],
        "-l", config.get("language", "en"),
        "-t", str(config.get("whisper_processing_threads", 4)),
        "-otxt",
        "-of", str(out_base),
        "-f", audio_path,
    ]
    if config.get("use_gpu", True):
        cmd.extend(["-ngl", "1"])

    try:
        current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info(f"Whisper PID: {current_process.pid}")
        while current_process.poll() is None:
            if shutdown_requested.is_set():
                current_process.terminate()
                try:
                    current_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                current_process = None
                return None
            if paused.is_set():
                paused.wait(timeout=1)
            time.sleep(0.5)
        rc = current_process.returncode
        current_process = None
        if rc == 0 and transcript_path.exists():
            logging.info(f"Transcription done: {transcript_path}")
            return str(transcript_path)
        logging.error(f"Transcription failed, exit code: {rc}")
        return None
    except FileNotFoundError:
        logging.error(f"Whisper executable not found: {whisper_exec}")
        return None
    except Exception as e:
        logging.error(f"Transcription error: {e}", exc_info=True)
        return None


def summarize_text(text: str) -> Optional[str]:
    model_name = config.get("ollama_model_name", "llama3:8b")
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    try:
        response = ollama.generate(model=model_name, prompt=prompt, stream=False)
        summary = response.get("response", "").strip()
        return summary if summary else None
    except Exception as e:
        logging.error(f"Ollama summarization error: {e}")
        return None


def save_text(text: str, output_path: str) -> None:
    try:
        Path(output_path).write_text(text, encoding="utf-8")
        logging.info(f"Saved: {output_path}")
    except IOError as e:
        logging.error(f"Save error: {e}")


def process_audio_file(audio_path: str) -> None:
    global current_task_info
    if shutdown_requested.is_set():
        return
    task_name = f"Processing {os.path.basename(audio_path)}"
    current_task_info = {"file_path": audio_path, "stage": "starting"}
    if state_manager:
        state_manager.update_status("rush", status="active", task=task_name, progress=0.0)

    # Transcription
    current_task_info["stage"] = "transcribing"
    transcript_file = transcribe_audio(audio_path)
    if shutdown_requested.is_set():
        return
    if not transcript_file:
        if state_manager:
            state_manager.update_status("rush", status="error", task=f"Transcription failed: {os.path.basename(audio_path)}")
        current_task_info = {}
        return
    if state_manager:
        state_manager.update_status("rush", status="active", task=task_name, progress=0.5)

    # Summarization
    current_task_info["stage"] = "summarizing"
    try:
        transcript_text = Path(transcript_file).read_text(encoding="utf-8")
    except IOError as e:
        logging.error(f"Read error: {e}")
        if state_manager:
            state_manager.update_status("rush", status="error", task=f"Read failed: {transcript_file}")
        current_task_info = {}
        return

    summary = summarize_text(transcript_text)
    if summary:
        summary_path = Path(config["summary_output_dir"]) / f"{Path(audio_path).stem}_summary.txt"
        save_text(summary, str(summary_path))
        if vector_store and vector_store.is_ready():
            vector_store.add_document(
                text=summary,
                metadata={"source": "rush", "type": "summary", "file": audio_path},
            )
            vector_store.add_document(
                text=transcript_text[:2000],
                metadata={"source": "rush", "type": "transcript", "file": audio_path},
            )
    else:
        logging.warning("Summarization returned no output.")

    if state_manager:
        state_manager.update_status("rush", status="idle")
    current_task_info = {}
    logging.info(f"Finished: {audio_path}")


class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        path = event.src_path
        if path.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
            logging.info(f"New audio file: {path}")
            process_audio_file(path)


def main() -> None:
    global state_manager, checkpoint_manager, vector_store
    if not load_config():
        sys.exit(1)
    setup_logging()
    logging.info("RUSH starting...")

    state_manager = StateManager()
    checkpoint_manager = CheckpointManager()
    vector_store = VectorStoreClient()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGUSR2, handle_signal)

    # Check for checkpoint resume
    ckpt_id = os.environ.get("TRIO_CHECKPOINT_ID")
    if ckpt_id:
        logging.info(f"Resuming from checkpoint: {ckpt_id}")
        state_manager.update_status("rush", status="active", task="Resumed")
    else:
        state_manager.update_status("rush", status="idle")

    observer = Observer()
    handler = AudioFileHandler()
    audio_dir = config["audio_input_dir"]
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    observer.schedule(handler, path=audio_dir, recursive=False)
    observer.start()
    logging.info(f"Watching: {audio_dir}")

    try:
        while not shutdown_requested.is_set():
            if paused.is_set():
                if state_manager and state_manager.load_state("rush").status != "paused":
                    state_manager.update_status("rush", status="paused")
                paused.wait(timeout=1)
                continue
            else:
                if state_manager and state_manager.load_state("rush").status == "paused":
                    state_manager.update_status("rush", status="active")
            shutdown_requested.wait(timeout=2)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        if state_manager:
            state_manager.update_status("rush", status="idle")
        logging.info("RUSH finished.")


if __name__ == "__main__":
    main()
