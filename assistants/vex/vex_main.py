"""VEX - Video Exploration Helper for scene detection and clip extraction."""

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
from typing import Any, Dict, List, Optional, Tuple

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.frame_timecode import FrameTimecode
except ImportError:
    print("Error: scenedetect not installed. pip install scenedetect")
    sys.exit(1)

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
        required = ["video_input_dir", "clip_output_dir", "metadata_output_dir"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing key: {key}")
            if "dir" in key:
                config[key] = str(Path(config[key]).expanduser())
                Path(config[key]).mkdir(parents=True, exist_ok=True)
        config["scenedetect_threshold"] = float(config.get("scenedetect_threshold", 30.0))
        config["min_scene_duration_sec"] = float(config.get("min_scene_duration_sec", 2.0))
        config["clip_padding_sec"] = float(config.get("clip_padding_sec", 0.5))
        return True
    except Exception as e:
        logging.error(f"Config error: {e}")
        return False


def setup_logging() -> None:
    level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    log_dir = Path(config["video_input_dir"]).parents[1] / "logs" / "vex"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "vex_assistant.log"),
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


def detect_scenes(video_path: str) -> Optional[List[Tuple[FrameTimecode, FrameTimecode]]]:
    logging.info(f"Detecting scenes: {video_path}")
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=config["scenedetect_threshold"]))
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()

        min_dur = config["min_scene_duration_sec"]
        fps = video.frame_rate
        min_frames = int(min_dur * fps)

        filtered = [
            (start, end) for start, end in scene_list
            if (end.get_frames() - start.get_frames()) >= min_frames
        ]
        logging.info(f"Scenes: {len(filtered)} (filtered from {len(scene_list)})")
        return filtered
    except Exception as e:
        logging.error(f"Scene detection error: {e}", exc_info=True)
        return None


def extract_clip(
    video_path: str,
    start_time: FrameTimecode,
    end_time: FrameTimecode,
    output_path: str,
) -> bool:
    global current_process
    fps = start_time.frame_rate
    pad_frames = int(config["clip_padding_sec"] * fps)
    start_frame = max(0, start_time.get_frames() - pad_frames)
    end_frame = end_time.get_frames() + pad_frames

    start_tc = FrameTimecode(start_frame, fps).get_timecode()
    end_tc = FrameTimecode(end_frame, fps).get_timecode()

    logging.info(f"Extracting clip: {output_path} ({start_tc} -> {end_tc})")

    # FIX BUG-TRIO-006: Use re-encoding instead of -c copy to avoid
    # corrupted clips at non-keyframe cut points. Added -preset fast
    # for reasonable speed with good quality.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", video_path,
        "-ss", start_tc,
        "-to", end_tc,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path,
    ]
    try:
        current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while current_process.poll() is None:
            if shutdown_requested.is_set():
                current_process.terminate()
                try:
                    current_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                current_process = None
                return False
            if paused.is_set():
                paused.wait(timeout=1)
            time.sleep(0.2)
        rc = current_process.returncode
        current_process = None
        if rc == 0:
            logging.info(f"Clip saved: {output_path}")
            return True
        logging.error(f"ffmpeg failed with code {rc}")
        return False
    except FileNotFoundError:
        logging.error("ffmpeg not found in PATH")
        return False
    except Exception as e:
        logging.error(f"Clip extraction error: {e}", exc_info=True)
        return False


def save_metadata(metadata: Dict[str, Any], output_path: str) -> None:
    try:
        Path(output_path).write_text(json.dumps(metadata, indent=4), encoding="utf-8")
        logging.info(f"Metadata saved: {output_path}")
    except IOError as e:
        logging.error(f"Metadata save error: {e}")


def _wait_for_file_stability(file_path: str, checks: int = 3, interval: float = 1.0) -> bool:
    """FIX BUG-TRIO-008: Wait for file size to stabilize before processing.
    
    This prevents processing partially-written files that trigger the
    file watcher before the copy/download is complete.
    """
    path = Path(file_path)
    if not path.exists():
        return False
    
    last_size = -1
    for i in range(checks):
        current_size = path.stat().st_size
        if current_size == last_size and current_size > 0:
            return True
        last_size = current_size
        time.sleep(interval)
    
    logging.warning(f"File {file_path} size not stable after {checks} checks, proceeding anyway")
    return True


def process_video_file(video_path: str) -> None:
    global current_task_info
    if shutdown_requested.is_set():
        return
    
    # FIX BUG-TRIO-008: Verify file integrity before processing
    if not _wait_for_file_stability(video_path):
        logging.warning(f"File not stable or empty: {video_path}")
        return
    
    task_name = f"Processing {os.path.basename(video_path)}"
    current_task_info = {"file_path": video_path, "stage": "starting"}
    if state_manager:
        state_manager.update_status("vex", status="active", task=task_name, progress=0.0)

    if paused.is_set():
        paused.wait()
    if shutdown_requested.is_set():
        return

    current_task_info["stage"] = "detecting scenes"
    scene_list = detect_scenes(video_path)
    if shutdown_requested.is_set():
        return
    if scene_list is None:
        if state_manager:
            state_manager.update_status("vex", status="error", task=f"Detection failed: {os.path.basename(video_path)}")
        current_task_info = {}
        return
    if not scene_list:
        logging.info(f"No scenes found in {video_path}")
        if state_manager:
            state_manager.update_status("vex", status="idle")
        current_task_info = {}
        return

    if state_manager:
        state_manager.update_status("vex", status="active", task=task_name, progress=0.2)

    current_task_info["stage"] = "extracting clips"
    base_name = Path(video_path).stem
    clip_dir = Path(config["clip_output_dir"])
    metadata_dir = Path(config["metadata_output_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)

    video_metadata = {
        "source_video": video_path,
        "total_scenes": len(scene_list),
        "clips": [],
    }

    total_clips = len(scene_list)
    extracted = 0
    for i, (start_tc, end_tc) in enumerate(scene_list):
        if paused.is_set():
            paused.wait()
        if shutdown_requested.is_set():
            return

        clip_name = f"{base_name}_scene_{i+1:03d}.mp4"
        clip_path = str(clip_dir / clip_name)
        success = extract_clip(video_path, start_tc, end_tc, clip_path)
        if success:
            extracted += 1
            video_metadata["clips"].append({
                "clip_filename": clip_name,
                "clip_path": clip_path,
                "scene_index": i + 1,
                "start_timecode": start_tc.get_timecode(),
                "end_timecode": end_tc.get_timecode(),
                "start_seconds": start_tc.get_seconds(),
                "end_seconds": end_tc.get_seconds(),
                "duration_seconds": end_tc.get_seconds() - start_tc.get_seconds(),
            })
            if vector_store and vector_store.is_ready():
                vector_store.add_document(
                    text=f"Video clip from {os.path.basename(video_path)}: scene {i+1}, "
                         f"duration {end_tc.get_seconds() - start_tc.get_seconds():.1f}s",
                    metadata={
                        "source": "vex",
                        "type": "video_clip",
                        "video_file": video_path,
                        "scene_index": i + 1,
                    },
                )
        if state_manager:
            progress = 0.2 + 0.7 * (extracted / total_clips)
            state_manager.update_status("vex", status="active", task=task_name, progress=progress)

    meta_path = str(metadata_dir / f"{base_name}_metadata.json")
    save_metadata(video_metadata, meta_path)
    if state_manager:
        state_manager.update_status("vex", status="idle")
    current_task_info = {}
    logging.info(f"Finished {video_path}: {extracted}/{total_clips} clips extracted.")


class VideoFileHandler(FileSystemEventHandler):
    def on_created(self, event: FileCreatedEvent):
        if event.is_directory:
            return
        path = event.src_path
        if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv")):
            logging.info(f"New video: {path}")
            process_video_file(path)


def main() -> None:
    global state_manager, checkpoint_manager, vector_store
    if not load_config():
        sys.exit(1)
    setup_logging()
    logging.info("VEX starting...")

    state_manager = StateManager()
    checkpoint_manager = CheckpointManager()
    vector_store = VectorStoreClient()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGUSR2, handle_signal)

    ckpt_id = os.environ.get("TRIO_CHECKPOINT_ID")
    if ckpt_id:
        logging.info(f"Resuming from checkpoint: {ckpt_id}")
        state_manager.update_status("vex", status="active", task="Resumed")
    else:
        state_manager.update_status("vex", status="idle")

    observer = Observer()
    handler = VideoFileHandler()
    video_dir = config["video_input_dir"]
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    observer.schedule(handler, path=video_dir, recursive=False)
    observer.start()
    logging.info(f"Watching: {video_dir}")

    try:
        while not shutdown_requested.is_set():
            if paused.is_set():
                if state_manager and state_manager.load_state("vex").status != "paused":
                    state_manager.update_status("vex", status="paused")
                paused.wait(timeout=1)
                continue
            else:
                if state_manager and state_manager.load_state("vex").status == "paused":
                    state_manager.update_status("vex", status="active")
            shutdown_requested.wait(timeout=2)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        if state_manager:
            state_manager.update_status("vex", status="idle")
        logging.info("VEX finished.")


if __name__ == "__main__":
    main()
