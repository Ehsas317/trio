"""NAMI - Network and Machine Intelligence assistant via Telegram."""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Core imports
CORE_DIR = Path(__file__).resolve().parents[2] / "core"
sys.path.insert(0, str(CORE_DIR.parent))

from core.controller import ModelController
from core.memory_manager import MemoryManager
from core.state_manager import StateManager
from core.vector_store import VectorStoreClient

CONFIG_FILE = Path(__file__).with_suffix(".json")
CONFIRM_EXECUTE = 0
pending_commands: Dict[int, str] = {}
config: Dict[str, Any] = {}


def load_config() -> bool:
    global config
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if not config.get("telegram_token"):
            raise ValueError("telegram_token is required")
        if not isinstance(config.get("allowed_user_ids"), list):
            raise ValueError("allowed_user_ids must be a list")
        config["project_base_dir"] = str(Path(config.get("project_base_dir", "~")).expanduser())
        return True
    except Exception as e:
        logging.error(f"Config error: {e}")
        return False


def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, **kwargs):
        user_id = update.effective_user.id
        if user_id not in config.get("allowed_user_ids", []):
            logging.warning(f"Unauthorized access from user {user_id}")
            await update.message.reply_text("Not authorized.")
            return
        return await func(update, context, **kwargs)
    return wrapped


# ---- Command Handlers ----

@restricted
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Hello {update.effective_user.first_name}! I am NAMI, your Mac Mini M4 controller.\n"
        "Type /help for available commands."
    )

@restricted
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "*NAMI Commands:*\n"
        "/start - Welcome message\n"
        "/help - This help\n"
        "/status - System & assistant status\n"
        "/run <cmd> - Execute shell command (with confirmation)\n"
        "/ls [path] - List directory\n"
        "/cd <path> - Change directory\n"
        "/pwd - Show current directory\n"
        "/read <file> [lines] - Read file content\n"
        "/control <assistant> <action> - Control assistants (start/stop/pause/resume)\n"
        "/query <text> - Search vector knowledge base"
    )
    await update.message.reply_text(text, parse_mode="Markdown")

@restricted
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        mem = MemoryManager()
        sys_mem = mem.get_system_memory_info()
        pressure = mem.get_memory_pressure_macos()
        state_mgr = StateManager()
        states = state_mgr.get_all_states()

        lines = ["*System Status:*"]
        if "error" not in sys_mem:
            lines.append(f"  RAM: {sys_mem['used_mb']} MB / {sys_mem['total_mb']} MB ({sys_mem['percent_used']}%)")
        if "error" not in pressure:
            lines.append(f"  Wired: {pressure.get('wired_mb', 'N/A')} MB | Active: {pressure.get('active_mb', 'N/A')} MB | Compressed: {pressure.get('compressed_mb', 'N/A')} MB")

        lines.append("\n*Assistants:*")
        for name, st in states.items():
            line = f"  *{name.upper()}*: {st.status.capitalize()}"
            if st.current_task:
                line += f" (Task: {st.current_task}, Progress: {st.task_progress:.0%})"
            lines.append(line)
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Status error: {e}", exc_info=True)
        await update.message.reply_text(f"Error: {e}")


# ---- Shell command with confirmation ----

@restricted
async def cmd_run_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    command_text = update.message.text.partition(" ")[2].strip()
    if not command_text:
        await update.message.reply_text("Usage: /run <command>")
        return ConversationHandler.END
    pending_commands[chat_id] = command_text
    await update.message.reply_text(
        f"Execute this command?\n`{command_text}`\n\nReply **yes** to confirm, **no** to cancel.",
        parse_mode="Markdown",
    )
    return CONFIRM_EXECUTE

async def cmd_run_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    command_text = pending_commands.pop(chat_id, None)
    reply_text = update.message.text.strip().lower()
    if reply_text not in ("yes", "y"):
        await update.message.reply_text("Command cancelled.")
        return ConversationHandler.END
    if not command_text:
        await update.message.reply_text("Command not found.")
        return ConversationHandler.END
    await update.message.reply_text("Executing...")
    try:
        cwd = context.user_data.get("cwd", str(Path.home()))
        args = shlex.split(command_text)
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=30, cwd=cwd
        )
        output = (result.stdout or "")[:3000]
        err_output = (result.stderr or "")[:3000]
        text = f"*Command:* `{command_text}`\n*Exit code:* {result.returncode}\n"
        if output:
            text += f"*Output:*\n```\n{output}\n```\n"
        if err_output:
            text += f"*Stderr:*\n```\n{err_output}\n```"
        if not output and not err_output:
            text += "_(No output)_"
        await update.message.reply_text(text, parse_mode="Markdown")
    except subprocess.TimeoutExpired:
        await update.message.reply_text("Command timed out after 30s.")
    except FileNotFoundError:
        await update.message.reply_text("Command not found.")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
    return ConversationHandler.END

async def cmd_run_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending_commands.pop(update.message.chat_id, None)
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ---- File system commands ----

def _get_cwd(context: ContextTypes.DEFAULT_TYPE, new_path: Optional[str] = None) -> str:
    base_dir = str(Path.home())
    current = context.user_data.get("cwd", base_dir)
    if new_path is None:
        return current
    proposed = str(Path(current) / Path(new_path).expanduser())
    proposed = str(Path(proposed).resolve())
    if proposed.startswith(base_dir) and os.path.isdir(proposed):
        context.user_data["cwd"] = proposed
        return proposed
    return f"Error: Invalid path: {proposed}"

@restricted
async def cmd_pwd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"`{_get_cwd(context)}`", parse_mode="Markdown")

@restricted
async def cmd_cd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = update.message.text.partition(" ")[2].strip()
    if not path:
        await update.message.reply_text("Usage: /cd <path>")
        return
    result = _get_cwd(context, path)
    if result.startswith("Error:"):
        await update.message.reply_text(result)
    else:
        await update.message.reply_text(f"Changed to: `{result}`", parse_mode="Markdown")

@restricted
async def cmd_ls(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path_arg = update.message.text.partition(" ")[2].strip()
    cwd = _get_cwd(context)
    target = str(Path(cwd) / Path(path_arg).expanduser()) if path_arg else cwd
    base_dir = str(Path.home())
    if not target.startswith(base_dir):
        await update.message.reply_text("Access denied.")
        return
    try:
        if not os.path.isdir(target):
            await update.message.reply_text(f"Not a directory: `{target}`", parse_mode="Markdown")
            return
        entries = sorted(os.listdir(target))[:50]
        if not entries:
            await update.message.reply_text("(empty directory)")
            return
        lines = []
        for e in entries:
            full = os.path.join(target, e)
            lines.append(f"{'📁' if os.path.isdir(full) else '📄'} {e}{'/' if os.path.isdir(full) else ''}")
        msg = f"*Contents of* `{target}`:\n\n" + "\n".join(lines)
        if len(os.listdir(target)) > 50:
            msg += "\n\n_(truncated)_"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

@restricted
async def cmd_read(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = update.message.text.partition(" ")[2].strip()
    parts = shlex.split(args) if args else []
    if not parts:
        await update.message.reply_text("Usage: /read <file> [lines]")
        return
    file_path = parts[0]
    num_lines = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 25
    cwd = _get_cwd(context)
    target = str(Path(cwd) / Path(file_path).expanduser())
    if not target.startswith(str(Path.home())):
        await update.message.reply_text("Access denied.")
        return
    try:
        if not os.path.isfile(target):
            await update.message.reply_text("File not found.")
            return
        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total = len(lines)
        shown = lines[:num_lines]
        content = "".join(shown)
        msg = f"*{os.path.basename(target)}* ({num_lines}/{total} lines):\n```\n{content}\n```"
        if total > num_lines:
            msg += "\n\n_(truncated)_"
        if len(msg) > 4000:
            msg = msg[:3950] + "\n\n...(truncated)"
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error reading file: {e}")


# ---- Assistant control ----

@restricted
async def cmd_control(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = update.message.text.partition(" ")[2].strip()
    parts = shlex.split(args) if args else []
    if len(parts) != 2:
        await update.message.reply_text("Usage: /control <assistant> <start|stop|pause|resume>")
        return
    name, action = parts[0].lower(), parts[1].lower()
    valid = {"nami", "rush", "vex"}
    valid_actions = {"start", "stop", "pause", "resume"}
    if name not in valid or action not in valid_actions:
        await update.message.reply_text(f"Invalid. Assistants: {', '.join(valid)} | Actions: start, stop, pause, resume")
        return
    ctrl: ModelController = context.application.bot_data.get("controller")
    if not ctrl:
        await update.message.reply_text("Controller not available.")
        return
    ctrl.send_command({"action": action, "assistant": name})
    await update.message.reply_text(f"Sent '{action}' to {name.upper()}.")


# ---- Vector DB query ----

@restricted
async def cmd_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query_text = update.message.text.partition(" ")[2].strip()
    if not query_text:
        await update.message.reply_text("Usage: /query <search text>")
        return
    vs: VectorStoreClient = context.application.bot_data.get("vector_store")
    if not vs or not vs.is_ready():
        await update.message.reply_text("Vector store not available.")
        return
    results = vs.query(query_text, n_results=5)
    if not results:
        await update.message.reply_text("No results found.")
        return
    lines = [f"*Results for:* _{query_text}_\n"]
    for i, res in enumerate(results, 1):
        src = res["metadata"].get("source", "unknown")
        dist = res["distance"]
        text_preview = res["text"][:200].replace("\n", " ")
        lines.append(f"{i}. *{src.upper()}* (dist: {dist:.3f}): {text_preview}...")
    msg = "\n\n".join(lines)
    if len(msg) > 4000:
        msg = msg[:3950] + "\n\n...(truncated)"
    await update.message.reply_text(msg, parse_mode="Markdown")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Unknown command. Type /help for help.")


def _signal_handler(sig, frame):
    logging.info(f"Signal {sig} received. Shutting down...")
    sys.exit(0)


def main() -> None:
    if not load_config():
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    token = config["telegram_token"]
    controller = ModelController()
    vector_store = VectorStoreClient()

    application = Application.builder().token(token).build()
    application.bot_data["controller"] = controller
    application.bot_data["vector_store"] = vector_store

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CommandHandler("pwd", cmd_pwd))
    application.add_handler(CommandHandler("cd", cmd_cd))
    application.add_handler(CommandHandler("ls", cmd_ls))
    application.add_handler(CommandHandler("read", cmd_read))
    application.add_handler(CommandHandler("control", cmd_control))
    application.add_handler(CommandHandler("query", cmd_query))

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("run", cmd_run_entry)],
        states={
            CONFIRM_EXECUTE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_run_confirm),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_run_cancel)],
        conversation_timeout=60,
    )
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logging.info("NAMI starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
