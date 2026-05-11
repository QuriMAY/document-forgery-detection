import datetime
import logging
import os
import platform
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Default log directory. Resolved lazily inside setup_logging() so tests and
# containers can override it via the LOG_DIR env var.
LOG_DIR = Path("logs")


def _resolve_log_dir() -> Path:
    return Path(os.environ.get("LOG_DIR", str(LOG_DIR)))

# Shared format used by both console and file handlers
_FMT      = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(script_name: str, level: str = "INFO") -> Path:
    """
    Configure logging for a script run and load .env into the process environment.

    Creates:
        logs/<script_name>/<script_name>_YYYY-MM-DD_HH-MM-SS.log

    Console  → INFO and above (clean, readable)
    Log file → DEBUG and above (everything, for post-run inspection)

    Returns the path to the log file.
    """
    # Load .env before anything else so env vars are available to all code that follows
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)  # override=False: real env vars win

    script_log_dir = _resolve_log_dir() / script_name
    script_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file  = script_log_dir / f"{script_name}_{timestamp}.log"

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Write a header to the log file so each run is easy to identify
    header = _build_header(script_name, log_file)
    for line in header.splitlines():
        logging.getLogger(script_name).info(line)

    return log_file


def _build_header(script_name: str, log_file: Path) -> str:
    sep = "=" * 70
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"\n{sep}\n"
        f"  Script  : {script_name}\n"
        f"  Started : {now}\n"
        f"  Log     : {log_file}\n"
        f"  Python  : {sys.version.split()[0]}  |  Platform: {platform.platform()}\n"
        f"{sep}"
    )


def log_config(cfg: dict, logger: logging.Logger) -> None:
    """Pretty-print a config dict to the logger at INFO level."""
    logger.info("─── Configuration ───────────────────────────────────────────")
    _log_dict(cfg, logger, indent=0)
    logger.info("─────────────────────────────────────────────────────────────")


def _log_dict(d: dict, logger: logging.Logger, indent: int) -> None:
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            logger.info(f"{pad}{k}:")
            _log_dict(v, logger, indent + 1)
        else:
            logger.info(f"{pad}{k}: {v}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
