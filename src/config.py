from dotenv import load_dotenv
load_dotenv() # Loading vars from .env

import os
import os.path as osp

from pathlib import Path
from pprint import pprint

def get_project_root() -> Path:
    """Finds the root by looking for a marker file."""
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    return current_path.parent

class CFG:
    PROJECT_ROOT            = get_project_root()
    LOGS_DIR                = osp.join(PROJECT_ROOT, "logs")
    DATA_DIR                = osp.join(PROJECT_ROOT, "data")
    PROCESSED_DATA_DIR      = osp.join(DATA_DIR, "processed")
    SEED_VAL                = 2026
    DEBUG_MODE              = True

    # model
    MODEL_ID                = os.getenv("BASE_MODEL", "google/gemma-4-E2B-it")
    MAX_NEW_TOKENS          = 1024
    DEVICE_MAP              = "auto"
    MODEL_CTX_SIZE          = os.getenv("CTX_SIZE", 32_000)
    MODEL_PORT              = os.getenv("MODEL_PORT", "8080")
    STREAMING               = True

