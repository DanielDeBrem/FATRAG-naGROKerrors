import json
import os
from typing import Any, Dict, Optional, List
from datetime import datetime

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
BACKUP_DIR = os.path.join(CONFIG_DIR, "backups")


def _ensure_backup_dir() -> None:
    os.makedirs(BACKUP_DIR, exist_ok=True)


def _ensure_dirs() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)


def default_config() -> Dict[str, Any]:
    # Defaults mirror main.py env-based defaults
    return {
        "LLM_MODEL": os.getenv("OLLAMA_LLM_MODEL", "llama3.1:70b"),
        "EMBED_MODEL": os.getenv("OLLAMA_EMBED_MODEL", "gemma2:2b"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "CHROMA_DIR": os.getenv("CHROMA_DIR", "./fatrag_chroma_db"),
        "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION", "fatrag"),
        "RETRIEVER_K": 5,
        "TEMPERATURE": 0.7,
        "FEEDBACK_ENABLED": True,
    }


def load_config() -> Dict[str, Any]:
    cfg = default_config()
    try:
        if os.path.isfile(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                disk = json.load(f)
            # Only merge known keys to avoid arbitrary file injection
            for k in cfg.keys():
                if k in disk:
                    cfg[k] = disk[k]
    except Exception:
        # Corrupt config should not break the server; fallback to defaults
        pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    # Persist only known keys
    current = load_config()
    for k in current.keys():
        if k in cfg:
            current[k] = cfg[k]
    _ensure_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)


def backup_config(label: Optional[str] = None) -> str:
    """
    Create a timestamped backup snapshot of the current on-disk config
    and return the backup filename.
    """
    _ensure_dirs()
    _ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_label = f"-{label}" if (label and label.strip()) else ""
    name = f"config-{timestamp}{safe_label}.json"
    path = os.path.join(BACKUP_DIR, name)
    # Use load_config to persist only known keys in snapshot
    snapshot = load_config()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return name


def list_backups() -> List[Dict[str, Any]]:
    """
    List available backup snapshots with name, size, and mtime (epoch seconds).
    """
    _ensure_backup_dir()
    items: List[Dict[str, Any]] = []
    try:
        for fname in os.listdir(BACKUP_DIR):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(BACKUP_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            st = os.stat(fpath)
            items.append({
                "name": fname,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            })
        # Newest first
        items.sort(key=lambda x: x["mtime"], reverse=True)
    except Exception:
        # On any error, return what we have
        pass
    return items


def restore_backup(name: str) -> None:
    """
    Restore a backup by name back into config.json using save_config to
    enforce known-key filtering.
    """
    if not name or "/" in name or "\\" in name:
        raise ValueError("Invalid backup name")
    _ensure_backup_dir()
    bpath = os.path.join(BACKUP_DIR, name)
    if not os.path.isfile(bpath):
        raise FileNotFoundError("Backup not found")
    with open(bpath, "r", encoding="utf-8") as f:
        disk = json.load(f)
    # Persist only known keys using save_config
    save_config(disk)


def update_runtime_from_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional: overlay runtime env on config (for containerized overrides).
    Only applies to keys we recognize; returns a new merged dict.
    """
    merged = dict(cfg)
    env_overrides = {
        "LLM_MODEL": os.getenv("OLLAMA_LLM_MODEL"),
        "EMBED_MODEL": os.getenv("OLLAMA_EMBED_MODEL"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
        "CHROMA_DIR": os.getenv("CHROMA_DIR"),
        "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION"),
    }
    for k, v in env_overrides.items():
        if v:
            merged[k] = v
    return merged
