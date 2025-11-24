"""
MySQL-based configuration store for FATRAG
Replaces file-based config_store.py
"""
import os
from typing import Any, Dict, Optional, List
from datetime import datetime
from db_models import SessionLocal, Config
from sqlalchemy import text
import json

def _parse_value(value: str, value_type: str) -> Any:
    """Convert string value to appropriate Python type"""
    if value_type == "int":
        return int(value)
    elif value_type == "float":
        return float(value)
    elif value_type == "bool":
        return value.lower() in ("true", "1", "yes")
    elif value_type == "json":
        return json.loads(value)
    return value  # string


def _serialize_value(value: Any) -> tuple[str, str]:
    """Convert Python value to string and determine type"""
    if isinstance(value, bool):
        return (str(value).lower(), "bool")
    elif isinstance(value, int):
        return (str(value), "int")
    elif isinstance(value, float):
        return (str(value), "float")
    elif isinstance(value, (dict, list)):
        return (json.dumps(value), "json")
    return (str(value), "string")


def default_config() -> Dict[str, Any]:
    """Return default configuration values"""
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
    """Load configuration from MySQL database"""
    cfg = default_config()
    
    try:
        db = SessionLocal()
        try:
            # Get all config entries
            config_entries = db.query(Config).all()
            
            for entry in config_entries:
                if entry.config_key in cfg:
                    cfg[entry.config_key] = _parse_value(entry.config_value, entry.value_type)
        finally:
            db.close()
    except Exception as e:
        # If database fails, return defaults
        print(f"Warning: Could not load config from database: {e}")
        pass
    
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """Save configuration to MySQL database"""
    db = SessionLocal()
    try:
        # Get current defaults to know which keys are valid
        valid_keys = default_config().keys()
        
        for key in valid_keys:
            if key in cfg:
                value_str, value_type = _serialize_value(cfg[key])
                
                # Check if config key exists
                existing = db.query(Config).filter(Config.config_key == key).first()
                
                if existing:
                    # Update existing
                    existing.config_value = value_str
                    existing.value_type = value_type
                else:
                    # Insert new
                    new_config = Config(
                        config_key=key,
                        config_value=value_str,
                        value_type=value_type,
                        description=f"Configuration for {key}"
                    )
                    db.add(new_config)
        
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def backup_config(label: Optional[str] = None) -> str:
    """
    Create a backup snapshot by creating an audit log entry.
    Returns a timestamp-based identifier.
    """
    from db_models import AuditLog
    
    db = SessionLocal()
    try:
        # Get current config
        current_config = load_config()
        
        # Create audit log entry
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label_str = f"-{label}" if label else ""
        backup_name = f"config-{timestamp}{label_str}"
        
        audit_entry = AuditLog(
            event="config_backup",
            event_type="info",
            payload={
                "backup_name": backup_name,
                "config": current_config,
                "label": label
            }
        )
        db.add(audit_entry)
        db.commit()
        
        return backup_name
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def list_backups() -> List[Dict[str, Any]]:
    """List available config backups from audit log"""
    from db_models import AuditLog
    
    db = SessionLocal()
    try:
        backups = db.query(AuditLog).filter(
            AuditLog.event == "config_backup"
        ).order_by(AuditLog.created_at.desc()).all()
        
        items = []
        for backup in backups:
            payload = backup.payload or {}
            items.append({
                "name": payload.get("backup_name", f"backup-{backup.id}"),
                "size": len(json.dumps(payload)),
                "mtime": int(backup.created_at.timestamp()) if backup.created_at else 0,
            })
        
        return items
    except Exception as e:
        print(f"Error listing backups: {e}")
        return []
    finally:
        db.close()


def restore_backup(name: str) -> None:
    """Restore configuration from backup"""
    from db_models import AuditLog
    
    db = SessionLocal()
    try:
        # Find backup by name
        backup = db.query(AuditLog).filter(
            AuditLog.event == "config_backup"
        ).order_by(AuditLog.created_at.desc()).all()
        
        target_backup = None
        for b in backup:
            payload = b.payload or {}
            if payload.get("backup_name") == name:
                target_backup = payload.get("config")
                break
        
        if not target_backup:
            raise FileNotFoundError(f"Backup {name} not found")
        
        # Restore config
        save_config(target_backup)
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def update_runtime_from_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overlay runtime environment variables on config.
    Returns merged configuration dictionary.
    
    IMPORTANT: Only applies env overrides if config value matches default.
    This ensures UI-configured values take precedence over .env settings.
    """
    merged = dict(cfg)
    defaults = default_config()
    
    env_overrides = {
        "LLM_MODEL": os.getenv("OLLAMA_LLM_MODEL"),
        "EMBED_MODEL": os.getenv("OLLAMA_EMBED_MODEL"),
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL"),
        "CHROMA_DIR": os.getenv("CHROMA_DIR"),
        "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION"),
    }
    
    for k, v in env_overrides.items():
        if v:
            # Only override if config value equals default (meaning user hasn't customized it via UI)
            # This gives UI configuration priority over environment variables
            if k in defaults and cfg.get(k) == defaults[k]:
                merged[k] = v
    
    return merged
