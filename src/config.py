"""Config loading. Paths in config.yaml are relative to the project root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a cfg['paths'][key] entry to an absolute path."""
    return ROOT / cfg["paths"][key]
