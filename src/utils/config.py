from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments
    raise SystemExit(
        "Missing dependency: PyYAML. Install project dependencies first with:\n"
        "  pip install -r requirements.txt"
    ) from exc


class ConfigError(ValueError):
    """Raised when an experiment config is missing required fields."""


REQUIRED_BY_MODE: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "experiment": ("name", "seed", "output_dir"),
        "model": ("type", "backbone", "num_classes"),
        "data": ("dataset", "data_dir", "image_size", "batch_size", "num_workers"),
        "training": ("epochs", "learning_rate", "weight_decay", "optimizer"),
    },
    "evaluate": {
        "experiment": ("name", "seed", "output_dir"),
        "model": ("type", "backbone", "num_classes"),
        "data": ("dataset", "data_dir", "image_size", "batch_size", "num_workers"),
    },
    "analyze": {
        "experiment": ("name", "seed", "output_dir"),
        "model": ("type", "backbone", "num_classes"),
        "analysis": ("layers_to_analyze",),
    },
    "smoke": {
        "experiment": ("name", "seed", "output_dir"),
        "model": ("type", "backbone", "num_classes"),
    },
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")

    config = copy.deepcopy(config)
    config["_config_path"] = str(path)
    return config


def validate_config(config: dict[str, Any], mode: str) -> None:
    if mode == "all":
        modes = ("train", "evaluate", "analyze")
    else:
        modes = (mode,)

    for selected_mode in modes:
        required_sections = REQUIRED_BY_MODE.get(selected_mode)
        if required_sections is None:
            raise ConfigError(f"Unknown mode for config validation: {selected_mode}")

        for section, keys in required_sections.items():
            value = config.get(section)
            if not isinstance(value, dict):
                raise ConfigError(f"Missing or invalid config section: {section}")

            missing = [key for key in keys if key not in value]
            if missing:
                missing_text = ", ".join(missing)
                raise ConfigError(f"Missing config key(s) in {section}: {missing_text}")


def get_output_dir(config: dict[str, Any]) -> Path:
    output_dir = config.get("experiment", {}).get("output_dir")
    if not output_dir:
        raise ConfigError("Missing experiment.output_dir")
    return Path(output_dir)


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    snapshot_path = output_path / "config_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return snapshot_path
