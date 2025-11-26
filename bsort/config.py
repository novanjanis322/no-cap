"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Args:
        config_path: Path to the YAML configuration file.
    Returns:
        Dictionary containing configuration parameters.
    Raises:
        FileNotFoundError: If configuration file does not exist.
        yaml.YAMLError: If configuration file is not valid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to YAML file.
    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the configuration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
