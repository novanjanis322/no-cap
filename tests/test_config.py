"""Unit tests for configuration module."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from bsort.config import load_config, save_config


class TestConfig:
    """Test cases for configuration loading and saving."""

    def test_load_config_success(self):
        """Test successful configuration loading."""
        config_data = {
            "model": {"architecture": "yolov8n"},
            "training": {"epochs": 100},
        }
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loaded_config = load_config(config_path)
            assert loaded_config["model"]["architecture"] == "yolov8n"
            assert loaded_config["training"]["epochs"] == 100
        finally:
            config_path.unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))

    def test_save_config(self, tmp_path):
        """Test configuration saving."""
        config_data = {"test_key": "test_value", "nested": {"key": 123}}
        output_path = tmp_path / "test_config.yaml"

        save_config(config_data, output_path)

        assert output_path.exists()
        loaded_config = load_config(output_path)
        assert loaded_config["test_key"] == "test_value"
        assert loaded_config["nested"]["key"] == 123
