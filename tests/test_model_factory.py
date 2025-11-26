"""Unit tests for model factory."""

import pytest

from bsort.models.model_factory import ModelFactory


class TestModelFactory:
    """Test cases for ModelFactory."""

    def test_create_yolov8n_model(self):
        """Test creating YOLOv8n model."""
        config = {
            "architecture": "yolov8n",
            "pretrained": True,
            "num_classes": 3,
        }
        model = ModelFactory.create_model(config)
        assert model is not None

    def test_create_yolov8s_model(self):
        """Test creating YOLOv8s model."""
        config = {
            "architecture": "yolov8s",
            "pretrained": True,
            "num_classes": 3,
        }
        model = ModelFactory.create_model(config)
        assert model is not None

    def test_create_default_model(self):
        """Test creating model with defaults."""
        config = {}
        model = ModelFactory.create_model(config)
        assert model is not None
