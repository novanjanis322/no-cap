"""Unit tests for visualizer."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from bsort.inference.visualizer import Visualizer


class TestVisualizer:
    """Test cases for Visualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance."""
        config = {
            "output": {"box_thickness": 2, "font_scale": 0.5},
        }
        return Visualizer(config)

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "light_blue",
            },
            {
                "bbox": [300, 300, 400, 400],
                "confidence": 0.87,
                "class_id": 1,
                "class_name": "dark_blue",
            },
        ]

    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.box_thickness == 2
        assert visualizer.font_scale == 0.5
        assert len(visualizer.color_map) == 3

    def test_draw_predictions(
        self, visualizer, sample_image, sample_detections, tmp_path
    ):
        """Test drawing predictions on image."""
        output_path = tmp_path / "output.jpg"
        visualizer.draw_predictions(sample_image, sample_detections, str(output_path))

        assert output_path.exists()
        output_image = cv2.imread(str(output_path))
        assert output_image is not None
        assert output_image.shape == sample_image.shape
