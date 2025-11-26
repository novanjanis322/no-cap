"""Unit tests for utility functions."""

import cv2
import numpy as np
import pytest

from bsort.utils.color_classifier import ColorClassifier


class TestColorClassifier:
    """Test cases for ColorClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create a color classifier instance."""
        return ColorClassifier()

    def test_light_blue_classification(self, classifier):
        """Test classification of light blue color."""  # Create a light blue image crop
        # HSV: [110, 150, 200] - blue hue, high saturation, high value (bright)
        light_blue_crop = np.full((50, 50, 3), [110, 150, 200], dtype=np.uint8)
        light_blue_crop = cv2.cvtColor(light_blue_crop, cv2.COLOR_HSV2BGR)
        class_id = classifier.classify_color(light_blue_crop)
        assert class_id == 0, "Light blue should be classified as class 0"

    def test_dark_blue_classification(self, classifier):
        """Test classification of dark blue color."""  # Create a dark blue image crop
        # HSV: [110, 150, 100] - blue hue, high saturation, low value (dark)
        dark_blue_crop = np.full((50, 50, 3), [110, 150, 100], dtype=np.uint8)
        dark_blue_crop = cv2.cvtColor(dark_blue_crop, cv2.COLOR_HSV2BGR)
        class_id = classifier.classify_color(dark_blue_crop)
        assert class_id == 1, "Dark blue should be classified as class 1"

    def test_other_color_classification(self, classifier):
        """Test classification of other colors."""  # Create a red image crop
        red_crop = np.full((50, 50, 3), [0, 255, 255], dtype=np.uint8)
        red_crop = cv2.cvtColor(red_crop, cv2.COLOR_HSV2BGR)
        class_id = classifier.classify_color(red_crop)
        assert class_id == 2, "Red should be classified as class 2 (others)"

    def test_get_color_name(self, classifier):
        """Test color name retrieval."""
        assert classifier.get_color_name(0) == "light_blue"
        assert classifier.get_color_name(1) == "dark_blue"
        assert classifier.get_color_name(2) == "others"
