"""Color classification utilities for bottle caps."""

from typing import Tuple

import cv2
import numpy as np


class ColorClassifier:
    """Classifier for bottle cap colors (light blue, dark blue, others).

    Note: OpenCV HSV uses H: 0-179, S: 0-255, V: 0-255.
    Blue colors typically have hue values around 100-130 in OpenCV HSV.
    These default thresholds should be adjusted based on actual dataset analysis.
    Run scripts/analyze_colors.py to determine optimal thresholds.
    """

    def __init__(
        self,
        light_blue_hue_range: Tuple[int, int] = (90, 130),
        light_blue_sat_min: int = 50,
        light_blue_val_min: int = 150,
        dark_blue_hue_range: Tuple[int, int] = (90, 130),
        dark_blue_sat_min: int = 50,
        dark_blue_val_max: int = 149,
    ):
        """Initialize color classifier with HSV thresholds.

        Args:
            light_blue_hue_range: Hue range for light blue (OpenCV uses 0-179 for hue).
            light_blue_sat_min: Minimum saturation for light blue.
            light_blue_val_min: Minimum value (brightness) for light blue.
            dark_blue_hue_range: Hue range for dark blue.
            dark_blue_sat_min: Minimum saturation for dark blue.
            dark_blue_val_max: Maximum value (brightness) for dark blue.
        """
        self.light_blue_hue_range = light_blue_hue_range
        self.light_blue_sat_min = light_blue_sat_min
        self.light_blue_val_min = light_blue_val_min
        self.dark_blue_hue_range = dark_blue_hue_range
        self.dark_blue_sat_min = dark_blue_sat_min
        self.dark_blue_val_max = dark_blue_val_max

    def classify_color(self, image_crop: np.ndarray) -> int:
        """Classify the dominant color of a bottle cap crop.
        Args:
            image_crop: BGR image crop of the bottle cap.
        Returns:
            Class label: 0 for light_blue, 1 for dark_blue, 2 for others.
        """
        hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        mean_hue = np.mean(hsv[:, :, 0])
        mean_sat = np.mean(hsv[:, :, 1])
        mean_val = np.mean(hsv[:, :, 2])
        if (
            self.light_blue_hue_range[0] <= mean_hue <= self.light_blue_hue_range[1]
            and mean_sat >= self.light_blue_sat_min
            and mean_val >= self.light_blue_val_min
        ):
            return 0
        if (
            self.dark_blue_hue_range[0] <= mean_hue <= self.dark_blue_hue_range[1]
            and mean_sat >= self.dark_blue_sat_min
            and mean_val <= self.dark_blue_val_max
        ):
            return 1
        return 2

    def get_color_name(self, class_id: int) -> str:
        """
        Get color name from class ID.
        Args:
            class_id: Class identifier (0, 1, or 2).
        Returns:
            Color name string.
        """
        colors = {0: "light_blue", 1: "dark_blue", 2: "others"}
        return colors.get(class_id, "unknown")
