"""Visualization utilities for detection results."""

from typing import Any, Dict, List

import cv2
import numpy as np


class Visualizer:
    """Visualizer for drawing detection results on images."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        output_cfg = config.get("output", {})
        self.box_thickness = output_cfg.get("box_thickness", 2)
        self.font_scale = output_cfg.get("font_scale", 0.5)
        # Color map for classes
        self.color_map = {
            0: (255, 200, 100),  # Light blue in BGR
            1: (139, 69, 19),  # Dark blue in BGR
            2: (128, 128, 128),  # Gray for others
        }

    def draw_predictions(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Draw bounding boxes and labels on image.

        Args:
            image: Input image in BGR format.
            detections: List of detection dictionaries.
            output_path: Path to save the output image.
        """
        output_image = image.copy()
        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            # Get color
            color = self.color_map.get(class_id, (255, 255, 255))
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, self.box_thickness)
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            # Draw label background
            cv2.rectangle(
                output_image,
                (x1, y1 - label_size[1] - 5),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            # Draw label text
            cv2.putText(
                output_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
            )
        # Save output
        cv2.imwrite(output_path, output_image)
        print(f"Output saved to {output_path}")
